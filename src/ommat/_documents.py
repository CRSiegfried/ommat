"""Stored documents as the lens and the outline read them: one strict JSON object per document,
each classified as a record or a design, and the files under a folder walked in one order.

``read_document`` reads bytes as ommat reads a record's bytes. ``kind_of`` calls a JSON object
with a ``log`` list a record and one with a ``rules`` object and no ``log`` list a design.
``gather`` reads files, a ``.jsonl`` file line by line, skips anything else with its reason, and
keys each design by the SHA-256 of its exact bytes, the hash a record's ``design`` item holds.
"""

import hashlib
import json
import math
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

from ommat._canonical import integer_literal
from ommat._values import MAX_DOCUMENT_DEPTH


def read_document(document: object, what: str) -> dict[str, Any]:
    """``document`` read as ommat reads a record's bytes, one JSON object in strict UTF-8 with no
    byte-order mark, no repeated key, only finite numbers and no deeper nesting than a record
    holds, else ``ValueError`` naming ``what``."""
    if type(document) is not bytes:
        raise ValueError(f"{what} must be bytes, not {type(document).__name__}")
    try:
        source = document.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{what} is not UTF-8 text") from None
    try:
        value: object = json.loads(
            source,
            parse_constant=_refuse_constant,
            parse_float=_finite,
            parse_int=_exact,
            object_pairs_hook=_unrepeated,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{what} is not JSON: {error}") from None
    except RecursionError:
        raise ValueError(f"{what} nests deeper than {MAX_DOCUMENT_DEPTH}") from None
    except ValueError as error:  # an integer over Python's digit limit, or a repeated key
        raise ValueError(f"{what} is not readable JSON: {error}") from None
    # Depth first: whether the parse above reached the recursion limit depends on the C stack,
    # so a deep document must be refused here whatever its top-level type.
    if _depth(value) > MAX_DOCUMENT_DEPTH:
        raise ValueError(f"{what} nests deeper than {MAX_DOCUMENT_DEPTH}")
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return cast("dict[str, Any]", value)


def kind_of(value: dict[str, Any]) -> Literal["record", "design"] | None:
    """``record`` for a JSON object holding a ``log`` list, ``design`` for one holding a
    ``rules`` object and no ``log`` list, else ``None``."""
    if isinstance(value.get("log"), list):
        return "record"
    if isinstance(value.get("rules"), dict):
        return "design"
    return None


def design_key(document: bytes) -> str:
    """The hash a record's ``design`` item holds for the design document ``document``."""
    return "sha256:" + hashlib.sha256(document).hexdigest()


def design_hash(record: dict[str, Any]) -> object:
    """The hash in the record's first ``design`` item, or ``None``."""
    log = cast("list[object]", record.get("log") or [])
    for item in log:
        held = cast("dict[str, object]", item) if isinstance(item, dict) else {}
        if held.get("item") == "design":
            found = held.get("hash")
            return found if isinstance(found, str) else None
    return None


def walk(folder: Path, skipped: list[str]) -> list[Path]:
    """Every .json and .jsonl file under ``folder``, by relative path; a folder it cannot open
    is skipped and named."""

    def refused(error: OSError) -> None:
        where = Path(error.filename or folder)
        name = where.relative_to(folder).as_posix() if where != folder else "."
        print(f"skipped {name}/: {error.strerror or error}", file=sys.stderr)
        skipped.append(name + "/")

    found = [
        Path(root) / name
        for root, _, names in os.walk(folder, onerror=refused, followlinks=False)
        for name in names
        if Path(name).suffix.lower() in {".json", ".jsonl"} and (Path(root) / name).is_file()
    ]
    return sorted(found, key=lambda p: p.relative_to(folder).as_posix())


def identity(file: Path) -> tuple[int, int] | None:
    """The file's device and inode, so a symbolic or hard link to it is the same file."""
    try:
        held = file.stat()
    except OSError:
        return None
    return held.st_dev, held.st_ino


def gather(
    found: Iterable[tuple[Path, str]], skipped: list[str]
) -> tuple[dict[str, bytes], dict[str, tuple[str, bytes]]]:
    """The records, by name, and the designs, by ``design_key``, each with its name, in the
    files ``found``, each with its name; a ``.jsonl`` file's lines read one by one as
    ``<name> · line <n>``. A file reached twice is read once. A file it cannot read and a
    document that is neither is skipped, named on stderr with the reason and in ``skipped``."""
    records: dict[str, bytes] = {}
    designs: dict[str, tuple[str, bytes]] = {}
    seen: set[object] = set()
    for file, name in found:
        held = identity(file)
        if held is not None and held in seen:
            continue  # a link to a file already read
        seen.add(held)
        try:
            data = file.read_bytes()
        except OSError as error:
            print(f"skipped {name}: {error.strerror or error}", file=sys.stderr)
            skipped.append(name)
            continue
        lines = (
            [
                (f"{name} · line {n}", line.strip())
                for n, line in enumerate(data.split(b"\n"), 1)
                if line.strip()
            ]
            if file.suffix.lower() == ".jsonl"
            else [(name, data)]
        )
        for label, document in lines:
            try:
                value = read_document(document, label)
            except ValueError as error:
                reason = str(error)
            else:
                kind = kind_of(value)
                if kind == "record":
                    records[label] = document
                    continue
                if kind == "design":
                    designs[design_key(document)] = (label, document)
                    continue
                reason = f"{label} is not an ommat record or design"
            print(f"skipped {reason}", file=sys.stderr)
            skipped.append(label)
    return records, designs


def note_unpaired(records: dict[str, bytes], designs: dict[str, tuple[str, bytes]]) -> None:
    """Name on stderr each design no record pairs with, when there is a record."""
    used = {design_hash(read_document(document, "record")) for document in records.values()}
    for key, (label, _) in designs.items():
        if records and key not in used:
            print(
                f"note: design {label} matches no record; a record pairs with its design's exact "
                "bytes, as loaded.document gives them",
                file=sys.stderr,
            )


def _refuse_constant(name: str) -> object:
    raise json.JSONDecodeError(f"{name} is not a JSON number", name, 0)


def _finite(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"{text} is out of a JSON number's range")
    return number


def _exact(text: str) -> int | float:
    number = integer_literal(text)  # ommat's own rule for an integer past 2^53 - 1
    if number is None:
        raise ValueError(f"the integer {text} names no double exactly")
    return number


def _unrepeated(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    held = dict(pairs)
    if len(held) != len(pairs):
        seen: set[str] = set()
        repeated = next(key for key, _ in pairs if key in seen or seen.add(key))
        raise ValueError(f"the key {repeated!r} is repeated")
    return held


def _depth(value: object) -> int:
    deepest, stack = 0, [(value, 1)]
    while stack:
        held, depth = stack.pop()
        children: list[object] = []
        if isinstance(held, dict):
            children = list(cast("dict[str, object]", held).values())
        elif isinstance(held, list):
            children = cast("list[object]", held)
        else:
            continue
        deepest = max(deepest, depth)
        stack.extend((child, depth + 1) for child in children)
    return deepest
