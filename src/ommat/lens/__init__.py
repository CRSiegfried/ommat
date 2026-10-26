"""The record lens: stored records and designs rendered as one HTML page a person reads, outside
the compatibility promise while it is validated.

``page(records, designs=())`` returns a self-contained page with three views. Design shows a
design document with or without a record: its targets, questions, rules, policy reads,
requirements and parameters, and the lineage of any card, a target's declared links included.
Turn shows each record round by round: its calls and entries, each choice's probabilities and an
extraction's confidence as percentages against the boundaries its design declares, each marker by
kind and cause, then the plan and the verdict, and the lineage of any entry, call or the plan: the
links the record holds, and, with the design the record ran under, each rule's and the policy's
declared reads, marked as declared. Turns lists the records grouped by design hash or by a
question's identity. A design pairs with a record by its hash. ``python -m ommat.lens PATH``
writes the page for a record or design file or a folder.

The page embeds the records whole, user text included, so keep it as you keep them, and loads
nothing from the network. ``import ommat`` does not import this package, and it takes nothing
beyond the standard library.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from importlib.resources import files
from typing import Any, cast

from ommat._canonical import integer_literal
from ommat._values import MAX_DOCUMENT_DEPTH

__all__ = ["page"]

_SLOT = "__OMMAT_LENS_DATA__"


def page(
    records: Sequence[bytes] | Mapping[str, bytes],
    designs: Sequence[bytes] = (),
    *,
    skipped: Sequence[str] = (),
) -> str:
    """The page for ``records``, record documents as ``bytes`` (``decision.record.document``),
    named by a mapping's keys or else ``record 1``, ``record 2`` and so on, with ``designs``,
    design documents as ``bytes`` (``loaded.document``), each paired with the records whose
    ``design`` item holds the SHA-256 of its exact bytes. ``skipped`` names files a caller passed
    over, which the page counts. Raises ``ValueError`` for an argument of the wrong type, a
    document that is not strict JSON, a record that is not a JSON object holding a ``log`` list,
    or a design that is not a JSON object."""
    named = (
        [(str(name), document) for name, document in records.items()]
        if isinstance(records, Mapping)
        else [(f"record {i}", r) for i, r in enumerate(_listed(records, "records"), 1)]
    )
    shown: list[dict[str, Any]] = []
    for name, document in named:
        value = read_document(document, name)
        if not isinstance(value.get("log"), list):
            raise ValueError(f"{name} is not a record document: it holds no 'log' list")
        shown.append({"name": name, "record": value})
    keyed: dict[str, Any] = {}
    for i, document in enumerate(_listed(designs, "designs"), 1):
        value = read_document(document, f"design {i}")
        keyed["sha256:" + hashlib.sha256(cast(bytes, document)).hexdigest()] = value
    names = _listed(skipped, "skipped")
    if not all(type(name) is str for name in names):
        raise ValueError("skipped must be a sequence of str")
    data = {"records": shown, "designs": keyed, "skipped": names, "read": len(shown)}
    # ASCII only, and every "<" escaped, so no text in a record can close the script element
    # and the page reads the same under any charset.
    embedded = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    return _template().replace(_SLOT, embedded, 1)


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
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    held = cast("dict[str, Any]", value)
    if _depth(held) > MAX_DOCUMENT_DEPTH:
        raise ValueError(f"{what} nests deeper than {MAX_DOCUMENT_DEPTH}")
    return held


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


def _listed(values: object, what: str) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{what} must be a sequence, not {type(values).__name__}")
    return list(cast("Iterable[Any]", values))


def _template() -> str:
    return files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
