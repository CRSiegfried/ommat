"""``python -m ommat.lens PATH [-o FILE] [--open]``: write the record lens page for a record
file, or for every ``.json`` and ``.jsonl`` record and design under a folder."""

import argparse
import hashlib
import os
import sys
import webbrowser
from pathlib import Path
from typing import cast

from ommat.lens import page, read_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ommat.lens",
        description=(
            "Write an HTML page showing ommat records round by round, with lineage. The page "
            "holds the records whole, user text included: keep it as you keep them."
        ),
    )
    parser.add_argument("path", type=Path, help="a record file, or a folder of records and designs")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("ommat-lens.html"), help="the page to write"
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    path: Path = args.path
    output: Path = args.output
    skipped: list[str] = []
    if path.is_dir():
        found, base = _walk(path, skipped), path
    elif path.is_file():
        found, base = [path], path.parent
    else:
        print(f"error: {path}: no such file or folder", file=sys.stderr)
        return 1
    if _identity(output) in {_identity(file) for file in found} - {None}:
        print(f"error: {output} is one of the files read; choose another output", file=sys.stderr)
        return 1

    records: dict[str, bytes] = {}
    designs: dict[str, tuple[str, bytes]] = {}
    seen: set[object] = set()
    for file in found:
        identity = _identity(file)
        if identity is not None and identity in seen:
            continue  # a link to a file already read
        seen.add(identity)
        name = file.relative_to(base).as_posix()
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
                if isinstance(value.get("log"), list):
                    records[label] = document
                    continue
                if "policy" in value:
                    designs["sha256:" + hashlib.sha256(document).hexdigest()] = (label, document)
                    continue
                reason = f"{label} is not an ommat record or design"
            print(f"skipped {reason}", file=sys.stderr)
            skipped.append(label)

    if not records:
        print(f"error: {path}: no ommat record found", file=sys.stderr)
        return 1
    used = {_design_hash(document) for document in records.values()}
    for key, (label, _) in designs.items():
        if key not in used:
            print(
                f"note: design {label} matches no record; a record pairs with its design's exact "
                "bytes, as loaded.document gives them",
                file=sys.stderr,
            )
    try:
        html = page(records, [document for _, document in designs.values()], skipped=skipped)
    except ValueError as error:  # read_document passed each document, so this is a defect
        print(f"error: cannot build the page: {error}", file=sys.stderr)
        return 1
    try:
        output.write_text(html, encoding="utf-8")
    except OSError as error:
        print(f"error: cannot write {output}: {error.strerror or error}", file=sys.stderr)
        return 1
    print(f"wrote {output} with {len(records)} record(s) and {len(designs)} design(s)")
    if args.open and not webbrowser.open(output.resolve().as_uri()):
        print(f"note: no browser opened; open {output} yourself", file=sys.stderr)
    return 0


def _walk(folder: Path, skipped: list[str]) -> list[Path]:
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


def _identity(file: Path) -> tuple[int, int] | None:
    """The file's device and inode, so a symbolic or hard link to it is the same file."""
    try:
        held = file.stat()
    except OSError:
        return None
    return held.st_dev, held.st_ino


def _design_hash(document: bytes) -> object:
    log: list[object] = read_document(document, "record")["log"]
    first = log[0] if log else None
    return cast("dict[str, object]", first).get("hash") if isinstance(first, dict) else None


if __name__ == "__main__":
    sys.exit(main())
