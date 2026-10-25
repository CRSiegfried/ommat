"""``python -m ommat.lens PATH [-o FILE] [--open]``: write the record lens page for a record
file, or for every ``.json`` and ``.jsonl`` record and design under a folder."""

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import cast

from ommat.lens import page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ommat.lens",
        description="Write an HTML page showing ommat records round by round, with lineage.",
    )
    parser.add_argument("path", type=Path, help="a record file, or a folder of records and designs")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("ommat-lens.html"), help="the page to write"
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    path: Path = args.path
    if path.is_dir():
        found = sorted(
            (p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"} and p.is_file()),
            key=lambda p: p.relative_to(path).as_posix(),
        )
        base = path
    elif path.is_file():
        found, base = [path], path.parent
    else:
        print(f"{path}: no such file or folder", file=sys.stderr)
        return 1

    records: dict[str, bytes] = {}
    designs: list[bytes] = []
    skipped: list[str] = []
    for file in found:
        name = file.relative_to(base).as_posix()
        data = file.read_bytes()
        lines = (
            [
                (f"{name} · line {n}", line)
                for n, line in enumerate(data.split(b"\n"), 1)
                if line.strip()
            ]
            if file.suffix.lower() == ".jsonl"
            else [(name, data)]
        )
        for label, document in lines:
            kind = _kind(document)
            if kind == "record":
                records[label] = document
            elif kind == "design":
                designs.append(document)
            else:
                skipped.append(label)

    for label in skipped:
        print(f"skipped {label}: not an ommat record or design", file=sys.stderr)
    if not records:
        print(f"{path}: no ommat record found", file=sys.stderr)
        return 1
    output: Path = args.output
    output.write_text(page(records, designs, skipped=skipped), encoding="utf-8")
    print(f"wrote {output} with {len(records)} record(s) and {len(designs)} design(s)")
    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


def _kind(document: bytes) -> str | None:
    try:
        value: object = json.loads(document)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    held = cast("dict[str, object]", value)
    if isinstance(held.get("log"), list):
        return "record"
    if "policy" in held and "log" not in held:
        return "design"
    return None


if __name__ == "__main__":
    sys.exit(main())
