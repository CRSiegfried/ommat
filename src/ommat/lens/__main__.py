"""``python -m ommat.lens PATH [-o FILE] [--open]``: write the lens page for a record or design
file, or for every ``.json`` and ``.jsonl`` record and design under a folder."""

import argparse
import sys
import webbrowser
from pathlib import Path

from ommat._documents import gather, identity, note_unpaired, walk
from ommat.lens import page


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ommat.lens",
        description=(
            "Write an HTML page drawing an ommat design and its records from their outlines: "
            "Gravity, the design as bodies with a turn's entries, lineage and rounds over it and "
            "a panel saying what the design and the turn say of each body, and Turn, a record "
            "in plain words. The page holds the records' values, user text included: "
            "keep it as you keep them."
        ),
    )
    parser.add_argument(
        "path", type=Path, help="a record or design file, or a folder of records and designs"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("ommat-lens.html"), help="the page to write"
    )
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    args = parser.parse_args(argv)

    path: Path = args.path
    output: Path = args.output
    skipped: list[str] = []
    if path.is_dir():
        found, base = walk(path, skipped), path
    elif path.is_file():
        found, base = [path], path.parent
    else:
        print(f"error: {path}: no such file or folder", file=sys.stderr)
        return 1
    if identity(output) in {identity(file) for file in found} - {None}:
        print(f"error: {output} is one of the files read; choose another output", file=sys.stderr)
        return 1

    records, designs = gather(
        ((file, file.relative_to(base).as_posix()) for file in found), skipped
    )

    if not records and not designs:
        print(f"error: {path}: no ommat record or design found", file=sys.stderr)
        return 1
    note_unpaired(records, designs)
    try:
        html = page(records, dict(designs.values()), skipped=skipped)
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


if __name__ == "__main__":
    sys.exit(main())
