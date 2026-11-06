"""``python -m ommat.outline PATH... [--json | --oneline] [--detail] [-o FILE]``: print the outline
of each record and design file, or of every ``.json`` and ``.jsonl`` record and design under a
folder."""

import argparse
import os
import sys
from pathlib import Path

from ommat._canonical import ordered
from ommat._documents import design_hash, gather, identity, note_unpaired, read_document, walk
from ommat.outline import design_outline, render_design, render_turn, turn_outline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ommat.outline",
        description=(
            "Print an ommat design or turn as text, one node, call or entry per line, or as "
            "JSON: each record beside the design it ran under. The text holds the records' "
            "values, user text included: keep it as you keep them."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="a record or design file, or a folder of records and designs",
    )
    form = parser.add_mutually_exclusive_group()
    form.add_argument(
        "--json", action="store_true", help="print each outline as one JSON object per line"
    )
    form.add_argument(
        "--oneline",
        action="store_true",
        help="print each record's summary line alone, and nothing for a design",
    )
    parser.add_argument(
        "--detail", action="store_true", help="print each call's detail members a line each"
    )
    parser.add_argument("-o", "--output", type=Path, help="write to this file, not stdout")
    args = parser.parse_args(argv)

    paths: list[Path] = args.paths
    output: Path | None = args.output
    skipped: list[str] = []
    found: list[tuple[Path, str]] = []
    for path in paths:
        if path.is_dir():
            found += [(f, (path / f.relative_to(path)).as_posix()) for f in walk(path, skipped)]
        elif path.is_file():
            found.append((path, path.as_posix()))
        else:
            print(f"error: {path}: no such file or folder", file=sys.stderr)
            return 1
    if output is not None and identity(output) in {identity(f) for f, _ in found} - {None}:
        print(f"error: {output} is one of the files read; choose another output", file=sys.stderr)
        return 1

    records, designs = gather(found, skipped)
    if not records and not designs:
        shown = " ".join(str(p) for p in paths)
        print(f"error: {shown}: no ommat record or design found", file=sys.stderr)
        return 1
    note_unpaired(records, designs)

    chunks: list[str] = []
    for label, document in designs.values() if not args.oneline else ():
        outline = design_outline(document)
        chunks.append(_shown(label, outline, render_design, json=args.json))
    for label, document in records.items():
        paired = designs.get(str(design_hash(read_document(document, label))))
        outline = turn_outline(document, None if paired is None else paired[1])
        if args.oneline:
            chunks.append(outline["summary"] + "\n")
            continue
        detail: bool = args.detail

        def turn_text(o: dict[str, object], detail: bool = detail) -> str:
            return render_turn(o, detail=detail)

        chunks.append(_shown(label, outline, turn_text, json=args.json))
    text = ("" if args.json or args.oneline else "\n").join(chunks)
    if output is None:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except BrokenPipeError:
            # The reader closed the pipe (``| head``): stop quietly. Point stdout at devnull so
            # the flush at exit does not fail again.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            return 1
        return 0
    try:
        output.write_text(text, encoding="utf-8")
    except OSError as error:
        print(f"error: cannot write {output}: {error.strerror or error}", file=sys.stderr)
        return 1
    return 0


def _shown(label: str, outline: dict[str, object], render: object, *, json: bool) -> str:
    if json:
        return ordered({"path": label, "outline": outline}) + "\n"
    assert callable(render)
    return f"== {label}\n{render(outline)}"


if __name__ == "__main__":
    sys.exit(main())
