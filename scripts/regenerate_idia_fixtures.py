"""Rebuild, offline, the files ommat derives from its recorded idia logs: the end-to-end turn's
record document and the idia tutorial's printed output.

Run it from the repository root::

    uv run python scripts/regenerate_idia_fixtures.py

It replays each turn from the logs in ``tests/idia_logs`` with idia's ``Recorded`` leaf in mode
``none``, under placeholder credentials that are never sent, so it makes no wire call and reads
no credential. Run it when ommat's record format or the tutorial's printing changes. When the
wire input changes, the design, a prompt's form or the leaf, the logs themselves are recorded
again live, with ``scripts/record_idia_e2e.py`` and ``scripts/record_idia_tutorial.py``.
When idia's log changes form, its ``upgrade_log`` in ``idia.providers.recorded`` brings each log
up offline, editing it in place, before this script runs. The record document is rewritten only
when its hash or hashed log changed, its ``unhashed`` part not compared, so a run with nothing
to change keeps the timings of whichever run wrote it. A changed record hash changes the
``answered from record`` lines, so copy the new output into the tutorial page's printed blocks.
"""

import asyncio
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT / "tests"),
    str(ROOT / "docs/tutorials/street_reports"),
    str(ROOT / "docs/tutorials/decide_with_idia"),
]

import decide_live  # noqa: E402
import idia_e2e as e2e  # noqa: E402
from ommat import InvalidRecord, read_record  # noqa: E402


def main() -> None:
    record = e2e.decide(e2e.PLACEHOLDER, "none").record
    try:
        kept = read_record(e2e.RECORD.read_bytes()) if e2e.RECORD.exists() else None
    except InvalidRecord:  # a stored record that no longer reads is replaced
        kept = None
    if kept is None or (kept.record_hash, kept.log) != (record.record_hash, record.log):
        e2e.RECORD.write_bytes(record.document)
        record_state = "rewritten"
    else:
        record_state = "kept"
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        asyncio.run(decide_live.run(e2e.recorded_providers(e2e.PLACEHOLDER, "none")))
    output = e2e.TUTORIAL / "output.txt"
    text = printed.getvalue()
    output_state = "unchanged" if output.exists() and output.read_text("utf-8") == text else ""
    if not output_state:
        output.write_text(text, encoding="utf-8")
        output_state = "rewritten"
    print(f"{e2e.RECORD.relative_to(ROOT)}: {record_state}, {record.record_hash}")
    print(f"{output.relative_to(ROOT)}: {output_state}")


if __name__ == "__main__":
    main()
