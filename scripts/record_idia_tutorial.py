"""Record the idia tutorial's run live: ``docs/tutorials/decide_with_idia/decide_live.py``
over OpenAI's Decisions and Cloudflare's Clairic, each leaf wrapped in idia's ``Recorded``
leaf writing to ``tests/idia_logs/tutorial``.

Run it from the repository root with the credentials in the environment, and nowhere else::

    uv run python scripts/record_idia_tutorial.py

It reads the three variables ``record_idia_e2e.py`` reads and refuses to run without them,
replaces the two logs, keeps what the run printed as ``output.txt`` beside them for the
tutorial's text, then scans the logs and ``output.txt`` as that script does; a run that
raises deletes the logs.
"""

import asyncio
import contextlib
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT / "tests"),
    str(ROOT / "docs/tutorials/street_reports"),
    str(ROOT / "docs/tutorials/decide_with_idia"),
]

from record_idia_e2e import VARIABLES, masked, scrub  # noqa: E402

import decide_live  # noqa: E402
import idia_e2e as e2e  # noqa: E402

LOGS = (e2e.TUTORIAL / "classifier.json", e2e.TUTORIAL / "extractor.json")
OUTPUT = e2e.TUTORIAL / "output.txt"


def main() -> int:
    credentials = {name: os.environ.get(variable, "") for name, variable in VARIABLES.items()}
    missing = [VARIABLES[name] for name, value in credentials.items() if not value]
    if missing:
        print(f"set {', '.join(missing)} in the environment", file=sys.stderr)
        return 2
    for path in (*LOGS, OUTPUT):
        path.unlink(missing_ok=True)
    printed = io.StringIO()
    try:
        with contextlib.redirect_stdout(printed):
            asyncio.run(decide_live.run(e2e.recorded_providers(credentials, "once")))
    except Exception as exc:  # a failed run leaves no unscrubbed log behind
        for path in LOGS:
            path.unlink(missing_ok=True)
        print(masked(f"{type(exc).__name__}: {exc}", credentials), file=sys.stderr)
        return 1
    text = masked(printed.getvalue(), credentials)
    OUTPUT.write_text(text, encoding="utf-8")
    print(text, end="")
    problems: list[str] = []
    for path in (*LOGS, OUTPUT):
        if path.exists():
            problems += scrub(path, credentials)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        for path in (*LOGS, OUTPUT):
            path.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
