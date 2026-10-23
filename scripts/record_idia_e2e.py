"""Record the idia binding's end-to-end turn live: the street-reports design decided once,
OpenAI's Decisions on the classifier oracle and Cloudflare's Clairic on the extractor.

Run it from the repository root with the credentials in the environment, and nowhere else::

    uv run python scripts/record_idia_e2e.py

It writes the street-reports design's document beside the logs, then decides under it. It reads
``OPENAI_API_KEY``, ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN`` and refuses
to run without all three. It replaces ``tests/idia_logs``' two logs and the turn's record
document, then scans the logs and the record: in a log the account id is replaced by the
placeholder; a record holding it, whose hash covers its bytes, and any file holding a
credential, fail the run and the logs and the record are deleted. A run that raises deletes
them too. It prints what each call returned and never a credential.
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "docs/tutorials/street_reports")]

import idia_e2e as e2e  # noqa: E402
import street_reports_design as works  # noqa: E402
from ommat import load  # noqa: E402

VARIABLES = {
    "openai_key": "OPENAI_API_KEY",
    "account_id": "CLOUDFLARE_ACCOUNT_ID",
    "api_token": "CLOUDFLARE_API_TOKEN",
}
KEY_SHAPES = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}"),
)


def scrub(path: Path, credentials: dict[str, str]) -> list[str]:
    """Replace the account id in a log; the problems with what remains."""
    text = path.read_text(encoding="utf-8")
    account = credentials["account_id"]
    replaced = text.count(account)
    problems: list[str] = []
    if replaced and path == e2e.RECORD:
        problems.append(f"{path.name} holds the account id inside its hashed log")
    text = text.replace(account, e2e.PLACEHOLDER["account_id"])
    path.write_text(text, encoding="utf-8")
    for name in ("openai_key", "api_token"):
        if credentials[name] in text:
            problems.append(f"{path.name} holds the {VARIABLES[name]} value")
    problems += [f"{path.name} holds a key shape" for shape in KEY_SHAPES if shape.search(text)]
    print(f"{path.name}: account id replaced {replaced} time(s)")
    return problems


def masked(text: str, credentials: dict[str, str]) -> str:
    """``text`` with every credential and key shape replaced, as it may be printed."""
    for value in credentials.values():
        text = text.replace(value, "<credential>")
    for shape in KEY_SHAPES:
        text = shape.sub("<key>", text)
    return text


def main() -> int:
    credentials = {name: os.environ.get(variable, "") for name, variable in VARIABLES.items()}
    missing = [VARIABLES[name] for name, value in credentials.items() if not value]
    if missing:
        print(f"set {', '.join(missing)} in the environment", file=sys.stderr)
        return 2
    for path in (e2e.CLASSIFIER_LOG, e2e.EXTRACTOR_LOG, e2e.RECORD):
        path.unlink(missing_ok=True)
    e2e.DESIGN.write_bytes(load(works.DESIGN).document)
    try:
        decision = e2e.decide(credentials, "once")
    except Exception as exc:  # a failed run leaves no unscrubbed log behind
        for path in (e2e.CLASSIFIER_LOG, e2e.EXTRACTOR_LOG, e2e.RECORD):
            path.unlink(missing_ok=True)
        print(masked(f"{type(exc).__name__}: {exc}", credentials), file=sys.stderr)
        return 1
    e2e.RECORD.write_bytes(decision.record.document)
    problems: list[str] = []
    for path in (e2e.CLASSIFIER_LOG, e2e.EXTRACTOR_LOG, e2e.RECORD):
        if path.exists():
            problems += scrub(path, credentials)
    print("plan:", decision.plan.kind, "|", decision.plan.reply or decision.plan.reason)
    calls = json.loads(decision.record.document)["unhashed"]["calls"]
    for index, item in enumerate(decision.record.log):
        if item["item"] == "call":
            print("call", item["sources"], "outcome:", item["outcome"])
            for line in calls.get(str(index), {}).get("warnings", []):
                print("   ", masked(line, credentials))
        elif item["item"] == "entry" and item["ref"].startswith("q:"):
            print(" ", item["ref"], masked(str(item.get("value", item.get("marker"))), credentials))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        for path in (e2e.CLASSIFIER_LOG, e2e.EXTRACTOR_LOG, e2e.RECORD):
            path.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
