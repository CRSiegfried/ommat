"""Decide one request with the assistant design and print the turn as it runs.

``python assistant.py "<request>"`` decides the request, prints each round as ommat's observer
reports it, and saves the record. ``python assistant.py --replay`` replays every saved record.

No model is called: ``MODEL`` holds the answers a judgement model gave for each request, and the
binding waits a moment before returning them so the rounds can be seen. The engine is the real
one, and so are the order, the skipped sources, the markers and the replay.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

from assistant_design import DESIGN

from ommat import CallResult, SourceCall, decide_sync, escalation, load, read_record, replay_sync
from ommat.bindings import fixed

RECORDS = Path("records")
DIM, BOLD, OFF = "\033[2m", "\033[1m", "\033[0m"
GREEN, AMBER, CYAN = "\033[32m", "\033[33m", "\033[36m"


def found(request: str, *values: str) -> dict[str, Any]:
    """An extraction answer: one value, or a list of them, each with its span in the request."""
    spans = [[request.index(v), request.index(v) + len(v)] for v in values]
    one = len(values) == 1
    return {
        "value": values[0] if one else list(values),
        "guess": None,
        "p_null": 0.03,
        "confidence": 0.95,
        "span": spans[0] if one else spans,
    }


NOTHING = {"value": None, "guess": None, "p_null": 0.9, "confidence": 0.8, "span": None}
FIND = "Find my recent emails from Neo or Smith"
REMIND = "Remind me to call the dentist tomorrow at nine"
VAGUE = "Can you deal with that thing from yesterday"
MODEL: dict[str, dict[str, Any]] = {
    FIND: {
        "q:asks/find_email": 0.99,
        "q:asks/send_email": 0.03,
        "q:asks/set_reminder": 0.03,
        "q:find_email/senders": found(FIND, "Neo", "Smith"),
        "q:find_email/about": NOTHING,
    },
    REMIND: {
        "q:asks/find_email": 0.02,
        "q:asks/send_email": 0.04,
        "q:asks/set_reminder": 0.98,
        "q:set_reminder/what": found(REMIND, "call the dentist"),
        "q:set_reminder/when": found(REMIND, "tomorrow at nine"),
    },
    VAGUE: {"q:asks/find_email": 0.45, "q:asks/send_email": 0.3, "q:asks/set_reminder": 0.35},
}


def model(request: str) -> Any:
    """A binding that answers from ``MODEL`` after a pause, standing in for a model call."""
    answers = {
        ref: {"yes": a, "no": round(1 - a, 2)} if isinstance(a, float) else a
        for ref, a in MODEL[request].items()
    }
    answer = fixed(answers)

    async def binding(calls: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        await asyncio.sleep(0.9)
        return await answer(calls, seconds=seconds)

    return binding


def shown(entry: dict[str, Any]) -> str:
    """An entry's value or marker, in a few words."""
    if "marker" in entry:
        kind = entry["marker"]["kind"]
        return f"{DIM}not asked{OFF}" if kind == "Unasked" else f"{AMBER}{kind}{OFF}"
    value = entry["value"]
    if not isinstance(value, dict):
        return f"{GREEN}{value}{OFF}"
    if "probabilities" in value:
        yes = value["probabilities"]["yes"]
        return f"{'yes' if yes >= 0.5 else 'no '} {DIM}{max(yes, 1 - yes):.2f}{OFF}"
    found = value["value"]
    return f"{DIM}nothing given{OFF}" if found is None else str(found)


def decide(request: str) -> None:
    loaded = load(DESIGN)
    questions = sum(len(source.provides) for source in DESIGN.sources.values())
    asked = calls = 0
    skipped: list[str] = []

    def watch(index: int, item: dict[str, Any]) -> None:
        nonlocal asked, calls
        if item["item"] == "call":
            calls += 1
            print(
                f"\n  {CYAN}model call {calls}{OFF}  {DIM}source: {', '.join(item['sources'])}{OFF}"
            )
        elif item["item"] == "entry" and item["ref"] != "t:state":
            ref = item["ref"]
            if ref.startswith("d:"):
                print(f"    {BOLD}{ref:<24}{OFF}{shown(item)}")
            elif "value" in item:
                asked += 1
                print(f"    {ref:<24}{shown(item)}")
            else:
                source = ref[2:].split("/")[0]
                if source in skipped:
                    return
                skipped.append(source)
                print(f"    {DIM}{'q:' + source + '/…':<24}not asked{OFF}")
            time.sleep(0.15)

    print(f"\n  {DIM}the design holds {questions} questions{OFF}")
    bindings = {"default": model(request)}
    record = decide_sync(loaded, {"request": request}, bindings, observer=watch).record
    plural = "call" if calls == 1 else "calls"
    print(f"\n  {DIM}asked {asked} of {questions} questions in {calls} model {plural}{OFF}")
    why = escalation(record, "d:call")
    if why is None:
        print(f"  {BOLD}{GREEN}→ {record.entries['d:call']['value']}{OFF}\n")
    else:
        refs = ", ".join(why["root_cause"]["refs"])
        print(f"  {BOLD}{AMBER}→ undecided:{OFF} {why['kind']}, no clear answer at {refs}")
        print(f"  {DIM}  the caller hands this one to an LLM{OFF}\n")
    RECORDS.mkdir(exist_ok=True)
    (RECORDS / f"{record.record_hash[7:19]}.json").write_bytes(record.document)


def replay() -> None:
    loaded = load(DESIGN)
    differences = 0
    paths = sorted(RECORDS.glob("*.json"))
    for path in paths:
        _, findings = replay_sync(read_record(path.read_bytes()), loaded)
        differences += len(findings)
    print(f"\n  {BOLD}replayed {len(paths)} recorded turns{OFF}")
    print(f"  {GREEN}0 model calls, {differences} differences{OFF}\n")


if __name__ == "__main__":
    replay() if sys.argv[1:] == ["--replay"] else decide(sys.argv[1])
