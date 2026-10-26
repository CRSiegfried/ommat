"""The guide's per-candidate example: a question asked once per street a report names.

The how-to at docs/how-to/add-a-source.md shows it under "Ask a question per candidate".
"""

from ommat import (
    CallResult,
    Design,
    Plan,
    Policy,
    Question,
    ReadView,
    Rule,
    Source,
    SourceCall,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    decide_sync,
    load,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_each_with_margin

MARKERS = (Unsure, Unavailable, Unasked)

SOURCES = {
    "lamps": Source(
        {
            "q:lamps/out": Question(
                "noul",
                Text(
                    "Does the report say a lamp is out on {street}?",
                    [{"name": "street", "origin": {"candidate": "d:streets"}}],
                ),
            ),
            "q:lamps/urgent": Question("noul", Text("Does the report call it urgent?")),
        },
        needs=["t:state.text"],
        grounded="d:streets",
    ),
}


def out_streets(r: ReadView) -> object:
    """The streets the model is sure have a lamp out, in the set's order."""
    picks = pick_each_with_margin(r, "q:lamps/out", 0.8, 0.2)
    if not isinstance(picks, dict):
        return picks  # the set's own marker
    return [street for street, pick in picks.items() if pick == "yes"]


RULES = {
    "streets": Rule(
        "d:streets",
        t.list_(t.string),
        lambda r: r.read("t:state.streets"),
        version="1",
        reads=["t:state.streets"],
        candidates=True,
    ),
    "out": Rule("d:out", t.list_(t.string), out_streets, version="1", reads=["q:lamps/out"]),
}


def policy(r: ReadView) -> Plan:
    out = r.read("d:out")
    if isinstance(out, MARKERS) or not out:
        return Plan("escalate", reason="no lamp is out")
    return Plan("inform", reply="Logged.")


DESIGN = Design(
    name="lamps",
    state=t.record(text=t.string, streets=t.list_(t.string)),
    sources=SOURCES,
    rules=RULES,
    policy=Policy(policy, version="1", reads=["d:out"]),
)


async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """A stand-in: a lamp is out on Quay Street alone."""
    for source in batch:
        for ref, question in source.questions.items():
            print(" ", ref, "|", question.wording)
    return CallResult(
        {
            ref: {"yes": p, "no": 1 - p}
            for s in batch
            for ref, q in s.questions.items()
            for p in [0.9 if "Quay" in str(q.wording) else 0.1]
        }
    )


if __name__ == "__main__":
    loaded = load(DESIGN)
    for streets in (["Quay Street", "Elm Row"], []):
        decision = decide_sync(loaded, {"text": "x", "streets": streets}, {"default": binding})
        print(decision.plan.kind, "| replay findings:", replay_sync(decision.record, loaded)[1])
