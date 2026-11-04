"""The guide's per-candidate example: a question asked once per street a report names.

The how-to at docs/how-to/add-a-source.md shows it under "Ask a question per candidate".
"""

from ommat import (
    CallResult,
    Design,
    Parameter,
    Question,
    ReadView,
    Rule,
    Source,
    SourceCall,
    Text,
    Undetermined,
    decide_sync,
    load,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_each_with_margin

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


PARAMETERS = {
    "lamps/out_at": Parameter(t.number(0, 1), 0.8),
    "lamps/margin": Parameter(t.number(0, 1), 0.2),
}


def streets(r: ReadView) -> object:
    """The streets the caller names, the candidate set the lamps source is grounded over."""
    return r.read("t:state.streets")


def out_streets(r: ReadView) -> object:
    """The streets the model is sure have a lamp out, in the set's order."""
    picks = pick_each_with_margin(
        r, "q:lamps/out", r.param("lamps/out_at"), r.param("lamps/margin")
    )
    if isinstance(picks, Undetermined):
        return picks  # the set's own marker
    return [street for street, pick in picks.items() if pick == "yes"]


RULES = {
    "streets": Rule(
        "d:streets",
        t.list_(t.string),
        streets,
        version="1",
        reads=["t:state.streets"],
        candidates=True,
    ),
    "out": Rule(
        "d:out",
        t.list_(t.string),
        out_streets,
        version="1",
        reads=["q:lamps/out"],
        params=["lamps/out_at", "lamps/margin"],
    ),
}


def policy(r: ReadView) -> object:
    """Log a report with a lamp out; escalate one with none, or with none settled."""
    out = r.read("d:out")
    if isinstance(out, Undetermined) or not out:
        return "escalate"
    return "log"


RULES["policy"] = Rule("d:policy", t.label("escalate", "log"), policy, version="1", reads=["d:out"])

DESIGN = Design(
    name="lamps",
    state=t.record(text=t.string, streets=t.list_(t.string)),
    sources=SOURCES,
    rules=RULES,
    parameters=PARAMETERS,
    policy="d:policy",
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
    for named in (["Quay Street", "Elm Row"], []):
        decision = decide_sync(loaded, {"text": "x", "streets": named}, {"default": binding})
        print(decision.estimate, "| replay findings:", replay_sync(decision.record, loaded)[1])
