"""The guide's design: it reads the streets a lamp report names, from a local span model.

The how-to at docs/how-to/add-a-source.md wires it, and reports_run.py decides with it.
"""

from ommat import (
    Cause,
    Design,
    Question,
    ReadView,
    Rule,
    Source,
    Text,
    Unasked,
    Unavailable,
    Unsure,
)
from ommat import types as t
from ommat.helpers import extract

MARKERS = (Unsure, Unavailable, Unasked)

SOURCES = {
    "places": Source(
        {
            "q:places/street": Question(
                "extraction", Text("street"), None, value_type="text", text={"ref": "t:state.text"}
            ),
            "q:places/streets": Question(
                "extraction",
                Text("street"),
                None,
                value_type="text",
                is_list=True,
                text={"ref": "t:state.text"},
            ),
        },
        oracle="spans",
        timeout=5,
    ),
}


def sure_street(r: ReadView) -> object:
    """The street the model is sure of: the answer read whole, for its confidence."""
    answer = r.read("q:places/street")
    if isinstance(answer, MARKERS):
        return answer
    if answer.value is None or answer.confidence is None or answer.confidence < 0.8:
        return Unsure(Cause("abstained", ["q:places/street"]))
    return answer.value


RULES = {
    "street": Rule("d:street", t.string, sure_street, version="1", reads=["q:places/street"]),
    "streets": Rule(
        "d:streets",
        t.list_(t.string),
        lambda r: extract(r, "q:places/streets", 0.5),
        version="1",
        reads=["q:places/streets"],
    ),
}


def route(r: ReadView) -> object:
    """Log a report against the streets it names; hand on one that names none."""
    street = r.read("d:street")
    streets = r.read("d:streets")
    if isinstance(streets, MARKERS) or not streets:
        return Unsure(Cause("abstained", ["d:streets"]))
    if isinstance(street, MARKERS):
        return {"log": streets}
    return {"log": [street]}


RULES["route"] = Rule(
    "d:route",
    t.record(log=t.list_(t.string)),
    route,
    version="1",
    reads=["d:street", "d:streets"],
)

DESIGN = Design(
    name="lamp-reports",
    state=t.record(text=t.string),
    sources=SOURCES,
    rules=RULES,
    policy="d:route",
)
