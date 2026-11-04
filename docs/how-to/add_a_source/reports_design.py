"""The guide's design: it reads the streets a lamp report names, from a local span model.

The how-to at docs/how-to/add-a-source.md wires it, and reports_run.py decides with it.
"""

from ommat import (
    Cause,
    Design,
    Parameter,
    Question,
    ReadView,
    Rule,
    Source,
    Text,
    Undetermined,
    Unsure,
)
from ommat import types as t
from ommat.helpers import extract

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


PARAMETERS = {
    "places/sure_at": Parameter(t.number(0, 1), 0.8),
    "places/none_at": Parameter(t.number(0, 1), 0.5),
}


def sure_street(r: ReadView) -> object:
    """The street the model is sure of: the answer read whole, for its confidence."""
    answer = r.read("q:places/street")
    if isinstance(answer, Undetermined):
        return answer
    sure = r.param("places/sure_at")
    if answer.value is None or answer.confidence is None or answer.confidence < sure:
        return Unsure(Cause("abstained", ["q:places/street"]))
    return answer.value


def streets(r: ReadView) -> object:
    """Every street the report names, or abstain when the model likely found none."""
    return extract(r, "q:places/streets", r.param("places/none_at"))


RULES = {
    "street": Rule(
        "d:street",
        t.string,
        sure_street,
        version="1",
        reads=["q:places/street"],
        params=["places/sure_at"],
    ),
    "streets": Rule(
        "d:streets",
        t.list_(t.string),
        streets,
        version="1",
        reads=["q:places/streets"],
        params=["places/none_at"],
    ),
}


def route(r: ReadView) -> object:
    """Log a report against the streets it names; hand on one that names none."""
    street = r.read("d:street")
    named = r.read("d:streets")
    if isinstance(named, Undetermined) or not named:
        return Unsure(Cause("abstained", ["d:streets"]))
    if isinstance(street, Undetermined):
        return {"log": named}
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
    parameters=PARAMETERS,
    policy="d:route",
)
