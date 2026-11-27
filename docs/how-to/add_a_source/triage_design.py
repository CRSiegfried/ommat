"""Path C's design: it routes a fault report by a noul and a choice asked of a chat model.

The how-to at docs/how-to/add-a-source.md wires it, and chat_run.py decides with it.
"""

from ommat import Design, Parameter, Question, ReadView, Rule, Source, Text, Undetermined
from ommat import types as t
from ommat.helpers import pick_with_margin

SOURCES = {
    "triage": Source(
        {
            "q:triage/hazard": Question(
                "noul", Text("Does the report describe a danger to people?")
            ),
            "q:triage/asset": Question(
                "choice", Text("Which asset is the report about?"), ["lighting", "road", "other"]
            ),
        },
        needs=["t:state.text"],
        oracle="chat",
        timeout=10,
    ),
}

PARAMETERS = {
    "triage/bar": Parameter(t.number(0, 1), 0.6),
    "triage/margin": Parameter(t.number(0, 1), 0.2),
}


def hazard(r: ReadView) -> object:
    """``yes`` or ``no``, when the answer clears the bar and the margin; else abstain."""
    return pick_with_margin(r, "q:triage/hazard", r.param("triage/bar"), r.param("triage/margin"))


def asset(r: ReadView) -> object:
    """The asset, when its answer clears the bar and the margin; else abstain."""
    return pick_with_margin(r, "q:triage/asset", r.param("triage/bar"), r.param("triage/margin"))


def route(r: ReadView) -> object:
    """A danger goes to the emergency crew; anything else to the asset's team."""
    danger, kind = r.read("d:hazard"), r.read("d:asset")
    if danger == "yes":
        return "emergency crew"
    if isinstance(danger, Undetermined):
        return danger
    if isinstance(kind, Undetermined):
        return kind
    return {"lighting": "lighting team", "road": "roads team"}.get(kind, "front desk")


ASKED = ["triage/bar", "triage/margin"]
RULES = {
    "hazard": Rule(
        "d:hazard",
        t.label("yes", "no"),
        hazard,
        version="1",
        reads=["q:triage/hazard"],
        params=ASKED,
    ),
    "asset": Rule(
        "d:asset",
        t.label("lighting", "road", "other"),
        asset,
        version="1",
        reads=["q:triage/asset"],
        params=ASKED,
    ),
    "route": Rule("d:route", t.string, route, version="1", reads=["d:hazard", "d:asset"]),
}

DESIGN = Design(
    name="fault-triage",
    state=t.record(text=t.string),
    sources=SOURCES,
    rules=RULES,
    parameters=PARAMETERS,
)
