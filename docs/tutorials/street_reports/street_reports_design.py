"""A design for a city's public-works desk: triage one street report into a work order.

A resident files a report about something in the street. Each oracle stands for one judgement
model. The classifier answers ``triage``: whether people are in danger, which asset is involved
and how serious it is. The extractor reads from the report through two sources in one batch, each
with its own needs: ``details``, when the problem was first noticed, against the filing date, and
where it is; and ``timing``, the time of day it was seen. A registry, called through its own
oracle, lists the places that already have an open work order.

Every label is described but ``other``, so a binding meets both forms. The caller acts on
``d:route``, an ordinary rule read top to bottom; the first row that matches decides:

====================  ==============================  ==========================================
``d:hazard``          the other reads                 decision
====================  ==============================  ==========================================
``yes``               anything                        ``send_crew``: a hazard to people
a marker              anything                        that marker: undecided
``no``                a marker on any                 that marker: undecided
``no``                ``d:asset`` is ``other``        ``front_desk``: not a public-works asset
``no``                a place has an open order       ``already_open``: an order is open there
``no``                all settled                     ``open_work_order``
====================  ==============================  ==========================================

A rule that cannot settle abstains with ``Unsure`` naming the answer that fell short, and the
route rule hands that marker on, so the undecided route says which answer stopped the order.
``d:noticed`` and ``d:seen_at`` are where ``p_null`` matters: at or above its bar the report
gives no date or time and the order goes out without one; below it, a missing value abstains.
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
    Unasked,
    Unavailable,
    Unsure,
)
from ommat import types as t
from ommat.helpers import extract, pick_with_margin, pick_with_thresholds

ASSETS = [
    {"name": "road", "description": "The road surface, kerbs or a footpath"},
    {"name": "lighting", "description": "A street light or a lit sign"},
    {"name": "drainage", "description": "A drain, gully or flooding from one"},
    {"name": "signage", "description": "A road sign, marking or bollard"},
    "other",
]

SEVERITY = [
    {"name": "minor", "description": "Cosmetic; it can wait for the next scheduled round"},
    {"name": "moderate", "description": "Gets in the way; worth fixing within a few weeks"},
    {"name": "serious", "description": "Stops people using the street as they should"},
]

SOURCES = {
    "triage": Source(
        {
            "q:triage/hazard": Question(
                "noul", Text("Does the report describe a danger to people right now?")
            ),
            "q:triage/asset": Question(
                "choice", Text("Which street asset is the report about?"), ASSETS
            ),
            "q:triage/severity": Question(
                "score", Text("How serious is the problem the report describes?"), SEVERITY
            ),
        },
        needs=["t:state.report"],
        oracle="classifier",
    ),
    "details": Source(
        {
            "q:details/noticed": Question(
                "extraction",
                Text("The date the resident first noticed the problem"),
                None,
                value_type="date",
                text={"ref": "t:state.report"},
                reference_date={"ref": "t:state.filed_on"},
            ),
            "q:details/places": Question(
                "extraction",
                Text("Each street or place the report names"),
                None,
                value_type="text",
                is_list=True,
                text={"ref": "t:state.report"},
            ),
        },
        needs=[],
        oracle="extractor",
    ),
    "timing": Source(
        {
            "q:timing/seen_at": Question(
                "extraction",
                Text("The time of day the resident saw the problem"),
                None,
                value_type="time",
                text={"ref": "t:state.report"},
            ),
        },
        needs=[],
        oracle="extractor",
    ),
    "registry": Source(
        {"t:open_orders": t.list_(t.string)},
        needs=["t:state"],
        oracle="registry",
    ),
}

PARAMETERS = {
    "works/hazard_at": Parameter(t.number(0, 1), 0.6),
    "works/asset_at": Parameter(t.number(0, 1), 0.5),
    "works/severity_cuts": Parameter(
        t.mapping(t.number(0, 1)), {"minor": 0.5, "moderate": 0.5, "serious": 0.4}
    ),
    "works/margin": Parameter(t.number(0, 1), 0.2),
    "works/no_date_at": Parameter(t.number(0, 1), 0.5),
    "works/no_time_at": Parameter(t.number(0, 1), 0.5),
    "works/no_places_at": Parameter(t.number(0, 1), 0.5),
}

MARKERS = (Unsure, Unavailable, Unasked)


def hazard(r: ReadView) -> object:
    """``yes`` or ``no`` when the noul clears its bar and margin; else abstain."""
    return pick_with_margin(
        r, "q:triage/hazard", r.param("works/hazard_at"), r.param("works/margin")
    )


def asset(r: ReadView) -> object:
    """The asset the report is about, or abstain."""
    return pick_with_margin(r, "q:triage/asset", r.param("works/asset_at"), r.param("works/margin"))


def severity(r: ReadView) -> object:
    """The severity level whose probability clears its own cut score, or abstain."""
    return pick_with_thresholds(
        r, "q:triage/severity", r.param("works/severity_cuts"), r.param("works/margin")
    )


def optional(r: ReadView, ref: str, bar: str) -> object:
    """An extraction's value; ``None`` when ``p_null`` reaches ``bar``, so the report likely
    gives none; else abstain when the value is missing."""
    answer = r.read(ref)
    if isinstance(answer, MARKERS):
        return answer
    if answer.p_null is not None and answer.p_null >= r.param(bar):
        return None
    if answer.value is None:
        return Unsure(Cause("abstained", [ref]))
    return answer.value


def noticed(r: ReadView) -> object:
    """The date the problem was first noticed, ``None``, or abstain."""
    return optional(r, "q:details/noticed", "works/no_date_at")


def seen_at(r: ReadView) -> object:
    """The time of day the problem was seen, ``None``, or abstain."""
    return optional(r, "q:timing/seen_at", "works/no_time_at")


def places(r: ReadView) -> object:
    """The places the report names, or abstain when it likely names none."""
    return extract(r, "q:details/places", r.param("works/no_places_at"))


RULES = {
    "hazard": Rule(
        "d:hazard",
        t.label("no", "yes"),
        hazard,
        version="1",
        reads=["q:triage/hazard"],
        params=["works/hazard_at", "works/margin"],
    ),
    "asset": Rule(
        "d:asset",
        t.label("drainage", "lighting", "other", "road", "signage"),
        asset,
        version="1",
        reads=["q:triage/asset"],
        params=["works/asset_at", "works/margin"],
    ),
    "severity": Rule(
        "d:severity",
        t.label("minor", "moderate", "serious"),
        severity,
        version="1",
        reads=["q:triage/severity"],
        params=["works/severity_cuts", "works/margin"],
    ),
    "noticed": Rule(
        "d:noticed",
        t.nullable(t.string),
        noticed,
        version="1",
        reads=["q:details/noticed"],
        params=["works/no_date_at"],
    ),
    "seen_at": Rule(
        "d:seen_at",
        t.nullable(t.string),
        seen_at,
        version="1",
        reads=["q:timing/seen_at"],
        params=["works/no_time_at"],
    ),
    "places": Rule(
        "d:places",
        t.list_(t.string),
        places,
        version="1",
        reads=["q:details/places"],
        params=["works/no_places_at"],
    ),
}

SETTLED = ("d:asset", "d:severity", "d:noticed", "d:seen_at", "d:places", "t:open_orders")


def route(r: ReadView) -> object:
    """The table in the module docstring, row by row."""
    danger = r.read("d:hazard")
    if danger == "yes":
        return "send_crew"
    if isinstance(danger, MARKERS):
        return danger
    for ref in SETTLED:
        value = r.read(ref)
        if isinstance(value, MARKERS):
            return value
    if r.read("d:asset") == "other":
        return "front_desk"
    if set(r.read("d:places")) & set(r.read("t:open_orders")):
        return "already_open"
    return "open_work_order"


RULES["route"] = Rule(
    "d:route",
    t.label("already_open", "front_desk", "open_work_order", "send_crew"),
    route,
    version="1",
    reads=["d:hazard", *SETTLED],
)

DESIGN = Design(
    name="street-reports",
    state=t.record(report=t.string, filed_on=t.string),
    sources=SOURCES,
    rules=RULES,
    parameters=PARAMETERS,
)
