"""The how-to's design: a triage source, a projection of its intent, and a refund source
gated on that projection by a source requirement, so the intent is never sent."""

from ommat import (
    Design,
    Parameter,
    Question,
    ReadView,
    Rule,
    Source,
    SourceRequirement,
    Text,
    Undetermined,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

INTENTS = ("refund", "billing", "other")

PARAMETERS = {
    "gate/bar": Parameter(t.number(0, 1), 0.7),
    "gate/margin": Parameter(t.number(0, 1), 0.2),
}
BARS = ["gate/bar", "gate/margin"]


def picked(r: ReadView, ref: str) -> object:
    """The label at ``ref`` that clears the bar and the margin, or abstain."""
    return pick_with_margin(r, ref, r.param("gate/bar"), r.param("gate/margin"))


def triage(r: ReadView) -> object:
    """The intent and urgency, as one record estimate."""
    intent = picked(r, "q:ask/intent")
    urgent = picked(r, "q:ask/urgent")
    if isinstance(intent, Undetermined):
        return intent
    if isinstance(urgent, Undetermined):
        return urgent
    return {"intent": intent, "urgent": urgent}


def intent_label(r: ReadView) -> object:
    """The intent alone, as a label a source requirement can test."""
    return r.read("d:triage.intent")


def amount_stated(r: ReadView) -> object:
    """Whether the message states an amount, or abstain."""
    return picked(r, "q:refund/amount_stated")


def route(r: ReadView) -> object:
    """Queue a refund whose amount is stated, ask for one that is not, and note the rest."""
    intent = r.read("d:intent_label")
    if isinstance(intent, Undetermined):
        return intent
    if intent != "refund":
        return "note"
    # Only a refund asked the gated question, so only here is its estimate read for a value.
    if r.read("d:amount_stated") == "yes":
        return "queue_refund"
    return "ask_amount"


# --- the asking source, its estimate and the projection ---------------------------------
ASK = Source(
    {
        "q:ask/intent": Question("choice", Text("What does the customer want?"), labels=INTENTS),
        "q:ask/urgent": Question("noul", Text("Is it urgent?")),
    },
    needs=["t:state.message"],
)
RULES = {
    "triage": Rule(
        "d:triage",
        t.record(intent=t.label(*INTENTS), urgent=t.label("yes", "no")),
        triage,
        version="1",
        reads=["q:ask/intent", "q:ask/urgent"],
        params=BARS,
    ),
    "intent_label": Rule(
        "d:intent_label", t.label(*INTENTS), intent_label, version="1", reads=["d:triage.intent"]
    ),
    "route": Rule(
        "d:route",
        t.label("ask_amount", "note", "queue_refund"),
        route,
        version="1",
        reads=["d:intent_label", "d:amount_stated"],
    ),
    "amount_stated": Rule(
        "d:amount_stated",
        t.label("yes", "no"),
        amount_stated,
        version="1",
        reads=["q:refund/amount_stated"],
        params=BARS,
    ),
}

# --- the gated source --------------------------------------------------------------------
REFUND = Source(
    {"q:refund/amount_stated": Question("noul", Text("Does the message state an amount?"))},
    needs=["t:state.message"],
    requirement=SourceRequirement("d:intent_label", ["refund"]),
)

DESIGN = Design(
    name="refund gate",
    state=t.record(message=t.string),
    sources={"ask": ASK, "refund": REFUND},
    rules=RULES,
    parameters=PARAMETERS,
)
