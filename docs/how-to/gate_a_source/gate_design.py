"""The how-to's design: a triage source, a projection of its intent, and a refund source
gated on that projection by a source requirement, so the intent is never sent."""

from ommat import Design, Plan, Policy, Question, ReadView, Rule, Source, SourceRequirement, Text
from ommat import types as t
from ommat.helpers import pick_with_margin

INTENTS = ("refund", "billing", "other")


def triage(r: ReadView) -> object:
    """The intent and urgency, as one record estimate."""
    intent = pick_with_margin(r, "q:ask/intent", 0.7, 0.2)
    urgent = pick_with_margin(r, "q:ask/urgent", 0.7, 0.2)
    if not isinstance(intent, str):
        return intent
    if not isinstance(urgent, str):
        return urgent
    return {"intent": intent, "urgent": urgent}


def intent_label(r: ReadView) -> object:
    """The intent alone, as a label a source requirement can test."""
    return r.read("d:triage.intent")


def refund_asked(r: ReadView) -> object:
    return pick_with_margin(r, "q:refund/amount_stated", 0.7, 0.2)


def route(r: ReadView) -> Plan:
    intent = r.read("d:intent_label")
    if not isinstance(intent, str):
        return Plan("escalate", reason="the intent is not settled")
    if intent != "refund":
        return Plan("inform", reply="Noted.")
    # Only a refund asked the gated question, so only here is its estimate read for a value.
    if r.read("d:amount_stated") == "yes":
        return Plan("inform", reply="A refund with its amount: queued for review.")
    return Plan("clarify", reply="How much should we refund?")


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
    ),
    "intent_label": Rule(
        "d:intent_label", t.label(*INTENTS), intent_label, version="1", reads=["d:triage.intent"]
    ),
    "amount_stated": Rule(
        "d:amount_stated",
        t.label("yes", "no"),
        refund_asked,
        version="1",
        reads=["q:refund/amount_stated"],
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
    policy=Policy(route, version="1", reads=["d:intent_label", "d:amount_stated"]),
)
