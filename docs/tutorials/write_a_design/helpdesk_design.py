"""The tutorial's design: a shop's help desk that routes one customer message.

The tutorial at docs/tutorials/write-a-design.md builds it step by step, and helpdesk_run.py
decides with it.
"""

from ommat import Design, Parameter, Question, ReadView, Rule, Source, Text, Undetermined
from ommat import types as t
from ommat.helpers import extract, pick_with_margin

TOPICS = [
    {"name": "delivery", "description": "Where an order is, or when it will arrive"},
    {"name": "return", "description": "Sending something back, or a refund for it"},
    {"name": "damaged", "description": "An item that arrived broken or faulty"},
    {"name": "other", "description": "Anything that is not delivery, a return or damage"},
]

SOURCES = {
    "triage": Source(
        {
            "q:triage/urgent": Question("noul", Text("Does the customer need an answer today?")),
            "q:triage/topic": Question(
                "choice", Text("What is the customer's message about?"), TOPICS
            ),
        },
        needs=["t:state"],
        oracle="classifier",
    ),
    "lookup": Source(
        {
            "q:order/number": Question(
                "extraction",
                Text("The order number the customer quotes"),
                None,
                value_type="text",
                text={"ref": "t:state.message"},
            ),
        },
        needs=["t:state"],
        oracle="reader",
    ),
}

PARAMETERS = {
    "desk/topic_at": Parameter(t.number(0, 1), 0.5),
    "desk/urgent_at": Parameter(t.number(0, 1), 0.6),
    "desk/margin": Parameter(t.number(0, 1), 0.2),
    "desk/no_order_at": Parameter(t.number(0, 1), 0.5),
}


def topic_of(r: ReadView) -> object:
    """The topic, when its answer clears the bar and the margin; else abstain."""
    return pick_with_margin(r, "q:triage/topic", r.param("desk/topic_at"), r.param("desk/margin"))


def urgency_of(r: ReadView) -> object:
    """``yes`` or ``no``, when the answer clears the bar and the margin; else abstain."""
    return pick_with_margin(r, "q:triage/urgent", r.param("desk/urgent_at"), r.param("desk/margin"))


def order_of(r: ReadView) -> object:
    """The order number the message quotes, or abstain."""
    return extract(r, "q:order/number", r.param("desk/no_order_at"))


RULES = {
    "topic": Rule(
        "d:topic",
        t.label("damaged", "delivery", "other", "return"),
        topic_of,
        version="1",
        reads=["q:triage/topic"],
        params=["desk/topic_at", "desk/margin"],
    ),
    "urgent": Rule(
        "d:urgent",
        t.label("no", "yes"),
        urgency_of,
        version="1",
        reads=["q:triage/urgent"],
        params=["desk/urgent_at", "desk/margin"],
    ),
    "order": Rule(
        "d:order",
        t.string,
        order_of,
        version="1",
        reads=["q:order/number"],
        params=["desk/no_order_at"],
    ),
}


def route(r: ReadView) -> object:
    """Escalate what the desk cannot settle or must answer today; inform the rest."""
    topic = r.read("d:topic")
    urgent = r.read("d:urgent")
    order = r.read("d:order")
    if isinstance(topic, Undetermined):
        return topic
    if urgent == "yes":
        return "answer_today"
    if isinstance(order, Undetermined):
        return f"route_{topic}"
    return f"route_{topic}_with_order"


RULES["route"] = Rule(
    "d:route",
    t.string,
    route,
    version="1",
    reads=["d:topic", "d:urgent", "d:order"],
)

DESIGN = Design(
    name="help-desk",
    state=t.record(message=t.string),
    sources=SOURCES,
    rules=RULES,
    parameters=PARAMETERS,
    policy="d:route",
)
