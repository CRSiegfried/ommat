"""The support design: whether a customer asks for a refund and what the turn does about it."""

from ommat import Design, Question, ReadView, Rule, Source, Text, Unavailable, Undetermined
from ommat import types as t
from ommat.helpers import pick_with_margin

STATE = t.record(
    message=t.string,
    email=t.string,
    order=t.record(id=t.string, status=t.string),
)

ASK = Source(
    {
        "q:refund": Question(
            "noul",
            Text(
                "Does this customer ask for a refund? {message}",
                [{"name": "message", "origin": {"ref": "t:state.message"}}],
            ),
        )
    },
    needs=["t:state.message"],
)


def refund(r: ReadView) -> str | Undetermined:
    return pick_with_margin(r, "q:refund", 0.7, 0.2)


def policy(r: ReadView) -> str | Undetermined:
    wanted = r.read("d:refund")
    if isinstance(wanted, Unavailable):
        return wanted
    if wanted == "yes":
        return "refund_order"
    if wanted == "no":
        return "thank"
    return "ask_if_refund"


DESIGN = Design(
    name="support",
    state=STATE,
    sources={"ask": ASK},
    rules={
        "refund": Rule("d:refund", t.label("no", "yes"), refund, version="1", reads=["q:refund"]),
        "policy": Rule(
            "d:policy",
            t.label("ask_if_refund", "refund_order", "thank"),
            policy,
            version="1",
            reads=["d:refund"],
        ),
    },
    policy="d:policy",
)
