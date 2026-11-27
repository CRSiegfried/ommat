"""The support design: whether a customer asks for a refund and what the turn does about it."""

from ommat import (
    Design,
    Parameter,
    Question,
    ReadView,
    Rule,
    Source,
    Text,
    Unavailable,
    Undetermined,
)
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
    return pick_with_margin(r, "q:refund", r.param("refund/bar"), r.param("refund/margin"))


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
        "refund": Rule(
            "d:refund",
            t.label("no", "yes"),
            refund,
            version="1",
            reads=["q:refund"],
            params=["refund/bar", "refund/margin"],
        ),
        "policy": Rule(
            "d:policy",
            t.label("ask_if_refund", "refund_order", "thank"),
            policy,
            version="1",
            reads=["d:refund"],
        ),
    },
    parameters={
        "refund/bar": Parameter(t.number(0, 1), 0.7),
        "refund/margin": Parameter(t.number(0, 1), 0.2),
    },
)
