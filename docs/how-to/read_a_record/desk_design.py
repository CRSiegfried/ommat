"""The desk design: whether a customer asks for a refund, and how many days their order has
been open, which the policy does not read."""

import datetime as dt

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

STATE = t.record(message=t.string, ordered=t.string)

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

CLOCK = Source({"t:today": t.string}, oracle="clock")

PARAMETERS = {
    "desk/refund_at": Parameter(t.number(0, 1), 0.7),
    "desk/margin": Parameter(t.number(0, 1), 0.2),
}


def refund(r: ReadView) -> str | Undetermined:
    return pick_with_margin(r, "q:refund", r.param("desk/refund_at"), r.param("desk/margin"))


def days_open(r: ReadView) -> int | Undetermined:
    today = r.read("t:today")
    if isinstance(today, Undetermined):
        return today
    ordered = dt.date.fromisoformat(r.read("t:state.ordered"))
    return (dt.date.fromisoformat(today) - ordered).days


def policy(r: ReadView) -> str | Undetermined:
    wanted = r.read("d:refund")
    if isinstance(wanted, Unavailable):
        return wanted  # an outage escalates; an Unsure or Unasked refund asks
    return {"yes": "refund_order", "no": "thank"}.get(wanted, "ask_if_refund")


DESIGN = Design(
    name="desk",
    state=STATE,
    sources={"ask": ASK, "clock": CLOCK},
    rules={
        "refund": Rule(
            "d:refund",
            t.label("no", "yes"),
            refund,
            version="1",
            reads=["q:refund"],
            params=["desk/refund_at", "desk/margin"],
        ),
        "days_open": Rule(
            "d:days_open", t.integer(), days_open, version="1", reads=["t:today", "t:state.ordered"]
        ),
        "policy": Rule(
            "d:policy",
            t.label("ask_if_refund", "refund_order", "thank"),
            policy,
            version="1",
            reads=["d:refund"],
        ),
    },
    parameters=PARAMETERS,
    policy="d:policy",
)
