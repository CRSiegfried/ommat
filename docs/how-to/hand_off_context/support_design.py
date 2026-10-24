"""The support design: whether a customer asks for a refund, what the turn does about it, and
what the next model is told."""

from ommat import (
    Context,
    ContextEntry,
    Design,
    Origin,
    Plan,
    Policy,
    Question,
    ReadView,
    Requirement,
    Rule,
    Source,
    Step,
    Text,
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
    return pick_with_margin(r, "q:refund", 0.7, 0.2)


def policy(r: ReadView) -> Plan:
    wanted = r.read("d:refund")
    if wanted == "yes":
        step = Step(
            "refund_order",
            {"order": Origin(ref="t:state.order.id")},
            [Requirement("d:refund", "yes")],
        )
        return Plan("execute", [step], reason="the customer asked for a refund")
    if wanted == "no":
        return Plan("inform", reply="Thanks for your message.")
    return Plan("clarify", reply="Would you like a refund for this order?")


CONTEXT = Context(
    [
        ContextEntry(
            "d:refund",
            Text("Refund requested: {refund}", [{"name": "refund", "origin": {"ref": "d:refund"}}]),
        ),
        ContextEntry(
            "t:state.order.status",
            Text(
                "Order {id} is {status}",
                [
                    {"name": "id", "origin": {"ref": "t:state.order.id"}},
                    {"name": "status", "origin": {"ref": "t:state.order.status"}},
                ],
            ),
        ),
    ]
)

DESIGN = Design(
    name="support",
    state=STATE,
    sources={"ask": ASK},
    rules={
        "refund": Rule("d:refund", t.label("no", "yes"), refund, version="1", reads=["q:refund"])
    },
    policy=Policy(policy, version="1", reads=["d:refund", "t:state.order.id"]),
    tools=["refund_order"],
    classes={"t:state": {"email": "identifier"}},
    context=CONTEXT,
)
