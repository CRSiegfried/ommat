"""The design the stored format-6 record in ``tests/format_6`` ran under, kept in a module of
its own so its document loads from bytes. It declares no targets: a question asked about the
state, a rule estimating from it and a policy, each an object format 7 adds members to."""

from ommat import (
    CallResult,
    Design,
    Plan,
    Policy,
    Question,
    ReadView,
    Rule,
    Source,
    SourceCall,
    Text,
)
from ommat import types as t
from ommat.helpers import pick_with_margin


def urgent(r: ReadView) -> object:
    return pick_with_margin(r, "q:ask/urgent", 0.6, 0.2)


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply=f"urgent: {r.read('d:urgent')}")


DESIGN = Design(
    name="format-6",
    state=t.record(request=t.string),
    sources={
        "ask": Source(
            {
                "q:ask/urgent": Question(
                    "noul",
                    Text(
                        "Does {request} need action today?",
                        [{"name": "request", "origin": {"ref": "t:state.request"}}],
                    ),
                )
            },
            needs=["t:state"],
        ),
    },
    rules={
        "urgent": Rule(
            "d:urgent", t.label("no", "yes"), urgent, version="1", reads=["q:ask/urgent"]
        )
    },
    policy=Policy(plan, version="1", reads=["d:urgent"]),
)

STATE = {"request": "a cancelled flight"}


async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"q:ask/urgent": {"yes": 0.9, "no": 0.1}})
