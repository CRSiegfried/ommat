"""The design the stored format-5 record in ``tests/format_5`` ran under, kept in a module of
its own so its document loads from bytes. It holds what format 6 adds order to: a state record
declared out of code-point order, a payload, a structured wording and a mapping parameter, each
out of canonical order, a rule returning a mapping, and a tool returning one."""

from ommat import (
    CallResult,
    Design,
    Parameter,
    Plan,
    Policy,
    Question,
    ReadView,
    Rule,
    Source,
    SourceCall,
    Structured,
)
from ommat import types as t


def weights(r: ReadView) -> dict[str, int]:
    return {"zeta": r.param("order/weights")["zeta"], "alpha": 1}


def plan(r: ReadView) -> Plan:
    return Plan("escalate", reason="stored")


DESIGN = Design(
    name="format-5",
    state=t.record(subject=t.string, body=t.string),
    sources={
        "lookup": Source(
            {"t:found": t.mapping(t.integer())},
            needs=["t:state.subject"],
            payload={"zone": "b", "area": "a"},
        ),
        "ask": Source(
            {
                "q:ask/tone": Question(
                    "choice",
                    Structured(
                        {"task": "Pick the tone of {subject}", "about": "tone"},
                        [{"name": "subject", "origin": {"ref": "t:state.subject"}}],
                    ),
                    ["warm", "cold"],
                )
            },
            needs=["t:state.subject"],
        ),
    },
    rules={
        "weights": Rule(
            "d:weights", t.mapping(t.integer()), weights, version="1", params=["order/weights"]
        )
    },
    parameters={"order/weights": Parameter(t.mapping(t.integer()), {"zeta": 2, "alpha": 1})},
    policy=Policy(plan, version="1", reads=[]),
)

STATE = {"subject": "Lamp", "body": "out"}


async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """The tool's mapping and the question's answer, each out of canonical order."""
    if batch[0].id == "ask":
        return CallResult({"q:ask/tone": {"warm": 0.75, "cold": 0.25}})
    return CallResult({"t:found": {"zeta": 3, "alpha": 4}})
