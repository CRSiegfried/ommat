"""A design for the transcript view and the conversation renderers, and its four paths.

Round 1 asks ``mail/router`` and ``ask/again`` together, one call item to the ``model``
oracle, and the tool source ``lookup`` apart; round 2 asks ``confirm``, which needs ``d:route``,
through ``default``. The policy executes on a confirmed archive: a step with a board-ref and a
literal argument, then a ``notify`` step, which binds the first step's result when ``bound``.
The paths: ``clear`` passes; ``refused`` fails a plan-level requirement; ``unsure`` abstains on
``q:confirm`` and clarifies; ``down`` has the ``model`` oracle raise, so ``confirm`` is never
called.
"""

from typing import Any

from ommat import (
    Binding,
    CallResult,
    Context,
    ContextEntry,
    Decision,
    Design,
    LoadedDesign,
    Origin,
    Plan,
    Policy,
    Question,
    Requirement,
    Rule,
    Source,
    SourceCall,
    Step,
    Text,
    decide_sync,
    load,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

STATE = {"message": "Archive the newsletters", "account": "acct-7"}

SOURCES = {
    "mail/router": Source(
        {"q:route": Question("noul", Text("Archive this?"))},
        needs=["t:state.message"],
        oracle="model",
    ),
    "ask/again": Source(
        {"q:again": Question("noul", Text("Asked before?"))},
        needs=["t:state.message"],
        oracle="model",
    ),
    "lookup": Source({"t:order": t.record(id=t.string)}, needs=["t:state.message"], oracle="tools"),
    "confirm": Source({"q:confirm": Question("noul", Text("Sure?"))}, needs=["d:route"]),
}


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:route", 0.7, 0.2)


def sure(r: Any) -> Any:
    return pick_with_margin(r, "q:confirm", 0.7, 0.2)


RULES = {
    "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:route"]),
    "sure": Rule("d:sure", t.label("no", "yes"), sure, version="1", reads=["q:confirm"]),
}


def policy(bound: bool) -> Any:
    def plan(r: Any) -> Plan:
        if r.read("d:sure") == "yes":
            note = Binding(1, "receipt") if bound else Origin(literal="done")
            sure = [Requirement("d:sure", "yes")]
            return Plan(
                "execute",
                [
                    Step(
                        "archive",
                        {"order": Origin(ref="t:order.id"), "folder": Origin(literal="old")},
                        sure,
                    ),
                    Step("notify", {"receipt": note}, sure),
                ],
                reason="the user asked to archive",
            )
        return Plan("clarify", reply="Shall I archive them?")

    return plan


CONTEXT = Context(
    [
        ContextEntry(
            "d:route", Text("Archive asked: {r}", [{"name": "r", "origin": {"ref": "d:route"}}])
        )
    ]
)


def design(*, bound: bool = False, **changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "mail",
        "state": t.record(message=t.string, account=t.string),
        "sources": SOURCES,
        "rules": RULES,
        "policy": Policy(policy(bound), version="1", reads=["d:sure", "t:order.id"]),
        "tools": ["archive", "notify"],
        "classes": {"t:state": {"account": "identifier"}},
        "context": CONTEXT,
    }
    fields.update(changes)
    return Design(**fields)


YES = {"yes": 0.95, "no": 0.05}
HALF = {"yes": 0.5, "no": 0.5}


def model(*, down: bool = False) -> Any:
    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        if down:
            raise RuntimeError("the oracle is down")
        return CallResult({"q:route": YES, "q:again": HALF})

    return binding


def confirming(answer: dict[str, float]) -> Any:
    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({"q:confirm": answer})

    return binding


async def tools(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"t:order": {"id": "o-1"}})


PATHS = ("clear", "refused", "unsure", "down")


def turn(path: str, under: LoadedDesign | None = None, *, bound: bool = False) -> Decision:
    """One of the four paths: ``clear``, ``refused``, ``unsure`` or ``down``."""
    if under is None:
        changes: dict[str, Any] = {}
        if path == "refused":
            changes["requirements"] = [Requirement("d:route", "no")]
        under = load(design(bound=bound, **changes))
    bindings = {
        "default": confirming(HALF if path == "unsure" else YES),
        "model": model(down=path == "down"),
        "tools": tools,
    }
    return decide_sync(under, STATE, bindings)
