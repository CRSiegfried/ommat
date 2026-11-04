"""A design for the transcript view and the conversation renderers, and its four paths.

Round 1 asks ``mail/router`` and ``ask/again`` together, one call item to the ``model``
oracle, and the tool source ``lookup`` apart; round 2 asks ``confirm``, which needs ``d:route``,
through ``default``. The policy ``d:act`` reads ``d:sure`` and ``t:order.id``: it gives
``archive`` on a confirmed archive and ``hold`` otherwise, and takes an ``Unsure`` confirmation
as its own estimate. The paths: ``clear`` decides ``archive``; ``hold`` decides ``hold`` on a
``no`` confirmation; ``unsure`` abstains on ``q:confirm``, so the policy is undecided; ``down``
has the ``model`` oracle raise, so ``confirm`` is never called.
"""

from typing import Any

from ommat import (
    CallResult,
    Decision,
    Design,
    LoadedDesign,
    Question,
    Rule,
    Source,
    SourceCall,
    Text,
    Unsure,
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


def act(r: Any) -> Any:
    r.read("t:order.id")
    sure = r.read("d:sure")
    if isinstance(sure, Unsure):
        return sure
    return "archive" if sure == "yes" else "hold"


RULES = {
    "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:route"]),
    "sure": Rule("d:sure", t.label("no", "yes"), sure, version="1", reads=["q:confirm"]),
    "act": Rule(
        "d:act", t.label("archive", "hold"), act, version="1", reads=["d:sure", "t:order.id"]
    ),
}


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "mail",
        "state": t.record(message=t.string, account=t.string),
        "sources": SOURCES,
        "rules": RULES,
        "policy": "d:act",
        "classes": {"t:state": {"account": "identifier"}},
    }
    fields.update(changes)
    return Design(**fields)


YES = {"yes": 0.95, "no": 0.05}
HALF = {"yes": 0.5, "no": 0.5}
NO = {"yes": 0.05, "no": 0.95}


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


PATHS = ("clear", "hold", "unsure", "down")


def turn(path: str, under: LoadedDesign | None = None) -> Decision:
    """One of the four paths: ``clear``, ``hold``, ``unsure`` or ``down``."""
    if under is None:
        under = load(design())
    answer = {"unsure": HALF, "hold": NO}.get(path, YES)
    bindings = {
        "default": confirming(answer),
        "model": model(down=path == "down"),
        "tools": tools,
    }
    return decide_sync(under, STATE, bindings)
