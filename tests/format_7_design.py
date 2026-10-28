"""The design the stored format-6 record in ``tests/format_7`` ran under, a format-7 document, kept
in a module of its own so its document loads from bytes. Its candidate set is a ``list<string>``
whose ``candidates`` member is ``true``, and a source asks one question per candidate: the record
holds ``grounded`` and no ``sets``."""

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
    Undetermined,
)
from ommat import types as t
from ommat.helpers import pick_each_with_margin


def streets(r: ReadView) -> object:
    return r.read("t:state.streets")


def out(r: ReadView) -> object:
    picks = pick_each_with_margin(r, "q:lamps/out", 0.6, 0.2)
    if isinstance(picks, Undetermined):
        return picks
    return [street for street, pick in picks.items() if pick == "yes"]


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply=f"out: {r.read('d:out')}")


DESIGN = Design(
    name="format-7",
    state=t.record(streets=t.list_(t.string)),
    sources={
        "lamps": Source(
            {
                "q:lamps/out": Question(
                    "noul",
                    Text(
                        "Is a lamp out on {s}?",
                        [{"name": "s", "origin": {"candidate": "d:streets"}}],
                    ),
                )
            },
            oracle="model",
            grounded="d:streets",
        ),
    },
    rules={
        "streets": Rule(
            "d:streets",
            t.list_(t.string),
            streets,
            version="1",
            reads=["t:state.streets"],
            candidates=True,
        ),
        "out": Rule("d:out", t.list_(t.string), out, version="1", reads=["q:lamps/out"]),
    },
    policy=Policy(plan, version="1", reads=["d:out"]),
)

STATE = {"streets": ["Quay St", "Zoë Row"]}


async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({ref: {"yes": 0.9, "no": 0.1} for ref in batch[0].questions})
