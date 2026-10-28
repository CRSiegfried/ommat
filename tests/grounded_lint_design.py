"""A design for the lints over grounded rules: a candidate set of records, a per-candidate
question, a rule grounded over the set picking each candidate's answer, and a policy that
executes unless some candidate's pick is ``no``."""

from typing import Any, cast

from designs import slot
from ommat import Design, Origin, Plan, Policy, Question, ReadView, Rule, Source, Step, Text
from ommat import types as t
from ommat.helpers import pick_with_margin

PERSON = t.record(name=t.string, age=t.integer())
SET = "d:people"
OK = "q:ask/ok"
FIT = "d:fit"
ANA: dict[str, Any] = {"name": "Ana", "age": 30}
BO: dict[str, Any] = {"name": "Bo", "age": 41}


def people(r: ReadView) -> object:
    return r.read("t:state.people")


def fit(r: ReadView) -> object:
    return pick_with_margin(r, OK, 0.6, 0.2)


def policy(r: ReadView) -> Plan:
    picks = cast("dict[str, object]", r.read(FIT))
    if any(p == "no" for p in picks.values()):
        return Plan("clarify", reply="Which one?")
    return Plan("execute", [Step("send", {"to": Origin(literal="all")})])


DESIGN = Design(
    name="lints",
    state=t.record(people=t.list_(PERSON)),
    sources={
        "ask": Source(
            {OK: Question("noul", Text("Is {c} ok?", [slot("c", candidate=SET)]))}, grounded=SET
        )
    },
    rules={
        "people": Rule(
            SET, t.list_(PERSON), people, version="1", reads=["t:state.people"], candidates="name"
        ),
        "fit": Rule(FIT, t.label("no", "yes"), fit, version="1", reads=[OK], grounded=SET),
    },
    policy=Policy(policy, version="1", reads=[FIT]),
    tools=["send"],
    default_deny=False,
)
