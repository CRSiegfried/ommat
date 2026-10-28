# pyright: reportPrivateUsage=false
"""Grounded plan declarations: compile expands a grounded ``execute`` declaration into one copy
of its steps per candidate, a candidate origin as the candidate-value literal of the whole
member with its set, an origin on a grounded declared ref as the candidate's grounded ref, and a
binding as the same copy's step; a row naming one matches only when its set and every expanded
origin hold a value, and rule 2 names the set, then the first ``Unavailable`` expanded origin.
Load's checks on grounded plans and candidate origins; the ledger's checks of a literal's
``set``; redaction, replay and a function policy's candidate origin."""

import copy
import json
from typing import Any

import pytest

from designs import slot
from ledger_logs import rehashed
from ommat import (
    Binding,
    CallResult,
    Design,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Origin,
    Plan,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    ReadView,
    Record,
    Rule,
    Source,
    SourceCall,
    Step,
    Text,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._plan import candidate_literal
from ommat._refs import grounded_ref
from test_views import ROOT, digest

PERSON = t.record(name=t.string, age=t.integer())
SET = "d:people"
ADDR = "d:addr"

ANA: dict[str, Any] = {"name": "Ana", "age": 30}
BO: dict[str, Any] = {"name": "Bo", "age": 41}


def people(r: ReadView) -> object:
    value = r.read("t:state.people")
    if value == [{"name": "raise", "age": 0}]:
        raise RuntimeError("no people")
    return value


def ready(r: ReadView) -> object:
    return "yes" if r.read("t:state.go") else "no"


def addr(r: ReadView) -> object:
    name = r.read("candidate")
    if name == "Ana" and r.read("candidate.age") == 99:
        raise RuntimeError("no address")
    return f"{name}@example.org"


MAIL = PlanDeclaration(
    "execute",
    [
        Step("send", {"who": Origin(candidate=SET), "to": Origin(ref=ADDR)}),
        Step("log", {"sent": Binding(1, None), "note": Origin(literal="sent")}),
    ],
    reason=Text("Mail {all}", [slot("all", ref=SET)]),
    grounded=SET,
)
WAIT = PlanDeclaration("inform", reply=Text("Nothing to send"))


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "mail",
        "state": t.record(people=t.list_(PERSON), go=t.boolean),
        "sources": {
            "ask": Source({"q:ask/ok": Question("noul", Text("Is it ok?"))}, needs=["t:state"])
        },
        "rules": {
            "people": Rule(
                SET,
                t.list_(PERSON),
                people,
                version="1",
                reads=["t:state.people"],
                candidates="name",
            ),
            "ready": Rule(
                "d:ready", t.label("no", "yes"), ready, version="1", reads=["t:state.go"]
            ),
            "addr": Rule(
                ADDR,
                t.string,
                addr,
                version="1",
                reads=["candidate", "candidate.age"],
                grounded=SET,
            ),
        },
        "policy": PolicyTable(["d:ready"], [["yes", "mail"], ["-", "wait"]]),
        "plans": {"mail": MAIL, "wait": WAIT},
        "tools": ["send", "log"],
        "default_deny": False,
        "classes": {SET: {"name": "identifier"}},
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({ref: {"yes": 0.9, "no": 0.1} for s in batch for ref in s.questions})


def decided(
    members: list[Any] | None = None, go: bool = True, loaded: LoadedDesign = LOADED
) -> Record:
    state = {"people": [BO, ANA] if members is None else members, "go": go}
    return decide_sync(loaded, state, {"default": answer}).record


def plan_item(record: Record) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "plan")


def failures(d: Design) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as info:
        load(d)
    return [(f.check, f.message) for f in info.value.failures]


# ---------------------------------------------------------------------------------------------
# Load


def test_a_grounded_plan_loads_and_the_policy_reads_its_set_and_origins() -> None:
    assert set(LOADED._reads["policy"]) == {"d:ready", SET, ADDR}
    assert load(LOADED.document).design_hash == LOADED.design_hash
    doc = json.loads(LOADED.document)
    assert doc["plans"]["mail"]["steps"][0]["arguments"]["who"] == {"candidate": SET}


def test_check_9_refuses_a_grounded_plan_of_another_kind() -> None:
    plans = {"mail": PlanDeclaration("inform", reply=Text("hi"), grounded=SET), "wait": WAIT}
    assert (9, 'plan declaration "mail" is grounded, so of kind execute, not inform') in failures(
        design(plans=plans)
    )


def test_check_4_refuses_a_candidate_origin_in_an_ungrounded_plan() -> None:
    plans = {
        "mail": PlanDeclaration("execute", [Step("send", {"who": Origin(candidate=SET)})]),
        "wait": WAIT,
    }
    assert failures(design(plans=plans)) == [
        (
            4,
            'plan declaration "mail"\'s step 1 argument "who" is a candidate origin, which only a '
            "grounded plan declaration holds",
        )
    ]


def test_check_3_refuses_a_candidate_origin_naming_another_set() -> None:
    plans = {
        "mail": PlanDeclaration(
            "execute", [Step("send", {"who": Origin(candidate="d:ready")})], grounded=SET
        ),
        "wait": WAIT,
    }
    found = failures(design(plans=plans))
    assert any(c == 3 and "not its plan's grounding set" in m for c, m in found)


def test_check_9_refuses_an_expanded_origin_by_a_field_path() -> None:
    record_addr = Rule(
        ADDR, PERSON, addr, version="1", reads=["candidate", "candidate.age"], grounded=SET
    )
    plans = {
        "mail": PlanDeclaration(
            "execute", [Step("send", {"to": Origin(ref="d:addr.name")})], grounded=SET
        ),
        "wait": WAIT,
    }
    found = failures(design(rules={**design().rules, "addr": record_addr}, plans=plans))
    assert any(c == 9 and "a grounded declared ref by a field path" in m for c, m in found)


def test_check_9_refuses_a_grounded_plans_slot_on_a_grounded_ref() -> None:
    plans = {
        "mail": PlanDeclaration("execute", reason=Text("{a}", [slot("a", ref=ADDR)]), grounded=SET),
        "wait": WAIT,
    }
    found = failures(design(plans=plans))
    assert (
        9,
        'plan declaration "mail"\'s reason has a slot on "d:addr", a grounded declared ref',
    ) in found


def test_check_4_refuses_an_ungrounded_plans_origin_on_a_grounded_ref() -> None:
    plans = {
        "mail": PlanDeclaration("execute", [Step("send", {"to": Origin(ref=ADDR)})]),
        "wait": WAIT,
    }
    found = failures(design(plans=plans))
    assert any(
        c == 4 and "a grounded declared ref, which only a grounded plan's origin names" in m
        for c, m in found
    )


def test_check_9_refuses_a_grounded_plan_on_the_last_row() -> None:
    policy = PolicyTable(["d:ready"], [["yes", "wait"], ["-", "mail"]])
    found = failures(design(policy=policy))
    assert (9, 'the policy table\'s last row names "mail", a grounded plan') in found


# ---------------------------------------------------------------------------------------------
# A turn


def test_compile_expands_one_copy_of_the_steps_per_candidate_in_the_sets_order() -> None:
    plan = plan_item(decided())
    assert plan["kind"] == "execute"
    assert plan["reason"] == "Mail Bo, Ana"
    assert plan["steps"] == [
        {
            "tool": "send",
            "arguments": {
                "to": {"ref": grounded_ref(ADDR, "Bo")},
                "who": {"literal": BO, "set": SET},
            },
            "requirements": [],
        },
        {
            "tool": "log",
            "arguments": {"note": {"literal": "sent"}, "sent": {"step": 1, "path": None}},
            "requirements": [],
        },
        {
            "tool": "send",
            "arguments": {
                "to": {"ref": grounded_ref(ADDR, "Ana")},
                "who": {"literal": ANA, "set": SET},
            },
            "requirements": [],
        },
        {
            "tool": "log",
            "arguments": {"note": {"literal": "sent"}, "sent": {"step": 3, "path": None}},
            "requirements": [],
        },
    ]
    verdict = next(i for i in decided().log if i["item"] == "verdict")
    assert verdict["kind"] == "passed"


def test_over_an_empty_set_the_plan_has_no_steps_and_is_valid() -> None:
    record = decided([])
    plan = plan_item(record)
    assert (plan["kind"], plan["steps"], plan["reason"]) == ("execute", [], "Mail ")


def test_a_set_holding_unavailable_escalates_naming_the_set() -> None:
    plan = plan_item(decided([{"name": "raise", "age": 0}]))
    assert plan["kind"] == "escalate"
    assert plan["cause"] == {"code": "required", "refs": [SET]}


def test_an_unavailable_grounded_entry_escalates_naming_it() -> None:
    plan = plan_item(decided([BO, {"name": "Ana", "age": 99}]))
    assert plan["kind"] == "escalate"
    assert plan["cause"] == {"code": "required", "refs": [grounded_ref(ADDR, "Ana")]}


def test_a_row_naming_another_plan_is_untouched() -> None:
    plan = plan_item(decided(go=False))
    assert (plan["kind"], plan["reply"]) == ("inform", "Nothing to send")


# ---------------------------------------------------------------------------------------------
# A function policy


@pytest.mark.parametrize(
    "origin", [Origin(candidate=SET), candidate_literal(ANA, SET)], ids=["candidate", "set"]
)
def test_a_function_policys_candidate_origin_or_set_is_invalid_return(origin: Origin) -> None:
    def policy(r: ReadView) -> Plan:
        return Plan("execute", [Step("send", {"who": origin})])

    loaded = load(design(policy=Policy(policy, version="1", reads=["d:ready"]), plans={}))
    plan = plan_item(decided(loaded=loaded))
    assert (plan["kind"], plan["cause"]) == (
        "broken",
        {"code": "invalid-return", "refs": ["policy"]},
    )


def test_a_function_policy_naming_a_grounded_ref_breaks_rule_4() -> None:
    def policy(r: ReadView) -> Plan:
        return Plan("execute", [Step("send", {"to": Origin(ref=ADDR)})])

    loaded = load(design(policy=Policy(policy, version="1", reads=["d:ready", ADDR]), plans={}))
    plan = plan_item(decided(loaded=loaded))
    assert plan["cause"] == {"code": "invalid-return", "refs": ["rule:4", "step:1"]}


def test_origin_takes_exactly_one_keyword() -> None:
    with pytest.raises(TypeError):
        Origin(candidate=SET, ref=SET)
    with pytest.raises(TypeError):
        Origin(candidate=3)  # type: ignore[arg-type]
    origin = Origin(candidate=SET)
    assert (origin.candidate, origin.ref, origin.literal, origin.set) == (SET, None, None, None)


# ---------------------------------------------------------------------------------------------
# The record


def edited_plan(record: Record, change: Any) -> dict[str, Any]:
    log = copy.deepcopy(record.log)
    change(next(i for i in log if i["item"] == "plan"))
    return {"log": log, "unhashed": copy.deepcopy(record.unhashed)}


def refused(doc: dict[str, Any], says: str) -> None:
    with pytest.raises(InvalidRecord) as info:
        read_record(rehashed(doc))
    assert says in str(info.value)


def test_a_grounded_plan_turn_reads_back_and_replays_with_no_finding() -> None:
    record = decided()
    assert read_record(record.document).record_hash == record.record_hash
    rerun, found = replay_sync(record, LOADED)
    assert found == []
    assert rerun is not None
    assert rerun.record_hash == record.record_hash


def test_the_ledger_refuses_a_set_naming_no_set() -> None:
    def change(plan: dict[str, Any]) -> None:
        plan["steps"][0]["arguments"]["who"]["set"] = "d:ready"

    refused(edited_plan(decided(), change), "which is no key of sets")


def test_the_ledger_refuses_a_literal_that_does_not_fit_its_sets_tree() -> None:
    def change(plan: dict[str, Any]) -> None:
        plan["steps"][0]["arguments"]["who"]["literal"] = {"nick": "x"}

    refused(edited_plan(decided(), change), "which its class tree does not list")


def test_the_ledger_refuses_a_candidate_origin() -> None:
    def change(plan: dict[str, Any]) -> None:
        plan["steps"][0]["arguments"]["who"] = {"candidate": SET}

    refused(edited_plan(decided(), change), "is a candidate origin, which a record never holds")


def names(r: ReadView) -> object:
    return [p["name"] for p in r.read("t:state.people")]


def plain_addr(r: ReadView) -> object:
    return f"{r.read('candidate')}@example.org"


def test_the_ledger_refuses_a_set_in_a_format_6_record() -> None:
    rules = {
        **design().rules,
        "people": Rule(
            SET, t.list_(t.string), names, version="1", reads=["t:state.people"], candidates=True
        ),
        "addr": Rule(ADDR, t.string, plain_addr, version="1", reads=["candidate"], grounded=SET),
    }
    loaded = load(design(rules=rules, classes={}))

    def change(plan: dict[str, Any]) -> None:
        for step in plan["steps"]:
            step["arguments"].pop("to", None)  # a grounded ref, refused before format 7 too

    doc = edited_plan(decided(loaded=loaded), change)
    doc["log"][0]["format"] = 6
    del doc["log"][1]["sets"]
    refused(doc, "carries set, which a format-6 record does not")


def test_redact_digests_a_set_literal_under_its_sets_tree() -> None:
    record = decided()
    view = json.loads(redact(record, ROOT))
    plan = next(i for i in view["log"] if i["item"] == "plan")
    who = plan["steps"][0]["arguments"]["who"]
    assert who["set"] == SET
    assert who["literal"]["name"] == digest("Bo", record)
    to = plan["steps"][0]["arguments"]["to"]
    assert to["ref"] == grounded_ref(ADDR, digest("Bo", record))


def test_a_function_policy_naming_a_grounded_entry_breaks_rule_4() -> None:
    def policy(r: ReadView) -> Plan:
        return Plan("execute", [Step("send", {"to": Origin(ref=grounded_ref(ADDR, "Bo"))})])

    loaded = load(design(policy=Policy(policy, version="1", reads=["d:ready", ADDR]), plans={}))
    plan = plan_item(decided(loaded=loaded))
    assert plan["cause"] == {"code": "invalid-return", "refs": ["rule:4", "step:1"]}


def test_the_ledger_refuses_a_field_path_on_a_grounded_step_argument() -> None:
    def change(plan: dict[str, Any]) -> None:
        plan["steps"][0]["arguments"]["to"]["ref"] += ".x"

    refused(edited_plan(decided(), change), "which it never holds")


def test_check_3_refuses_an_origin_on_a_ref_grounded_over_another_set() -> None:
    def names(r: ReadView) -> object:
        return ["x"]

    rules = {
        **design().rules,
        "other": Rule("d:other", t.list_(t.string), names, version="1", candidates=True),
        "addr2": Rule(
            "d:addr2", t.string, plain_addr, version="1", reads=["candidate"], grounded="d:other"
        ),
    }
    plans = {
        "mail": PlanDeclaration(
            "execute", [Step("send", {"to": Origin(ref="d:addr2")})], grounded=SET
        ),
        "wait": WAIT,
    }
    found = failures(design(rules=rules, plans=plans))
    assert [c for c, m in found if "not its plan's set" in m] == [3]


def test_a_field_path_on_a_grounded_plan_origin_fails_once() -> None:
    record_addr = Rule(
        ADDR, PERSON, addr, version="1", reads=["candidate", "candidate.age"], grounded=SET
    )
    plans = {
        "mail": PlanDeclaration(
            "execute", [Step("send", {"to": Origin(ref="d:addr.name")})], grounded=SET
        ),
        "wait": WAIT,
    }
    found = failures(design(rules={**design().rules, "addr": record_addr}, plans=plans))
    assert len([m for _, m in found if "field path" in m]) == 1
