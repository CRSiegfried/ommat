# pyright: reportPrivateUsage=false
"""Grounded rules and tables: a function rule or a table grounded over a candidate set runs once
per candidate, reading ``candidate``, ``candidate.f`` and the current candidate's entry of a
declared ref grounded over the same set, and writes one entry per candidate in ref code-point
order; a set holding a marker gives the set's need marker, an empty set nothing. Load's checks
on ``candidate`` reads, field paths and tables over grounded refs; the record, replay and
redaction of a grounded rule's entries."""

import copy
import json
from typing import Any, cast

import pytest

from designs import minimal, rule, slot, with_rule
from ledger_logs import rehashed
from ommat import (
    CallResult,
    Design,
    FindingKind,
    Helper,
    InvalidDesign,
    LoadedDesign,
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
    Table,
    Text,
    Undetermined,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._refs import grounded_ref
from ommat.helpers import pick_with_margin, unique
from test_views import ROOT, digest

PERSON = t.record(name=t.string, age=t.integer())
SET = "d:people"
OUT = "q:ask/out"
FIT = "d:fit"
LEVEL = "d:level"

ANA: dict[str, Any] = {"name": "Ana", "age": 30}
BO: dict[str, Any] = {"name": "Bo", "age": 41}
ZOE: dict[str, Any] = {"name": "Zoë", "age": 7}


def people(r: ReadView) -> object:
    value = r.read("t:state.people")
    if value == []:
        return value
    if cast("list[dict[str, Any]]", value)[0]["name"] == "raise":
        raise RuntimeError("no people")
    return value


SEEN: list[tuple[object, object, object]] = []


def fit(r: ReadView) -> object:
    """Each candidate's own pick, its name and age as read."""
    name, age = r.read("candidate"), r.read("candidate.age")
    SEEN.append((name, age, r.read(OUT)))
    if age == 7:
        raise RuntimeError("too young")
    pick = pick_with_margin(r, OUT, 0.6, 0.2)
    if isinstance(pick, Undetermined):
        return pick
    return f"{name}:{pick}"


READ: list[object] = []


def summary(r: ReadView) -> object:
    READ.append(r.read(FIT))
    return "seen"


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply=f"fit: {r.read(FIT)}")


LEVEL_TABLE = Table(
    LEVEL,
    t.label("adult", "minor"),
    "F",
    ["candidate.age"],
    [[">= 18", "adult"], ["-", "minor"]],
    grounded=SET,
)


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "grounded",
        "state": t.record(people=t.list_(PERSON)),
        "sources": {
            "ask": Source(
                {OUT: Question("noul", Text("Is {c} in?", [slot("c", candidate=SET)]))},
                grounded=SET,
            )
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
            "fit": Rule(
                FIT,
                t.string,
                fit,
                version="1",
                reads=["candidate", "candidate.age", OUT],
                grounded=SET,
            ),
            "level": LEVEL_TABLE,
            "summary": Rule("d:summary", t.string, summary, version="1", reads=[FIT]),
        },
        "policy": Policy(plan, version="1", reads=[FIT, "d:summary"]),
        "classes": {SET: {"name": "identifier"}},
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({ref: {"yes": 0.9, "no": 0.1} for s in batch for ref in s.questions})


def decided(members: list[Any] | None = None, loaded: LoadedDesign = LOADED) -> Record:
    SEEN.clear()
    READ.clear()
    state = {"people": [BO, ANA] if members is None else members}
    return decide_sync(loaded, state, {"default": answer}).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def refs(record: Record) -> list[str]:
    return [i["ref"] for i in record.log if i["item"] == "entry"]


def failures(d: Design) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as info:
        load(d)
    return [(f.check, f.message) for f in info.value.failures]


# ---------------------------------------------------------------------------------------------
# Load


def test_a_grounded_rule_and_table_load_with_an_edge_from_the_sets_provider() -> None:
    assert dict(LOADED._grounded) == {FIT: SET, LEVEL: SET, OUT: SET}
    assert "people" in LOADED._graph["fit"]
    assert "people" in LOADED._graph["level"]
    assert "ask" in LOADED._graph["fit"]


def test_a_rule_grounded_over_a_set_that_reads_it_is_a_cycle() -> None:
    def finder(r: ReadView) -> object:
        return []

    rules = {
        **design().rules,
        "people": Rule(SET, t.list_(PERSON), finder, version="1", reads=[FIT], candidates="name"),
    }
    found = failures(design(rules=rules))
    assert any(c == 5 and "cycle" in m for c, m in found)


def test_check_4_refuses_a_candidate_read_in_an_ungrounded_rule() -> None:
    d = with_rule(minimal(), "r", rule("r", reads=("candidate",)))
    assert failures(d) == [
        (4, 'rule "r" reads "candidate", which only a grounded rule or table reads')
    ]


def test_check_4_refuses_a_candidate_read_in_the_policy() -> None:
    found = failures(design(policy=Policy(plan, version="1", reads=[FIT, "candidate"])))
    assert (4, 'the policy reads "candidate", which only a grounded rule or table reads') in found


@pytest.mark.parametrize("read", ["candidate.nick", "candidate.age.x", "candidate.name.x"])
def test_check_9_refuses_a_candidate_read_naming_no_field(read: str) -> None:
    rules = {
        **design().rules,
        "fit": Rule(FIT, t.string, fit, version="1", reads=[read, OUT], grounded=SET),
    }
    found = failures(design(rules=rules))
    assert found == [
        (9, f'rule "fit" reads "{read}", which names no field of its set\'s candidate')
    ]


def test_check_9_refuses_a_candidate_field_over_a_list_string_set() -> None:
    def names(r: ReadView) -> object:
        return ["a"]

    d = with_rule(
        with_rule(
            minimal(), "s", Rule("d:s", t.list_(t.string), names, version="1", candidates=True)
        ),
        "g",
        rule("g", reads=("candidate.name",), grounded="d:s"),
    )
    assert failures(d) == [
        (9, 'rule "g" reads "candidate.name", which names no field of its set\'s candidate')
    ]


def test_check_9_refuses_a_field_path_on_a_grounded_ref_outside_its_set() -> None:
    record_fit = Rule(FIT, PERSON, fit, version="1", reads=["candidate"], grounded=SET)
    reader = Rule("d:summary", t.string, summary, version="1", reads=["d:fit.age"])
    rules = {**design().rules, "fit": record_fit, "summary": reader}
    found = failures(design(rules=rules))
    assert found == [
        (
            9,
            'rule "summary" reads "d:fit.age" by a field path; only a component grounded over '
            '"d:people" reads "d:fit" so',
        )
    ]


def test_a_rule_grounded_over_the_same_set_reads_a_grounded_ref_by_a_field_path() -> None:
    record_fit = Rule(FIT, PERSON, fit, version="1", reads=["candidate"], grounded=SET)
    reader = Rule("d:summary", t.string, summary, version="1", reads=["d:fit.age"], grounded=SET)
    load(
        design(
            rules={**design().rules, "fit": record_fit, "summary": reader},
            policy=Policy(plan, version="1", reads=[FIT]),
        )
    )


def test_check_4_refuses_a_table_reading_a_grounded_ref_outside_its_set() -> None:
    table = Table("d:tab", t.boolean, "F", [OUT], [["-", True]])
    found = failures(design(rules={**design().rules, "tab": table}))
    assert found == [
        (
            4,
            'rule "tab"\'s input 1 reads "q:ask/out", a grounded declared ref; a table reads one '
            "only when grounded over its set",
        )
    ]


def test_check_4_refuses_a_policy_table_reading_a_grounded_ref() -> None:
    policy = PolicyTable([FIT], [["-", "done"]])
    found = failures(design(policy=policy, plans={"done": PlanDeclaration("inform")}))
    assert any(c == 4 and "a grounded declared ref" in m for c, m in found)


def test_check_4_refuses_a_pick_over_a_candidate_read() -> None:
    table = Table(
        "d:tab",
        t.boolean,
        "F",
        [Helper("pick_with_margin", "candidate", bar="b", margin="m")],
        [["-", True]],
        grounded=SET,
    )
    found = failures(design(rules={**design().rules, "tab": table}))
    assert any(
        c == 4 and 'pick_with_margin reads "candidate", which is not a question' in m
        for c, m in found
    )


def test_check_9_refuses_a_boundaries_field_naming_a_candidate_read() -> None:
    bounded = Rule(
        FIT,
        t.string,
        fit,
        version="1",
        reads=["candidate", "candidate.age", OUT],
        grounded=SET,
        boundaries={"candidate.age": ["x"]},
    )
    found = failures(design(rules={**design().rules, "fit": bounded}))
    assert any(c == 9 and 'names "candidate.age", a candidate read' in m for c, m in found)


def test_check_11_refuses_a_format_7_document_reading_candidate() -> None:
    doc = json.loads(LOADED.document)
    doc["format"] = 7
    for r in doc["rules"].values():
        r["candidates"] = r["candidates"] is not False and r["candidates"]
    doc["rules"]["people"]["candidates"] = True
    with pytest.raises(InvalidDesign) as info:
        load(doc)
    assert [(f.check, f.message) for f in info.value.failures] == [
        (11, 'rule "fit" reads candidate, which format 7 does not have'),
    ]
    del doc["rules"]["fit"]
    doc["policy"]["reads"] = []
    with pytest.raises(InvalidDesign) as info:
        load(doc)
    assert 'rule "level" reads candidate, which format 7 does not have' in str(info.value)


# ---------------------------------------------------------------------------------------------
# A turn


def test_each_candidate_gets_its_own_entry_in_ref_code_point_order() -> None:
    record = decided()
    e = entries(record)
    ana, bo = grounded_ref(FIT, "Ana"), grounded_ref(FIT, "Bo")
    assert e[ana]["value"] == "Ana:yes"
    assert e[bo]["value"] == "Bo:yes"
    assert refs(record).index(ana) + 1 == refs(record).index(bo)
    assert [s[:2] for s in SEEN] == [("Bo", 41), ("Ana", 30)]  # run in the set's order
    assert e[ana]["derivation"] == [
        {"helper": "pick_with_margin", "refs": [OUT], "rested_on": [OUT]}
    ]
    assert e[ana]["call"] is None
    assert FIT not in e  # no entry at the declared ref over a valued set


def test_a_grounded_rule_reads_its_candidates_own_entry() -> None:
    decided()
    answers = [cast(Any, s[2]).probabilities for s in SEEN]
    assert answers == [{"yes": 0.9, "no": 0.1}] * 2


def test_a_reader_of_the_declared_ref_gets_a_dict_in_the_sets_order() -> None:
    decided()
    (read,) = READ
    assert list(cast("dict[str, object]", read)) == ["Bo", "Ana"]


def test_a_raise_marks_its_own_candidates_entry_alone() -> None:
    record = decided([BO, ZOE, ANA])
    e = entries(record)
    zoe = grounded_ref(FIT, "Zoë")
    assert e[zoe]["marker"]["cause"] == {"code": "raised", "refs": ["fit"]}
    assert e[grounded_ref(FIT, "Bo")]["value"] == "Bo:yes"
    index = record.log.index(e[zoe])
    assert str(index) in record.unhashed["exceptions"]


def test_a_set_holding_a_marker_gives_the_declared_ref_its_need_marker() -> None:
    record = decided([{"name": "raise", "age": 1}])
    e = entries(record)
    assert e[FIT]["marker"]["cause"] == {"code": "need-unavailable", "refs": [SET]}
    assert "derivation" not in e[FIT]
    assert "rows" not in e[LEVEL]
    assert e[LEVEL]["marker"]["kind"] == "Unavailable"
    assert not any(r.startswith(FIT + "[") for r in e)
    assert SEEN == []
    (read,) = READ
    assert isinstance(read, Undetermined)
    assert read.cause.refs == (SET,)


def test_an_empty_set_writes_nothing_and_its_reader_gets_an_empty_dict() -> None:
    record = decided([])
    assert not any(r.startswith((FIT, LEVEL)) for r in refs(record))
    assert READ == [{}]


def test_a_grounded_table_evaluates_each_candidates_own_inputs() -> None:
    record = decided([BO, ZOE])
    e = entries(record)
    assert e[grounded_ref(LEVEL, "Bo")]["value"] == "adult"
    assert e[grounded_ref(LEVEL, "Bo")]["rows"] == [1, 2]
    assert e[grounded_ref(LEVEL, "Zoë")]["value"] == "minor"
    assert e[grounded_ref(LEVEL, "Zoë")]["rows"] == [2]


def test_a_grounded_c_table_sums_each_candidates_own_rows() -> None:
    table = Table(
        "d:score",
        t.integer(),
        "C+",
        ["candidate.age"],
        [[">= 18", 2], ["<= 40", 3]],
        grounded=SET,
    )
    loaded = load(design(rules={**design().rules, "score": table}))
    e = entries(decided([BO, ANA], loaded=loaded))
    assert e[grounded_ref("d:score", "Bo")]["value"] == 2
    assert e[grounded_ref("d:score", "Ana")]["value"] == 5
    assert e[grounded_ref("d:score", "Ana")]["rows"] == [1, 2]


def test_a_helper_given_a_candidate_read_names_the_sets_ref() -> None:
    def same(r: ReadView) -> object:
        return unique(r, "candidate", "candidate.name")

    rules = {
        **design().rules,
        "same": Rule(
            "d:same",
            t.string,
            same,
            version="1",
            reads=["candidate", "candidate.name"],
            grounded=SET,
        ),
    }
    e = entries(decided(loaded=load(design(rules=rules))))
    item = e[grounded_ref("d:same", "Ana")]
    assert item["marker"]["cause"] == {"code": "conflict", "refs": [SET]}
    assert item["derivation"] == [{"helper": "unique", "refs": [SET], "rested_on": [SET]}]


def test_the_policy_reads_a_grounded_rules_declared_ref_as_a_dict() -> None:
    record = decided()
    plan_item = next(i for i in record.log if i["item"] == "plan")
    assert plan_item["reply"] == "fit: {'Bo': 'Bo:yes', 'Ana': 'Ana:yes'}"


# ---------------------------------------------------------------------------------------------
# The record


def test_the_settings_item_names_each_grounded_rules_set() -> None:
    assert decided().log[1]["grounded"] == {FIT: SET, LEVEL: SET, OUT: SET}


def test_a_grounded_rule_turn_replays_with_no_finding() -> None:
    record = decided([BO, ZOE, ANA])
    rerun, found = replay_sync(record, LOADED)
    assert found == []
    assert rerun is not None
    assert rerun.record_hash == record.record_hash


def test_replay_finds_a_changed_grounded_entry() -> None:
    record = decided()
    log = copy.deepcopy(record.log)
    target = grounded_ref(FIT, "Ana")
    next(i for i in log if i["item"] == "entry" and i["ref"] == target)["value"] = "x"
    doc = {"log": log, "unhashed": copy.deepcopy(record.unhashed)}

    _, found = replay_sync(read_record(rehashed(doc)), LOADED)
    assert [(f["kind"], f.get("ref")) for f in found] == [(FindingKind.entry, target)]


def test_redact_digests_a_grounded_rules_candidate() -> None:
    record = decided()
    view = json.loads(redact(record, ROOT))
    written = [i["ref"] for i in view["log"] if i["item"] == "entry"]
    assert grounded_ref(FIT, digest("Ana", record)) in written
    assert grounded_ref(FIT, "Ana") not in written


def test_check_4_refuses_a_requirement_naming_a_grounded_declared_ref() -> None:
    from ommat import Requirement, SourceRequirement

    found = failures(design(requirements=[Requirement(FIT, "x")]))
    assert any(c == 4 and "which no requirement names" in m for c, m in found)
    gated = Source(
        {"q:other": Question("noul", Text("Other?"))},
        requirement=SourceRequirement(FIT, ["x"]),
    )
    found = failures(design(sources={**design().sources, "other": gated}))
    assert any(c == 4 and "which no requirement names" in m for c, m in found)


@pytest.mark.parametrize(
    "reads", [["candidate", "candidate", OUT], ["candidate.age", "candidate.age", OUT]]
)
def test_check_1_refuses_a_repeated_candidate_read(reads: list[str]) -> None:
    rules = {
        **design().rules,
        "fit": Rule(FIT, t.string, fit, version="1", reads=reads, grounded=SET),
    }
    found = failures(design(rules=rules))
    assert any(c == 1 and "twice" in m for c, m in found)


def test_check_9_refuses_an_in_cell_over_records_testing_a_nullable_string() -> None:
    person = t.record(name=t.string, nick=t.nullable(t.string))
    rules = {
        **design().rules,
        "people": Rule(
            SET, t.list_(person), people, version="1", reads=["t:state.people"], candidates="name"
        ),
        "fit": Rule(FIT, t.string, fit, version="1", reads=["candidate", OUT], grounded=SET),
        "level": Table(
            LEVEL,
            t.boolean,
            "F",
            ["candidate.nick", SET],
            [["in d:people", "-", True], ["-", "-", False]],
            grounded=SET,
        ),
    }
    found = failures(design(rules=rules, state=t.record(people=t.list_(person))))
    assert any(c == 9 and "is not a string" in m for c, m in found)
