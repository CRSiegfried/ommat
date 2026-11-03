# pyright: reportPrivateUsage=false
"""Candidate sets of records: a list of a record type whose ``candidates`` member names its
candidate field. Check 10 on the type and the field, check 11 on a format-7 document naming one,
check 9 on an ``in`` cell over one; distinctness, grounded refs and rendered labels by the
candidate value; the read view's ``dict`` keyed by it; the ``settings`` item's ``sets`` and the
ledger's checks of it; redaction under the candidate field's class; replay and from_record."""

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
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Plan,
    Policy,
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
    from_record,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._refs import grounded_ref
from ommat.helpers import pick_each_with_margin
from test_views import ROOT, digest

PERSON = t.record(name=t.string, age=t.integer(), tags=t.list_(t.string), note=t.nullable(t.string))
PEOPLE = t.list_(PERSON)
SET = "d:people"
OUT = "q:ask/out"
WHICH = "q:ask/which"

ANA: dict[str, Any] = {"name": "Ana", "age": 30, "tags": ["a"], "note": None}
BO: dict[str, Any] = {"name": "Bo", "age": 41, "tags": [], "note": "x"}


def people(r: ReadView) -> object:
    return r.read("t:state.people")


READS: list[object] = []


def out(r: ReadView) -> object:
    READS.append(r.read(OUT))
    picks = pick_each_with_margin(r, OUT, 0.6, 0.2)
    if isinstance(picks, Undetermined):
        return picks
    return [name for name, pick in picks.items() if pick == "yes"]


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply=f"out: {r.read('d:out')}")


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "people",
        "state": t.record(people=PEOPLE, who=t.string),
        "sources": {
            "ask": Source(
                {
                    OUT: Question("noul", Text("Is {c} in?", [slot("c", candidate=SET)])),
                    WHICH: Question(
                        "choice", Text("Who leads?"), {"fixed": ["none"], "candidates": SET}
                    ),
                },
                grounded=SET,
            )
        },
        "rules": {
            "people": Rule(
                SET, PEOPLE, people, version="1", reads=["t:state.people"], candidates="name"
            ),
            "out": Rule("d:out", t.list_(t.string), out, version="1", reads=[OUT]),
            "known": Table(
                "d:known",
                "boolean",
                "F",
                ["t:state.who", SET],
                [["in d:people", "-", True], ["-", "-", False]],
            ),
        },
        "policy": Policy(plan, version="1", reads=["d:out", "d:known"]),
        "classes": {SET: {"name": "identifier"}},
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())
ASKED: list[tuple[SourceCall, ...]] = []


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    ASKED.append(batch)
    values: dict[str, object] = {}
    for source in batch:
        for ref, question in source.questions.items():
            if ref == WHICH:
                values[ref] = {label: 0.8 if label == "Ana" else 0.1 for label in question.labels}
            else:
                values[ref] = {"yes": 0.9, "no": 0.1}
    return CallResult(values)


def decided(
    members: list[Any] | None = None, who: str = "Bo", loaded: LoadedDesign = LOADED
) -> Record:
    ASKED.clear()
    READS.clear()
    state = {"people": [ANA, BO] if members is None else members, "who": who}
    return decide_sync(loaded, state, {"default": answer}).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def failures(d: Design) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as info:
        load(d)
    return [(f.check, f.message) for f in info.value.failures]


# ---------------------------------------------------------------------------------------------
# Load


def test_a_record_set_loads_with_its_candidate_field() -> None:
    assert dict(LOADED._candidate_sets) == {SET: "name"}
    doc = json.loads(LOADED.document)
    assert doc["format"] == 9
    assert doc["rules"]["people"]["candidates"] == "name"


def test_a_t_set_of_records_names_its_field() -> None:
    sources = {
        "lookup": Source({"t:people": {"type": PEOPLE, "candidates": "name"}}, needs=["t:state"]),
        "ask": Source(
            {OUT: Question("noul", Text("{c}?", [slot("c", candidate="t:people")]))},
            grounded="t:people",
        ),
    }
    loaded = load(minimal(sources=sources))
    assert dict(loaded._candidate_sets) == {"t:people": "name"}


def set_rule(outputs: str, candidates: bool | str) -> Rule:
    return rule("r", outputs=outputs, candidates=candidates)


@pytest.mark.parametrize(
    ("outputs", "candidates", "says"),
    [
        (PEOPLE, True, "not list<string>"),
        (t.list_(t.string), "name", "is not a list of a record type"),
        (PEOPLE, "nick", 'candidate field "nick" is not a string field'),
        (PEOPLE, "age", 'candidate field "age" is not a string field'),
        (
            t.list_(t.record(name=t.string, inner=t.record(x=t.string))),
            "name",
            'field "inner" is not a scalar',
        ),
        (
            t.list_(t.record(name=t.string, deep=t.list_(t.list_(t.string)))),
            "name",
            'field "deep" is not a scalar',
        ),
        (
            t.list_(t.record(name=t.nullable(t.string))),
            "name",
            'candidate field "name" is not a string field',
        ),
    ],
)
def test_check_10_refuses_a_record_set_of_the_wrong_shape(
    outputs: str, candidates: bool | str, says: str
) -> None:
    found = failures(with_rule(minimal(), "r", set_rule(outputs, candidates)))
    assert [c for c, _ in found] == [10]
    assert says in found[0][1]


def test_check_10_takes_a_field_through_type_definitions() -> None:
    d = with_rule(
        minimal(types={"person": PERSON, "people": t.list_("person")}),
        "r",
        set_rule("people", "name"),
    )
    assert dict(load(d)._candidate_sets) == {"d:r": "name"}


@pytest.mark.parametrize("candidates", ["", "Name", "a.b"])
def test_check_form_refuses_a_candidates_member_that_is_no_field_name(candidates: str) -> None:
    found = failures(with_rule(minimal(), "r", set_rule(PEOPLE, candidates)))
    assert any("is not a boolean or a field name" in m for _, m in found)


def test_check_form_refuses_a_candidates_member_of_another_type() -> None:
    found = failures(with_rule(minimal(), "r", set_rule(PEOPLE, 3)))  # type: ignore[arg-type]
    assert "is not a boolean or a field name" in found[0][1]


def test_check_1_refuses_a_name_spelt_candidate() -> None:
    found = failures(with_rule(minimal(), "candidate", rule("candidate")))
    assert (1, 'rule "candidate" is the name a grounded component reads') in found


def test_check_11_refuses_a_format_7_document_naming_a_candidate_field() -> None:
    doc = json.loads(LOADED.document)
    doc["format"] = 7
    with pytest.raises(InvalidDesign) as info:
        load(doc)
    assert [f.check for f in info.value.failures] == [11]
    assert 'rule "people" names a candidate field, which format 7 does not have' in str(info.value)


def test_check_11_refuses_a_format_7_t_provision_naming_a_candidate_field() -> None:
    sources = {
        "lookup": Source({"t:people": {"type": PEOPLE, "candidates": "name"}}, needs=["t:state"])
    }
    doc = json.loads(load(minimal(sources=sources)).document)
    doc["format"] = 7
    with pytest.raises(InvalidDesign) as info:
        load(doc)
    assert [f.check for f in info.value.failures] == [11]
    assert '"t:people" names a candidate field, which format 7 does not have' in str(info.value)


def in_table(ref: str, who: str = "t:state.who") -> Table:
    return Table(
        "d:known", "boolean", "F", [who, ref], [[f"in {ref}", "-", True], ["-", "-", False]]
    )


def test_check_9_refuses_an_in_cell_over_records_that_are_no_candidate_set() -> None:
    d = design(state=t.record(people=PEOPLE, who=t.string, n=t.integer()))
    rules = {**d.rules, "known": in_table("t:state.people")}
    found = failures(design(state=d.state, rules=rules))
    assert found == [
        (
            9,
            'rule "known"\'s row 1\'s cell 1: "t:state.people" is a list of records that is no '
            "candidate set",
        )
    ]


def test_check_9_refuses_an_in_cell_over_a_record_set_testing_no_string() -> None:
    d = design(state=t.record(people=PEOPLE, who=t.string, n=t.integer()))
    rules = {**d.rules, "known": in_table(SET, "t:state.n")}
    found = failures(design(state=d.state, rules=rules))
    ((check, message),) = found
    assert check == 9
    assert '"t:state.n" is not a string, which the candidate field of "d:people" is' in message


# ---------------------------------------------------------------------------------------------
# A turn


def test_each_candidate_is_asked_under_its_candidate_field() -> None:
    record = decided()
    ((source,),) = ASKED
    assert sorted(source.questions) == sorted(
        [grounded_ref(OUT, "Ana"), grounded_ref(OUT, "Bo"), WHICH]
    )
    assert source.questions[grounded_ref(OUT, "Ana")].wording == "Is Ana in?"
    assert source.questions[WHICH].labels == ("none", "Ana", "Bo")
    e = entries(record)
    assert e[SET]["value"] == [ANA, BO]
    assert e[grounded_ref(OUT, "Bo")]["value"]["probabilities"] == {"yes": 0.9, "no": 0.1}
    assert e["d:out"]["value"] == ["Ana", "Bo"]
    assert e["d:known"]["value"] is True
    assert list(cast("dict[str, object]", READS[0])) == [
        "Ana",
        "Bo",
    ]  # the read view's dict, keyed by candidate value


def test_an_in_cell_over_a_record_set_tests_the_candidate_field_alone() -> None:
    assert entries(decided(who="Cy"))["d:known"]["value"] is False
    assert entries(decided(who="30"))["d:known"]["value"] is False


def test_members_repeating_a_candidate_value_make_the_set_invalid_return() -> None:
    record = decided([ANA, {**BO, "name": "Ana"}])
    item = entries(record)[SET]
    assert item["marker"]["cause"] == {"code": "invalid-return", "refs": ["people"]}
    assert ASKED == []


def test_members_differing_only_outside_the_candidate_field_are_still_repeats() -> None:
    record = decided([ANA, {**ANA, "age": 31}])
    assert entries(record)[SET]["marker"]["cause"]["code"] == "invalid-return"


def test_distinct_members_with_equal_other_fields_are_distinct() -> None:
    record = decided([ANA, {**ANA, "name": "Ann"}])
    assert entries(record)[SET]["value"] == [ANA, {**ANA, "name": "Ann"}]


def test_a_label_collision_is_judged_on_the_candidate_values() -> None:
    record = decided([ANA, {**BO, "name": "none"}])
    assert entries(record)[WHICH]["marker"]["cause"] == {
        "code": "label-collision",
        "refs": [SET],
    }


def test_a_t_record_set_with_a_repeated_candidate_value_is_invalid_return() -> None:
    sources = {
        "lookup": Source({"t:people": {"type": PEOPLE, "candidates": "name"}}, needs=["t:state"]),
    }
    loaded = load(minimal(sources=sources))

    async def lookup(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({"t:people": [ANA, {**BO, "name": "Ana"}]})

    record = decide_sync(
        loaded, {"subject": "s", "sender": {"address": "a"}}, {"default": lookup}
    ).record
    item = entries(record)["t:people"]
    assert item["marker"]["cause"] == {"code": "invalid-return", "refs": ["t:people"]}


# ---------------------------------------------------------------------------------------------
# The record


def test_the_settings_item_names_each_set_s_field() -> None:
    settings = decided().log[1]
    assert settings["sets"] == {SET: "name"}
    assert settings["grounded"] == {OUT: SET}


def test_a_list_string_set_is_null_under_sets() -> None:
    from format_7_design import DESIGN, STATE, binding

    record = decide_sync(load(DESIGN), STATE, {"model": binding}).record
    assert record.log[1]["sets"] == {"d:streets": None}


def with_settings(record: Record, **members: Any) -> dict[str, Any]:
    """The record's document, its settings item's members replaced, ``_DROP`` removing one."""
    log = copy.deepcopy(record.log)
    log[1].update(members)
    for k, v in members.items():
        if v is _DROP:
            del log[1][k]
    return {"log": log, "unhashed": copy.deepcopy(record.unhashed)}


_DROP = object()


@pytest.mark.parametrize(
    ("members", "says"),
    [
        ({"sets": _DROP}, "has no 'sets' member"),
        ({"sets": {SET: "nick"}}, "which is not null over a class name or a field"),
        ({"sets": {SET: None}}, "which is not null over a class name or a field"),
        ({"sets": {"d:nope": None}}, "not a provided declared ref"),
        ({"sets": {}}, "which is no key of sets"),
    ],
)
def test_the_ledger_refuses_a_settings_item_with_a_wrong_sets(
    members: dict[str, Any], says: str
) -> None:
    doc = with_settings(decided(), **members)
    with pytest.raises(InvalidRecord) as info:
        read_record(rehashed(doc))
    assert says in str(info.value)


def test_the_ledger_refuses_sets_in_a_format_6_record() -> None:
    doc = with_settings(decided(), sets={})
    doc["log"][0]["format"] = 6
    with pytest.raises(InvalidRecord) as info:
        read_record(rehashed(doc))
    assert "holds sets, which a format-6 record does not" in str(info.value)


def test_redact_digests_a_record_candidate_under_its_field_s_class() -> None:
    record = decided()
    view = json.loads(redact(record, ROOT))
    refs = [i["ref"] for i in view["log"] if i["item"] == "entry"]
    hidden = grounded_ref(OUT, digest("Ana", record))
    assert hidden in refs
    assert grounded_ref(OUT, "Ana") not in refs
    which = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == WHICH)
    assert digest("Ana", record) in json.dumps(which["value"])


def test_redact_passes_a_candidate_whose_field_is_open() -> None:
    loaded = load(design(classes={SET: {"name": "open", "note": "identifier"}}))
    record = decided(loaded=loaded)
    view = json.loads(redact(record, ROOT))
    refs = [i["ref"] for i in view["log"] if i["item"] == "entry"]
    assert grounded_ref(OUT, "Ana") in refs


def test_a_record_set_turn_replays_with_no_finding() -> None:
    record = decided()
    rerun, found = replay_sync(record, LOADED)
    assert found == []
    assert rerun is not None
    assert rerun.record_hash == record.record_hash


def replayed_with_set(members: list[Any]) -> list[Any]:
    """Replay's finding kinds over a record whose set entry, over which the label set was
    answered, is edited to ``members``."""
    record = decided()
    log = copy.deepcopy(record.log)
    next(i for i in log if i["item"] == "entry" and i["ref"] == SET)["value"] = members
    doc = {"log": log, "unhashed": copy.deepcopy(record.unhashed)}
    _, found = replay_sync(read_record(rehashed(doc)), LOADED)
    return [f["kind"] for f in found]


def test_replay_finds_no_stale_question_for_a_change_outside_the_candidate_field() -> None:
    kinds = replayed_with_set([{**ANA, "age": 99}, BO])
    assert FindingKind.stale_question not in kinds
    assert FindingKind.entry in kinds


def test_replay_finds_a_stale_question_for_a_changed_candidate_value() -> None:
    assert FindingKind.stale_question in replayed_with_set([{**ANA, "name": "Cy"}, BO])


def test_from_record_answers_a_record_set_turn() -> None:
    record = decided()
    again = decide_sync(
        LOADED, {"people": [ANA, BO], "who": "Bo"}, {"default": from_record(record, LOADED)}
    ).record
    assert entries(again)[WHICH]["value"] == entries(record)[WHICH]["value"]
    assert entries(again)[grounded_ref(OUT, "Ana")]["value"]["probabilities"] == {
        "yes": 0.9,
        "no": 0.1,
    }


def test_from_record_refuses_a_label_set_whose_candidate_values_changed() -> None:
    record = decided()
    again = decide_sync(
        LOADED,
        {"people": [ANA, {**BO, "name": "Cy"}], "who": "Bo"},
        {"default": from_record(record, LOADED)},
    ).record
    assert entries(again)[WHICH]["marker"]["cause"]["code"] == "unsupported"


def test_from_record_takes_a_label_set_whose_other_fields_changed() -> None:
    record = decided()
    again = decide_sync(
        LOADED,
        {"people": [{**ANA, "age": 99}, BO], "who": "Bo"},
        {"default": from_record(record, LOADED)},
    ).record
    assert entries(again)[WHICH]["value"] == entries(record)[WHICH]["value"]


def test_the_ledger_refuses_sets_in_a_format_5_record() -> None:
    from format_6_design import DESIGN as PLAIN
    from format_6_design import STATE as PLAIN_STATE
    from format_6_design import binding as plain_binding
    from ledger_logs import before_key_order

    record = decide_sync(load(PLAIN), PLAIN_STATE, {"default": plain_binding}).record
    doc = before_key_order(
        {"log": copy.deepcopy(record.log), "unhashed": copy.deepcopy(record.unhashed)}, 5
    )
    doc["log"][1]["sets"] = {}
    with pytest.raises(InvalidRecord) as info:
        read_record(rehashed(doc))
    assert "holds sets, which a format-5 record does not" in str(info.value)


def test_a_slot_on_a_record_set_renders_its_candidate_values() -> None:
    from ommat import Context, ContextEntry

    ask = Source(
        {"q:all/ok": Question("noul", Text("All {all} ok?", [slot("all", ref=SET)]))},
        needs=[SET],
    )
    d = design(
        sources={**design().sources, "all": ask},
        context=Context([ContextEntry(SET, Text("E {x}", [slot("x", ref=SET)]))]),
        classes={},
    )
    record = decided(loaded=load(d))
    found = [s for b in ASKED for s in b if "q:all/ok" in s.questions]
    assert found[0].questions["q:all/ok"].wording == "All Ana, Bo ok?"
    plan = next(i for i in record.log if i["item"] == "plan")
    assert plan["context"]["entries"][0]["text"] == "E Ana, Bo"
