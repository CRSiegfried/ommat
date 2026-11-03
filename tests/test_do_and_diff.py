# pyright: reportPrivateUsage=false
"""Do-and-diff's edits and ``intervene``: each job's edit gives exactly the findings
downstream of it, its refusals come in their order before any replay, the copy carries its own
hash and is read back, neither input changes, and no job's output reaches a ``Record``."""

import asyncio
import copy
import dataclasses
import json
import pathlib
from collections.abc import Callable
from typing import Any, cast

import pytest

import extraction
import test_context as context
import test_deciding_inputs as deciding
import test_grounded_rules as grounded
import test_label_sets as label_sets
from designs import minimal
from format_8_design import DESIGN
from ommat import (
    Cause,
    EntryRefusal,
    Failed,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Question,
    Record,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    decide_sync,
    load,
    read_record,
)
from ommat import types as t
from ommat._canonical import decode
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _diff,
    _edit,
    constant_rule,
    constant_rule_sync,
    fail_call,
    fail_call_sync,
    intervene,
    intervene_sync,
    set_entry,
    set_entry_sync,
    set_state_field,
    set_state_field_sync,
    use_answers_sync,
)
from scripted import Scripted, returns
from test_decide import ASK, BODY, LOADED, LOOK, RULES, SPAM, STATE, echo, go, route, slice_design
from test_extraction_turn import ran as extraction_ran
from test_replay import entry
from test_use_answers import HAM, REFUSING, broken, diff, live, unfinished

K = FindingKind
R = RefusalReason
NO = {"yes": 0.1, "no": 0.9}
YES = {"yes": 0.9, "no": 0.1}
TESTS = pathlib.Path(__file__).parent


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any replay fails the test: the refusal came before it."""

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)


def refusal(job: Callable[..., Any], *args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        job(*args, **kwargs)
    return raised.value


def kinds(found: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    return [(f["kind"], f.get("ref")) for f in found]


def downstream(record: Record, other: Record, *edited: str) -> list[Any]:
    """The findings a rerun reproducing ``other``'s turn gives against ``record``, less the
    edited entries themselves, which the copy holds as set."""
    return [f for f in diff(record, other, REFUSING) if f.get("ref") not in edited]


# ---------------------------------------------------------------------------------------------
# set_entry


def test_setting_an_entry_to_its_recorded_value_gives_no_finding() -> None:
    record = live(REFUSING)
    result = set_entry_sync(record, REFUSING, "q:spam", YES)
    assert result == {"entry": entry(record.log, "q:spam"), "findings": []}
    assert set_entry_sync(record, REFUSING, "t:body", "text of the mail")["findings"] == []


def test_setting_an_answer_gives_exactly_the_findings_downstream_of_it() -> None:
    record, other = live(REFUSING), live(REFUSING, ask=returns(HAM))
    result = set_entry_sync(record, REFUSING, "q:spam", NO)
    assert result["entry"] == entry(other.log, "q:spam")
    assert result["findings"] == downstream(record, other, "q:spam")
    assert kinds(result["findings"]) == [(K.entry, "d:route"), (K.plan, None), (K.verdict, None)]


def test_a_t_entry_is_set_to_a_value_of_its_type() -> None:
    result = set_entry_sync(go().record, LOADED, "t:size", 9)
    assert result["entry"]["value"] == 9
    assert result["findings"] == []  # nothing reads t:size


@pytest.mark.parametrize("code", ["refused", "unsupported", "invalid-answer", "timed-out"])
def test_an_entry_set_to_a_binding_marker_names_its_ref_alone(code: str) -> None:
    record = live(REFUSING)
    result = set_entry_sync(record, REFUSING, "q:spam", Unavailable(Cause(code, ["x", "y"])))
    assert result["entry"]["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": code, "refs": ["q:spam"]},
    }
    assert kinds(result["findings"]) == [(K.entry, "d:route"), (K.plan, None), (K.verdict, None)]


def test_a_nullable_t_entry_is_set_to_null() -> None:
    record = deciding.nullable_record()
    result = set_entry_sync(record, deciding.NULLABLE, "t:bb", None)
    assert result["entry"]["value"] is None
    assert kinds(result["findings"]) == [(K.entry, "d:m")]


def test_a_t_entry_set_to_a_marker_gives_the_findings_downstream_of_it() -> None:
    result = set_entry_sync(go().record, LOADED, "t:body", Unavailable(Cause("refused")))
    assert result["entry"]["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "refused", "refs": ["t:body"]},
    }
    assert result["findings"] == []  # echo declares t:body but never reads it


def test_a_candidate_valued_answer_takes_the_labels_as_asked() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    asked = entry(record.log, "q:sender")["value"]["labels"]
    assert asked == ["none", "several", "ann", "bo"]
    given = {"none": 0.1, "several": 0.1, "ann": 0.1, "bo": 0.7}
    result = set_entry_sync(record, loaded, "q:sender", given)
    assert result["entry"]["value"] == {"kind": "choice", "labels": asked, "probabilities": given}
    assert result["findings"] == []
    exc = refusal(set_entry_sync, record, loaded, "q:sender", {"none": 1.0})
    assert (exc.reason, exc.ref) == (R.nonconforming, "q:sender")


def test_candidate_labels_go_at_at_and_by_a_record_sets_candidate_field() -> None:
    members = [{"name": "ann", "age": 3}, {"name": "bo", "age": 4}]
    people = t.list_(t.record(name=t.string, age=t.integer()))
    finder = Source(
        {label_sets.SET: {"type": people, "candidates": "name"}}, needs=["t:state"], oracle="tools"
    )
    labels = {"fixed": label_sets.FIXED, "candidates": label_sets.SET, "at": 1}
    design = dataclasses.replace(
        label_sets.design(), sources={"finder": finder, "pick": label_sets.asks(labels)}
    )
    tools = Scripted({"finder": returns({label_sets.SET: members})})
    record = label_sets.run([], d=design, tools=tools)
    assert record.log[1]["sets"] == {label_sets.SET: "name"}
    given = {"none": 0.1, "ann": 0.1, "bo": 0.7, "several": 0.1}
    result = set_entry_sync(record, load(design), "q:sender", given)
    assert result["entry"]["value"]["labels"] == ["none", "ann", "bo", "several"]


def test_a_candidate_set_whose_members_repeat_is_nonconforming() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    ref = label_sets.SET
    assert set_entry_sync(record, loaded, ref, ["bo", "ann"])["entry"]["value"] == ["bo", "ann"]
    exc = refusal(set_entry_sync, record, loaded, ref, ["ann", "ann"])
    assert (exc.reason, exc.ref) == (R.nonconforming, ref)


def test_an_extraction_answer_is_set_its_spans_bounded_by_the_text_as_logged() -> None:
    record = extraction_ran()
    loaded = extraction.loaded()
    amount = {**extraction.AMOUNT, "value": 7, "guess": 7}
    result = set_entry_sync(record, loaded, "q:amount", amount)
    assert result["entry"]["value"] == {
        "kind": "extraction",
        "value_type": "number",
        "is_list": False,
        **amount,
    }
    assert kinds(result["findings"]) == [(K.entry, "d:amount"), (K.plan, None)]
    dates = {**extraction.DATES, "value": ["2024-03-01"], "guess": ["2024-03-01"]}
    listed = set_entry_sync(record, loaded, "q:dates", dates)["entry"]["value"]
    assert (listed["is_list"], listed["value"], listed["span"]) == (
        True,
        ["2024-03-01"],
        [[11, 21]],
    )
    past = len(extraction.BODY) + 1
    for ref, value in (
        ("q:amount", {**amount, "span": [4, past]}),
        ("q:dates", {**dates, "span": [[11, past]]}),
        ("q:dates", {**dates, "value": "2024-03-01"}),
    ):
        exc = refusal(set_entry_sync, record, loaded, ref, value)
        assert (exc.reason, exc.ref) == (R.nonconforming, ref)


def test_a_score_answer_is_set_over_its_levels() -> None:
    levels = ("low", "mid", "high")
    loaded = load(
        minimal(sources={"s": Source({"q:urgency": Question("score", Text("How?"), levels)})})
    )
    given = {"low": 0.25, "mid": 0.25, "high": 0.5}
    record = decide_sync(loaded, {}, {"default": Scripted({"s": returns({"q:urgency": given})})})
    changed = {"low": 0.7, "mid": 0.2, "high": 0.1}
    result = set_entry_sync(record.record, loaded, "q:urgency", changed)
    assert result["entry"]["value"] == {
        "kind": "score",
        "labels": list(levels),
        "probabilities": changed,
    }
    exc = refusal(set_entry_sync, record.record, loaded, "q:urgency", YES)
    assert exc.reason is R.nonconforming


@pytest.mark.usefixtures("no_replay")
def test_values_alone_are_checked_in_list_order() -> None:
    bad_q, bad_t = {"ref": "q:spam", "value": {"yes": 0.9}}, {"ref": "t:size", "value": "nine"}
    for edits, ref in (([bad_q, bad_t], "q:spam"), ([bad_t, bad_q], "t:size")):
        exc = refusal(intervene_sync, go().record, LOADED, edits)
        assert (exc.reason, exc.ref) == (R.nonconforming, ref)


def text_design() -> LoadedDesign:
    """The extraction design with its amount read from ``t:doc``, a source's entry."""
    amount = extraction.question(text={"ref": "t:doc"})
    fetch = Source({"t:doc": t.string}, needs=["t:state"])
    sources = {"read": extraction.reader(**{"q:amount": amount}), "fetch": fetch}
    return load(extraction.design(sources=sources))


def test_a_span_has_no_bound_where_its_text_holds_a_marker() -> None:
    loaded = text_design()
    answers = {"q:amount": extraction.AMOUNT, "q:dates": extraction.DATES}
    binding = Scripted({"fetch": returns({"t:doc": "Owe 120"}), "read": returns(answers)})
    record = decide_sync(loaded, extraction.STATE, {"default": binding}).record
    far = {"ref": "q:amount", "value": {**extraction.AMOUNT, "span": [4, 99]}}
    exc = refusal(intervene_sync, record, loaded, [far])
    assert (exc.reason, exc.ref) == (R.nonconforming, "q:amount")
    result = intervene_sync(record, loaded, [far, {"source": "fetch", "code": "raised"}])
    assert result["items"][-1]["value"]["span"] == [4, 99]


@pytest.mark.parametrize(
    ("ref", "value", "reason"),
    [
        ("q:nope", YES, R.not_found),
        (5, YES, R.not_found),
        ("d:route", "no", R.null_call),
        ("t:state", STATE, R.null_call),
        ("d:route", Unasked(Cause("when-false")), R.null_call),
        ("q:spam", Unasked(Cause("when-false")), R.marker),
        ("q:spam", Unavailable(Cause("not-returned")), R.marker),
        ("q:spam", Unsure(Cause("abstained")), R.marker),
        ("q:spam", Unavailable, R.marker),
        ("q:spam", Unsure(Cause("refused")), R.marker),
        ("q:spam", Unasked(Cause("refused")), R.marker),
        ("q:spam", {"yes": 0.9}, R.nonconforming),
        ("q:spam", "yes", R.nonconforming),
        ("t:size", "nine", R.nonconforming),
    ],
    ids=[
        "no entry",
        "a ref not a string",
        "a rule's entry",
        "the state entry",
        "null call before marker",
        "Unasked",
        "an engine code",
        "Unsure",
        "a marker class",
        "Unsure with a binding code",
        "Unasked with a binding code",
        "an answer missing a label",
        "not an answer",
        "not of the type",
    ],
)
@pytest.mark.usefixtures("no_replay")
def test_set_entry_refusals(ref: Any, value: object, reason: RefusalReason) -> None:
    exc = refusal(set_entry_sync, go().record, LOADED, ref, value)
    assert exc.reason is reason
    assert exc.ref == (ref if type(ref) is str else None)
    assert (exc.source, exc.path, exc.turn) == (None, None, None)


class Code(str):
    """A ``str`` subclass, which no job takes for a string."""


def failed_record() -> Record:
    return live(ask=Failed(Cause("raised", ["q:spam"])))


@pytest.mark.usefixtures("no_replay")
def test_an_entry_of_a_failed_call_is_failed_call_before_marker() -> None:
    for value in (YES, Unasked(Cause("when-false"))):
        exc = refusal(set_entry_sync, failed_record(), LOADED, "q:spam", value)
        assert (exc.reason, exc.ref) == (R.failed_call, "q:spam")


# ---------------------------------------------------------------------------------------------
# fail_call


def test_failing_a_call_gives_exactly_the_findings_downstream_of_it() -> None:
    record = live(REFUSING)
    other = live(REFUSING, ask=Failed(Cause("raised", ["q:spam"])))
    result = fail_call_sync(record, REFUSING, "body", "raised")
    assert result["findings"] == downstream(record, other, "q:spam", "t:body")
    assert kinds(result["findings"]) == [(K.entry, "d:route"), (K.plan, None), (K.verdict, None)]
    call = result["call"]
    recorded = next(i for i in other.log if i["item"] == "call" and "ask" in i["sources"])
    assert call == recorded
    assert call["outcome"] == {"failed": {"code": "raised", "refs": ["q:spam", "t:body"]}}
    assert call["claimed"] == []
    assert call["provider"] is None


@pytest.mark.parametrize("code", ["raised", "timed-out", "invalid-return", "unexpected-ref"])
def test_failing_a_call_nothing_reads_gives_no_finding(code: str) -> None:
    result = fail_call_sync(go().record, LOADED, "look", code)
    assert result["findings"] == []
    assert result["call"]["outcome"]["failed"] == {"code": code, "refs": ["t:size"]}


def test_a_failed_calls_cause_names_its_refs_in_code_point_order() -> None:
    loaded = slice_design(sources={"body": BODY, "zz": ASK, "look": LOOK})
    record = live(loaded, body=returns(SPAM))
    written = [i["ref"] for i in record.log if i["item"] == "entry" and i["call"] == 3]
    assert written == ["t:body", "q:spam"]
    call = fail_call_sync(record, loaded, "zz", "raised")["call"]
    assert call["outcome"]["failed"]["refs"] == ["q:spam", "t:body"]


@pytest.mark.parametrize(
    ("source", "code", "reason"),
    [
        ("nope", "raised", R.not_found),
        (["ask"], "raised", R.not_found),
        ("nope", "refused", R.not_found),
        ("ask", "refused", R.fault_code),
        ("ask", "not-returned", R.fault_code),
        ("ask", 5, R.fault_code),
        ("ask", Code("raised"), R.fault_code),
    ],
    ids=["no call", "not a string", "not-found first", "a ref's code", "engine", "int", "subclass"],
)
@pytest.mark.usefixtures("no_replay")
def test_fail_call_refusals(source: Any, code: Any, reason: RefusalReason) -> None:
    exc = refusal(fail_call_sync, go().record, LOADED, source, code)
    assert exc.reason is reason
    assert exc.source == (source if type(source) is str else None)
    assert (exc.ref, exc.path) == (None, None)


@pytest.mark.usefixtures("no_replay")
def test_a_failed_call_is_failed_call_before_fault_code() -> None:
    for code in ("raised", "refused"):
        exc = refusal(fail_call_sync, failed_record(), LOADED, "body", code)
        assert (exc.reason, exc.source) == (R.failed_call, "body")


# ---------------------------------------------------------------------------------------------
# set_state_field


def test_setting_a_state_field_to_its_value_gives_no_finding() -> None:
    record = go().record
    result = set_state_field_sync(record, LOADED, "t:state.sender.address", "a@example.com")
    assert result == {"entry": entry(record.log, "t:state"), "findings": []}


def test_setting_a_state_field_stales_each_question_that_needs_it() -> None:
    record, other = live(REFUSING), live(REFUSING, state={**STATE, "subject": "bye"})
    result = set_state_field_sync(record, REFUSING, "t:state.subject", "bye")
    assert result["entry"] == entry(other.log, "t:state")
    # Each source needing the subject was asked about the old one, so none is answered.
    found = result["findings"]
    assert [(f["kind"], f.get("ref"), f.get("clause")) for f in found[:3]] == [
        (K.stale_question, "q:spam", "needs"),
        (K.stale_question, "t:body", "needs"),
        (K.stale_question, "t:size", "needs"),
    ]
    for ref in ("q:spam", "t:body", "t:size"):
        assert entry_finding(found, ref)["replayed"]["marker"]["cause"]["code"] == "not-returned"
    assert entry_finding(found, "d:echo")["replayed"]["value"] == "bye"
    assert kinds(found[3:]) == [
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "q:spam"),
        (K.entry, "t:body"),
        (K.entry, "t:size"),
        (K.plan, None),
        (K.verdict, None),
    ]


def entry_finding(found: list[Any], ref: str) -> Any:
    return next(f for f in found if f["kind"] is K.entry and f["ref"] == ref)


def gate(r: Any) -> str:
    return "on" if r.read("t:state.weight") > 1 else "off"


def test_a_call_an_edit_fails_asked_without_a_source_of_it_is_stale() -> None:
    rules = {
        **RULES,
        "gate": Rule("d:gate", t.label("on", "off"), gate, version="1", reads=["t:state.weight"]),
    }
    body = Source(
        BODY.provides,
        needs=BODY.needs,
        oracle="mail",
        requirement=SourceRequirement("d:gate", ["on"]),
    )
    loaded = slice_design(sources={"ask": ASK, "body": body}, rules=rules)
    record = live(loaded)
    failed = {"source": "ask", "code": "raised"}
    alone = intervene_sync(record, loaded, [failed])["findings"]
    assert K.stale_question not in [f["kind"] for f in alone]
    # Shutting body's gate leaves ask asked alone, which the failed call says nothing of.
    found = intervene_sync(record, loaded, [failed, {"path": "t:state.weight", "value": 0}])
    stale = [(f["ref"], f["clause"]) for f in found["findings"] if f["kind"] is K.stale_question]
    assert stale == [("q:spam", "no-answer")]


def test_a_call_an_edit_fails_is_never_stale_though_its_needs_change() -> None:
    edits = [{"source": "ask", "code": "raised"}, {"path": "t:state.subject", "value": "bye"}]
    found = intervene_sync(live(REFUSING), REFUSING, edits)["findings"]
    # ask and body share the failed call, so only look, needing the whole state, is stale.
    assert [(f["kind"], f.get("ref"), f.get("clause")) for f in found[:1]] == [
        (K.stale_question, "t:size", "needs")
    ]
    assert K.stale_question not in [f["kind"] for f in found[1:]]
    route = entry_finding(found, "d:route")["replayed"]["marker"]["cause"]
    assert route == {"code": "raised", "refs": ["q:spam", "t:body"]}


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        ("t:state.nope", "x", R.not_found),
        ("t:state.subject.inner", "x", R.not_found),
        ("t:state.subject.ell", "x", R.not_found),
        ("subject", "x", R.not_found),
        ("t:state", STATE, R.not_found),
        ("t:body.x", "x", R.not_found),
        (None, "x", R.not_found),
        ("t:state.nope", Unasked(Cause("when-false")), R.not_found),
        ("t:state.subject", Unavailable(Cause("refused")), R.marker),
        ("t:state.subject", Unsure, R.marker),
        ("t:state.subject", 5, R.nonconforming),
        ("t:state.sender", {"address": 1}, R.nonconforming),
    ],
    ids=[
        "no field",
        "below a string",
        "below a string holding the name",
        "no prefix",
        "the whole state",
        "another ref",
        "not a string",
        "not-found before marker",
        "a marker",
        "a marker class",
        "a number for a string",
        "a nested misfit",
    ],
)
@pytest.mark.usefixtures("no_replay")
def test_set_state_field_refusals(path: Any, value: object, reason: RefusalReason) -> None:
    exc = refusal(set_state_field_sync, go().record, LOADED, path, value)
    assert exc.reason is reason
    assert exc.path == (path if type(path) is str else None)
    assert (exc.ref, exc.source) == (None, None)


# ---------------------------------------------------------------------------------------------
# constant_rule


def test_a_constant_rule_gives_exactly_the_findings_downstream_of_it() -> None:
    record, other = live(REFUSING), live(REFUSING, ask=returns(HAM))
    result = constant_rule_sync(record, REFUSING, "d:route", "no")
    found = result["findings"]
    assert kinds(found) == [
        (K.design_mismatch, None),
        (K.entry, "d:route"),
        (K.plan, None),
        (K.verdict, None),
    ]
    modified = load(result["document"])
    assert found[0] == {
        "kind": K.design_mismatch,
        "recorded": REFUSING.design_hash,
        "replayed": modified.design_hash,
    }
    route = found[1]["replayed"]
    assert (route["value"], route["rows"]) == ("no", [1])
    assert found[1]["rule"] is None  # a table now provides it
    assert found[2:] == downstream(record, other, "q:spam", "d:route")


def test_the_modified_document_replaces_the_declaration_under_its_name() -> None:
    document = constant_rule_sync(go().record, LOADED, "d:route", Unsure)["document"]
    rules = decode(LOADED.document)["rules"]  # type: ignore[index]
    assert document["rules"]["echo"] == rules["echo"]
    assert document["rules"]["route"] == {
        "form": "table",
        "provides": "d:route",
        "outputs": rules["route"]["outputs"],
        "candidates": False,
        "grounded": None,
        "target": None,
        "hit": "U",
        "order": None,
        "inputs": [{"read": "q:spam", "helper": None, "params": {}}],
        "rows": [{"cells": ["-"], "output": {"marker": "Unsure"}}],
    }
    assert json.loads(json.dumps(document)) == document


def test_a_constant_rule_at_its_recorded_value_changes_only_how_the_entry_was_derived() -> None:
    found = constant_rule_sync(go().record, LOADED, "d:route", "yes")["findings"]
    assert kinds(found) == [(K.design_mismatch, None), (K.entry, "d:route")]
    assert found[1]["recorded"]["value"] == found[1]["replayed"]["value"] == "yes"


def test_a_constant_unsure_rule_writes_unsure() -> None:
    found = constant_rule_sync(go().record, LOADED, "d:route", Unsure)["findings"]
    assert found[1]["replayed"]["marker"]["kind"] == "Unsure"


def test_a_table_rule_is_replaced_over_its_inputs_reads() -> None:
    loaded = load(DESIGN)
    record = decide_sync(loaded, {"score": 0.9, "name": "n"}, {}).record
    result = constant_rule_sync(record, loaded, "d:level", "low")
    assert result["document"]["rules"]["level"]["inputs"] == [
        {"read": "t:state.score", "helper": None, "params": {}}
    ]
    assert kinds(result["findings"]) == [
        (K.design_mismatch, None),
        (K.entry, "d:level"),
        (K.plan, None),
        (K.verdict, None),
    ]


@pytest.mark.parametrize(
    ("rule", "output", "reason"),
    [
        ("d:nope", "no", R.not_found),
        ("route", "no", R.not_found),
        (None, "no", R.not_found),
        ("d:route", Unavailable(Cause("refused")), R.marker),
        ("d:route", Unasked, R.marker),
        ("d:route", Unsure(Cause("abstained")), R.marker),
        ("d:route", "maybe", R.invalid_design),
    ],
    ids=[
        "no rule",
        "the rule's name",
        "not a string",
        "Unavailable",
        "Unasked",
        "an instance",
        "x",
    ],
)
@pytest.mark.usefixtures("no_replay")
def test_constant_rule_refusals(rule: Any, output: object, reason: RefusalReason) -> None:
    exc = refusal(constant_rule_sync, go().record, LOADED, rule, output)
    assert exc.reason is reason
    assert exc.ref == (rule if type(rule) is str else None)
    if reason is R.invalid_design:
        assert type(exc.error) is InvalidDesign
        assert exc.__cause__ is exc.error
    else:
        assert exc.error is None


@pytest.mark.parametrize("rule", [grounded.SET, grounded.FIT, grounded.LEVEL])
@pytest.mark.usefixtures("no_replay")
def test_a_candidate_set_or_grounded_rule_is_candidate_rule_before_marker(rule: str) -> None:
    record = grounded.decided()
    for output in ("x", Unasked):
        exc = refusal(constant_rule_sync, record, grounded.LOADED, rule, output)
        assert (exc.reason, exc.ref) == (R.candidate_rule, rule)


def test_a_constant_rule_loads_with_the_callers_functions() -> None:
    record = go().record
    same = constant_rule_sync(record, LOADED, "d:route", "no", functions={"echo": echo})
    assert same == constant_rule_sync(record, LOADED, "d:route", "no")
    exc = refusal(constant_rule_sync, record, LOADED, "d:route", "no", functions={"echo": route})
    assert exc.reason is R.invalid_design
    assert "echo" in str(exc.error)


def never(finding: object) -> bool:
    return False


def test_a_plan_change_of_context_alone_is_no_finding(monkeypatch: pytest.MonkeyPatch) -> None:
    record = context.run().record
    loaded = context.loaded()
    path, value = "t:state.order.status", "shipped"
    found = set_state_field_sync(record, loaded, path, value)["findings"]
    assert kinds(found) == [(K.entry, "d:status")]
    monkeypatch.setattr(_diff, "_context_alone", never)
    found = set_state_field_sync(record, loaded, path, value)["findings"]
    assert kinds(found) == [(K.entry, "d:status"), (K.plan, None)]


def test_an_output_outside_the_value_domain_is_invalid_design() -> None:
    record = go().record
    for given in ({1: "x"}, {"a": {2: "x"}}):
        exc = refusal(constant_rule_sync, record, LOADED, "d:route", given)
        assert (exc.reason, type(exc.error)) == (R.invalid_design, InvalidDesign)
        exc = refusal(intervene_sync, record, LOADED, [], rule={"rule": "d:route", "output": given})
        assert exc.reason is R.invalid_design


def test_an_output_holding_bytes_is_invalid_design_saying_why() -> None:
    for given in (b"x", {"a": [b"x"]}):
        exc = refusal(constant_rule_sync, go().record, LOADED, "d:route", given)
        assert (exc.reason, type(exc.error)) == (R.invalid_design, InvalidDesign)
        assert str(exc).endswith("a table has no bytes literal")


@pytest.mark.usefixtures("no_replay")
def test_a_rule_reading_a_grounded_ref_is_candidate_rule() -> None:
    exc = refusal(constant_rule_sync, grounded.decided(), grounded.LOADED, "d:summary", "x")
    assert (exc.reason, exc.ref) == (R.candidate_rule, "d:summary")


def nested(r: Any) -> Any:
    return [["a"], ["b"]]


def test_an_output_holding_one_list_at_two_places_is_copied() -> None:
    nest = Rule("d:nest", t.list_(t.list_(t.string)), nested, version="1", reads=["t:body"])
    loaded = slice_design(rules={**RULES, "nest": nest})
    shared = ["x"]
    result = constant_rule_sync(go(loaded).record, loaded, "d:nest", [shared, shared])
    output = result["document"]["rules"]["nest"]["rows"][0]["output"]
    assert output == {"literal": [["x"], ["x"]]}
    assert kinds(result["findings"]) == [(K.design_mismatch, None), (K.entry, "d:nest")]


def test_a_table_reading_one_ref_twice_is_replaced_over_it_once() -> None:
    twice = Table(
        "d:level",
        t.label("high", "low"),
        "F",
        ["t:state.score", "t:state.score"],
        [[">= bar", "-", "high"], ["-", "-", "low"]],
    )
    loaded = load(dataclasses.replace(DESIGN, rules={"level": twice}))
    record = decide_sync(loaded, {"score": 0.9, "name": "n"}, {}).record
    table = constant_rule_sync(record, loaded, "d:level", "low")["document"]["rules"]["level"]
    assert table["inputs"] == [{"read": "t:state.score", "helper": None, "params": {}}]
    assert table["rows"] == [{"cells": ["-"], "output": {"literal": "low"}}]


# ---------------------------------------------------------------------------------------------
# intervene


def test_several_edits_replay_once_on_one_copy_in_any_order() -> None:
    record = live(REFUSING)
    edits = [
        {"path": "t:state.subject", "value": "bye"},
        {"ref": "q:spam", "value": NO},
        {"source": "look", "code": "raised"},
    ]
    result = intervene_sync(record, REFUSING, edits)
    assert intervene_sync(record, REFUSING, edits[::-1]) == result
    assert result["document"] is None
    assert [(i["item"], i.get("ref"), i.get("sources")) for i in result["items"]] == [
        ("entry", "t:state", None),
        ("call", None, ["look"]),
        ("entry", "q:spam", None),
        ("entry", "t:size", None),
    ]
    # The set answer and the failed call are pinned; body, needing the subject, is stale.
    assert kinds(result["findings"]) == [
        (K.stale_question, "t:body"),
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "t:body"),
        (K.plan, None),
        (K.verdict, None),
    ]


def test_intervene_matches_each_form_alone() -> None:
    record = live(REFUSING)
    alone = set_entry_sync(record, REFUSING, "q:spam", NO)
    both = intervene_sync(record, REFUSING, [{"ref": "q:spam", "value": NO}])
    assert both == {"items": [alone["entry"]], "document": None, "findings": alone["findings"]}
    alone = constant_rule_sync(record, REFUSING, "d:route", "no")
    both = intervene_sync(record, REFUSING, [], rule={"rule": "d:route", "output": "no"})
    assert both == {"items": [], **alone}


def test_edits_and_a_rule_replay_together_under_the_modified_design() -> None:
    record = go().record
    result = intervene_sync(
        record,
        LOADED,
        [{"path": "t:state.subject", "value": "bye"}],
        rule={"rule": "d:route", "output": "no"},
    )
    assert result["document"]["rules"]["route"]["form"] == "table"
    assert kinds(result["findings"]) == [
        (K.design_mismatch, None),
        (K.stale_question, "q:spam"),
        (K.stale_question, "t:body"),
        (K.stale_question, "t:size"),
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "q:spam"),
        (K.entry, "t:body"),
        (K.entry, "t:size"),
        (K.plan, None),
    ]


def test_two_disjoint_state_fields_are_both_set() -> None:
    edits = [
        {"path": "t:state.subject", "value": "bye"},
        {"path": "t:state.sender.address", "value": "b@x"},
    ]
    (state,) = intervene_sync(go().record, LOADED, edits)["items"]
    assert (state["value"]["subject"], state["value"]["sender"]) == ("bye", {"address": "b@x"})


def test_sibling_fields_sharing_a_prefix_are_not_one_target() -> None:
    loaded = slice_design(state=t.record(sub=t.string, subject=t.string))
    record = go(loaded, {"sub": "a", "subject": "b"}).record
    edits = [{"path": "t:state.sub", "value": "x"}, {"path": "t:state.subject", "value": "y"}]
    (state,) = intervene_sync(record, loaded, edits)["items"]
    assert state["value"] == {"sub": "x", "subject": "y"}


def test_an_answer_over_a_set_another_edit_sets_takes_its_labels_in_either_order() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    over_cy = {"none": 0.1, "several": 0.1, "ann": 0.7, "cy": 0.1}
    edits = [{"ref": label_sets.SET, "value": ["ann", "cy"]}, {"ref": "q:sender", "value": over_cy}]
    result = intervene_sync(record, loaded, edits)
    assert intervene_sync(record, loaded, edits[::-1]) == result
    assert result["items"][1]["value"]["labels"] == ["none", "several", "ann", "cy"]
    assert result["findings"] == []
    over_bo = {"none": 0.1, "several": 0.1, "ann": 0.7, "bo": 0.1}
    gone = Unavailable(Cause("raised", []))
    for candidates, answer in ((["ann", "cy"], over_bo), (gone, over_cy)):
        pair = [
            {"ref": label_sets.SET, "value": candidates},
            {"ref": "q:sender", "value": answer},
        ]
        for given in (pair, pair[::-1]):
            exc = refusal(intervene_sync, record, loaded, given)
            assert (exc.reason, exc.ref) == (R.nonconforming, "q:sender")


@pytest.mark.usefixtures("no_replay")
def test_an_invalid_set_is_blamed_on_its_own_edit_in_either_order() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    over_cy = {"none": 0.1, "several": 0.1, "ann": 0.7, "cy": 0.1}
    edits = [{"ref": label_sets.SET, "value": ["cy", "cy"]}, {"ref": "q:sender", "value": over_cy}]
    for given in (edits, edits[::-1]):
        exc = refusal(intervene_sync, record, loaded, given)
        assert (exc.reason, exc.ref) == (R.nonconforming, label_sets.SET)


def test_a_span_is_bounded_by_the_text_another_edit_sets_in_either_order() -> None:
    record, loaded = extraction_ran(), extraction.loaded()
    owe = {"path": "t:state.body", "value": "Owe"}
    within = {"ref": "q:amount", "value": {**extraction.AMOUNT, "span": [0, 3]}}
    past = {"ref": "q:amount", "value": extraction.AMOUNT}  # its span [4, 7]
    assert intervene_sync(record, loaded, [owe, within]) == intervene_sync(
        record, loaded, [within, owe]
    )
    for given in ([owe, past], [past, owe]):
        exc = refusal(intervene_sync, record, loaded, given)
        assert (exc.reason, exc.ref) == (R.nonconforming, "q:amount")


@pytest.mark.parametrize(
    ("edits", "rule", "reason", "named"),
    [
        ({"ref": "q:spam", "value": NO}, None, R.invalid_edit, None),
        ([{"ref": "q:spam"}], None, R.invalid_edit, "edit 0"),
        ([{"ref": "q:spam", "value": NO}, ["q:spam", NO]], None, R.invalid_edit, "edit 1"),
        ([{"ref": "q:spam", "value": NO, "path": "x"}], None, R.invalid_edit, "edit 0"),
        ([{"source": "ask", "value": "raised"}], None, R.invalid_edit, "edit 0"),
        ([], {"rule": "d:route"}, R.invalid_edit, "rule"),
        ([], "d:route", R.invalid_edit, "rule"),
        ([], {}, R.invalid_edit, "rule"),
        ([], [], R.invalid_edit, "rule"),
        ([], "", R.invalid_edit, "rule"),
        ([], None, R.no_edit, None),
        ([{"ref": "q:nope"}], None, R.invalid_edit, "edit 0"),
    ],
    ids=[
        "not a list",
        "a member missing",
        "a list item",
        "an extra member",
        "mixed members",
        "a rule missing output",
        "a rule not an object",
        "an empty rule",
        "a rule an empty list",
        "a rule an empty string",
        "nothing to do",
        "invalid-edit before not-found",
    ],
)
@pytest.mark.usefixtures("no_replay")
def test_intervene_refuses_a_misshapen_or_empty_request(
    edits: Any, rule: Any, reason: RefusalReason, named: str | None
) -> None:
    exc = refusal(intervene_sync, go().record, LOADED, edits, rule=rule)
    assert exc.reason is reason
    if named is not None:
        assert named in str(exc)


@pytest.mark.parametrize(
    ("edits", "named"),
    [
        ([{"ref": "q:spam", "value": NO}, {"ref": "q:spam", "value": YES}], {"ref": "q:spam"}),
        (
            [{"source": "ask", "code": "raised"}, {"source": "body", "code": "timed-out"}],
            {"source": "body"},
        ),
        (
            [{"path": "t:state.subject", "value": "a"}, {"path": "t:state.subject", "value": "b"}],
            {"path": "t:state.subject"},
        ),
        (
            [
                {"path": "t:state.sender.address", "value": "a"},
                {"path": "t:state.sender", "value": {"address": "b"}},
            ],
            {"path": "t:state.sender"},
        ),
        (
            [
                {"path": "t:state.sender", "value": {"address": "b"}},
                {"path": "t:state.sender.address", "value": "a"},
            ],
            {"path": "t:state.sender.address"},
        ),
    ],
    ids=["one ref", "one call", "one path", "a path above", "a path below"],
)
@pytest.mark.usefixtures("no_replay")
def test_two_edits_of_one_target_are_duplicate_target(
    edits: list[Any], named: dict[str, str]
) -> None:
    exc = refusal(intervene_sync, go().record, LOADED, edits)
    assert exc.reason is R.duplicate_target
    assert {k: getattr(exc, k) for k in ("ref", "source", "path") if getattr(exc, k)} == named


@pytest.mark.usefixtures("no_replay")
def test_an_entry_of_a_failed_edits_call_is_conflicting_edits_in_either_order() -> None:
    edits = [{"ref": "t:body", "value": "x"}, {"source": "ask", "code": "raised"}]
    for given in (edits, edits[::-1]):
        exc = refusal(intervene_sync, go().record, LOADED, given)
        assert (exc.reason, exc.ref, exc.source) == (R.conflicting_edits, "t:body", "ask")


@pytest.mark.usefixtures("no_replay")
def test_intervenes_refusals_come_in_their_order() -> None:
    record = go().record
    bad = {"ref": "q:spam"}
    dup = [{"ref": "t:body", "value": "x"}, {"ref": "t:body", "value": "y"}]
    conflict = [{"ref": "q:spam", "value": NO}, {"source": "body", "code": "raised"}]
    own = [{"ref": "d:route", "value": "no"}, {"path": "t:state.nope", "value": 1}]
    rule = {"rule": "d:nope", "output": "x"}

    def reason(edits: Any, rule: Any = None, given: object = record) -> RefusalReason:
        return refusal(intervene_sync, given, LOADED, edits, rule=rule).reason

    assert reason([bad], given=unfinished()) is R.unfinished
    assert reason([*dup, bad]) is R.invalid_edit
    assert reason([*conflict, *dup]) is R.duplicate_target
    assert reason([*own, *conflict]) is R.conflicting_edits
    assert reason(own, rule) is R.null_call
    assert reason(own[::-1], rule) is R.not_found
    assert refusal(intervene_sync, record, LOADED, own[::-1], rule=rule).path == "t:state.nope"
    assert reason([{"ref": "t:body", "value": "x"}], rule) is R.not_found
    assert reason([{"ref": "t:body", "value": "x"}], {"rule": "d:route", "output": 1}) is (
        R.invalid_design
    )


# ---------------------------------------------------------------------------------------------
# Every job: the record checks, the copy, the inputs and the output


def job_calls() -> list[tuple[str, Callable[[object], Any]]]:
    return [
        ("set_entry", lambda r: set_entry_sync(r, LOADED, "q:spam", NO)),
        ("fail_call", lambda r: fail_call_sync(r, LOADED, "ask", "raised")),
        ("set_state_field", lambda r: set_state_field_sync(r, LOADED, "t:state.subject", "b")),
        ("constant_rule", lambda r: constant_rule_sync(r, LOADED, "d:route", "no")),
        (
            "intervene",
            lambda r: intervene_sync(
                r,
                LOADED,
                [{"ref": "q:spam", "value": NO}, {"path": "t:state.subject", "value": "b"}],
                rule={"rule": "d:route", "output": "no"},
            ),
        ),
    ]


JOBS = job_calls()
IDS = [name for name, _ in JOBS]


@pytest.mark.parametrize("job", [j for _, j in JOBS], ids=IDS)
@pytest.mark.usefixtures("no_replay")
def test_every_job_refuses_the_record_in_order_before_any_edit(
    job: Callable[[object], Any],
) -> None:
    other_design = live(slice_design(tools=["archive", "send"]))
    for given, reason in (
        (b"not json", R.invalid_record),
        (unfinished(), R.unfinished),
        (broken(), R.nothing_to_replay),
        (other_design, R.wrong_design),
    ):
        with pytest.raises(EvaluationRefusal) as raised:
            job(given)
        exc = raised.value
        assert exc.reason is reason
        if reason is R.invalid_record:
            assert type(exc.error) is InvalidRecord
        else:
            assert exc.turn == given.record_hash  # type: ignore[union-attr]
            assert str(exc).startswith("record")


@pytest.mark.parametrize(
    "job",
    [
        lambda: set_entry_sync(b"x", object(), "q:spam", NO),  # type: ignore[arg-type]
        lambda: fail_call_sync(b"x", object(), "ask", "raised"),  # type: ignore[arg-type]
        lambda: set_state_field_sync(b"x", object(), "t:state.a", 1),  # type: ignore[arg-type]
        lambda: constant_rule_sync(b"x", object(), "d:route", "no"),  # type: ignore[arg-type]
        lambda: intervene_sync(b"x", object(), []),  # type: ignore[arg-type]
    ],
    ids=IDS,
)
def test_a_loaded_not_exactly_a_loaded_design_is_refused_first(job: Callable[[], Any]) -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        job()


@pytest.mark.parametrize(
    "job", [j for name, j in JOBS if name != "constant_rule"], ids=IDS[:3] + IDS[4:]
)
def test_the_copy_carries_its_own_hash_and_is_read_back(
    job: Callable[[object], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    copies: list[tuple[Any, Record]] = []
    real = _edit._read

    def spy(document: object, name: str) -> Record:
        read = real(document, name)
        if name == "the copy":
            copies.append((document, read))
        return read

    monkeypatch.setattr(_edit, "_read", spy)
    record = go().record
    job(record)
    ((document, read),) = copies
    assert type(document) is dict
    assert document["hash"] == read.record_hash != record.record_hash
    assert read_record(read.document).record_hash == read.record_hash


def test_a_constant_rule_alone_makes_no_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_edit, "_record_copy", None)
    constant_rule_sync(go().record, LOADED, "d:route", "no")


@pytest.mark.parametrize("job", [j for _, j in JOBS], ids=IDS)
def test_neither_input_changes(job: Callable[[object], Any]) -> None:
    record = go().record
    before = (record.document, LOADED.document)
    parsed = json.loads(record.document)
    kept = copy.deepcopy(parsed)
    job(record)
    job(parsed)
    assert (record.document, LOADED.document) == before
    assert parsed == kept


def records_in(value: object) -> list[object]:
    """Every ``Record`` reachable from a plain return, through dicts, lists and tuples."""
    if type(value) is Record:
        return [value]
    if type(value) is dict:
        value = list(cast("dict[object, object]", value).values())
    if type(value) in (list, tuple):
        return [r for v in cast("list[object]", value) for r in records_in(v)]
    return []


ALL_JOBS: list[tuple[str, Callable[[object], Any]]] = [
    *JOBS,
    ("use_answers", lambda r: use_answers_sync(r, LOADED, live(ask=returns(HAM)))),
]


@pytest.mark.parametrize("job", [j for _, j in ALL_JOBS], ids=[n for n, _ in ALL_JOBS])
def test_no_record_is_reachable_from_a_jobs_output(job: Callable[[object], Any]) -> None:
    result = job(go().record)
    assert records_in(result) == []
    assert json.loads(json.dumps(result)) == result
    assert copy.deepcopy(result) == result


# ---------------------------------------------------------------------------------------------
# Records of earlier formats


PROFILE = ["elicitation", "model_reported", "model_requested", "precision", "prompt", "provider"]
NULLED: dict[int, list[str]] = {
    2: ["elicitation", "prompt", "snapshot"],
    3: ["elicitation", "precision", "prompt", "snapshot"],
    4: ["elicitation", "precision", "prompt", "snapshot"],
    5: [*PROFILE, "revision", "runtime"],
    6: [*PROFILE, "revision", "runtime"],
    7: [*PROFILE, "revision", "runtime"],
}
"""The hashed provenance members each stored record's call item holds, which fail_call nulls."""


def stored(fmt: int) -> tuple[LoadedDesign, Record]:
    folder = TESTS / f"format_{fmt}"
    loaded = load((folder / "design.json").read_bytes())
    return loaded, read_record((folder / "record.json").read_bytes())


@pytest.mark.parametrize("fmt", sorted(NULLED))
def test_a_record_of_an_earlier_format_is_copied_and_read_back(fmt: int) -> None:
    loaded, record = stored(fmt)
    asked = next(i for i in record.log if i["item"] == "entry" and i["ref"].startswith("q:"))
    probabilities = asked["value"]["probabilities"]
    same = set_entry_sync(record, loaded, asked["ref"], probabilities)
    assert same["entry"] == asked
    assert [f for f in same["findings"] if f["kind"] is not K.helpers_mismatch] == []
    index, call = next((n, i) for n, i in enumerate(record.log) if i["item"] == "call")
    refs = sorted(i["ref"] for i in record.log if i["item"] == "entry" and i["call"] == index)
    failed = fail_call_sync(record, loaded, call["sources"][0], "raised")["call"]
    nulled = dict[str, object].fromkeys(NULLED[fmt])
    if fmt >= 5:
        nulled["claimed"] = []
    assert failed == {**call, "outcome": {"failed": {"code": "raised", "refs": refs}}, **nulled}
    held = set(call) - {"item", "outcome", "questions", "round", "sources"}
    assert held == set(nulled)


@pytest.mark.parametrize(
    ("fmt", "code", "held"),
    [
        (2, "unsupported", False),
        (3, "unsupported", True),
        (3, "timed-out", False),
        (3, "raised", False),
        (4, "timed-out", True),
        (4, "raised", True),
    ],
)
def test_a_marker_the_records_format_does_not_hold_is_refused_marker(
    fmt: int, code: str, held: bool
) -> None:
    loaded, record = stored(fmt)
    ref = next(i["ref"] for i in record.log if i["item"] == "entry" and i["ref"].startswith("q:"))
    marker = Unavailable(Cause(code, [ref]))
    if held:
        assert set_entry_sync(record, loaded, ref, marker)["entry"]["marker"]["cause"]["code"] == (
            code
        )
    else:
        exc = refusal(set_entry_sync, record, loaded, ref, marker)
        assert (exc.reason, exc.ref) == (R.marker, ref)
        assert f"format-{fmt}" in str(exc)


# ---------------------------------------------------------------------------------------------
# The sync twins


def test_each_job_is_awaitable_and_equals_its_sync_twin() -> None:
    record = go().record

    async def main() -> list[Any]:
        return [
            await set_entry(record, LOADED, "q:spam", NO),
            await fail_call(record, LOADED, "ask", "raised"),
            await set_state_field(record, LOADED, "t:state.subject", "b"),
            await constant_rule(record, LOADED, "d:route", "no"),
            await intervene(record, LOADED, [{"ref": "q:spam", "value": NO}]),
        ]

    assert asyncio.run(main()) == [
        set_entry_sync(record, LOADED, "q:spam", NO),
        fail_call_sync(record, LOADED, "ask", "raised"),
        set_state_field_sync(record, LOADED, "t:state.subject", "b"),
        constant_rule_sync(record, LOADED, "d:route", "no"),
        intervene_sync(record, LOADED, [{"ref": "q:spam", "value": NO}]),
    ]


@pytest.mark.parametrize(
    ("name", "job"),
    [
        ("set_entry", lambda: set_entry_sync(object(), object(), "", 0)),  # type: ignore[arg-type]
        ("fail_call", lambda: fail_call_sync(object(), object(), "", "")),  # type: ignore[arg-type]
        ("set_state_field", lambda: set_state_field_sync(object(), object(), "", 0)),  # type: ignore[arg-type]
        ("constant_rule", lambda: constant_rule_sync(object(), object(), "", 0)),  # type: ignore[arg-type]
        ("intervene", lambda: intervene_sync(object(), object(), [])),  # type: ignore[arg-type]
    ],
)
def test_a_sync_twin_inside_a_running_loop_raises_before_doing_anything(
    name: str, job: Callable[[], Any]
) -> None:
    async def main() -> None:
        with pytest.raises(RuntimeError, match=f"{name}_sync was called inside a running event"):
            job()

    asyncio.run(main())
