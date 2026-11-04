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
from typing import Any

import pytest

import extraction
import test_deciding_inputs as deciding
import test_grounded_rules as grounded
import test_label_sets as label_sets
from designs import minimal
from ommat import (
    Cause,
    Design,
    EntryRefusal,
    Failed,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Parameter,
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
    intervene,
    intervene_sync,
    use_answers_sync,
)
from scripted import Scripted, returns
from test_decide import (
    ASK,
    BODY,
    LOADED,
    LOOK,
    RULES,
    SPAM,
    STATE,
    bindings,
    echo,
    go,
    route,
    slice_design,
)
from test_deciding_inputs import records_in
from test_extraction_turn import ran as extraction_ran
from test_replay import entry
from test_use_answers import CHANGED, HAM, broken, diff, format_7, live, unfinished

K = FindingKind
R = RefusalReason
NO = {"yes": 0.1, "no": 0.9}
YES = {"yes": 0.9, "no": 0.1}
TESTS = pathlib.Path(__file__).parent
TABLED = Design(
    name="tabled",
    state=t.record(score=t.number(0, 1), name=t.string),
    parameters={"bar": Parameter(t.number(0, 1), 0.8)},
    rules={
        "level": Table(
            "d:level",
            t.label("high", "low"),
            "F",
            ["t:state.score"],
            [[">= bar", "high"], ["-", "low"]],
        ),
        "act": Table(
            "d:act", t.label("page", "tell"), "F", ["d:level"], [["high", "page"], ["-", "tell"]]
        ),
    },
    policy="d:act",
)
"""All tables, the policy a table over ``d:level``."""


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any replay fails the test: the refusal came before it."""

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)


def edited(
    record: object,
    loaded: LoadedDesign,
    interventions: object,
    *,
    rerun_design: LoadedDesign | None = None,
    functions: Any = None,
) -> dict[str, Any]:
    """``intervene_sync`` read through its one procedure, so a test sees the edited copy the
    job does not return: ``{"items", "document", "findings", "stale"}``, ``document`` the
    design run's, parsed, when a rule was made constant, else ``None``."""
    read, log = _edit._checked(record, loaded, rerun_design)
    listed = _edit._shaped(interventions, rerun_design)
    done = asyncio.run(_edit._applied(read, log, loaded, listed, functions, rerun_design))
    constant = any(type(i) is _edit._Constant for i in listed)
    return {
        "items": done.items,
        "document": decode(done.design.document) if constant else None,
        "findings": done.findings,
        "stale": done.stale,
    }


def entry_set(record: object, loaded: LoadedDesign, ref: str, value: object) -> Any:
    """``intervene`` with one entry set, read through ``edited``: the edited entry and the
    findings. This helper and the three after it are the test's, not public jobs."""
    out = edited(record, loaded, [{"ref": ref, "value": value}])
    return {"entry": out["items"][0], "findings": out["findings"]}


def call_failed(record: object, loaded: LoadedDesign, source: str, code: str) -> Any:
    """One call failed: the edited call item, which precedes its entries, and the findings."""
    out = edited(record, loaded, [{"source": source, "code": code}])
    return {"call": out["items"][0], "findings": out["findings"]}


def field_set(record: object, loaded: LoadedDesign, path: str, value: object) -> Any:
    """One state field set: the edited state entry and the findings."""
    out = edited(record, loaded, [{"path": path, "value": value}])
    return {"entry": out["items"][0], "findings": out["findings"]}


def rule_constant(
    record: object, loaded: LoadedDesign, rule: str, output: object, *, functions: Any = None
) -> Any:
    """One rule made constant: the modified document and the findings."""
    out = edited(record, loaded, [{"rule": rule, "output": output}], functions=functions)
    return {"document": out["document"], "findings": out["findings"]}


def refusal(job: Callable[..., Any], *args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        job(*args, **kwargs)
    return raised.value


def kinds(found: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    return [(f["kind"], f.get("ref")) for f in found]


def downstream(record: Record, other: Record, *edited: str) -> list[Any]:
    """The findings a rerun reproducing ``other``'s turn gives against ``record``, less the
    edited entries themselves, which the copy holds as set."""
    return [f for f in diff(record, other, LOADED) if f.get("ref") not in edited]


# ---------------------------------------------------------------------------------------------
# An entry set


def test_setting_an_entry_to_its_recorded_value_gives_no_finding() -> None:
    record = live(LOADED)
    result = entry_set(record, LOADED, "q:spam", YES)
    assert result == {"entry": entry(record.log, "q:spam"), "findings": []}
    assert entry_set(record, LOADED, "t:body", "text of the mail")["findings"] == []


def test_setting_an_answer_gives_exactly_the_findings_downstream_of_it() -> None:
    record, other = live(LOADED), live(LOADED, ask=returns(HAM))
    result = entry_set(record, LOADED, "q:spam", NO)
    assert result["entry"] == entry(other.log, "q:spam")
    assert result["findings"] == downstream(record, other, "q:spam")
    assert kinds(result["findings"]) == [(K.entry, "d:act"), (K.entry, "d:route")]


def test_a_t_entry_is_set_to_a_value_of_its_type() -> None:
    result = entry_set(go().record, LOADED, "t:size", 9)
    assert result["entry"]["value"] == 9
    assert result["findings"] == []  # nothing reads t:size


@pytest.mark.parametrize("code", ["refused", "unsupported", "invalid-answer", "timed-out"])
def test_an_entry_set_to_a_binding_marker_names_its_ref_alone(code: str) -> None:
    record = live(LOADED)
    result = entry_set(record, LOADED, "q:spam", Unavailable(Cause(code, ["x", "y"])))
    assert result["entry"]["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": code, "refs": ["q:spam"]},
    }
    assert kinds(result["findings"]) == [(K.entry, "d:act"), (K.entry, "d:route")]


def test_a_nullable_t_entry_is_set_to_null() -> None:
    record = deciding.nullable_record()
    result = entry_set(record, deciding.NULLABLE, "t:bb", None)
    assert result["entry"]["value"] is None
    assert kinds(result["findings"]) == [(K.entry, "d:m")]


def test_a_t_entry_set_to_a_marker_gives_the_findings_downstream_of_it() -> None:
    result = entry_set(go().record, LOADED, "t:body", Unavailable(Cause("refused")))
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
    result = entry_set(record, loaded, "q:sender", given)
    assert result["entry"]["value"] == {"kind": "choice", "labels": asked, "probabilities": given}
    assert result["findings"] == []
    exc = refusal(entry_set, record, loaded, "q:sender", {"none": 1.0})
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
    result = entry_set(record, load(design), "q:sender", given)
    assert result["entry"]["value"]["labels"] == ["none", "ann", "bo", "several"]


def test_a_candidate_set_whose_members_repeat_is_nonconforming() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    ref = label_sets.SET
    assert entry_set(record, loaded, ref, ["bo", "ann"])["entry"]["value"] == ["bo", "ann"]
    exc = refusal(entry_set, record, loaded, ref, ["ann", "ann"])
    assert (exc.reason, exc.ref) == (R.nonconforming, ref)


def test_an_extraction_answer_is_set_its_spans_bounded_by_the_text_as_logged() -> None:
    record = extraction_ran()
    loaded = extraction.loaded()
    amount = {**extraction.AMOUNT, "value": 7, "guess": 7}
    result = entry_set(record, loaded, "q:amount", amount)
    assert result["entry"]["value"] == {
        "kind": "extraction",
        "value_type": "number",
        "is_list": False,
        **amount,
    }
    assert kinds(result["findings"]) == [(K.entry, "d:amount"), (K.entry, "d:reply")]
    dates = {**extraction.DATES, "value": ["2024-03-01"], "guess": ["2024-03-01"]}
    listed = entry_set(record, loaded, "q:dates", dates)["entry"]["value"]
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
        exc = refusal(entry_set, record, loaded, ref, value)
        assert (exc.reason, exc.ref) == (R.nonconforming, ref)


def test_a_score_answer_is_set_over_its_levels() -> None:
    levels = ("low", "mid", "high")
    loaded = load(
        minimal(sources={"s": Source({"q:urgency": Question("score", Text("How?"), levels)})})
    )
    given = {"low": 0.25, "mid": 0.25, "high": 0.5}
    record = decide_sync(loaded, {}, {"default": Scripted({"s": returns({"q:urgency": given})})})
    changed = {"low": 0.7, "mid": 0.2, "high": 0.1}
    result = entry_set(record.record, loaded, "q:urgency", changed)
    assert result["entry"]["value"] == {
        "kind": "score",
        "labels": list(levels),
        "probabilities": changed,
    }
    exc = refusal(entry_set, record.record, loaded, "q:urgency", YES)
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
    result = edited(record, loaded, [far, {"source": "fetch", "code": "raised"}])
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
    exc = refusal(entry_set, go().record, LOADED, ref, value)
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
        exc = refusal(entry_set, failed_record(), LOADED, "q:spam", value)
        assert (exc.reason, exc.ref) == (R.failed_call, "q:spam")


# ---------------------------------------------------------------------------------------------
# A call failed


def test_failing_a_call_gives_exactly_the_findings_downstream_of_it() -> None:
    record = live(LOADED)
    other = live(LOADED, ask=Failed(Cause("raised", ["q:spam"])))
    result = call_failed(record, LOADED, "body", "raised")
    assert result["findings"] == downstream(record, other, "q:spam", "t:body")
    assert kinds(result["findings"]) == [(K.entry, "d:act"), (K.entry, "d:route")]
    call = result["call"]
    recorded = next(i for i in other.log if i["item"] == "call" and "ask" in i["sources"])
    assert call == recorded
    assert call["outcome"] == {"failed": {"code": "raised", "refs": ["q:spam", "t:body"]}}
    assert call["claimed"] == []
    assert call["provider"] is None


@pytest.mark.parametrize("code", ["raised", "timed-out", "invalid-return", "unexpected-ref"])
def test_failing_a_call_nothing_reads_gives_no_finding(code: str) -> None:
    result = call_failed(go().record, LOADED, "look", code)
    assert result["findings"] == []
    assert result["call"]["outcome"]["failed"] == {"code": code, "refs": ["t:size"]}


def test_a_failed_calls_cause_names_its_refs_in_code_point_order() -> None:
    loaded = slice_design(sources={"body": BODY, "zz": ASK, "look": LOOK})
    record = live(loaded, body=returns(SPAM))
    written = [i["ref"] for i in record.log if i["item"] == "entry" and i["call"] == 3]
    assert written == ["t:body", "q:spam"]
    call = call_failed(record, loaded, "zz", "raised")["call"]
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
    exc = refusal(call_failed, go().record, LOADED, source, code)
    assert exc.reason is reason
    assert exc.source == (source if type(source) is str else None)
    assert (exc.ref, exc.path) == (None, None)


@pytest.mark.usefixtures("no_replay")
def test_a_failed_call_is_failed_call_before_fault_code() -> None:
    for code in ("raised", "refused"):
        exc = refusal(call_failed, failed_record(), LOADED, "body", code)
        assert (exc.reason, exc.source) == (R.failed_call, "body")


# ---------------------------------------------------------------------------------------------
# A state field set


def test_setting_a_state_field_to_its_value_gives_no_finding() -> None:
    record = go().record
    result = field_set(record, LOADED, "t:state.sender.address", "a@example.com")
    assert result == {"entry": entry(record.log, "t:state"), "findings": []}


def test_setting_a_state_field_stales_each_question_that_needs_it() -> None:
    record, other = live(LOADED), live(LOADED, state={**STATE, "subject": "bye"})
    result = field_set(record, LOADED, "t:state.subject", "bye")
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
        (K.entry, "d:act"),
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "q:spam"),
        (K.entry, "t:body"),
        (K.entry, "t:size"),
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
    found = intervene_sync(live(LOADED), LOADED, edits)["findings"]
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
    exc = refusal(field_set, go().record, LOADED, path, value)
    assert exc.reason is reason
    assert exc.path == (path if type(path) is str else None)
    assert (exc.ref, exc.source) == (None, None)


# ---------------------------------------------------------------------------------------------
# A rule made constant


def test_a_constant_rule_gives_exactly_the_findings_downstream_of_it() -> None:
    record, other = live(LOADED), live(LOADED, ask=returns(HAM))
    result = rule_constant(record, LOADED, "d:route", "no")
    found = result["findings"]
    assert kinds(found) == [(K.design_mismatch, None), (K.entry, "d:act"), (K.entry, "d:route")]
    modified = load(result["document"])
    assert found[0] == {
        "kind": K.design_mismatch,
        "recorded": LOADED.design_hash,
        "replayed": modified.design_hash,
    }
    route = found[2]["replayed"]
    assert (route["value"], route["rows"]) == ("no", [1])
    assert found[2]["rule"] is None  # a one-row table provides it
    assert found[1:2] == downstream(record, other, "q:spam", "d:route")


def test_the_modified_document_replaces_the_declaration_under_its_name() -> None:
    document = rule_constant(go().record, LOADED, "d:route", Unsure)["document"]
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
    found = rule_constant(go().record, LOADED, "d:route", "yes")["findings"]
    assert kinds(found) == [(K.design_mismatch, None), (K.entry, "d:route")]
    assert found[1]["recorded"]["value"] == found[1]["replayed"]["value"] == "yes"


def test_a_constant_unsure_rule_writes_unsure() -> None:
    found = rule_constant(go().record, LOADED, "d:route", Unsure)["findings"]
    assert entry_finding(found, "d:route")["replayed"]["marker"]["kind"] == "Unsure"


def test_a_table_rule_is_replaced_over_its_inputs_reads() -> None:
    loaded = load(TABLED)
    record = decide_sync(loaded, {"score": 0.9, "name": "n"}, {}).record
    result = rule_constant(record, loaded, "d:level", "low")
    assert result["document"]["rules"]["level"]["inputs"] == [
        {"read": "t:state.score", "helper": None, "params": {}}
    ]
    assert kinds(result["findings"]) == [
        (K.design_mismatch, None),
        (K.entry, "d:act"),
        (K.entry, "d:level"),
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
    exc = refusal(rule_constant, go().record, LOADED, rule, output)
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
        exc = refusal(rule_constant, record, grounded.LOADED, rule, output)
        assert (exc.reason, exc.ref) == (R.candidate_rule, rule)


def test_a_constant_rule_loads_with_the_callers_functions() -> None:
    record = go().record
    same = rule_constant(record, LOADED, "d:route", "no", functions={"echo": echo})
    assert same == rule_constant(record, LOADED, "d:route", "no")
    exc = refusal(rule_constant, record, LOADED, "d:route", "no", functions={"echo": route})
    assert exc.reason is R.invalid_design
    assert "echo" in str(exc.error)


def test_an_output_outside_the_value_domain_is_invalid_design() -> None:
    record = go().record
    for given in ({1: "x"}, {"a": {2: "x"}}):
        exc = refusal(rule_constant, record, LOADED, "d:route", given)
        assert (exc.reason, type(exc.error)) == (R.invalid_design, InvalidDesign)
        exc = refusal(intervene_sync, record, LOADED, [{"rule": "d:route", "output": given}])
        assert exc.reason is R.invalid_design


def test_an_output_holding_bytes_is_invalid_design_saying_why() -> None:
    for given in (b"x", {"a": [b"x"]}):
        exc = refusal(rule_constant, go().record, LOADED, "d:route", given)
        assert (exc.reason, type(exc.error)) == (R.invalid_design, InvalidDesign)
        assert str(exc).endswith("a table has no bytes literal")


def test_the_string_unsure_that_does_not_load_is_invalid_design_naming_the_class() -> None:
    exc = refusal(rule_constant, go().record, LOADED, "d:route", "Unsure")
    assert (exc.reason, type(exc.error)) == (R.invalid_design, InvalidDesign)
    assert str(exc).endswith("; Unsure is given as the class ommat.Unsure, not the string")


@pytest.mark.usefixtures("no_replay")
def test_a_rule_reading_a_grounded_ref_is_candidate_rule() -> None:
    exc = refusal(rule_constant, grounded.decided(), grounded.LOADED, "d:summary", "x")
    assert (exc.reason, exc.ref) == (R.candidate_rule, "d:summary")


def nested(r: Any) -> Any:
    return [["a"], ["b"]]


def test_an_output_holding_one_list_at_two_places_is_copied() -> None:
    nest = Rule("d:nest", t.list_(t.list_(t.string)), nested, version="1", reads=["t:body"])
    loaded = slice_design(rules={**RULES, "nest": nest})
    shared = ["x"]
    result = rule_constant(go(loaded).record, loaded, "d:nest", [shared, shared])
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
    loaded = load(dataclasses.replace(TABLED, rules={**TABLED.rules, "level": twice}))
    record = decide_sync(loaded, {"score": 0.9, "name": "n"}, {}).record
    table = rule_constant(record, loaded, "d:level", "low")["document"]["rules"]["level"]
    assert table["inputs"] == [{"read": "t:state.score", "helper": None, "params": {}}]
    assert table["rows"] == [{"cells": ["-"], "output": {"literal": "low"}}]


# ---------------------------------------------------------------------------------------------
# intervene


def test_several_edits_replay_once_on_one_copy_in_any_order() -> None:
    record = live(LOADED)
    edits = [
        {"path": "t:state.subject", "value": "bye"},
        {"ref": "q:spam", "value": NO},
        {"source": "look", "code": "raised"},
    ]
    result = edited(record, LOADED, edits)
    assert edited(record, LOADED, edits[::-1]) == result
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
        (K.entry, "d:act"),
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "t:body"),
    ]


def test_intervene_returns_the_replay_record_and_the_stale_refs() -> None:
    record = go().record
    out = intervene_sync(record, LOADED, [{"path": "t:state.subject", "value": "bye"}])
    assert set(out) == {"record", "findings", "stale"}
    assert type(out["record"]) is Record
    assert out["stale"] == ["q:spam", "t:body", "t:size"]  # look needs the whole state
    assert out["stale"] == [f["ref"] for f in out["findings"] if f["kind"] is K.stale_question]
    state = entry(out["record"].log, "t:state")
    assert state["value"]["subject"] == "bye"
    # The returned record is the rerun's: its stale answers are written not-returned.
    assert entry(out["record"].log, "q:spam")["marker"]["cause"]["code"] == "not-returned"


def test_a_pinned_question_under_a_changed_need_stays_pinned() -> None:
    record = go().record
    subject = {"path": "t:state.subject", "value": "bye"}
    alone = intervene_sync(record, LOADED, [subject])
    assert "q:spam" in alone["stale"]
    both = intervene_sync(record, LOADED, [subject, {"ref": "q:spam", "value": NO}])
    assert "q:spam" not in both["stale"]
    assert both["stale"] == ["t:body", "t:size"]
    assert entry(both["record"].log, "q:spam")["value"]["probabilities"] == NO


def test_edits_and_a_rule_replay_together_under_the_modified_design() -> None:
    record = go().record
    result = edited(
        record,
        LOADED,
        [{"path": "t:state.subject", "value": "bye"}, {"rule": "d:route", "output": "no"}],
    )
    assert result["document"]["rules"]["route"]["form"] == "table"
    assert kinds(result["findings"]) == [
        (K.design_mismatch, None),
        (K.stale_question, "q:spam"),
        (K.stale_question, "t:body"),
        (K.stale_question, "t:size"),
        (K.entry, "d:act"),
        (K.entry, "d:echo"),
        (K.entry, "d:route"),
        (K.entry, "q:spam"),
        (K.entry, "t:body"),
        (K.entry, "t:size"),
    ]


def test_two_disjoint_state_fields_are_both_set() -> None:
    edits = [
        {"path": "t:state.subject", "value": "bye"},
        {"path": "t:state.sender.address", "value": "b@x"},
    ]
    (state,) = edited(go().record, LOADED, edits)["items"]
    assert (state["value"]["subject"], state["value"]["sender"]) == ("bye", {"address": "b@x"})


def test_sibling_fields_sharing_a_prefix_are_not_one_target() -> None:
    loaded = slice_design(state=t.record(sub=t.string, subject=t.string))
    record = go(loaded, {"sub": "a", "subject": "b"}).record
    edits = [{"path": "t:state.sub", "value": "x"}, {"path": "t:state.subject", "value": "y"}]
    (state,) = edited(record, loaded, edits)["items"]
    assert state["value"] == {"sub": "x", "subject": "y"}


def test_an_answer_over_a_set_another_edit_sets_takes_its_labels_in_either_order() -> None:
    loaded = load(label_sets.design())
    record = label_sets.run(["ann", "bo"])
    over_cy = {"none": 0.1, "several": 0.1, "ann": 0.7, "cy": 0.1}
    edits = [{"ref": label_sets.SET, "value": ["ann", "cy"]}, {"ref": "q:sender", "value": over_cy}]
    result = edited(record, loaded, edits)
    assert edited(record, loaded, edits[::-1]) == result
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
    assert edited(record, loaded, [owe, within]) == edited(record, loaded, [within, owe])
    for given in ([owe, past], [past, owe]):
        exc = refusal(intervene_sync, record, loaded, given)
        assert (exc.reason, exc.ref) == (R.nonconforming, "q:amount")


@pytest.mark.parametrize(
    ("edits", "reason", "named"),
    [
        ({"ref": "q:spam", "value": NO}, R.invalid_edit, None),
        ([{"ref": "q:spam"}], R.invalid_edit, "intervention 0"),
        ([{"ref": "q:spam", "value": NO}, ["q:spam", NO]], R.invalid_edit, "intervention 1"),
        ([{"ref": "q:spam", "value": NO, "path": "x"}], R.invalid_edit, "intervention 0"),
        ([{"source": "ask", "value": "raised"}], R.invalid_edit, "intervention 0"),
        ([{"rule": "d:route"}], R.invalid_edit, "intervention 0"),
        ([{"ask": "look", "code": "raised"}], R.invalid_edit, "intervention 0"),
        (["d:route"], R.invalid_edit, "intervention 0"),
        ([dict[str, object]()], R.invalid_edit, "intervention 0"),
        ((), R.invalid_edit, None),
        ([type("D", (dict,), {})(ask="look")], R.invalid_edit, "intervention 0"),
        ([{Code("ask"): "look"}], R.invalid_edit, "intervention 0"),
        ([], R.no_edit, None),
        ([{"ref": "q:nope"}], R.invalid_edit, "intervention 0"),
    ],
    ids=[
        "not a list",
        "a member missing",
        "a list item",
        "an extra member",
        "mixed members",
        "a rule missing output",
        "an ask with a code",
        "a string",
        "an empty object",
        "a tuple",
        "a dict subclass",
        "a str subclass key",
        "nothing to do",
        "invalid-edit before not-found",
    ],
)
@pytest.mark.usefixtures("no_replay")
def test_intervene_refuses_a_misshapen_or_empty_request(
    edits: Any, reason: RefusalReason, named: str | None
) -> None:
    exc = refusal(intervene_sync, go().record, LOADED, edits)
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
        (
            [{"rule": "d:route", "output": "no"}, {"rule": "d:route", "output": "yes"}],
            {"ref": "d:route"},
        ),
        ([{"ask": "look"}, {"ask": "look"}], {"source": "look"}),
    ],
    ids=[
        "one ref",
        "one call",
        "one path",
        "a path above",
        "a path below",
        "one rule",
        "one asked source",
    ],
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


@pytest.mark.parametrize(
    ("edits", "named"),
    [
        ([{"ref": "q:spam", "value": NO}, {"ask": "ask"}], ("q:spam", "ask")),
        ([{"ask": "ask"}, {"source": "body", "code": "raised"}], (None, "ask")),
    ],
    ids=["a set entry it provides", "its failed call"],
)
@pytest.mark.usefixtures("no_replay")
def test_an_asked_source_beside_an_edit_of_its_answers_is_conflicting_edits(
    edits: list[Any], named: tuple[str | None, str]
) -> None:
    for given in (edits, edits[::-1]):
        exc = refusal(intervene_sync, go().record, LOADED, given)
        assert (exc.reason, exc.ref, exc.source) == (R.conflicting_edits, *named)


@pytest.mark.usefixtures("no_replay")
def test_an_asked_source_the_design_run_does_not_declare_is_not_found() -> None:
    for given in ("nope", 5):
        exc = refusal(intervene_sync, go().record, LOADED, [{"ask": given}])
        assert exc.reason is R.not_found
    asked = [{"ask": "look"}]
    exc = refusal(intervene_sync, go().record, LOADED, asked, rerun_design=without_look())
    assert (exc.reason, exc.source) == (R.not_found, "look")


@pytest.mark.usefixtures("no_replay")
def test_intervenes_refusals_come_in_their_order() -> None:
    record = go().record
    bad = {"ref": "q:spam"}
    dup = [{"ref": "t:body", "value": "x"}, {"ref": "t:body", "value": "y"}]
    conflict = [{"ref": "q:spam", "value": NO}, {"source": "body", "code": "raised"}]
    own = [{"ref": "d:route", "value": "no"}, {"path": "t:state.nope", "value": 1}]
    rule = {"rule": "d:nope", "output": "x"}

    def reason(edits: Any, given: object = record) -> RefusalReason:
        return refusal(intervene_sync, given, LOADED, edits).reason

    assert reason([bad], given=unfinished()) is R.unfinished
    assert reason([*dup, bad]) is R.invalid_edit
    assert reason([*conflict, *dup]) is R.duplicate_target
    assert reason([*own, *conflict]) is R.conflicting_edits
    assert reason([*own, rule]) is R.null_call
    assert reason([*own[::-1], rule]) is R.not_found
    assert refusal(intervene_sync, record, LOADED, [*own[::-1], rule]).path == "t:state.nope"
    assert refusal(intervene_sync, record, LOADED, [rule, *own]).ref == "d:nope"
    assert reason([{"ref": "t:body", "value": "x"}, rule]) is R.not_found
    assert reason([{"ref": "t:body", "value": 1}, {"rule": "d:route", "output": 1}]) is (
        R.nonconforming
    )
    assert reason([{"ref": "t:body", "value": "x"}, {"rule": "d:route", "output": 1}]) is (
        R.invalid_design
    )


# ---------------------------------------------------------------------------------------------
# A rerun design, and the sources it stales


def look_as(look: Source) -> LoadedDesign:
    return slice_design(sources={"ask": ASK, "body": BODY, "look": look})


def without_look() -> LoadedDesign:
    return slice_design(sources={"ask": ASK, "body": BODY})


def stale_of(out: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        f["ref"]: (f["source"], f["clause"])
        for f in out["findings"]
        if f["kind"] is K.stale_question
    }


@pytest.mark.parametrize(
    ("look", "clause"),
    [
        (Source({"t:size": t.integer()}, needs=["t:state"], payload={"k": 1}), "payload"),
        (Source({"t:size": t.integer()}, needs=["t:state"], oracle="other"), "oracle"),
        (Source({"t:size": t.integer(0, 100)}, needs=["t:state"]), "question"),
    ],
    ids=["its payload", "its oracle key", "a t: ref's type"],
)
def test_a_rerun_design_changing_a_source_stales_it(look: Source, clause: str) -> None:
    record = go().record
    rerun = look_as(look)
    assert rerun.design_hash != LOADED.design_hash
    out = intervene_sync(record, LOADED, [], rerun_design=rerun)
    assert stale_of(out) == {"t:size": ("look", clause)}
    assert entry(out["record"].log, "t:size")["marker"]["cause"]["code"] == "not-returned"
    assert out["findings"][0]["kind"] is K.design_mismatch


@pytest.mark.parametrize(
    ("look", "asked", "clause"),
    [
        (
            Source({"t:size": t.integer()}, needs=["t:state"], payload={"k": 1}, oracle="o"),
            [],
            "payload",
        ),
        (
            Source({"t:size": t.integer(0, 100)}, needs=["t:state"], payload={"k": 1}),
            [],
            "question",
        ),
        (Source({"t:size": t.integer()}, needs=["t:state"], payload={"k": 1}), ["look"], "payload"),
    ],
    ids=["payload before oracle", "type before payload", "payload before ask"],
)
def test_the_first_clause_that_holds_is_reported(
    look: Source, asked: list[str], clause: str
) -> None:
    given = [{"ask": a} for a in asked]
    out = intervene_sync(go().record, LOADED, given, rerun_design=look_as(look))
    assert stale_of(out) == {"t:size": ("look", clause)}


def test_a_ref_the_recorded_design_does_not_declare_is_no_answer() -> None:
    look = Source({"t:size": t.integer(), "t:extra": t.string}, needs=["t:state"])
    out = intervene_sync(go().record, LOADED, [], rerun_design=look_as(look))
    assert stale_of(out) == {"t:extra": ("look", "no-answer")}


def test_a_pinned_ref_stays_pinned_under_a_changed_payload() -> None:
    look = Source({"t:size": t.integer()}, needs=["t:state"], payload={"k": 1})
    out = intervene_sync(
        go().record, LOADED, [{"ref": "t:size", "value": 9}], rerun_design=look_as(look)
    )
    assert out["stale"] == []
    assert entry(out["record"].log, "t:size")["value"] == 9


def test_a_state_field_is_checked_against_the_recorded_design() -> None:
    numbered = slice_design(
        state=t.record(
            subject=t.integer(),
            to=t.string,
            blob=t.bytes_,
            weight=t.number(),
            sender=t.record(address=t.string),
        )
    )
    record = go().record
    out = intervene_sync(
        record, LOADED, [{"path": "t:state.subject", "value": "bye"}], rerun_design=numbered
    )
    assert out["findings"][0]["kind"] is K.design_mismatch
    exc = refusal(
        intervene_sync,
        record,
        LOADED,
        [{"path": "t:state.subject", "value": 5}],
        rerun_design=numbered,
    )
    assert (exc.reason, exc.path) == (R.nonconforming, "t:state.subject")


def test_a_value_is_checked_against_the_recorded_design() -> None:
    narrow = look_as(Source({"t:size": t.integer(0, 3)}, needs=["t:state"]))
    out = intervene_sync(go().record, LOADED, [{"ref": "t:size", "value": 9}], rerun_design=narrow)
    assert "t:size" not in out["stale"]  # pinned: set under the recorded type, never checked


@pytest.mark.usefixtures("no_replay")
def test_an_ask_conflicts_with_a_set_entry_its_source_provides_in_the_design_run() -> None:
    moved = slice_design(sources={"ask": ASK, "body2": BODY, "look": LOOK})
    given = [{"ref": "t:body", "value": "x"}, {"ask": "body2"}]
    exc = refusal(intervene_sync, go().record, LOADED, given, rerun_design=moved)
    assert (exc.reason, exc.ref, exc.source) == (R.conflicting_edits, "t:body", "body2")


def test_an_ask_of_a_source_the_rerun_never_calls_stales_nothing() -> None:
    replies = Scripted({"src": returns({"t:lab": "q"}), "gated": returns({"t:other": True})})
    record = decide_sync(deciding.GATED, {}, {"default": replies}).record
    out = intervene_sync(record, deciding.GATED, [{"ask": "gated"}])
    assert (out["findings"], out["stale"]) == ([], [])


def test_several_rules_are_made_constant_in_one_document() -> None:
    given = [{"rule": "d:route", "output": "no"}, {"rule": "d:echo", "output": "x"}]
    out = edited(go().record, LOADED, given)
    rules = out["document"]["rules"]
    assert (rules["route"]["form"], rules["echo"]["form"]) == ("table", "table")
    bad = [{"rule": "d:route", "output": "no"}, {"rule": "d:echo", "output": b"x"}]
    exc = refusal(intervene_sync, go().record, LOADED, bad)
    assert (exc.reason, exc.ref) == (R.invalid_design, "d:echo")
    assert "rule echo" in str(exc)
    assert str(exc).endswith("a table has no bytes literal")


def test_a_rerun_design_changing_no_source_stales_none() -> None:
    record = go().record
    out = intervene_sync(record, LOADED, [], rerun_design=CHANGED)
    assert out["stale"] == []
    assert kinds(out["findings"]) == [(K.design_mismatch, None)]


def test_an_asked_source_is_stale_by_its_oracle() -> None:
    out = intervene_sync(go().record, LOADED, [{"ask": "look"}])
    assert stale_of(out) == {"t:size": ("look", "oracle")}


def test_an_asked_source_whose_call_failed_is_stale_and_not_failed_again() -> None:
    record = go(binds=bindings(look=Failed(Cause("raised", ["t:size"])))).record
    assert intervene_sync(record, LOADED, [], rerun_design=LOADED)["findings"] == []
    out = intervene_sync(record, LOADED, [{"ask": "look"}])
    assert stale_of(out) == {"t:size": ("look", "oracle")}
    (call,) = [i for i in out["record"].log if i["item"] == "call" and i["sources"] == ["look"]]
    assert call["outcome"] == "returned"


def test_a_rule_made_constant_is_the_rerun_designs_rule() -> None:
    record = go().record
    out = edited(record, LOADED, [{"rule": "d:route", "output": "no"}], rerun_design=CHANGED)
    assert out["document"]["name"] == "changed"
    assert out["document"]["rules"]["route"]["form"] == "table"


@pytest.mark.usefixtures("no_replay")
def test_each_design_not_exactly_a_loaded_design_is_refused_by_name_before_the_record() -> None:
    with pytest.raises(EntryRefusal, match="rerun_design is a"):
        intervene_sync(b"x", LOADED, [], rerun_design=object())  # type: ignore[arg-type]
    with pytest.raises(EntryRefusal, match="recorded_design is a"):
        intervene_sync(b"x", object(), [])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Every job: the record checks, the copy, the inputs and the output


def job_calls() -> list[tuple[str, Callable[[object], Any]]]:
    return [
        ("an entry set", lambda r: entry_set(r, LOADED, "q:spam", NO)),
        ("a call failed", lambda r: call_failed(r, LOADED, "ask", "raised")),
        ("a state field set", lambda r: field_set(r, LOADED, "t:state.subject", "b")),
        ("a rule made constant", lambda r: rule_constant(r, LOADED, "d:route", "no")),
        (
            "intervene",
            lambda r: intervene_sync(
                r,
                LOADED,
                [
                    {"ref": "q:spam", "value": NO},
                    {"path": "t:state.subject", "value": "b"},
                    {"rule": "d:route", "output": "no"},
                    {"ask": "look"},
                ],
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
    other_design = live(CHANGED)
    for given, reason in (
        (b"not json", R.invalid_record),
        (format_7(), R.wrong_design),
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
        lambda: intervene_sync(b"x", object(), []),  # type: ignore[arg-type]
        lambda: intervene_sync(b"x", object(), [], rerun_design=LOADED),  # type: ignore[arg-type]
    ],
    ids=["intervene", "intervene with a rerun design"],
)
def test_a_loaded_not_exactly_a_loaded_design_is_refused_first(job: Callable[[], Any]) -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        job()


@pytest.mark.parametrize(
    "job", [j for name, j in JOBS if name != "a rule made constant"], ids=IDS[:3] + IDS[4:]
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
    rule_constant(go().record, LOADED, "d:route", "no")


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


ALL_JOBS: list[tuple[str, Callable[[object], Any]]] = [
    *JOBS,
    ("use_answers", lambda r: use_answers_sync(r, LOADED, live(ask=returns(HAM)))),
]


@pytest.mark.parametrize("job", [j for _, j in ALL_JOBS], ids=[n for n, _ in ALL_JOBS])
def test_no_record_is_reachable_from_a_jobs_output_but_intervenes_replay_record(
    job: Callable[[object], Any],
) -> None:
    result = job(go().record)
    rerun = result.pop("record", None)
    assert rerun is None or type(rerun) is Record
    assert records_in(result) == []
    assert json.loads(json.dumps(result)) == result
    assert copy.deepcopy(result) == result


# ---------------------------------------------------------------------------------------------
# Records of earlier formats


@pytest.mark.parametrize("folder", [f"format_{n}" for n in (1, 2, 3, 4, 5, 6, 7, 9)])
@pytest.mark.usefixtures("no_replay")
def test_a_stored_record_before_format_8_is_wrong_design(folder: str) -> None:
    record = read_record((TESTS / folder / "record.json").read_bytes())
    fmt = record.log[0]["format"]
    assert fmt < 8
    exc = refusal(intervene_sync, record, LOADED, [{"ref": "q:spam", "value": NO}])
    assert (exc.reason, exc.turn) == (R.wrong_design, record.record_hash)
    assert str(exc) == (
        f"record is a format-{fmt} record, before format 8, so it holds no policy's entry"
    )


# ---------------------------------------------------------------------------------------------
# The sync twins


def test_intervene_is_awaitable_and_equals_its_sync_twin() -> None:
    record = go().record
    given = [{"ref": "q:spam", "value": NO}, {"rule": "d:route", "output": "no"}]

    def plain(out: dict[str, Any]) -> dict[str, Any]:
        return {**out, "record": out["record"].record_hash}  # its timings differ

    awaited = asyncio.run(intervene(record, LOADED, given, rerun_design=LOADED))
    assert plain(awaited) == plain(intervene_sync(record, LOADED, given, rerun_design=LOADED))


def test_intervene_sync_inside_a_running_loop_raises_before_doing_anything() -> None:
    async def main() -> None:
        with pytest.raises(RuntimeError, match="intervene_sync was called inside a running event"):
            intervene_sync(object(), object(), [])  # type: ignore[arg-type]

    asyncio.run(main())
