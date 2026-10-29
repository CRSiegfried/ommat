# pyright: reportPrivateUsage=false
"""The deciding inputs: each declared read of a rule classed by whether some do-and-diff
intervention on it changes the rule's entry, each kind of read tried by its form, a finite
type at every value, the record checks and ``not-found`` before any replay, and plain data out
that reaches no ``Record``."""

import asyncio
import copy
import json
from typing import Any

import pytest

import test_grounded_rules as grounded
from ommat import Cause, EntryRefusal, Failed, FindingKind, InvalidRecord, Rule, Source, When
from ommat import types as t
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _deciding,
    _diff,
    _edit,
    deciding_inputs,
    deciding_inputs_sync,
)
from scripted import returns
from test_decide import ASK, BODY, LOADED, RULES, slice_design
from test_do_and_diff import records_in
from test_use_answers import broken, live, unfinished

R = RefusalReason
STATE_T = t.record(flag=t.boolean, mood=t.nullable(t.label("calm", "wild")), subject=t.string)
STATE: dict[str, Any] = {"flag": True, "mood": "calm", "subject": "hello"}


def never(r: Any) -> bool:
    return False


def pick(r: Any) -> str:
    return "x" if r.read("t:kind") == "a" and r.read("t:state.mood") != "wild" else "y"


def after(r: Any) -> object:
    return r.read("d:pick")


def note(r: Any) -> str:
    return "n"


PICK_READS = ["t:kind", "t:state.flag", "t:state.mood", "t:state.subject", "t:body", "t:gone"]
DESIGN = slice_design(
    state=STATE_T,
    sources={
        "ask": ASK,
        "body": BODY,
        "look": Source({"t:kind": t.label("a", "b")}, needs=["t:state"]),
        "skip": Source({"t:gone": t.boolean}, needs=["t:state"], when=When(never, version="1")),
    },
    rules={
        "route": RULES["route"],
        "pick": Rule("d:pick", t.label("x", "y"), pick, version="1", reads=PICK_READS),
        "note": Rule("d:note", t.string, note, version="1", reads=["t:state.subject"]),
        "after": Rule(
            "d:after", t.string, after, version="1", reads=["d:pick", "d:route", "d:note"]
        ),
    },
)


def decided() -> Any:
    return live(DESIGN, state=STATE, look=returns({"t:kind": "a"}))


@pytest.fixture
def tried(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Each intervention tried, as its edits and its rule."""
    seen: list[Any] = []
    real = _deciding._applied

    async def spy(read: Any, log: Any, loaded: Any, edits: Any, rule: Any, f: Any) -> Any:
        seen.append((edits, rule))
        return await real(read, log, loaded, edits, rule, f)

    monkeypatch.setattr(_deciding, "_applied", spy)
    return seen


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)


def refusal(*args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        deciding_inputs_sync(*args, **kwargs)
    return raised.value


# ---------------------------------------------------------------------------------------------
# The classes, by kind of read


def test_each_read_is_classed_in_declared_order() -> None:
    result = deciding_inputs_sync(decided(), DESIGN, "d:pick")
    assert list(result.items()) == [
        ("t:kind", "deciding"),  # a value and every marker change it
        ("t:state.flag", "not-deciding"),  # a finite field tried at every value
        ("t:state.mood", "deciding"),  # wild changes it
        ("t:state.subject", "not-determined"),  # a string field: form 3 sets values only
        ("t:body", "markers-only"),  # a string entry, never read
        ("t:gone", "not-determined"),  # its call is null: its source was not asked
    ]


def test_a_d_read_is_tried_through_form_4() -> None:
    result = deciding_inputs_sync(decided(), DESIGN, "d:after")
    assert result == {
        "d:pick": "deciding",
        "d:route": "not-deciding",  # a label, every value and Unsure tried
        "d:note": "markers-only",  # a string: Unsure only
    }


def test_a_q_answer_is_deciding_at_refused() -> None:
    assert deciding_inputs_sync(decided(), DESIGN, "d:route") == {"q:spam": "deciding"}


def test_an_entry_of_a_failed_call_is_not_determined() -> None:
    record = live(ask=Failed(Cause("raised", ["q:spam"])))
    assert deciding_inputs_sync(record, LOADED, "d:route") == {"q:spam": "not-determined"}


def test_a_grounded_ref_is_intervened_on_through_any_of_its_entries(tried: list[Any]) -> None:
    record = grounded.decided()
    result = deciding_inputs_sync(record, grounded.LOADED, 'd:fit["Bo"]')
    assert result == {
        "candidate": "not-determined",
        "candidate.age": "not-determined",
        "q:ask/out": "deciding",
    }
    # Ana's entry first, which leaves Bo's unchanged, then Bo's; one call wrote both
    assert [(e.ref, e.value.cause.code) for edits, _ in tried for e in edits] == [
        ('q:ask/out["Ana"]', "refused"),
        ('q:ask/out["Bo"]', "refused"),
    ]


def test_a_read_of_a_grounded_rule_is_not_determined() -> None:
    record = grounded.decided()
    assert deciding_inputs_sync(record, grounded.LOADED, "d:summary") == {"d:fit": "not-determined"}


# ---------------------------------------------------------------------------------------------
# The interventions tried


def test_a_finite_entry_is_tried_at_every_marker_and_value_and_its_lone_call_failed() -> None:
    trials, finite = _deciding._trials("t:kind", decided().log, DESIGN)
    assert finite
    set_to: list[Any] = [x.value for x in trials if type(x) is _edit._SetEntry]
    assert [m.cause.code for m in set_to[:-2]] == sorted(_deciding._BINDING_MARKERS)
    assert set_to[-2:] == ["a", "b"]
    failed = [(x.source, x.code) for x in trials if type(x) is _edit._FailCall]
    assert failed == [("look", code) for code in sorted(_deciding._STRUCTURAL)]


def test_a_q_answer_is_tried_at_refused_only_and_a_shared_call_never_failed() -> None:
    trials, finite = _deciding._trials("q:spam", decided().log, DESIGN)
    assert not finite
    assert [(type(x), x.value.cause.code) for x in trials] == [(_edit._SetEntry, "refused")]  # type: ignore[union-attr]


def test_a_nullable_state_field_is_tried_at_each_value_and_null() -> None:
    trials, finite = _deciding._trials("t:state.mood", decided().log, DESIGN)
    assert finite
    assert [(x.path, x.value) for x in trials] == [  # type: ignore[union-attr]
        ("t:state.mood", "calm"),
        ("t:state.mood", "wild"),
        ("t:state.mood", None),
    ]


def test_a_d_read_is_tried_at_unsure_then_each_value() -> None:
    trials, finite = _deciding._trials("d:pick", decided().log, DESIGN)
    assert finite
    assert [(x.rule, x.output) for x in trials] == [  # type: ignore[union-attr]
        ("d:pick", _deciding.Unsure),
        ("d:pick", "x"),
        ("d:pick", "y"),
    ]


def test_a_trial_whose_replay_leaves_the_record_reaches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def outside(*args: object) -> object:
        found: list[dict[str, Any]] = [{"kind": FindingKind.not_in_record, "sources": []}]
        return list[Any](), None, found

    monkeypatch.setattr(_deciding, "_applied", outside)
    assert deciding_inputs_sync(decided(), DESIGN, "d:route") == {"q:spam": "not-determined"}


# ---------------------------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize("ref", ["d:nope", "q:spam", "t:state", 'd:pick["a"]', 5, None])
@pytest.mark.usefixtures("no_replay")
def test_a_ref_no_rule_provides_in_the_record_is_not_found(ref: Any) -> None:
    exc = refusal(decided(), DESIGN, ref)
    assert exc.reason is R.not_found
    assert exc.ref == (ref if type(ref) is str else None)


@pytest.mark.usefixtures("no_replay")
def test_the_record_is_refused_in_order_before_the_ref() -> None:
    other_design = live(slice_design(tools=["archive", "send"]))
    for given, reason in (
        (b"not json", R.invalid_record),
        (unfinished(), R.unfinished),
        (broken(), R.nothing_to_replay),
        (other_design, R.wrong_design),
    ):
        exc = refusal(given, LOADED, "d:nope")
        assert exc.reason is reason
        if reason is R.invalid_record:
            assert type(exc.error) is InvalidRecord
        else:
            assert exc.turn == given.record_hash  # type: ignore[union-attr]


def test_a_loaded_not_exactly_a_loaded_design_is_refused_first() -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        deciding_inputs_sync(b"x", object(), "d:route")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The output and the twin


def test_the_output_is_plain_data_reaching_no_record_and_neither_input_changes() -> None:
    record = decided()
    parsed = json.loads(record.document)
    kept = copy.deepcopy(parsed)
    before = (record.document, DESIGN.document)
    result = deciding_inputs_sync(record, DESIGN, "d:after")
    assert records_in(result) == []
    assert json.loads(json.dumps(result)) == result
    assert deciding_inputs_sync(parsed, DESIGN, "d:after") == result
    assert (parsed, (record.document, DESIGN.document)) == (kept, before)


def test_the_job_is_awaitable_and_equals_its_sync_twin() -> None:
    record = decided()
    awaited = asyncio.run(deciding_inputs(record, DESIGN, "d:pick"))
    assert awaited == deciding_inputs_sync(record, DESIGN, "d:pick")


def test_the_sync_twin_inside_a_running_loop_raises_before_doing_anything() -> None:
    async def main() -> None:
        with pytest.raises(RuntimeError, match="deciding_inputs_sync was called inside a running"):
            deciding_inputs_sync(object(), object(), "")  # type: ignore[arg-type]

    asyncio.run(main())
