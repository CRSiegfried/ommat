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
from designs import minimal
from ommat import (
    Cause,
    EntryRefusal,
    Failed,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    Policy,
    Rule,
    Source,
    Table,
    Unsure,
    When,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._interpret import _BINDING_MARKERS, _STRUCTURAL
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _deciding,
    _diff,
    _edit,
    deciding_inputs,
    deciding_inputs_sync,
    set_entry_sync,
    set_state_field_sync,
)
from scripted import Scripted, returns
from test_decide import ASK, BODY, LOADED, RULES, ok_policy, slice_design
from test_do_and_diff import records_in
from test_use_answers import broken, live, unfinished

R = RefusalReason
STATE_T = t.record(flag=t.boolean, mood=t.nullable(t.label("calm", "wild")), subject=t.string)
STATE: dict[str, Any] = {"flag": True, "mood": "calm", "subject": "hello"}


def never(r: Any) -> bool:
    return False


def pick(r: Any) -> str:
    return "x" if r.read("t:kind") == "a" and r.read("t:state.mood") != "wild" else "y"


def still(r: Any) -> str:
    return "x"


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
        "still": Rule("d:still", t.label("x", "y"), still, version="1", reads=["t:kind"]),
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
    assert [(e.ref, e.value.cause.code) for e in flat(tried)] == [
        ('q:ask/out["Ana"]', "refused"),
        ('q:ask/out["Bo"]', "refused"),
    ]


def test_a_read_of_a_grounded_rule_is_not_determined() -> None:
    record = grounded.decided()
    assert deciding_inputs_sync(record, grounded.LOADED, "d:summary") == {"d:fit": "not-determined"}


# ---------------------------------------------------------------------------------------------
# The interventions tried


def flat(trials: list[Any]) -> list[Any]:
    """Each trial's one edit, or its rule."""
    return [edits[0] if edits else rule for edits, rule in trials]


def test_a_finite_entry_is_tried_at_every_marker_and_value_and_its_lone_call_failed() -> None:
    trials, exhaustive = _deciding._trials("t:kind", decided().log, DESIGN)
    assert exhaustive
    tried = flat(trials)
    set_to: list[Any] = [x.value for x in tried if type(x) is _edit._SetEntry]
    assert [m.cause.code for m in set_to[:-2]] == sorted(_BINDING_MARKERS)
    assert set_to[-2:] == ["a", "b"]
    failed = [(x.source, x.code) for x in tried if type(x) is _edit._FailCall]
    assert failed == [("look", str(code)) for code in sorted(_STRUCTURAL)]
    assert all(type(code) is str for _, code in failed)


def test_a_read_no_intervention_changes_runs_every_trial_through_the_job(
    tried: list[Any],
) -> None:
    assert deciding_inputs_sync(decided(), DESIGN, "d:still") == {"t:kind": "not-deciding"}
    trials, _ = _deciding._trials("t:kind", decided().log, DESIGN)
    assert tried == trials
    assert any(type(e) is _edit._FailCall for edits, _ in tried for e in edits)


@pytest.mark.parametrize(
    ("fmt", "left_out"), [(2, {"unsupported", "timed-out", "raised"}), (3, {"timed-out", "raised"})]
)
def test_a_marker_the_records_format_does_not_hold_is_not_tried(
    fmt: int, left_out: set[str]
) -> None:
    log = decided().log
    log[0] = {**log[0], "format": fmt}
    tried = flat(_deciding._trials("t:kind", log, DESIGN)[0])
    markers: list[Any] = [x.value for x in tried if type(x) is _edit._SetEntry]
    codes = {m.cause.code for m in markers if m not in ("a", "b")}
    assert codes == set(_BINDING_MARKERS) - left_out


def test_a_q_answer_is_tried_at_refused_only_and_a_shared_call_never_failed() -> None:
    trials, exhaustive = _deciding._trials("q:spam", decided().log, DESIGN)
    assert not exhaustive
    assert [(type(x), x.value.cause.code) for x in flat(trials)] == [(_edit._SetEntry, "refused")]


def test_a_grounded_batch_is_failed_once_for_its_declared_ref() -> None:
    log = grounded.decided().log
    tried = flat(_deciding._trials("q:ask/out", log, grounded.LOADED)[0])
    assert [(type(x), getattr(x, "ref", None)) for x in tried if type(x) is _edit._SetEntry] == [
        (_edit._SetEntry, 'q:ask/out["Ana"]'),
        (_edit._SetEntry, 'q:ask/out["Bo"]'),
    ]
    failed = [(x.source, x.code) for x in tried if type(x) is _edit._FailCall]
    assert failed == [("ask", str(code)) for code in sorted(_STRUCTURAL)]


def test_a_nullable_state_field_is_tried_at_each_value_and_null() -> None:
    trials, exhaustive = _deciding._trials("t:state.mood", decided().log, DESIGN)
    assert exhaustive
    assert [(x.path, x.value) for x in flat(trials)] == [
        ("t:state.mood", "calm"),
        ("t:state.mood", "wild"),
        ("t:state.mood", None),
    ]


def flag(r: Any) -> object:
    return r.read("t:state.s.flag")


NESTED = load(
    minimal(
        policy=Policy(ok_policy, version="1"),
        state=t.record(s=t.nullable(t.record(flag=t.boolean, name=t.string))),
        rules={"m": Rule("d:m", t.boolean, flag, version="1", reads=["t:state.s.flag"])},
    )
)


def test_a_field_under_a_nullable_record_is_tried_at_its_own_values() -> None:
    record = decide_sync(NESTED, {"s": {"flag": True, "name": "x"}}, {}).record
    trials, _ = _deciding._trials("t:state.s.flag", record.log, NESTED)
    assert [x.value for x in flat(trials)] == [False, True]
    assert deciding_inputs_sync(record, NESTED, "d:m") == {"t:state.s.flag": "deciding"}
    record = decide_sync(NESTED, {"s": None}, {}).record
    assert deciding_inputs_sync(record, NESTED, "d:m") == {"t:state.s.flag": "not-determined"}


def maybe(r: Any) -> object:
    return r.read("t:bb")


NULLABLE = load(
    minimal(
        policy=Policy(ok_policy, version="1"),
        state=t.record(k=t.label("a", "b", "c")),
        sources={"src": Source({"t:bb": t.nullable(t.boolean)}, needs=["t:state"])},
        rules={
            "m": Rule("d:m", t.nullable(t.boolean), maybe, version="1", reads=["t:bb"]),
            "tab": Table(
                "d:tab", t.label("x", "y"), "F", ["t:state.k"], [["a", "x"], ["b", "x"], ["c", "x"]]
            ),
        },
    )
)


def nullable_record() -> Any:
    binds = {"default": Scripted({"src": returns({"t:bb": True})})}
    return decide_sync(NULLABLE, {"k": "a"}, binds).record


def test_a_nullable_finite_t_entry_is_tried_at_null_too(tried: list[Any]) -> None:
    assert deciding_inputs_sync(nullable_record(), NULLABLE, "d:m") == {"t:bb": "deciding"}
    trials, exhaustive = _deciding._trials("t:bb", nullable_record().log, NULLABLE)
    assert exhaustive
    values = [x.value for x in flat(trials) if type(x) is _edit._SetEntry]
    assert values[-3:] == [False, True, None]


def test_a_change_of_rows_alone_is_no_change_of_the_entry() -> None:
    record = nullable_record()
    found = set_state_field_sync(record, NULLABLE, "t:state.k", "b")["findings"]
    assert [(f["ref"], f["replayed"]["rows"]) for f in found] == [("d:tab", [2])]
    assert deciding_inputs_sync(record, NULLABLE, "d:tab") == {"t:state.k": "not-deciding"}


def marked(code: str, *refs: str) -> dict[str, Any]:
    return {"ref": "d:x", "marker": {"kind": "Unavailable", "cause": {"code": code, "refs": refs}}}


def test_a_change_of_a_markers_refs_alone_is_no_change() -> None:
    entry = marked("refused", "a")
    assert not _diff.changed({"recorded": entry, "replayed": marked("refused", "b")})
    assert _diff.changed({"recorded": entry, "replayed": marked("raised", "a")})
    assert _diff.changed({"recorded": entry, "replayed": {"ref": "d:x", "value": None}})
    assert _diff.changed({"recorded": entry, "replayed": None})


def test_a_d_read_is_tried_at_unsure_then_each_value() -> None:
    trials, exhaustive = _deciding._trials("d:pick", decided().log, DESIGN)
    assert exhaustive
    assert [(x.rule, x.output) for x in flat(trials)] == [
        ("d:pick", Unsure),
        ("d:pick", "x"),
        ("d:pick", "y"),
    ]


def stubbed(monkeypatch: pytest.MonkeyPatch, *outcomes: object) -> None:
    """``_applied`` giving each outcome in turn, a refusal raised, then no finding."""
    given = list(outcomes)

    async def applied(*args: object) -> object:
        outcome: object = given.pop(0) if given else list[Any]()
        if isinstance(outcome, EvaluationRefusal):
            raise outcome
        return list[Any](), None, outcome

    monkeypatch.setattr(_deciding, "_applied", applied)


@pytest.mark.parametrize(
    "kind", [FindingKind.not_in_record, FindingKind.ref_not_in_record, FindingKind.question_changed]
)
def test_a_trial_whose_replay_leaves_the_record_is_undetermined_even_if_it_changes(
    kind: FindingKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    changed = {"kind": FindingKind.entry, "ref": "d:still", "recorded": None, "replayed": None}
    stubbed(monkeypatch, [{"kind": kind}, changed])
    assert deciding_inputs_sync(decided(), DESIGN, "d:still") == {"t:kind": "not-determined"}
    stubbed(monkeypatch, [{"kind": kind}], [changed])
    assert deciding_inputs_sync(decided(), DESIGN, "d:still") == {"t:kind": "deciding"}


def gate(r: Any) -> bool:
    return r.read("t:lab") == "p"


def lab(r: Any) -> str:
    return "y" if r.read("t:lab") == "p" else "n"


GATED = load(
    minimal(
        policy=Policy(ok_policy, version="1"),
        sources={
            "src": Source({"t:lab": t.label("p", "q")}),
            "gated": Source({"t:other": t.boolean}, needs=["t:lab"], when=When(gate, version="1")),
        },
        rules={"r": Rule("d:r", t.label("y", "n"), lab, version="1", reads=["t:lab"])},
    )
)


def test_a_value_whose_replay_asks_outside_the_record_leaves_the_read_not_determined() -> None:
    replies = Scripted({"src": returns({"t:lab": "q"}), "gated": returns({"t:other": True})})
    record = decide_sync(GATED, {}, {"default": replies}).record
    found = set_entry_sync(record, GATED, "t:lab", "p")["findings"]
    assert FindingKind.not_in_record in [f["kind"] for f in found]
    assert deciding_inputs_sync(record, GATED, "d:r") == {"t:lab": "not-determined"}


def test_one_undetermined_trial_among_unchanged_ones_is_not_determined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stubbed(monkeypatch, [], [{"kind": FindingKind.question_changed}])
    assert deciding_inputs_sync(decided(), DESIGN, "d:still") == {"t:kind": "not-determined"}


def test_a_trial_refused_at_its_target_is_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """As for a carried or unasked entry beside an asked one: the asked one's trials class it."""
    stubbed(monkeypatch, EvaluationRefusal("x", reason=R.null_call))
    assert deciding_inputs_sync(decided(), DESIGN, "d:still") == {"t:kind": "not-deciding"}
    stubbed(monkeypatch, EvaluationRefusal("x", reason=R.null_call))
    assert deciding_inputs_sync(decided(), DESIGN, "d:route") == {"q:spam": "not-determined"}


DRIFT = {"to": "x"}


def drifting(r: Any) -> str:
    r.read("t:bb")
    return DRIFT["to"]


def test_a_record_its_design_no_longer_replays_is_baseline_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = {"d": Rule("d:d", t.label("x", "y"), drifting, version="1", reads=["t:bb"])}
    loaded = load(
        minimal(
            policy=Policy(ok_policy, version="1"),
            sources={"src": Source({"t:bb": t.nullable(t.boolean)})},
            rules=rules,
        )
    )
    binds = {"default": Scripted({"src": returns({"t:bb": True})})}
    record = decide_sync(loaded, {}, binds).record
    assert deciding_inputs_sync(record, loaded, "d:d") == {"t:bb": "not-deciding"}
    monkeypatch.setitem(DRIFT, "to", "y")
    exc = refusal(record, loaded, "d:d")
    assert exc.reason is R.baseline_differs
    assert str(exc).endswith(": entry")
    monkeypatch.setattr(_deciding, "_applied", None)  # refused before any trial
    assert refusal(record, loaded, "d:d").reason is R.baseline_differs


def flagged(r: Any) -> object:
    return r.read("t:rec.f")


def test_a_field_path_read_of_an_entry_is_tried_through_the_whole_entry(
    tried: list[Any],
) -> None:
    loaded = load(
        minimal(
            policy=Policy(ok_policy, version="1"),
            sources={"src": Source({"t:rec": t.record(f=t.boolean, g=t.string)})},
            rules={"m": Rule("d:m", t.boolean, flagged, version="1", reads=["t:rec.f"])},
        )
    )
    binds = {"default": Scripted({"src": returns({"t:rec": {"f": True, "g": "z"}})})}
    record = decide_sync(loaded, {}, binds).record
    assert deciding_inputs_sync(record, loaded, "d:m") == {"t:rec.f": "deciding"}
    trials, exhaustive = _deciding._trials("t:rec.f", record.log, loaded)
    assert not exhaustive  # a record type: the entry is tried at markers only
    assert {x.ref for x in flat(trials) if type(x) is _edit._SetEntry} == {"t:rec"}
    assert all(type(e) is not _edit._SetField for edits, _ in tried for e in edits)


def test_an_unexpected_refusal_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    stubbed(monkeypatch, EvaluationRefusal("x", reason=R.marker))
    assert refusal(decided(), DESIGN, "d:route").reason is R.marker


def test_a_rule_that_does_not_reload_without_functions_propagates() -> None:
    def local(r: Any) -> object:
        return str(r.read("d:a"))

    loaded = load(
        minimal(
            policy=Policy(ok_policy, version="1"),
            rules={
                "a": Rule("d:a", t.boolean, never, version="1"),
                "b": Rule("d:b", t.string, local, version="1", reads=["d:a"]),
            },
        )
    )
    record = decide_sync(loaded, {}, {}).record
    exc = refusal(record, loaded, "d:b")
    assert exc.reason is R.invalid_design
    assert type(exc.error) is InvalidDesign
    functions = dict(loaded._callables)
    assert deciding_inputs_sync(record, loaded, "d:b", functions=functions) == {"d:a": "deciding"}


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
