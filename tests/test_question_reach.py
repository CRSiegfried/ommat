# pyright: reportPrivateUsage=false
"""The question-reach job: one form 1 trial per point of each ``q:`` entry a returned call
wrote, the recorded value skipped and the marker last; each status, the witness by record,
log and point order, ``values`` checked and narrowing as the lints take it before any record
is read, the records refused one by one before any trial, a per-candidate and a
candidate-valued question, and a report of plain data."""

import asyncio
import copy
import json
from typing import Any, cast

import pytest

import extraction
import test_grounded_rules as grounded
import test_label_sets as label_sets
from designs import roles
from ommat import (
    Cause,
    Design,
    EntryRefusal,
    InvalidRecord,
    Parameter,
    Plan,
    Policy,
    Question,
    Record,
    Requirement,
    Rule,
    Source,
    Text,
    Unavailable,
    Undetermined,
    When,
    decide_sync,
    load,
)
from ommat import types as t
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _diff,
    _reach,
    question_reach,
    question_reach_sync,
)
from scripted import Scripted, returns
from test_deciding_inputs import records_in
from test_replay import edited, entry
from test_use_answers import live

R = RefusalReason


def go(r: Any) -> object:
    from ommat.helpers import pick_with_margin

    picked = pick_with_margin(r, "q:go", r.param("go/bar"), r.param("go/margin"))
    return picked if isinstance(picked, Undetermined) else picked == "yes"


def by_go(r: Any) -> Plan:
    if r.read("d:go") is True:
        return Plan("execute")
    return Plan("escalate", reason="not sure")


def never(r: Any) -> bool:
    return False


def cold(r: Any) -> bool:
    tone = r.read("q:tone")
    return not isinstance(tone, Undetermined) and tone.probabilities["cold"] >= 0.5


def known(r: Any) -> bool:
    return not isinstance(r.read("q:tone"), Undetermined)


def by_known(r: Any) -> Plan:
    if r.read("d:known") is True:
        return Plan("execute")
    return Plan("escalate", reason="no tone")


GO = Rule(
    "d:go",
    t.boolean,
    go,
    version="1",
    reads=["q:go"],
    params=["go/bar", "go/margin"],
    boundaries={"q:go": roles(bar=["go/bar"], margin=["go/margin"])},
)
KNOWN = Rule("d:known", t.boolean, known, version="1", reads=["q:tone"])
ASK = Source(
    {
        "q:go": Question("noul", Text("Go?")),
        "q:tone": Question("choice", Text("Tone?"), labels=("warm", "cold")),
    }
)
LATE = Source(
    {"q:late": Question("noul", Text("Late?"))}, needs=["t:state"], when=When(never, version="1")
)
MORE = Source({"t:more": t.boolean}, needs=["q:tone"], when=When(cold, version="1"))


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "state": t.record(),
        "policy": Policy(by_go, version="p", reads=["d:go"]),
        "sources": {"ask": ASK, "late": LATE},
        "rules": {"go": GO},
        "parameters": {
            "go/bar": Parameter(t.number(0, 1), 0.8),
            "go/margin": Parameter(t.number(0, 1), 0.2),
        },
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())
GATED = load(design(sources={"ask": ASK, "late": LATE, "more": MORE}))
MARKED = load(
    design(
        policy=Policy(by_known, version="p", reads=["d:known"]), rules={"go": GO, "known": KNOWN}
    )
)

WARMISH = {"warm": 0.6, "cold": 0.4}


def turn(
    p_yes: float = 0.95, tone: object = WARMISH, loaded: Any = LOADED, state: Any = None
) -> Record:
    replies = Scripted(
        {
            "ask": returns({"q:go": {"yes": p_yes, "no": 1 - p_yes}, "q:tone": tone}),
            "late": returns({"q:late": {"yes": 1.0, "no": 0.0}}),
            "more": returns({"t:more": True}),
        }
    )
    return decide_sync(loaded, {} if state is None else state, {"default": replies}).record


def answer(kind: str, labels: list[str], **probabilities: float) -> dict[str, Any]:
    return {"kind": kind, "labels": labels, "probabilities": probabilities}


@pytest.fixture
def tried(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, list[str]]]:
    """Each trial run: its entry's ref, the value set and its findings' kinds."""
    seen: list[tuple[str, Any, list[str]]] = []
    real = _reach._applied

    async def spy(read: Any, log: Any, loaded: Any, edits: Any, rule: Any, f: Any) -> Any:
        out = await real(read, log, loaded, edits, rule, f)
        seen.append((edits[0].ref, edits[0].value, [str(x["kind"]) for x in out[2]]))
        return out

    monkeypatch.setattr(_reach, "_applied", spy)
    return seen


@pytest.fixture
def no_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    async def applied(*args: object) -> object:
        raise AssertionError("tried")

    monkeypatch.setattr(_reach, "_applied", applied)


def refusal(*args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        question_reach_sync(*args, **kwargs)
    return raised.value


def p_yes(value: Any) -> object:
    return cast("dict[str, float]", value)["yes"] if type(value) is dict else value


# ---------------------------------------------------------------------------------------------
# Statuses


def test_each_question_is_reported_with_its_status_and_witness() -> None:
    record = turn()
    report = question_reach_sync([record], LOADED)
    assert report["records"] == [record.record_hash]
    assert report["narrowed"] == []
    assert report["questions"] == {
        "q:go": {
            "status": "live",
            "witness": {
                "record": record.record_hash,
                "ref": "q:go",
                "value": answer("noul", ["yes", "no"], yes=0.0, no=1.0),
            },
        },
        "q:late": {"status": "never-asked", "witness": None},  # its call is null
        "q:tone": {"status": "no-change-found", "witness": None},
    }


def test_a_trial_whose_replay_asks_outside_the_record_is_not_determined(
    tried: list[Any],
) -> None:
    report = question_reach_sync([turn(loaded=GATED)], GATED)
    assert report["questions"]["q:tone"] == {"status": "not-determined", "witness": None}
    tone = [kinds for ref, _, kinds in tried if ref == "q:tone"]
    # warm at 1 leaves the gate shut; cold at 1 opens it on a source never called
    assert ["not-in-record" in kinds for kinds in tone] == [False, True, False]


def test_a_reaching_trial_beats_an_undetermined_one() -> None:
    loaded = load(
        design(
            sources={"ask": ASK, "late": LATE, "more": MORE},
            policy=Policy(by_known, version="p", reads=["d:known"]),
            rules={"go": GO, "known": KNOWN},
        )
    )
    report = question_reach_sync([turn(loaded=loaded)], loaded)
    assert report["questions"]["q:tone"]["status"] == "live"  # the marker reaches


def by_cold(r: Any) -> Plan:
    if r.read("d:cold") is True:
        return Plan("execute")
    return Plan("escalate", reason="warm")


def test_a_trial_outside_the_record_is_undetermined_though_its_plan_changes(
    tried: list[Any],
) -> None:
    loaded = load(
        design(
            sources={"ask": ASK, "late": LATE, "more": MORE},
            policy=Policy(by_cold, version="p", reads=["d:cold"]),
            rules={"cold": Rule("d:cold", t.boolean, cold, version="1", reads=["q:tone"])},
        )
    )
    report = question_reach_sync([turn(loaded=loaded)], loaded)
    tone = [kinds for ref, _, kinds in tried if ref == "q:tone"]
    # cold at 1 opens the gate on a source never called and changes the plan
    assert "not-in-record" in tone[1]
    assert "plan" in tone[1]
    assert not any("plan" in kinds or "verdict" in kinds for kinds in (tone[0], tone[2]))
    assert report["questions"]["q:tone"] == {"status": "not-determined", "witness": None}


def always(r: Any) -> Plan:
    return Plan("execute")


def test_a_trial_changing_the_verdict_alone_reaches(tried: list[Any]) -> None:
    ok = Rule(
        "d:ok",
        t.boolean,
        go,
        version="1",
        reads=["q:go"],
        params=["go/bar", "go/margin"],
        boundaries={"q:go": roles(bar=["go/bar"], margin=["go/margin"])},
    )
    loaded = load(
        design(
            policy=Policy(always, version="p"),
            rules={"ok": ok},
            requirements=[Requirement("d:ok", True)],
            default_deny=False,
        )
    )
    record = turn(loaded=loaded)
    report = question_reach_sync([record], loaded)
    assert report["questions"]["q:go"]["status"] == "live"
    first = next(kinds for ref, _, kinds in tried if ref == "q:go" and "verdict" in kinds)
    assert "plan" not in first
    witness = report["questions"]["q:go"]["witness"]
    assert witness["value"]["probabilities"] == {"yes": 0.0, "no": 1.0}


def by_both(r: Any) -> Plan:
    if r.read("d:go") is True and r.read("d:known") is True:
        return Plan("execute")
    return Plan("escalate", reason="not both")


BOTH = load(
    design(
        policy=Policy(by_both, version="p", reads=["d:go", "d:known"]),
        rules={"go": GO, "known": KNOWN},
    )
)


def test_the_witness_is_the_second_record_when_only_it_reaches(tried: list[Any]) -> None:
    shut, open_ = turn(p_yes=0.0, loaded=BOTH), turn(p_yes=0.95, loaded=BOTH)
    report = question_reach_sync([shut, open_], BOTH)
    tone = [kinds for ref, _, kinds in tried if ref == "q:tone"]
    assert len(tone) == 6  # three in each record
    assert not any("plan" in kinds for kinds in tone[:3])  # q:go escalates the first anyway
    witness = report["questions"]["q:tone"]["witness"]
    assert witness["record"] == open_.record_hash
    assert witness["value"]["kind"] == "Unavailable"


def test_no_record_asks_nothing_and_runs_no_trial() -> None:
    report = question_reach_sync([], LOADED)
    assert report["records"] == []
    assert report["trials"] == 0
    assert {q: v["status"] for q, v in report["questions"].items()} == dict.fromkeys(
        ["q:go", "q:late", "q:tone"], "never-asked"
    )


def test_an_entry_of_a_failed_call_runs_no_trial(tried: list[Any]) -> None:
    from ommat import Failed

    replies = Scripted({"ask": Failed(Cause("raised", ["q:go", "q:tone"])), "late": returns({})})
    record = decide_sync(LOADED, {}, {"default": replies}).record
    report = question_reach_sync([record], LOADED)
    assert report["trials"] == 0
    assert tried == []
    assert report["questions"]["q:go"]["status"] == "never-asked"


# ---------------------------------------------------------------------------------------------
# The trials


def test_each_point_but_the_recorded_one_is_tried_in_order_the_marker_last(
    tried: list[Any],
) -> None:
    report = question_reach_sync([turn(p_yes=0.0)], LOADED)
    go_tried = [value for ref, value, _ in tried if ref == "q:go"]
    # q:go's nine axis points, the bar, margin and their mirrors, the gaps and the ends, less
    # the recorded 0.0, then the marker
    assert [p_yes(v) for v in go_tried[:-1]] == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    marker = go_tried[-1]
    assert type(marker) is Unavailable
    assert marker.cause.code == "refused"
    assert [v for ref, v, _ in tried if ref == "q:tone"][:2] == [
        {"warm": 1.0, "cold": 0.0},
        {"warm": 0.0, "cold": 1.0},
    ]
    assert report["trials"] == len(tried) == 9 + 3


def test_a_recorded_value_off_the_points_skips_nothing(tried: list[Any]) -> None:
    report = question_reach_sync([turn(p_yes=0.95)], LOADED)
    assert len([ref for ref, _, _ in tried if ref == "q:go"]) == 10
    assert report["trials"] == 10 + 3


def test_a_recorded_marker_skips_the_marker_trial(tried: list[Any]) -> None:
    refused = Unavailable(Cause("refused", ["q:tone"]))
    record = turn(tone=refused, loaded=MARKED)
    assert entry(record.log, "q:tone")["marker"]["cause"]["code"] == "refused"
    report = question_reach_sync([record], MARKED)
    assert [v for ref, v, _ in tried if ref == "q:tone"] == [
        {"warm": 1.0, "cold": 0.0},
        {"warm": 0.0, "cold": 1.0},
    ]
    witness = report["questions"]["q:tone"]["witness"]
    assert witness["value"] == answer("choice", ["warm", "cold"], warm=1.0, cold=0.0)


def test_the_marker_trial_names_the_entry_and_its_witness_is_the_marker_form() -> None:
    report = question_reach_sync([turn(loaded=MARKED)], MARKED)
    assert report["questions"]["q:tone"] == {
        "status": "live",
        "witness": {
            "record": turn(loaded=MARKED).record_hash,
            "ref": "q:tone",
            "value": {"kind": "Unavailable", "cause": {"code": "refused", "refs": ["q:tone"]}},
        },
    }


def test_the_witness_is_the_first_reaching_trial_by_record_then_point() -> None:
    low, high = turn(p_yes=0.0), turn(p_yes=0.95)
    first = question_reach_sync([low, high], LOADED)["questions"]["q:go"]["witness"]
    # from 0.0, escalating, the first point that executes is the bar
    assert (first["record"], first["value"]["probabilities"]) == (
        low.record_hash,
        {"yes": 0.8, "no": 0.2},
    )
    second = question_reach_sync([high, low], LOADED)["questions"]["q:go"]["witness"]
    assert (second["record"], second["value"]["probabilities"]) == (
        high.record_hash,
        {"yes": 0.0, "no": 1.0},
    )


def test_the_trials_count_every_record_given_a_repeat_included() -> None:
    record = turn()
    once = question_reach_sync([record], LOADED)
    twice = question_reach_sync([record, record], LOADED)
    assert twice["trials"] == 2 * once["trials"]
    assert twice["records"] == [record.record_hash] * 2


# ---------------------------------------------------------------------------------------------
# A per-candidate and a candidate-valued question


def test_a_per_candidate_question_is_tried_through_each_grounded_entry_in_log_order(
    tried: list[Any],
) -> None:
    record = grounded.decided()
    report = question_reach_sync([record], grounded.LOADED)
    assert list(report["questions"]) == [grounded.OUT]
    refs = [ref for ref, _, _ in tried]
    ana, bo = f'{grounded.OUT}["Ana"]', f'{grounded.OUT}["Bo"]'
    assert refs == [ana] * 3 + [bo] * 3
    # each entry's marker names that entry
    assert [list(v.cause.refs) for _, v, _ in tried if isinstance(v, Unavailable)] == [[ana], [bo]]
    assert report["questions"][grounded.OUT]["status"] == "live"
    assert report["questions"][grounded.OUT]["witness"]["ref"] == ana


def who(r: Any) -> str:
    sender = r.read("q:sender")
    if isinstance(sender, Undetermined):
        return "unknown"
    return max(sender.probabilities, key=sender.probabilities.__getitem__)


def by_who(r: Any) -> Plan:
    if r.read("d:who") == "ann":
        return Plan("inform", reply="from ann")
    return Plan("inform", reply="ok")


def test_a_candidate_valued_question_is_tried_at_each_fixed_label_then_the_first_candidate(
    tried: list[Any],
) -> None:
    import dataclasses

    d = dataclasses.replace(
        label_sets.design(),
        rules={"who": Rule("d:who", t.string, who, version="1", reads=["q:sender"])},
        policy=Policy(by_who, version="1", reads=["d:who"]),
    )
    loaded = load(d)
    record = label_sets.run(["ann", "bo"], label_sets.Model(top="bo"), d=d)
    report = question_reach_sync([record], loaded)
    labels = ["none", "several", "ann", "bo"]
    assert [v for ref, v, _ in tried if ref == "q:sender"][:3] == [
        dict(zip(labels, [1.0, 0.0, 0.0, 0.0], strict=True)),
        dict(zip(labels, [0.0, 1.0, 0.0, 0.0], strict=True)),
        dict(zip(labels, [0.0, 0.0, 1.0, 0.0], strict=True)),  # ann, the first candidate
    ]
    assert len([ref for ref, _, _ in tried if ref == "q:sender"]) == 4
    assert report["questions"]["q:sender"] == {
        "status": "live",
        "witness": {
            "record": record.record_hash,
            "ref": "q:sender",
            "value": answer("choice", labels, none=0.0, several=0.0, ann=1.0, bo=0.0),
        },
    }
    assert report["questions"]["q:other"]["status"] == "no-change-found"


def test_a_candidate_valued_answer_recorded_at_a_point_is_skipped(tried: list[Any]) -> None:
    answers = {"q:sender": {"none": 0.0, "several": 0.0, "ann": 1.0, "bo": 0.0}}
    record = label_sets.run(["ann", "bo"], label_sets.Model(answers=answers))
    question_reach_sync([record], load(label_sets.design()))
    sender = [v for ref, v, _ in tried if ref == "q:sender"]
    assert len(sender) == 3
    assert isinstance(sender[-1], Unavailable)


# ---------------------------------------------------------------------------------------------
# values


def test_a_candidate_valued_question_over_an_empty_set_is_never_asked(tried: list[Any]) -> None:
    # An empty set leaves the question out with its call null, so a returned call always
    # rendered a candidate and no trial is ever made without one.
    record = label_sets.run([])
    assert entry(record.log, "q:sender")["call"] is None
    report = question_reach_sync([record], load(label_sets.design()))
    assert [ref for ref, _, _ in tried] == ["q:other"] * 3
    assert report["trials"] == 3
    assert report["questions"]["q:sender"] == {"status": "never-asked", "witness": None}


def test_values_narrows_a_label_question_with_no_grid_entry(tried: list[Any]) -> None:
    report = question_reach_sync([turn()], LOADED, values={"q:tone": ["cold"], "q:late": ["no"]})
    assert report["narrowed"] == ["q:late", "q:tone"]
    tone = [v for ref, v, _ in tried if ref == "q:tone"]
    assert tone[0] == {"warm": 0.0, "cold": 1.0}
    assert isinstance(tone[1], Unavailable)
    assert len(tone) == 2


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"t:state": [{"a": 1}]}, "a read the lint does not fill"),
        ({"d:go": [True]}, "a read the lint does not fill"),
        ({"q:nope": ["x"]}, "a read the lint does not fill"),
        ({"q:go": ["yes"]}, "its grid points or its type fill"),
        ({"q:tone": ["hot"]}, "is not one of its labels"),
        ({"q:tone": ["cold", "cold"]}, "lists one value twice"),
        ({"q:tone": []}, "is empty"),
    ],
)
@pytest.mark.usefixtures("no_trial")
def test_a_slip_in_values_is_a_value_error_before_any_record_is_read(
    values: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        question_reach_sync([b"not json"], LOADED, values=values)


def test_an_unfilled_question_is_a_value_error_and_an_extraction_value_is_checked() -> None:
    loaded = load(extraction.design())
    with pytest.raises(ValueError, match=r'unfilled: "q:dates"'):
        question_reach_sync([b"not json"], loaded)
    with pytest.raises(ValueError, match="does not conform"):
        question_reach_sync([b"not json"], loaded, values={"q:dates": [["not a date"]]})
    record = decide_sync(
        loaded,
        extraction.STATE,
        {
            "default": Scripted(
                {"read": returns({"q:amount": extraction.AMOUNT, "q:dates": extraction.DATES})}
            )
        },
    ).record
    report = question_reach_sync([record], loaded, values={"q:dates": [["2024-03-01"]]})
    assert report["questions"]["q:dates"]["status"] == "live"
    assert report["questions"]["q:dates"]["witness"]["value"]["value"] == ["2024-03-01"]


def test_values_that_is_not_a_mapping_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        question_reach_sync([], LOADED, values=["q:tone"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Refusals


def unfinished() -> Record:
    return edited(turn(), lambda doc: doc["log"].pop())  # the verdict item, last


def broken() -> Record:
    return turn(state=5)  # a state the encoder refuses: no state entry


def drifted() -> Record:
    def flip(doc: dict[str, Any]) -> None:
        go_entry = entry(doc["log"], "d:go")
        go_entry["value"] = not go_entry["value"]

    return edited(turn(), flip)


@pytest.mark.usefixtures("no_trial")
def test_each_record_is_refused_with_its_turn_before_any_trial() -> None:
    for given, reason in (
        (unfinished(), R.unfinished),
        (broken(), R.nothing_to_replay),
        (live(), R.wrong_design),
        (drifted(), R.baseline_differs),
    ):
        exc = refusal([turn(), given], LOADED)
        assert (exc.reason, exc.turn) == (reason, given.record_hash)
    exc = refusal([turn(), b"not json"], LOADED)
    assert (exc.reason, exc.turn) == (R.invalid_record, None)
    assert type(exc.error) is InvalidRecord


@pytest.mark.usefixtures("no_trial")
def test_the_records_are_refused_one_by_one_in_list_order() -> None:
    # Each record is read and refused in full before the next is read.
    exc = refusal([drifted(), b"not json"], LOADED)
    assert exc.reason is R.baseline_differs
    exc = refusal([unfinished(), broken()], LOADED)
    assert exc.reason is R.unfinished
    exc = refusal([broken(), unfinished()], LOADED)
    assert exc.reason is R.nothing_to_replay


def test_a_record_differing_only_in_helpers_mismatch_is_tried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ommat._replay import FindingKind

    real = _diff._replayed

    async def noted(log: Any, loaded: Any, answers: Any) -> Any:
        return [*(await real(log, loaded, answers)), {"kind": FindingKind.helpers_mismatch}]

    monkeypatch.setattr(_diff, "_replayed", noted)
    assert question_reach_sync([turn()], LOADED)["questions"]["q:go"]["status"] == "live"


def test_a_loaded_not_exactly_a_loaded_design_is_refused_first() -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        question_reach_sync([b"x"], object())  # type: ignore[arg-type]


@pytest.mark.parametrize("records", ["abc", b"abc", {"a": 1}, 5])
def test_records_that_are_not_an_iterable_of_records_are_a_type_error(records: Any) -> None:
    with pytest.raises(TypeError):
        question_reach_sync(records, LOADED)


# ---------------------------------------------------------------------------------------------
# The output and the twin


def test_the_output_is_plain_data_reaching_no_record_and_no_input_changes() -> None:
    record = turn(loaded=MARKED)
    parsed = json.loads(record.document)
    kept = copy.deepcopy(parsed)
    before = (record.document, MARKED.document)
    result = question_reach_sync([record], MARKED)
    assert records_in(result) == []
    assert json.loads(json.dumps(result)) == result
    assert copy.deepcopy(result) == result
    assert question_reach_sync([parsed], MARKED) == result
    assert (parsed, (record.document, MARKED.document)) == (kept, before)


def test_the_job_is_awaitable_and_equals_its_sync_twin() -> None:
    record = turn()
    awaited = asyncio.run(question_reach([record], LOADED))
    assert awaited == question_reach_sync([record], LOADED)


def test_the_sync_twin_inside_a_running_loop_raises_before_doing_anything() -> None:
    async def main() -> None:
        with pytest.raises(RuntimeError, match="question_reach_sync was called inside a running"):
            question_reach_sync(object(), object())  # type: ignore[arg-type]

    asyncio.run(main())


def test_witness_and_trial_values_are_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    record = turn(loaded=MARKED)
    parsed = json.loads(record.document)
    kept = copy.deepcopy(parsed)
    held: list[Any] = []  # each trial's value and the entry its copy wrote
    real = _reach._applied

    async def holding(read: Any, log: Any, loaded: Any, edits: Any, rule: Any, f: Any) -> Any:
        out = await real(read, log, loaded, edits, rule, f)
        held.append((edits[0].value, out[0][0]))
        return out

    monkeypatch.setattr(_reach, "_applied", holding)
    report = question_reach_sync([parsed], MARKED)
    expected = copy.deepcopy(report)
    assert report["questions"]["q:tone"]["status"] == "live"
    for value, item in held:  # spoil every trial's value and written entry
        if type(value) is dict:
            cast("dict[str, Any]", value).clear()
        cast("dict[str, Any]", item.get("value", item.get("marker"))).clear()
    assert report == expected  # the witness shares nothing with a trial
    report["questions"]["q:tone"]["witness"]["value"]["cause"]["refs"].append("q:go")
    report["records"].clear()
    assert question_reach_sync([parsed], MARKED) == expected  # nor with a later run
    assert question_reach_sync([record], MARKED) == expected
    assert parsed == kept


def test_a_trial_value_spoiled_by_its_trial_reaches_no_later_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = turn()
    given: list[Any] = []
    real = _reach._applied

    async def spoiling(read: Any, log: Any, loaded: Any, edits: Any, rule: Any, f: Any) -> Any:
        value = edits[0].value
        given.append(copy.deepcopy(value))
        out = await real(read, log, loaded, edits, rule, f)
        if type(value) is dict:
            cast("dict[str, Any]", value).clear()  # the points are shared across records
        return out

    monkeypatch.setattr(_reach, "_applied", spoiling)
    report = question_reach_sync([record, record], LOADED)
    half = len(given) // 2
    assert repr(given[:half]) == repr(given[half:])
    assert report["trials"] == len(given)
