"""The gold-answers job: labels for the questions finished turns asked, written under a gold
set by the original turn id, question identity and candidate; each refusal in its order before
any write; a copy from an old set, its withdrawal and its emptiness; and a held key before the
writes and at one (§evaluation-store)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import extraction as x
import test_label_sets as label_sets
import test_per_candidate as pc
from designs import minimal
from ommat import CallResult, InvalidRecord, Question, Record, Source, Text, decide_sync, load
from ommat.evaluation import EvaluationRefusal, EvaluationStore, RefusalReason, write_gold_answers

R = RefusalReason
TYPES = ("text", "number", "date", "time", "datetime", "duration")
SPAM = Question("noul", Text("Is it spam?"))
SOURCE = Source(
    {
        "q:spam": SPAM,
        "q:junk": SPAM,  # one question at two refs: one gold key
        "q:tone": Question(
            "choice", Text("Which tone?"), [{"name": "warm", "description": "kind words"}, "cold"]
        ),
        "q:urgency": Question("score", Text("How urgent?"), ["low", "high"]),
        **{f"q:{v}": x.question(v) for v in TYPES},
        "q:days": x.question("date", True),
    },
    needs=["t:state.body", "t:state.today"],
)
LOADED = load(minimal(state=x.STATE_T, sources={"ask": SOURCE}))


async def silent(batch: Any, *, seconds: float) -> CallResult:
    """Returns nothing: every entry is a marker its call wrote."""
    return CallResult({})


def turn(body: str = x.BODY) -> Record:
    return decide_sync(LOADED, {**x.STATE, "body": body}, {"default": silent}).record


ONE, TWO = turn(), turn("Owe 7")


def identity(record: Record, ref: str) -> str:
    call = next(i for i in record.log if i["item"] == "call")
    return call["questions"][ref]["identity"]


def key(gold_set: str, record: Record, ref: str, candidate: str = "") -> dict[str, str]:
    return {
        "gold_set": gold_set,
        "question": identity(record, ref.split("[")[0]),
        "turn": record.record_hash,
        "candidate": candidate,
    }


def gold(gold_set: str, record: Record, ref: str, value: object, candidate: str = "") -> Any:
    return {
        "kind": "gold-answer",
        "format": 2,
        "key": key(gold_set, record, ref, candidate),
        "value": value,
    }


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def refused(root: Path, *args: Any, **kwargs: Any) -> EvaluationRefusal:
    """The job's refusal, after which the store holds the files it held before."""
    before = files(root)
    with pytest.raises(EvaluationRefusal) as raised:
        write_gold_answers(*args, results=EvaluationStore(root), **kwargs)
    assert files(root) == before
    return raised.value


# ---------------------------------------------------------------------------------------------
# Writing


def test_a_noul_choice_and_score_label_are_written_under_their_keys(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    labels = {ONE.record_hash: {"q:spam": "yes", "q:tone": "cold", "q:urgency": "high"}}
    written = write_gold_answers("v1", [ONE], labels, results=store)
    assert written == [
        gold("v1", ONE, "q:spam", "yes"),
        gold("v1", ONE, "q:tone", "cold"),
        gold("v1", ONE, "q:urgency", "high"),
    ]
    for entry in written:
        assert store.lookup("gold-answer", entry["key"]) == entry
    assert json.loads(json.dumps(written)) == copy.deepcopy(written) == written


def test_a_described_label_is_given_by_its_name_and_refused_by_its_description(
    tmp_path: Path,
) -> None:
    by_name = {ONE.record_hash: {"q:tone": "warm"}}
    assert write_gold_answers("v1", [ONE], by_name, results=EvaluationStore(tmp_path)) == [
        gold("v1", ONE, "q:tone", "warm")
    ]
    exc = refused(tmp_path, "v2", [ONE], {ONE.record_hash: {"q:tone": "kind words"}})
    assert (exc.reason, exc.turn, exc.ref) == (R.invalid_label, ONE.record_hash, "q:tone")
    assert "'kind words'" in str(exc)


def test_a_conforming_and_a_null_extraction_label_are_written(tmp_path: Path) -> None:
    labels = {
        ONE.record_hash: {
            "q:number": {"value": 120},
            "q:date": {"value": None},
            "q:days": {"value": ["2024-02-29", "2024-03-01"]},
            "q:duration": {"value": "P1DT2H"},
        }
    }
    written = write_gold_answers("v1", [ONE], labels, results=EvaluationStore(tmp_path))
    assert written == [
        gold("v1", ONE, "q:date", {"value": None}),
        gold("v1", ONE, "q:days", {"value": ["2024-02-29", "2024-03-01"]}),
        gold("v1", ONE, "q:duration", {"value": "P1DT2H"}),
        gold("v1", ONE, "q:number", {"value": 120.0}),  # a number as the answer holds it
    ]


@pytest.mark.parametrize(
    ("ref", "label"),
    [
        ("q:text", {"value": 7}),
        ("q:number", {"value": "120"}),
        ("q:number", {"value": True}),
        ("q:number", {"value": 2**53}),
        ("q:date", {"value": "2023-02-29"}),
        ("q:time", {"value": "24:00:00"}),
        ("q:datetime", {"value": "2024-01-01 10:00:00"}),
        ("q:duration", {"value": "P1W"}),
        ("q:days", {"value": "2024-02-29"}),
        ("q:days", {"value": ["2024-02-29", None]}),
        ("q:number", 120),
        ("q:number", {"value": 120, "span": None}),
        ("q:spam", "maybe"),
        ("q:spam", {"value": "yes"}),
        ("q:urgency", "medium"),
        ("q:tone", ["cold"]),
    ],
)
def test_a_label_not_of_its_question_is_invalid(tmp_path: Path, ref: str, label: object) -> None:
    exc = refused(tmp_path, "v1", [ONE], {ONE.record_hash: {ref: label}})
    assert (exc.reason, exc.turn, exc.ref, exc.key) == (R.invalid_label, ONE.record_hash, ref, None)


def test_a_grounded_refs_candidate_is_in_its_key(tmp_path: Path) -> None:
    record = pc.decided()
    labels = {record.record_hash: {pc.REFS[0]: "yes", pc.URGENT: "no"}}
    written = write_gold_answers("v1", [record], labels, results=EvaluationStore(tmp_path))
    call = next(i for i in record.log if i["item"] == "call")
    out, urgent = (call["questions"][r]["identity"] for r in (pc.OUT, pc.URGENT))
    assert [e["key"]["question"] for e in written] == [out, urgent]
    assert [e["key"]["candidate"] for e in written] == [pc.STREETS[0], ""]  # 'Quay "St"'


def test_a_candidate_valued_label_set_takes_the_turns_candidates(tmp_path: Path) -> None:
    record = label_sets.run(["ann", "bo"])
    turn_id = record.record_hash
    written = write_gold_answers(
        "v1", [record], {turn_id: {"q:sender": "bo"}}, results=EvaluationStore(tmp_path)
    )
    assert [e["value"] for e in written] == ["bo"]
    assert write_gold_answers(
        "v2", [record], {turn_id: {"q:sender": "none"}}, results=EvaluationStore(tmp_path)
    )
    exc = refused(tmp_path, "v3", [record], {turn_id: {"q:sender": "cy"}})
    assert (exc.reason, exc.ref) == (R.invalid_label, "q:sender")


def test_an_unlabelled_question_writes_nothing(tmp_path: Path) -> None:
    assert write_gold_answers("v1", [ONE, TWO], {}, results=EvaluationStore(tmp_path)) == []
    assert files(tmp_path) == []


def test_records_are_read_as_records_bytes_or_parsed_documents_under_the_original_turn_id(
    tmp_path: Path,
) -> None:
    decision = decide_sync(LOADED, x.STATE, {"default": silent})
    labels = {decision.record.record_hash: {"q:spam": "no"}}
    given = [decision.record, decision.record.document, json.loads(decision.record.document)]
    for i, record in enumerate(given):
        (entry,) = write_gold_answers(
            f"v{i}", [record, record], labels, results=EvaluationStore(tmp_path)
        )
        assert entry["key"]["turn"] == decision.record.record_hash == ONE.record_hash


# ---------------------------------------------------------------------------------------------
# One key's refs


def test_refs_of_one_key_write_one_entry_and_an_absent_ref_takes_the_given_label(
    tmp_path: Path,
) -> None:
    store = EvaluationStore(tmp_path)
    both = {ONE.record_hash: {"q:spam": "yes", "q:junk": "yes"}}
    assert write_gold_answers("v1", [ONE], both, results=store) == [
        gold("v1", ONE, "q:junk", "yes")
    ]
    one = {ONE.record_hash: {"q:spam": "no"}}
    assert write_gold_answers("v2", [ONE], one, results=store) == [gold("v2", ONE, "q:junk", "no")]


@pytest.mark.parametrize("other", ["no", None])
def test_differing_labels_for_one_key_conflict(tmp_path: Path, other: object) -> None:
    labels = {ONE.record_hash: {"q:spam": "yes", "q:junk": other}}
    exc = refused(tmp_path, "v1", [ONE], labels)
    assert (exc.reason, exc.turn, exc.ref) == (R.conflicting_labels, ONE.record_hash, None)
    assert exc.key == [key("v1", ONE, "q:spam")]
    assert "q:junk" in str(exc)
    assert "q:spam" in str(exc)


def test_a_none_for_every_ref_of_a_key_writes_nothing(tmp_path: Path) -> None:
    labels = {ONE.record_hash: {"q:spam": None, "q:junk": None, "q:tone": "warm"}}
    written = write_gold_answers("v1", [ONE], labels, results=EvaluationStore(tmp_path))
    assert written == [gold("v1", ONE, "q:tone", "warm")]


# ---------------------------------------------------------------------------------------------
# A copy


def old_set(root: Path) -> None:
    labels = {
        ONE.record_hash: {"q:spam": "yes", "q:tone": "warm", "q:number": {"value": 1}},
        TWO.record_hash: {"q:spam": "no"},
    }
    write_gold_answers("v1", [ONE, TWO], labels, results=EvaluationStore(root))


def test_a_copy_writes_the_given_label_else_the_old_one(tmp_path: Path) -> None:
    old_set(tmp_path)
    labels = {ONE.record_hash: {"q:tone": "cold", "q:urgency": "low"}}
    written = write_gold_answers(
        "v2", [ONE], labels, results=EvaluationStore(tmp_path), copy_from="v1"
    )
    assert written == [
        gold("v2", ONE, "q:junk", "yes"),
        gold("v2", ONE, "q:number", {"value": 1}),
        gold("v2", ONE, "q:tone", "cold"),
        gold("v2", ONE, "q:urgency", "low"),
    ]  # TWO, not passed, carries nothing over


def test_a_given_none_withdraws_an_old_label(tmp_path: Path) -> None:
    old_set(tmp_path)
    labels = {ONE.record_hash: {"q:spam": None}, TWO.record_hash: {"q:junk": None}}
    written = write_gold_answers(
        "v2", [ONE, TWO], labels, results=EvaluationStore(tmp_path), copy_from="v1"
    )
    assert written == [gold("v2", ONE, "q:number", {"value": 1}), gold("v2", ONE, "q:tone", "warm")]


def test_a_format_1_old_entry_is_copied_as_its_label(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    store.write(
        {"kind": "gold-answer", "format": 1, "key": key("v1", ONE, "q:spam"), "value": "no"}
    )
    written = write_gold_answers("v2", [ONE], {}, results=store, copy_from="v1")
    assert written == [gold("v2", ONE, "q:junk", "no")]


def test_an_old_entry_of_an_unknown_format_is_refused(tmp_path: Path) -> None:
    held = key("v1", ONE, "q:spam")
    EvaluationStore(tmp_path).write(
        {"kind": "gold-answer", "format": 3, "key": held, "value": "no"}
    )
    exc = refused(tmp_path, "v2", [ONE], {}, copy_from="v1")
    assert (exc.reason, exc.key) == (R.unknown_entry, [held])


def test_a_copy_from_a_set_holding_none_of_the_keys_is_empty(tmp_path: Path) -> None:
    old_set(tmp_path)
    exc = refused(tmp_path, "v2", [ONE], {ONE.record_hash: {"q:tone": "cold"}}, copy_from="v0")
    assert (exc.reason, exc.key, exc.turn) == (R.empty_copy, None, None)
    assert "'v0'" in str(exc)
    third = turn("Owe 3")  # no key of it in v1
    assert refused(tmp_path, "v2", [third], {}, copy_from="v1").reason is R.empty_copy


# ---------------------------------------------------------------------------------------------
# Held keys


def test_held_keys_are_refused_before_any_write_naming_every_one(tmp_path: Path) -> None:
    old_set(tmp_path)
    labels = {
        ONE.record_hash: {"q:urgency": "low", "q:spam": "yes", "q:number": {"value": 1}},
        TWO.record_hash: {"q:junk": "no"},
    }
    exc = refused(tmp_path, "v1", [ONE, TWO], labels)
    assert exc.reason is R.held_key
    assert exc.key == [
        key("v1", ONE, "q:junk"),
        key("v1", ONE, "q:number"),
        key("v1", TWO, "q:spam"),
    ]


class Racing(EvaluationStore):
    """A store another run writes the second entry to first, between the lookup and the write."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writes = 0

    def write(self, entry: dict[str, Any]) -> None:
        self.writes += 1
        if self.writes == 2:
            super().write(entry)
        super().write(entry)


def test_a_key_another_run_wrote_first_is_held_then_this_runs_keys_in_write_order(
    tmp_path: Path,
) -> None:
    labels = {ONE.record_hash: {"q:spam": "yes", "q:tone": "cold", "q:urgency": "high"}}
    with pytest.raises(EvaluationRefusal) as raised:
        write_gold_answers("v1", [ONE], labels, results=Racing(tmp_path))
    assert raised.value.reason is R.held_key
    assert raised.value.key == [key("v1", ONE, "q:tone"), key("v1", ONE, "q:junk")]
    assert type(raised.value.__cause__) is EvaluationRefusal


# ---------------------------------------------------------------------------------------------
# Refusals and their order


def test_an_invalid_record_is_refused_before_any_label_is_checked(tmp_path: Path) -> None:
    exc = refused(tmp_path, "v1", [ONE, b"not json"], {"sha256:nope": {"q:spam": "yes"}})
    assert (exc.reason, exc.turn) == (R.invalid_record, None)
    assert type(exc.error) is InvalidRecord
    assert exc.__cause__ is exc.error
    assert "records[1]" in str(exc)


@pytest.mark.parametrize(
    ("turn_id", "ref"),
    [
        ("sha256:" + "0" * 64, "q:spam"),  # a turn not passed
        (ONE.record_hash, "q:nope"),  # no entry
        (ONE.record_hash, "t:state"),  # no q: entry
        (ONE.record_hash, 'q:spam["a"]'),  # not as written
    ],
)
def test_a_label_for_no_question_a_call_wrote_is_not_found(
    tmp_path: Path, turn_id: str, ref: str
) -> None:
    exc = refused(tmp_path, "v1", [ONE], {turn_id: {ref: "yes"}})
    assert (exc.reason, exc.turn, exc.ref) == (R.not_found, turn_id, ref)


def test_an_entry_no_call_wrote_is_not_found(tmp_path: Path) -> None:
    record = label_sets.run([])  # an empty set: q:sender Unasked, its call null
    exc = refused(tmp_path, "v1", [record], {record.record_hash: {"q:sender": "none"}})
    assert (exc.reason, exc.ref) == (R.not_found, "q:sender")


def test_the_refusals_come_in_their_order(tmp_path: Path) -> None:
    old_set(tmp_path)
    one, two = ONE.record_hash, TWO.record_hash
    bad, conflict = {"q:tone": "hot"}, {"q:spam": "yes", "q:junk": "no"}
    order = [
        ({one: bad, two: {"q:nope": "yes"}}, "v0", R.not_found),
        ({one: conflict, two: bad}, "v0", R.invalid_label),
        ({one: conflict}, "v0", R.conflicting_labels),
        ({one: {"q:spam": "yes"}}, "v0", R.empty_copy),
        ({one: {"q:spam": "yes"}}, "v1", R.held_key),
    ]
    for labels, copy_from, reason in order:
        exc = refused(tmp_path, "v1", [ONE, TWO], labels, copy_from=copy_from)
        assert exc.reason is reason


def test_labels_not_mappings_are_type_errors(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    with pytest.raises(TypeError, match="labels is a list"):
        write_gold_answers("v1", [ONE], [], results=store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a mapping of refs"):
        write_gold_answers("v1", [ONE], {ONE.record_hash: "yes"}, results=store)  # type: ignore[dict-item]
