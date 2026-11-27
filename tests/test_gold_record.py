# pyright: reportPrivateUsage=false
"""``gold_record``: a replay of a turn with every question it asked set to its gold answer.
Its record holds the gold answers, its findings are what they change downstream, a question
with no gold answer is stale and listed, and it refuses as ``intervene`` does."""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import extraction
import test_grounded_rules as grounded
import test_label_sets as label_sets
from ledger_logs import rehashed
from ommat import (
    Cause,
    CauseCode,
    EntryRefusal,
    Failed,
    Record,
    Unavailable,
    load,
    read_record,
)
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    gold_record,
    gold_record_sync,
    write_gold_answers,
)
from scripted import returns
from test_decide import LOADED, bindings, go, slice_design
from test_extraction_turn import ran as extraction_ran

R = RefusalReason


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def kinds(found: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
    return [(f["kind"], f.get("ref")) for f in found]


def labelled(root: Path, record: Record, labels: Mapping[str, object]) -> EvaluationStore:
    store = EvaluationStore(root)
    write_gold_answers("g", [record], {record.record_hash: labels}, results=store)
    return store


def test_a_gold_answer_on_the_answers_label_changes_nothing_downstream(tmp_path: Path) -> None:
    base = go().record  # q:spam answered yes at 0.9
    out = gold_record_sync(base, LOADED, "g", results=labelled(tmp_path, base, {"q:spam": "yes"}))
    assert entry(out["record"], "q:spam")["value"]["probabilities"] == {"yes": 1.0, "no": 0.0}
    assert out["findings"] == []  # an entry set is never a finding itself
    assert (out["stale"], out["missing"]) == ([], [])


def test_a_gold_answer_that_differs_changes_what_follows(tmp_path: Path) -> None:
    base = go().record
    out = gold_record_sync(base, LOADED, "g", results=labelled(tmp_path, base, {"q:spam": "no"}))
    assert kinds(out["findings"]) == [("entry", "d:act"), ("entry", "d:route")]


def test_a_question_with_no_gold_answer_is_stale_and_listed(tmp_path: Path) -> None:
    base = go().record
    out = gold_record_sync(base, LOADED, "g", results=EvaluationStore(tmp_path))
    assert out["missing"] == ["q:spam"]
    assert out["stale"] == ["q:spam"]
    found = [f for f in out["findings"] if f["kind"] == "stale-question"]
    assert found == [
        {"kind": "stale-question", "ref": "q:spam", "source": "ask", "clause": "no-answer"}
    ]
    assert entry(out["record"], "q:spam")["marker"]["cause"]["code"] == "not-returned"


def test_another_gold_set_is_not_read(tmp_path: Path) -> None:
    base = go().record
    store = labelled(tmp_path, base, {"q:spam": "no"})
    out = gold_record_sync(base, LOADED, "other", results=store)
    assert out["missing"] == ["q:spam"]


def test_a_question_whose_call_failed_keeps_the_failure_and_is_missing(tmp_path: Path) -> None:
    base = go(binds=bindings(ask=Failed(Cause(CauseCode.raised, ["q:spam", "t:body"])))).record
    store = labelled(tmp_path, base, {"q:spam": "no"})
    out = gold_record_sync(base, LOADED, "g", results=store)
    assert out["missing"] == ["q:spam"]
    assert out["stale"] == []  # the call is given back failed, whole, as recorded
    assert entry(out["record"], "q:spam")["marker"]["cause"]["code"] == "raised"
    assert entry(out["record"], "t:body")["marker"]["cause"]["code"] == "raised"


def test_an_extraction_gold_answer_is_held_with_no_doubt_and_no_span(tmp_path: Path) -> None:
    base, loaded = extraction_ran(), extraction.loaded()
    labels = {"q:amount": {"value": 99}, "q:dates": {"value": None}}
    out = gold_record_sync(base, loaded, "g", results=labelled(tmp_path, base, labels))
    amount = entry(out["record"], "q:amount")["value"]
    assert (amount["value"], amount["p_null"], amount["confidence"], amount["span"]) == (
        99.0,
        0.0,
        1.0,
        None,
    )
    dates = entry(out["record"], "q:dates")["value"]
    assert (dates["value"], dates["p_null"]) == (None, 1.0)
    assert out["missing"] == []


def test_a_grounded_question_takes_each_candidates_gold_answer(tmp_path: Path) -> None:
    base = grounded.decided()
    labels = {'q:ask/out["Ana"]': "no", 'q:ask/out["Bo"]': "yes"}
    out = gold_record_sync(base, grounded.LOADED, "g", results=labelled(tmp_path, base, labels))
    assert entry(out["record"], 'q:ask/out["Ana"]')["value"]["probabilities"]["no"] == 1.0
    assert entry(out["record"], 'q:ask/out["Bo"]')["value"]["probabilities"]["yes"] == 1.0
    assert out["missing"] == []


def test_a_gold_record_under_a_rerun_design(tmp_path: Path) -> None:
    base = go().record
    rerun = slice_design(name="changed")
    store = labelled(tmp_path, base, {"q:spam": "yes"})
    out = gold_record_sync(base, LOADED, "g", results=store, rerun_design=rerun)
    assert out["record"].design_hash == rerun.design_hash
    assert out["findings"][0]["kind"] == "design-mismatch"


def test_a_gold_entry_of_an_unknown_format_is_unknown_entry(tmp_path: Path) -> None:
    base = go().record
    store = labelled(tmp_path, base, {"q:spam": "no"})
    (key,) = [e["key"] for e in _entries(store, base)]
    other = EvaluationStore(tmp_path / "other")
    other.write({"kind": "gold-answer", "format": 9, "key": key, "value": "no"})
    with pytest.raises(EvaluationRefusal) as raised:
        gold_record_sync(base, LOADED, "g", results=other)
    assert raised.value.reason is R.unknown_entry


def test_a_gold_value_not_of_its_questions_form_is_unknown_entry(tmp_path: Path) -> None:
    base = go().record
    store = labelled(tmp_path, base, {"q:spam": "no"})
    (key,) = [e["key"] for e in _entries(store, base)]
    other = EvaluationStore(tmp_path / "other")
    other.write({"kind": "gold-answer", "format": 2, "key": key, "value": "maybe"})
    with pytest.raises(EvaluationRefusal) as raised:
        gold_record_sync(base, LOADED, "g", results=other)
    assert raised.value.reason is R.unknown_entry


def _entries(store: EvaluationStore, base: Record) -> list[dict[str, Any]]:
    """The gold entries written for ``base``, read back by their keys."""
    question = next(i for i in base.log if i["item"] == "call" and "ask" in i["sources"])
    identity = question["questions"]["q:spam"]["identity"]
    key = {"gold_set": "g", "question": identity, "turn": base.record_hash, "candidate": ""}
    held = store.lookup("gold-answer", key)
    assert held is not None
    return [held]


def test_the_record_checks_and_argument_types_are_refused(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    with pytest.raises(EvaluationRefusal) as raised:
        gold_record_sync(b"not json", LOADED, "g", results=store)
    assert raised.value.reason is R.invalid_record
    with pytest.raises(EntryRefusal, match="recorded_design is a"):
        gold_record_sync(go().record, object(), "g", results=store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gold_set"):
        gold_record_sync(go().record, LOADED, 5, results=store)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="results"):
        gold_record_sync(go().record, LOADED, "g", results=tmp_path)  # type: ignore[arg-type]


def test_a_record_before_format_8_is_wrong_design(tmp_path: Path) -> None:
    old = read_record((Path(__file__).parent / "format_9" / "record.json").read_bytes())
    assert old.log[0]["format"] == 7
    with pytest.raises(EvaluationRefusal) as raised:
        gold_record_sync(old, LOADED, "g", results=EvaluationStore(tmp_path))
    assert (raised.value.reason, raised.value.turn) == (R.wrong_design, old.record_hash)


def test_a_cancelled_record_is_unfinished(tmp_path: Path) -> None:
    from test_use_answers import unfinished

    cancelled = unfinished()
    with pytest.raises(EvaluationRefusal) as raised:
        gold_record_sync(cancelled, LOADED, "g", results=EvaluationStore(tmp_path))
    assert (raised.value.reason, raised.value.turn) == (R.unfinished, cancelled.record_hash)


def test_a_record_with_an_entry_cut_is_rerun_and_the_entry_found(tmp_path: Path) -> None:
    """No ``cause`` item, so finished: the rerun writes the entry the record lacks."""
    doc = json.loads(go().record.document)
    policy = "d:act"
    doc["log"] = [i for i in doc["log"] if not (i["item"] == "entry" and i["ref"] == policy)]
    cut = read_record(rehashed(doc))
    store = labelled(tmp_path, cut, {"q:spam": "yes"})
    found = gold_record_sync(cut, LOADED, "g", results=store)["findings"]
    assert [(f["kind"], f["ref"], f["recorded"]) for f in found] == [("entry", policy, None)]


def test_gold_record_is_awaitable_and_its_twin_refuses_inside_a_loop(tmp_path: Path) -> None:
    base = go().record
    store = labelled(tmp_path, base, {"q:spam": "no"})

    async def main() -> dict[str, Any]:
        with pytest.raises(RuntimeError, match="gold_record_sync was called inside a running"):
            gold_record_sync(base, LOADED, "g", results=store)
        return await gold_record(base, LOADED, "g", results=store)

    out = asyncio.run(main())
    assert out["missing"] == []


def test_a_marker_answer_is_replaced_by_its_gold_answer(tmp_path: Path) -> None:
    refused = Unavailable(Cause(CauseCode.refused, ["q:spam"]))
    base = go(binds=bindings(ask=returns({"q:spam": refused, "t:body": "b"}))).record
    assert "marker" in entry(base, "q:spam")
    out = gold_record_sync(base, LOADED, "g", results=labelled(tmp_path, base, {"q:spam": "no"}))
    assert entry(out["record"], "q:spam")["value"]["probabilities"] == {"yes": 0.0, "no": 1.0}


def test_a_candidate_valued_gold_label_takes_the_labels_as_asked(tmp_path: Path) -> None:
    loaded = load(label_sets.design())
    base = label_sets.run(["ann", "bo"])
    out = gold_record_sync(base, loaded, "g", results=labelled(tmp_path, base, {"q:sender": "bo"}))
    probabilities = entry(out["record"], "q:sender")["value"]["probabilities"]
    assert probabilities["bo"] == 1.0
    assert sum(probabilities.values()) == 1.0
    assert set(probabilities) == {"none", "several", "ann", "bo"}
