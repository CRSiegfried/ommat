# pyright: reportPrivateUsage=false
"""The sweep job: the gold copy, carrying its own hash, read back and replayed, from format-1
and format-2 gold entries and an extraction's (§gold-copy)."""

import asyncio
from pathlib import Path
from typing import Any

import pytest

import extraction as x
from ommat import CallResult, Record, decide_sync, read_record
from ommat._replay import FindingKind, _replay_with
from ommat.evaluation import EvaluationRefusal, EvaluationStore, RefusalReason
from ommat.evaluation._sweep import gold_answers, gold_copy
from test_gold_answers import LOADED, ONE, TYPES, key, unfinished
from test_use_answers import broken

R = RefusalReason
_OUTSIDE = {FindingKind.not_in_record, FindingKind.ref_not_in_record, FindingKind.question_changed}


def put(store: EvaluationStore, record: Record, golds: dict[str, object], fmt: int = 2) -> None:
    for ref, value in golds.items():
        store.write(
            {"kind": "gold-answer", "format": fmt, "key": key("g", record, ref), "value": value}
        )


LABELS: dict[str, object] = {
    "q:spam": "yes",  # and q:junk, one question at two refs
    "q:tone": "warm",
    "q:urgency": "high",
    **{f"q:{v}": {"value": None} for v in TYPES},
    "q:number": {"value": 120},
    "q:days": {"value": ["2024-02-29"]},
}


def copy_of(record: Record, store: EvaluationStore) -> Record | str:
    log = record.log
    return gold_copy(record, log, gold_answers(record.record_hash, log, "g", store))


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def replayed(copy: Record, loaded: Any) -> tuple[Record, list[dict[str, Any]]]:
    return asyncio.run(_replay_with(copy.log, loaded, copy.log, observer=None))


# ---------------------------------------------------------------------------------------------
# The gold copy


def test_the_gold_copy_sets_every_question_and_carries_its_own_hash(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, LABELS)
    copy = copy_of(ONE, store)
    assert isinstance(copy, Record)
    assert copy.record_hash != ONE.record_hash
    assert copy.log[0]["hash"] == ONE.log[0]["hash"]  # the design item is kept
    assert read_record(copy.document).record_hash == copy.record_hash
    assert entry(copy, "q:spam")["value"]["probabilities"] == {"yes": 1, "no": 0}
    assert entry(copy, "q:junk")["value"] == entry(copy, "q:spam")["value"]
    assert entry(copy, "q:tone")["value"]["probabilities"] == {"warm": 1, "cold": 0}
    assert entry(copy, "q:urgency")["value"]["probabilities"] == {"low": 0, "high": 1}
    number = entry(copy, "q:number")["value"]
    assert (number["value"], number["p_null"], number["confidence"]) == (120, 0, 1)
    assert (number["guess"], number["span"]) == (None, None)
    null = entry(copy, "q:date")["value"]
    assert (null["value"], null["p_null"], null["confidence"]) == (None, 1, None)
    assert entry(copy, "q:days")["value"]["value"] == ["2024-02-29"]
    assert entry(copy, "t:state") == entry(ONE, "t:state")
    rerun, found = replayed(copy, LOADED)
    assert not [f for f in found if f["kind"] in _OUTSIDE]
    assert entry(rerun, "q:spam") == entry(copy, "q:spam")  # answered from the copy


def test_a_format_1_gold_entry_is_read(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {k: v for k, v in LABELS.items() if type(v) is str}, fmt=1)
    put(store, ONE, {k: v for k, v in LABELS.items() if type(v) is not str})
    copy = copy_of(ONE, store)
    assert isinstance(copy, Record)
    assert entry(copy, "q:tone")["value"]["probabilities"] == {"warm": 1, "cold": 0}


def test_an_extraction_gold_copy_is_read_back_and_replayed(tmp_path: Path) -> None:
    loaded = x.loaded()

    async def silent(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({})

    record = decide_sync(loaded, x.STATE, {"default": silent}).record
    store = EvaluationStore(tmp_path)
    put(store, record, {"q:amount": {"value": 120}, "q:dates": {"value": None}})
    copy = copy_of(record, store)
    assert isinstance(copy, Record)
    back = read_record(copy.document)
    rerun, found = replayed(back, loaded)
    assert not [f for f in found if f["kind"] in _OUTSIDE]
    assert entry(rerun, "d:amount")["value"] == "big"
    assert "marker" in entry(rerun, "d:dates")  # a null gold value extracts to none
    assert entry(record, "d:amount") != entry(rerun, "d:amount")


def test_an_unknown_gold_entry_is_refused_naming_its_key(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {"q:tone": "warm"}, fmt=3)
    with pytest.raises(EvaluationRefusal) as raised:
        copy_of(ONE, store)
    assert (raised.value.reason, raised.value.key) == (R.unknown_entry, [key("g", ONE, "q:tone")])


def test_a_turn_with_no_gold_copy_says_why(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {k: v for k, v in LABELS.items() if k != "q:tone"})
    assert copy_of(ONE, store) == "q:tone has no gold answer"
    assert copy_of(unfinished(), store) == "the record has no verdict item"
    assert copy_of(broken(), store) == "the record has no state entry"


def test_a_label_for_an_extraction_question_is_no_answer(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {**LABELS, "q:number": "yes"}, fmt=1)
    assert copy_of(ONE, store) == "the gold answer for q:number is no answer to its question"


def test_a_failed_call_leaves_no_gold_copy(tmp_path: Path) -> None:
    async def boom(batch: Any, *, seconds: float) -> CallResult:
        raise RuntimeError("down")

    record = decide_sync(LOADED, x.STATE, {"default": boom}).record
    store = EvaluationStore(tmp_path)
    put(store, record, LABELS)
    assert copy_of(record, store) == "the call that wrote q:date is Failed"
