# pyright: reportPrivateUsage=false
"""``answers``: the answers table. One row per question a call answered in each record, per
candidate for a per-candidate question, with the answer or marker, the top label, the
question's targets and groups; with a gold set, the gold answer and whether the answer erred,
a missing gold answer kept as a row. A record of another design is refused."""

import asyncio
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import extraction
import test_grounded_rules as grounded
import test_label_sets as label_sets
from ommat import (
    Cause,
    CauseCode,
    EntryRefusal,
    Group,
    Source,
    Unavailable,
    When,
    decide_sync,
    escalation,
    load,
    read_record,
)
from ommat import types as t
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    answers,
    write_gold_answers,
)
from scripted import returns
from test_decide import (
    ASK,
    BODY,
    LOADED,
    LOOK,
    bindings,
    caught,
    deciding,
    go,
    record_of,
    slice_design,
)
from test_extraction_turn import ran as extraction_ran
from test_targets import STATE as TARGET_STATE
from test_targets import binding as target_binding
from test_targets import targeted

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")  # defects are the point here

R = RefusalReason
NO = {"yes": 0.1, "no": 0.9}


def test_a_row_per_answered_question_with_its_answer_and_top_label() -> None:
    base = go().record
    (row,) = answers([base], LOADED)
    assert row == {
        "turn": base.record_hash,
        "ref": "q:spam",
        "question": "q:spam",
        "candidate": "",
        "identity": row["identity"],
        "kind": "noul",
        "labels": ["yes", "no"],
        "answer": {
            "kind": "noul",
            "labels": ["yes", "no"],
            "probabilities": {"yes": 0.9, "no": 0.1},
        },
        "marker": None,
        "top": "yes",
        "target": None,
        "latent_targets": [],
        "groups": [],
    }
    assert row["identity"].startswith("sha256:")


def test_a_marker_row_has_no_answer_and_no_top() -> None:
    refused = Unavailable(Cause(CauseCode.refused, ["q:spam"]))
    base = go(binds=bindings(ask=returns({"q:spam": refused, "t:body": "b"}))).record
    (row,) = answers([base], LOADED)
    assert (row["answer"], row["top"]) == (None, None)
    assert row["marker"]["cause"]["code"] == "refused"


def test_a_tie_for_the_top_has_no_top_label() -> None:
    base = go(
        binds=bindings(ask=returns({"q:spam": {"yes": 0.5, "no": 0.5}, "t:body": "b"}))
    ).record
    (row,) = answers([base], LOADED)
    assert row["top"] is None


def test_rows_follow_the_records_and_their_logs_in_order() -> None:
    first, second = (
        go().record,
        go(binds=bindings(ask=returns({"q:spam": NO, "t:body": "b"}))).record,
    )
    rows = answers([first, second, first], LOADED)
    assert [r["turn"] for r in rows] == [first.record_hash, second.record_hash, first.record_hash]
    assert [r["top"] for r in rows] == ["yes", "no", "yes"]


def test_a_per_candidate_question_has_a_row_per_candidate() -> None:
    rows = answers([grounded.decided()], grounded.LOADED)
    assert [(r["ref"], r["question"], r["candidate"]) for r in rows] == [
        ('q:ask/out["Ana"]', "q:ask/out", "Ana"),
        ('q:ask/out["Bo"]', "q:ask/out", "Bo"),
    ]


def test_an_extraction_row_has_no_labels_and_no_top() -> None:
    rows = answers([extraction_ran()], extraction.loaded())
    amount = next(r for r in rows if r["ref"] == "q:amount")
    assert (amount["kind"], amount["labels"], amount["top"]) == ("extraction", None, None)
    assert amount["answer"]["value"] == 120


def test_a_row_names_its_questions_targets() -> None:
    loaded = load(targeted())
    record = decide_sync(loaded, TARGET_STATE, {"default": target_binding}).record
    (row,) = answers([record], loaded)
    assert row["target"] == "tg:urgency"
    assert row["latent_targets"] == [{"role": "reflective", "target": "tg:distress"}]


def test_a_row_names_the_groups_holding_its_question_with_their_reasons() -> None:
    design = slice_design(
        groups={
            "spam": Group(["q:spam", "t:body"], ["exclusive", "convergent"]),
            "other": Group(["t:body"], ["consistent"]),
        }
    )
    base = go(design).record
    (row,) = answers([base], design)
    assert row["groups"] == [{"group": "spam", "reason": ["convergent", "exclusive"]}]


# ---------------------------------------------------------------------------------------------
# Gold


def labelled(root: Path, record: Any, labels: Mapping[str, object]) -> EvaluationStore:
    store = EvaluationStore(root)
    write_gold_answers("g", [record], {record.record_hash: labels}, results=store)
    return store


@pytest.mark.parametrize(("gold", "erred"), [("yes", False), ("no", True)])
def test_a_probability_answer_errs_when_its_top_label_is_not_gold(
    tmp_path: Path, gold: str, erred: bool
) -> None:
    base = go().record
    (row,) = answers(
        [base], LOADED, gold_set="g", results=labelled(tmp_path, base, {"q:spam": gold})
    )
    assert (row["gold"], row["erred"]) == (gold, erred)


def test_a_tie_errs(tmp_path: Path) -> None:
    base = go(
        binds=bindings(ask=returns({"q:spam": {"yes": 0.5, "no": 0.5}, "t:body": "b"}))
    ).record
    store = labelled(tmp_path, base, {"q:spam": "yes"})
    (row,) = answers([base], LOADED, gold_set="g", results=store)
    assert row["erred"] is True


def test_a_missing_gold_answer_is_a_row_with_nulls(tmp_path: Path) -> None:
    (row,) = answers([go().record], LOADED, gold_set="g", results=EvaluationStore(tmp_path))
    assert (row["gold"], row["erred"]) == (None, None)


def test_a_marker_with_a_gold_answer_has_erred_null(tmp_path: Path) -> None:
    refused = Unavailable(Cause(CauseCode.refused, ["q:spam"]))
    base = go(binds=bindings(ask=returns({"q:spam": refused, "t:body": "b"}))).record
    store = labelled(tmp_path, base, {"q:spam": "no"})
    (row,) = answers([base], LOADED, gold_set="g", results=store)
    assert (row["gold"], row["erred"]) == ("no", None)


def test_an_extraction_answer_errs_by_its_values_bytes(tmp_path: Path) -> None:
    base, loaded = extraction_ran(), extraction.loaded()
    labels = {"q:amount": {"value": 120}, "q:dates": {"value": None}}
    rows = answers([base], loaded, gold_set="g", results=labelled(tmp_path, base, labels))
    erred = {r["ref"]: r["erred"] for r in rows}
    assert erred == {"q:amount": False, "q:dates": True}


def test_a_per_candidate_gold_answer_is_found_by_its_candidate(tmp_path: Path) -> None:
    base = grounded.decided()
    labels = {'q:ask/out["Ana"]': "no", 'q:ask/out["Bo"]': "yes"}
    rows = answers([base], grounded.LOADED, gold_set="g", results=labelled(tmp_path, base, labels))
    assert [(r["candidate"], r["gold"]) for r in rows] == [("Ana", "no"), ("Bo", "yes")]


# ---------------------------------------------------------------------------------------------
# Refusals


def test_a_record_of_another_design_is_wrong_design() -> None:
    other = go(slice_design(name="changed")).record
    with pytest.raises(EvaluationRefusal) as raised:
        answers([go().record, other], LOADED)
    assert (raised.value.reason, raised.value.turn) == (R.wrong_design, other.record_hash)


def test_every_records_format_is_checked_before_any_design_hash() -> None:
    old = read_record((Path(__file__).parent / "format_9" / "record.json").read_bytes())
    other = go(slice_design(name="changed")).record
    for given in ([other, old], [old, other]):
        with pytest.raises(EvaluationRefusal) as raised:
            answers(given, LOADED)
        assert (raised.value.reason, raised.value.turn) == (R.wrong_design, old.record_hash)
        assert "records[" in str(raised.value)
        assert "format-7 record, before format 8" in str(raised.value)
    with pytest.raises(EvaluationRefusal) as raised:
        answers([old, b"not json"], LOADED)  # every record is read before any is checked
    assert raised.value.reason is R.invalid_record


def cancels(r: Any) -> Any:
    raise asyncio.CancelledError


def test_a_cancelled_records_answered_calls_are_rows() -> None:
    later = Source({"t:later": t.string}, needs=["d:route"], when=When(cancels, version="1"))
    loaded = slice_design(sources={"ask": ASK, "body": BODY, "look": LOOK, "later": later})
    record = record_of(caught(deciding(loaded)))
    assert record.log[-1]["item"] == "cause"
    (row,) = answers([record], loaded)
    assert (row["ref"], row["turn"], row["top"]) == ("q:spam", record.record_hash, "yes")


def test_a_broken_entry_gives_no_rows_and_no_refusal() -> None:
    broken = go(state={}).record
    assert escalation(broken) is not None
    assert answers([broken], LOADED) == []


def test_a_record_that_does_not_read_is_invalid_record() -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        answers([b"not json"], LOADED)
    assert raised.value.reason is R.invalid_record


def test_gold_arguments_of_another_form_are_a_type_error(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    for kwargs in (
        {"gold_set": "g"},
        {"results": store},
        {"gold_set": 5, "results": store},
        {"gold_set": "g", "results": "a store"},
    ):
        with pytest.raises(TypeError):
            answers([go().record], LOADED, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("records", ["abc", b"abc", {"a": 1}, 5])
def test_records_that_are_not_an_iterable_of_records_are_a_type_error(records: Any) -> None:
    with pytest.raises(TypeError):
        answers(records, LOADED)


def test_a_loaded_not_exactly_a_loaded_design_is_refused_first() -> None:
    with pytest.raises(EntryRefusal, match="exactly a LoadedDesign"):
        answers([b"x"], object())  # type: ignore[arg-type]


def test_the_rows_are_plain_data_and_share_nothing_with_the_record() -> None:
    base = go().record
    (row,) = answers([base], LOADED)
    row["answer"]["probabilities"]["yes"] = 0.0
    assert answers([base], LOADED)[0]["answer"]["probabilities"]["yes"] == 0.9
    assert dataclasses.is_dataclass(row) is False


def test_rows_share_no_target_link_or_group_reason() -> None:
    loaded = load(targeted())
    record = decide_sync(loaded, TARGET_STATE, {"default": target_binding}).record
    first, second = answers([record, record], loaded)
    first["latent_targets"].append("x")
    first["latent_targets"][0]["role"] = "x"
    assert second["latent_targets"] == [{"role": "reflective", "target": "tg:distress"}]
    design = slice_design(groups={"spam": Group(["q:spam"], ["convergent"])})
    base = go(design).record
    first, second = answers([base, base], design)
    first["groups"][0]["reason"].append("x")
    assert second["groups"] == [{"group": "spam", "reason": ["convergent"]}]


def test_a_candidate_valued_row_has_the_labels_as_asked() -> None:
    loaded = load(label_sets.design())
    rows = answers([label_sets.run(["ann", "bo"])], loaded)
    row = next(r for r in rows if r["ref"] == "q:sender")
    assert row["labels"] == ["none", "several", "ann", "bo"]
