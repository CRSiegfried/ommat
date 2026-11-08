"""The rescoring job: each recorded turn decided again under each of several designs, building
no record. Every decision equals the policy's entry a full replay under that design writes; a
design that asks a question the record does not answer is not-determined; records are checked
as replay checks them, against the design they were decided under."""

from typing import Any

import pytest

from ommat import (
    CallResult,
    Design,
    EntryRefusal,
    LoadedDesign,
    Parameter,
    Question,
    Record,
    Rule,
    Source,
    SourceRequirement,
    Text,
    Undetermined,
    decide_sync,
    load,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import decode
from ommat._ledger import decision_entry
from ommat.evaluation import EvaluationRefusal, RefusalReason, rescore_sync
from ommat.helpers import mass, pick_with_margin

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")

STATE_T = t.record(text=t.string)
STATE = {"text": "I want my money back"}
INTENTS = ("refund", "other")


def intent(r: Any) -> object:
    return pick_with_margin(r, "q:intent", r.param("g/bar"), r.param("g/margin"))


def act(r: Any) -> object:
    chosen = r.read("d:intent")
    if isinstance(chosen, Undetermined):
        return "ask"
    if chosen == "other":
        return "pass"
    stated = mass(r, "q:amount", ["yes"])
    if isinstance(stated, Undetermined):
        return "ask"
    return "refund" if stated >= r.param("g/stated") else "ask"


def gated(bar: float, margin: float = 0.0, stated: float = 0.5) -> LoadedDesign:
    """A refund question asked only when the intent is refund, the intent picked at ``bar``."""
    return load(
        Design(
            name="gated",
            state=STATE_T,
            sources={
                "ask": Source(
                    {"q:intent": Question("choice", Text("Intent?"), labels=INTENTS)},
                    needs=["t:state"],
                ),
                "more": Source(
                    {"q:amount": Question("noul", Text("Does it state an amount?"))},
                    needs=["t:state"],
                    requirement=SourceRequirement("d:intent", ["refund"]),
                ),
            },
            rules={
                "intent": Rule(
                    "d:intent",
                    t.label(*INTENTS),
                    intent,
                    version="1",
                    reads=["q:intent"],
                    params=["g/bar", "g/margin"],
                ),
                "act": Rule(
                    "d:act",
                    t.label("refund", "ask", "pass"),
                    act,
                    version="1",
                    reads=["d:intent", "q:amount"],
                    params=["g/stated"],
                ),
            },
            parameters={
                "g/bar": Parameter(t.number(0, 1), bar),
                "g/margin": Parameter(t.number(0, 1), margin),
                "g/stated": Parameter(t.number(0, 1), stated),
            },
            policy="d:act",
        )
    )


def turn(loaded: LoadedDesign, refund: float, amount: float) -> Record:
    async def answer(batch: Any, *, seconds: float) -> object:
        given = {
            "q:intent": {"refund": refund, "other": round(1 - refund, 6)},
            "q:amount": {"yes": amount, "no": round(1 - amount, 6)},
        }
        return CallResult({ref: given[ref] for call in batch for ref in call.questions})

    return decide_sync(loaded, STATE, {"default": answer}).record


def replayed(record: Record, loaded: LoadedDesign) -> tuple[dict[str, Any] | str, list[str]]:
    """What a full replay says: the policy's entry, or not-determined with its stale refs."""
    rerun, findings = replay_sync(record, loaded)
    assert rerun is not None
    stale = sorted(f["ref"] for f in findings if f["kind"] == "stale-question")
    if stale:
        return "not-determined", stale
    entry = decode(_policy_bytes(rerun))
    return entry, []


def _policy_bytes(record: Record) -> bytes:
    from ommat._canonical import encode

    entry = decision_entry(record.log)
    assert entry is not None
    return encode({k: entry[k] for k in ("value", "marker") if k in entry})


SPLITS = [(0.9, 0.9), (0.9, 0.2), (0.55, 0.9), (0.5, 0.5), (0.2, 0.9), (0.62, 0.61)]
BARS = [0.0, 0.3, 0.5, 0.55, 0.6, 0.7, 0.95]


def test_every_rescored_decision_is_replays() -> None:
    recorded = gated(0.6)  # a turn whose refund falls short asks no amount
    records = [turn(recorded, *split) for split in SPLITS]
    designs = [gated(bar, margin) for bar in BARS for margin in (0.0, 0.2)]
    rows = rescore_sync(records, recorded, designs)
    assert len(rows) == len(designs) * len(records)
    determined = 0
    for row, (design, record) in zip(rows, [(d, r) for d in designs for r in records], strict=True):
        assert row["design"] == design.design_hash
        assert row["record"] == record.record_hash
        assert (row["decision"], row["stale"]) == replayed(record, design)
        determined += row["decision"] != "not-determined"
    assert 0 < determined < len(rows)  # both kinds of row are exercised


def test_a_design_asking_what_the_record_never_asked_is_not_determined() -> None:
    # Decided at bar 0.6, 0.55 for refund is no pick, so the amount question was never asked.
    recorded = gated(0.6)
    record = turn(recorded, 0.55, 0.9)
    (same, opened) = rescore_sync([record], recorded, [recorded, gated(0.5)])
    assert same["decision"] == {"value": "ask"}
    assert same["stale"] == []
    # At 0.5 refund is picked, the gate opens, and the record holds no answer to the amount.
    assert opened["decision"] == "not-determined"
    assert opened["stale"] == ["q:amount"]


def test_a_rescored_parameter_changes_the_decision_alone() -> None:
    recorded = gated(0.5, stated=0.5)
    record = turn(recorded, 0.9, 0.6)
    rows = rescore_sync([record], recorded, [gated(0.5, stated=s) for s in (0.5, 0.7)])
    assert [row["decision"] for row in rows] == [{"value": "refund"}, {"value": "ask"}]


def test_a_record_of_another_design_is_refused_before_any_run() -> None:
    other = gated(0.7)
    record = turn(other, 0.9, 0.9)
    with pytest.raises(EvaluationRefusal) as caught:
        rescore_sync([record], gated(0.5), [gated(0.5)])
    assert caught.value.reason is RefusalReason.wrong_design


def test_a_stored_record_document_reads_as_a_record() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    (row,) = rescore_sync([record.document], recorded, [recorded])
    assert row["record"] == record.record_hash
    assert row["decision"] == {"value": "refund"}


def test_designs_must_be_loaded_designs() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    with pytest.raises(EntryRefusal):
        rescore_sync([record], recorded, [object()])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        rescore_sync([record], recorded, "designs")  # type: ignore[arg-type]


def test_tables_and_helpers_rescore_as_they_replay() -> None:
    import json

    import test_switching_values as sv

    recorded = sv.MOOD
    records = [sv.mood_record()]
    document = json.loads(recorded.document)
    designs: list[LoadedDesign] = []
    for bar in (0.2, 0.5, 0.71, 0.9):
        for th in (0.3, 0.6, 0.8):
            edited = json.loads(json.dumps(document))
            edited["parameters"]["mood/bar"]["value"] = bar
            edited["parameters"]["mood/th"]["value"] = {"calm": th, "wild": th}
            designs.append(load(edited, functions=recorded._callables))  # pyright: ignore[reportPrivateUsage]
    rows = rescore_sync(records, recorded, designs)
    for row, design in zip(rows, designs, strict=True):
        assert (row["decision"], row["stale"]) == replayed(records[0], design)
    assert len({json.dumps(r["decision"]) for r in rows}) > 1  # the grid moves the decision
