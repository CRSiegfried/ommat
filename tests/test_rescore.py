"""The rescoring job: each recorded turn decided again under each of several designs, building no
record. Every row's entries equal the entries at the named refs a full replay under that design
writes, except that each asked source is checked against the record under the recorded design,
so a design changing a source's payload, oracle key or a provided t: type is not-determined
where replay under it decides; a design that asks a question the record does not answer is
not-determined; records are checked as replay checks them, against the design they were decided
under."""

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


ACT = ["d:act"]


def replayed(
    record: Record, loaded: LoadedDesign, refs: tuple[str, ...] = ("d:act",)
) -> tuple[dict[str, Any] | str, list[str]]:
    """What a full replay says: the entries it writes at ``refs``, or not-determined with its
    stale refs."""
    rerun, findings = replay_sync(record, loaded)
    assert rerun is not None
    stale = sorted(f["ref"] for f in findings if f["kind"] == "stale-question")
    if stale:
        return "not-determined", stale
    held = rerun.entries
    return {
        ref: {k: held[ref][k] for k in ("value", "marker") if k in held[ref]}
        for ref in sorted(refs)
        if ref in held
    }, []


def act_is(value: str) -> dict[str, Any]:
    """A row's entries holding ``value`` at ``d:act``."""
    return {"d:act": {"value": value}}


SPLITS = [(0.9, 0.9), (0.9, 0.2), (0.55, 0.9), (0.5, 0.5), (0.2, 0.9), (0.62, 0.61)]
BARS = [0.0, 0.3, 0.5, 0.55, 0.6, 0.7, 0.95]


def test_every_rescored_entry_is_replays() -> None:
    recorded = gated(0.6)  # a turn whose refund falls short asks no amount
    records = [turn(recorded, *split) for split in SPLITS]
    designs = [gated(bar, margin) for bar in BARS for margin in (0.0, 0.2)]
    rows = rescore_sync(records, recorded, designs, refs=ACT)
    assert len(rows) == len(designs) * len(records)
    determined = 0
    for row, (design, record) in zip(rows, [(d, r) for d in designs for r in records], strict=True):
        assert row["design"] == design.design_hash
        assert row["record"] == record.record_hash
        assert (row["entries"], row["stale"]) == replayed(record, design)
        determined += row["entries"] != "not-determined"
    assert 0 < determined < len(rows)  # both kinds of row are exercised


def test_a_design_asking_what_the_record_never_asked_is_not_determined() -> None:
    # Decided at bar 0.6, 0.55 for refund is no pick, so the amount question was never asked.
    recorded = gated(0.6)
    record = turn(recorded, 0.55, 0.9)
    (same, opened) = rescore_sync([record], recorded, [recorded, gated(0.5)], refs=ACT)
    assert same["entries"] == act_is("ask")
    assert same["stale"] == []
    # At 0.5 refund is picked, the gate opens, and the record holds no answer to the amount.
    assert opened["entries"] == "not-determined"
    assert opened["stale"] == ["q:amount"]


def test_a_rescored_parameter_changes_the_entry_alone() -> None:
    recorded = gated(0.5, stated=0.5)
    record = turn(recorded, 0.9, 0.6)
    rows = rescore_sync([record], recorded, [gated(0.5, stated=s) for s in (0.5, 0.7)], refs=ACT)
    assert [row["entries"] for row in rows] == [act_is("refund"), act_is("ask")]


def test_a_record_of_another_design_is_refused_before_any_run() -> None:
    other = gated(0.7)
    record = turn(other, 0.9, 0.9)
    with pytest.raises(EvaluationRefusal) as caught:
        rescore_sync([record], gated(0.5), [gated(0.5)], refs=ACT)
    assert caught.value.reason is RefusalReason.wrong_design


def test_a_stored_record_document_reads_as_a_record() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    (row,) = rescore_sync([record.document], recorded, [recorded], refs=ACT)
    assert row["record"] == record.record_hash
    assert row["entries"] == act_is("refund")


def test_designs_must_be_loaded_designs() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    with pytest.raises(EntryRefusal):
        rescore_sync([record], recorded, [object()], refs=ACT)  # type: ignore[list-item]
    with pytest.raises(TypeError):
        rescore_sync([record], recorded, "designs", refs=ACT)  # type: ignore[arg-type]


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
    rows = rescore_sync(records, recorded, designs, refs=ACT)
    for row, design in zip(rows, designs, strict=True):
        assert (row["entries"], row["stale"]) == replayed(records[0], design)
    assert len({json.dumps(r["entries"]) for r in rows}) > 1  # the grid moves the entry


def whole(r: Any) -> float:
    return 1.0


def test_a_rescored_entry_reads_as_the_record_does() -> None:
    loaded = load(
        Design(
            name="whole",
            state=STATE_T,
            sources={},
            rules={"act": Rule("d:act", t.number(), whole, version="1")},
            parameters={},
        )
    )
    record = decide_sync(loaded, STATE, {}).record
    (row,) = rescore_sync([record], loaded, [loaded], refs=ACT)
    entry = record.entries["d:act"]
    assert row["entries"] == {"d:act": {"value": entry["value"]}}
    # the integral float as the int 1
    assert repr(row["entries"]) == repr({"d:act": {"value": entry["value"]}})
    assert type(row["entries"]["d:act"]["value"]) is int


# ---------------------------------------------------------------------------------------------
# Where rescore and replay part: the source check runs under the recorded design


def wording(bar: float, payload: Any = None, oracle: str | None = None) -> LoadedDesign:
    """A design whose intent question's wording takes the bar, so a changed bar stales it; its
    source's payload and oracle key as given."""

    def intent(r: Any) -> object:
        return pick_with_margin(r, "q:intent", 0.5, 0.0)

    def act(r: Any) -> object:
        return "a" if r.read("d:intent") == "refund" else "b"

    extra: dict[str, Any] = {} if oracle is None else {"oracle": oracle}
    return load(
        Design(
            name="worded",
            state=STATE_T,
            sources={
                "ask": Source(
                    {
                        "q:intent": Question(
                            "choice",
                            Text(
                                "Intent at {bar}?", [{"name": "bar", "origin": {"param": "w/bar"}}]
                            ),
                            labels=INTENTS,
                        )
                    },
                    needs=["t:state"],
                    payload=payload if payload is not None else {"m": 1},
                    **extra,
                )
            },
            rules={
                "intent": Rule(
                    "d:intent", t.label(*INTENTS), intent, version="1", reads=["q:intent"]
                ),
                "act": Rule("d:act", t.label("a", "b"), act, version="1", reads=["d:intent"]),
            },
            parameters={"w/bar": Parameter(t.number(0, 1), bar)},
        )
    )


def worded_turn(loaded: LoadedDesign) -> Record:
    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({"q:intent": {"refund": 0.8, "other": 0.2}})

    return decide_sync(loaded, STATE, {"default": answer}).record


@pytest.mark.parametrize(
    "changed",
    [wording(0.3, payload={"m": 2}), wording(0.3, oracle="x")],
    ids=["payload", "oracle"],
)
def test_a_changed_payload_or_oracle_is_not_determined_where_replay_decides(
    changed: LoadedDesign,
) -> None:
    recorded = wording(0.3)
    record = worded_turn(recorded)
    (row,) = rescore_sync([record], recorded, [changed], refs=ACT)
    assert (row["entries"], row["stale"]) == ("not-determined", ["q:intent"])
    assert replayed(record, changed) == (act_is("a"), [])  # replay checks under ``changed``
    (same,) = rescore_sync([record], recorded, [recorded], refs=ACT)
    assert (same["entries"], same["stale"]) == (act_is("a"), [])


def test_a_wording_slot_leaf_stales_the_question_as_replay_does() -> None:
    recorded = wording(0.3)
    record = worded_turn(recorded)
    (row,) = rescore_sync([record], recorded, [wording(0.4)], refs=ACT)
    assert (row["entries"], row["stale"]) == ("not-determined", ["q:intent"])
    assert replayed(record, wording(0.4)) == ("not-determined", ["q:intent"])


def test_a_state_that_fails_the_designs_type_gives_the_broken_entry() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    retyped = load(
        Design(
            name="gated",
            state=t.record(text=t.number()),
            sources={},
            rules={"act": Rule("d:act", t.label("a"), whole_a, version="1")},
            parameters={},
        )
    )
    (row,) = rescore_sync([record], recorded, [retyped], refs=ACT)
    assert row["entries"] == {}  # a broken entry writes t:state alone
    assert row["stale"] == []
    assert (row["entries"], row["stale"]) == replayed(record, retyped)


def whole_a(r: Any) -> str:
    return "a"


def test_a_cut_record_rescores_as_it_replays() -> None:
    from ommat import RunSettings

    recorded = gated(0.5)
    whole = turn(recorded, 0.9, 0.9)
    assert whole.entries["d:act"]["value"] == "refund"

    async def answer(batch: Any, *, seconds: float) -> object:
        given = {
            "q:intent": {"refund": 0.9, "other": 0.1},
            "q:amount": {"yes": 0.9, "no": 0.1},
        }
        return CallResult({ref: given[ref] for call in batch for ref in call.questions})

    # The amount question is asked in round 2, past the bound: the turn is cut after round 1.
    cut = decide_sync(recorded, STATE, {"default": answer}, RunSettings(round_bound=1)).record
    assert cut.entries["d:act"]["value"] == "ask"
    assert "q:amount" not in cut.entries or "marker" in cut.entries["q:amount"]
    designs = [recorded, gated(0.95), gated(0.5, stated=0.95)]
    rows = rescore_sync([cut, whole], recorded, designs, refs=ACT)
    for row, (design, record) in zip(
        rows, [(d, r) for d in designs for r in (cut, whole)], strict=True
    ):
        assert (row["entries"], row["stale"]) == replayed(record, design)
    assert rows[0]["entries"] == act_is("ask")  # the cut holds under its own design
    assert rows[1]["entries"] == act_is("refund")


# ---------------------------------------------------------------------------------------------
# The refusals, each check over every record before the next


def test_the_records_are_refused_in_replays_order() -> None:
    from test_decide import LOADED
    from test_use_answers import broken, format_7, unfinished

    def refusal(records: list[Any]) -> EvaluationRefusal:
        with pytest.raises(EvaluationRefusal) as caught:
            rescore_sync(records, LOADED, [LOADED], refs=ACT)
        return caught.value

    exc = refusal([{"not": "a record"}])
    assert exc.reason is RefusalReason.invalid_record
    old, half, marker = format_7(), unfinished(), broken()
    exc = refusal([half, old])  # the format check runs over every record first
    assert exc.reason is RefusalReason.wrong_design
    assert exc.turn == old.record_hash
    assert "before format 8" in str(exc)
    exc = refusal([marker, half])
    assert exc.reason is RefusalReason.unfinished
    assert exc.turn == half.record_hash
    exc = refusal([marker])
    assert exc.reason is RefusalReason.nothing_to_replay
    assert exc.turn == marker.record_hash
    other = turn(gated(0.5), 0.9, 0.9)
    exc = refusal([other])
    assert exc.reason is RefusalReason.wrong_design
    assert exc.turn == other.record_hash
    # Invalid records are refused as they are read, before the checks on the records read.
    exc = refusal([old, {"not": "a record"}])
    assert exc.reason is RefusalReason.invalid_record


def test_a_record_its_observer_cut_short_is_rescored_as_the_whole_turn() -> None:
    """No ``cause`` item, so not ``unfinished``: a cut that kept every answer is rescored as if
    the turn were whole, one that lost answers gives ``not-determined`` naming them."""
    from test_decide import LOADED
    from test_use_answers import cut

    after, before = cut(9), cut()
    rows = rescore_sync([after, before], LOADED, [LOADED], refs=ACT)
    assert [(r["record"], r["entries"], r["stale"]) for r in rows] == [
        (after.record_hash, act_is("archive"), []),
        (before.record_hash, "not-determined", ["t:body", "t:size"]),
    ]


def test_records_must_be_a_sequence() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    with pytest.raises(TypeError):
        rescore_sync((r for r in [record]), recorded, [recorded], refs=ACT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rescore_sync("records", recorded, [recorded], refs=ACT)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rescore_sync([record], recorded, (d for d in [recorded]), refs=ACT)  # type: ignore[arg-type]
    assert rescore_sync([], recorded, [recorded], refs=ACT) == []
    assert rescore_sync([record], recorded, [], refs=ACT) == []


def test_a_record_whose_settings_decide_refuses_is_refused_before_any_run() -> None:
    import test_replay as tp
    from ommat import EntryRefusal
    from ommat.evaluation import rescore_grid_sync

    good = tp.go().record
    bad = tp.edited(good, lambda d: d["log"][1]["settings"].update({"timeout": 0}))
    for designs in ([tp.LOADED], []):
        with pytest.raises(EntryRefusal):
            rescore_sync([good, bad], tp.LOADED, designs, refs=ACT)
    with pytest.raises(EntryRefusal):
        rescore_grid_sync([good, bad], tp.LOADED, tp.LOADED, [], refs=ACT)


# ---------------------------------------------------------------------------------------------
# The refs: each a declared d: ref an ungrounded rule of every design provides


def test_each_named_ref_is_rescored_as_replay_writes_it_in_code_point_order() -> None:
    recorded = gated(0.6)
    records = [turn(recorded, *split) for split in SPLITS]
    designs = [gated(bar) for bar in BARS]
    named = ("d:intent", "d:act")
    rows = rescore_sync(records, recorded, designs, refs=list(named))
    for row, (design, record) in zip(rows, [(d, r) for d in designs for r in records], strict=True):
        assert (row["entries"], row["stale"]) == replayed(record, design, named)
        if row["entries"] != "not-determined":
            assert list(row["entries"]) == ["d:act", "d:intent"]
    # Naming a ref twice names it once.
    again = rescore_sync(records, recorded, designs, refs=["d:act", "d:act"])
    assert again == rescore_sync(records, recorded, designs, refs=ACT)


def test_a_row_has_no_decision_member() -> None:
    recorded = gated(0.5)
    (row,) = rescore_sync([turn(recorded, 0.9, 0.9)], recorded, [recorded], refs=["d:intent"])
    assert set(row) == {"design", "record", "entries", "stale"}
    assert row["entries"] == {"d:intent": {"value": "refund"}}


def test_a_broken_state_writes_no_named_entry() -> None:
    """A state that does not type writes ``t:state`` alone, so no named ref has an entry."""
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    retyped = load(
        Design(
            name="gated",
            state=t.record(text=t.number()),
            sources={},
            rules={
                "act": Rule("d:act", t.label("a"), whole_a, version="1"),
                "other": Rule("d:other", t.label("a"), whole_a, version="1"),
            },
            parameters={},
        )
    )
    (row,) = rescore_sync([record], recorded, [retyped], refs=["d:other"])
    assert row["entries"] == {}
    (both,) = rescore_sync([record], recorded, [retyped], refs=["d:other", "d:act"])
    assert both["entries"] == {}
    assert (both["entries"], both["stale"]) == replayed(record, retyped, ("d:act", "d:other"))


def test_refs_must_be_a_non_empty_list_or_tuple() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    assert rescore_sync([record], recorded, [recorded], refs=("d:act",)) == rescore_sync(
        [record], recorded, [recorded], refs=ACT
    )
    for given in ("d:act", b"d:act", None, (r for r in ACT), {"d:act": 1}):
        with pytest.raises(TypeError, match="refs is a"):
            rescore_sync([record], recorded, [recorded], refs=given)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refs is empty"):
        rescore_sync([record], recorded, [recorded], refs=[])


@pytest.mark.parametrize(
    "ref",
    ["d:none", "q:intent", "t:state", "d:act.x", "d:act[a]", "act", "", 3],
)
def test_a_ref_no_rule_provides_whole_is_refused_as_dead_questions_refuses_it(ref: object) -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    with pytest.raises(ValueError, match=r"^designs\[0\]: refs ") as caught:
        rescore_sync([record], recorded, [recorded], refs=["d:act", ref])  # type: ignore[list-item]
    assert type(caught.value) is ValueError
    assert "d:act," not in str(caught.value)  # the refs it provides are not named


def test_a_grounded_rules_declared_ref_is_refused() -> None:
    import test_grounded_rules as grounded

    record = grounded.decided()
    assert grounded.LEVEL in grounded.LOADED._providers  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="d:level"):
        rescore_sync([record], grounded.LOADED, [grounded.LOADED], refs=[grounded.LEVEL])
    (row,) = rescore_sync([record], grounded.LOADED, [grounded.LOADED], refs=[grounded.SET])
    assert row["entries"] == {grounded.SET: {"value": record.entries[grounded.SET]["value"]}}


def test_every_design_must_provide_each_ref_and_the_refs_are_checked_before_the_records() -> None:
    recorded = gated(0.5)
    record = turn(recorded, 0.9, 0.9)
    lacking = load(
        Design(
            name="lacking",
            state=STATE_T,
            sources={},
            rules={"act": Rule("d:act", t.label("a"), whole_a, version="1")},
            parameters={},
        )
    )
    with pytest.raises(ValueError, match=r"^designs\[1\]: .*d:intent"):
        rescore_sync([record], recorded, [recorded, lacking], refs=["d:intent"])
    with pytest.raises(ValueError, match="d:intent"):
        rescore_sync([{"not": "a record"}], recorded, [lacking], refs=["d:intent"])
    # With no designs the refs are checked against the recorded design.
    assert rescore_sync([record], recorded, [], refs=["d:intent"]) == []
    with pytest.raises(ValueError, match=r"^recorded: "):
        rescore_sync([record], recorded, [], refs=["d:none"])
