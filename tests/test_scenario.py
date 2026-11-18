# pyright: reportPrivateUsage=false
"""``scenario``: ``intervene`` with the stale questions asked live. Only stale sources reach
the caller's binding, each whole; the rest are pinned from the base; a live answer equal to
the base's stops the cascade; a mixed batch's warnings name the base and its pinned refs; a
live failure fails the whole call; and the returned record replays clean under the design it
ran under."""

import asyncio
from typing import Any

import pytest

import test_grounded_rules as grounded
from designs import minimal
from ommat import (
    CallResult,
    Cause,
    CauseCode,
    EntryRefusal,
    Failed,
    LoadedDesign,
    Record,
    Rule,
    RunSettings,
    Source,
    decide_sync,
    load,
    replay_sync,
)
from ommat import types as t
from ommat._ledger import finished
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    intervene_sync,
    scenario,
    scenario_sync,
)
from scripted import Scripted, returns, sleeping
from test_decide import ASK, BODY, LOADED, LOOK, SPAM, Slow, bindings, go, rounds, slice_design

R = RefusalReason
NO = {"yes": 0.1, "no": 0.9}
YES = {"yes": 0.9, "no": 0.1}


def warned(record: Record, sid: str) -> list[str]:
    """The warnings of the call item listing ``sid``."""
    log = record.log
    index = next(n for n, i in enumerate(log) if i["item"] == "call" and sid in i["sources"])
    return record.unhashed["calls"][str(index)].get("warnings", [])


def unhashed_call(record: Record, sid: str) -> dict[str, Any]:
    """The unhashed members of the call item listing ``sid``."""
    log = record.log
    index = next(n for n, i in enumerate(log) if i["item"] == "call" and sid in i["sources"])
    return record.unhashed["calls"][str(index)]


def call_of(record: Record, sid: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "call" and sid in i["sources"])


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def live(**replies: object) -> dict[str, Scripted]:
    binding = Scripted(replies)
    return {"default": binding, "mail": binding}


# ---------------------------------------------------------------------------------------------
# What reaches the live binding


def test_only_a_stale_source_reaches_the_live_binding_whole() -> None:
    base = go().record
    given = live(ask=returns({"q:spam": NO}, provider="p"))
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], given)
    (batch,) = given["mail"].batches
    assert [s.id for s in batch] == ["ask"]
    assert list(batch[0].questions) == ["q:spam"]
    assert out["stale"] == ["q:spam"]
    assert out["pinned"] == ["t:body", "t:size"]
    assert entry(out["record"], "q:spam")["value"]["probabilities"] == NO
    assert entry(out["record"], "t:body") == entry(base, "t:body")


def test_nothing_stale_calls_nothing_and_matches_intervene() -> None:
    base = go().record
    given = live()
    out = scenario_sync(base, LOADED, [{"ref": "q:spam", "value": NO}], given)
    assert given["mail"].batches == []
    offline = intervene_sync(base, LOADED, [{"ref": "q:spam", "value": NO}])
    assert out["findings"] == offline["findings"]
    assert out["stale"] == offline["stale"] == []
    assert out["pinned"] == ["q:spam", "t:body", "t:size"]


def test_a_changed_need_asks_every_source_needing_it_live() -> None:
    base = go().record
    given = live(
        ask=returns({"q:spam": NO, "t:body": "new body"}),
        look=returns({"t:size": 4}),
    )
    out = scenario_sync(base, LOADED, [{"path": "t:state.subject", "value": "bye"}], given)
    assert out["stale"] == ["q:spam", "t:body", "t:size"]
    assert out["pinned"] == []
    assert entry(out["record"], "t:body")["value"] == "new body"
    assert [[s.id for s in b] for b in given["mail"].batches] == [["ask", "body"], ["look"]]


# ---------------------------------------------------------------------------------------------
# Early cutoff


def big(r: Any) -> bool:
    return bool(r.read("t:a") > 5)


def say(r: Any) -> str:
    return str(r.read("t:b"))


CHAIN = load(
    minimal(
        state=t.record(n=t.integer()),
        sources={
            "a": Source({"t:a": t.integer()}, needs=["t:state"]),
            "b": Source({"t:b": t.string}, needs=["d:big"]),
        },
        rules={
            "big": Rule("d:big", t.boolean, big, version="1", reads=["t:a"]),
            "say": Rule("d:say", t.string, say, version="1", reads=["t:b"]),
        },
        policy="d:say",
    )
)


def chained(a: int) -> Record:
    replies = Scripted({"a": returns({"t:a": a}), "b": returns({"t:b": "hi"})})
    return decide_sync(CHAIN, {"n": 1}, {"default": replies}).record


def test_a_live_answer_that_leaves_the_rule_unchanged_stops_the_cascade() -> None:
    base = chained(7)
    replies = Scripted({"a": returns({"t:a": 9}), "b": returns({"t:b": "other"})})
    out = scenario_sync(base, CHAIN, [{"ask": "a"}], {"default": replies})
    assert [[s.id for s in b] for b in replies.batches] == [["a"]]
    assert out["stale"] == ["t:a"]
    assert out["pinned"] == ["t:b"]


def test_a_live_answer_that_changes_the_rule_asks_the_next_source_live() -> None:
    base = chained(7)
    replies = Scripted({"a": returns({"t:a": 3}), "b": returns({"t:b": "other"})})
    out = scenario_sync(base, CHAIN, [{"ask": "a"}], {"default": replies})
    assert [[s.id for s in b] for b in replies.batches] == [["a"], ["b"]]
    assert out["stale"] == ["t:a", "t:b"]
    assert out["pinned"] == []
    found = {f["ref"]: f["clause"] for f in out["findings"] if f["kind"] == "stale-question"}
    assert found == {"t:a": "oracle", "t:b": "needs"}


# ---------------------------------------------------------------------------------------------
# Provenance and failure


def test_a_mixed_batch_keeps_the_live_provenance_and_names_the_base_and_its_pins() -> None:
    base = go().record
    out = scenario_sync(
        base,
        LOADED,
        [{"ask": "ask"}],
        live(ask=returns({"q:spam": NO}, provider="p", warnings=("live line",))),
    )
    call = call_of(out["record"], "ask")
    assert call["provider"] == "p"
    assert warned(out["record"], "ask") == [
        f"answered from record {base.record_hash}",
        "'t:body': pinned",
        "live line",
    ]


def test_a_wholly_pinned_batch_has_null_provenance() -> None:
    base = go(binds=bindings(look=returns({"t:size": 4}, provider="seen"))).record
    assert call_of(base, "look")["provider"] == "seen"
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], live(ask=returns({"q:spam": NO})))
    assert call_of(out["record"], "look")["provider"] is None
    assert warned(out["record"], "look") == []


def test_a_live_failure_fails_the_whole_call_pinned_batchmates_included() -> None:
    base = go().record
    failed = Failed(Cause(CauseCode.raised, ["q:spam"]))
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], live(ask=failed))
    call = call_of(out["record"], "ask")
    assert call["outcome"]["failed"]["code"] == "raised"
    assert entry(out["record"], "t:body")["marker"]["cause"]["code"] == "raised"
    assert out["pinned"] == ["t:size"]  # the failed call gave back no recorded answer


def test_a_live_raise_fails_the_whole_call() -> None:
    async def boom(batch: Any, *, seconds: float) -> object:
        raise RuntimeError("down")

    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=boom))
    assert call_of(out["record"], "ask")["outcome"]["failed"]["code"] == "raised"


def test_a_live_value_for_a_pinned_batchmate_is_unexpected_ref() -> None:
    base = go().record
    answer = returns({"q:spam": NO, "t:body": "sneaked"})
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], live(ask=answer))
    call = call_of(out["record"], "ask")
    assert call["outcome"]["failed"]["code"] == "unexpected-ref"


PAIR = load(
    minimal(
        state=t.record(n=t.integer()),
        sources={"s": Source({"t:x": t.string, "t:y": t.string}, needs=["t:state"])},
    )
)


def test_a_pinned_ref_of_a_live_source_keeps_its_set_value() -> None:
    replies = Scripted({"s": returns({"t:x": "a", "t:y": "b"})})
    base = decide_sync(PAIR, {"n": 1}, {"default": replies}).record
    given = Scripted({"s": returns({"t:x": "live", "t:y": "live"})})
    out = scenario_sync(
        base,
        PAIR,
        [{"path": "t:state.n", "value": 2}, {"ref": "t:x", "value": "set"}],
        {"default": given},
    )
    assert [[s.id for s in b] for b in given.batches] == [["s"]]
    assert entry(out["record"], "t:x")["value"] == "set"
    assert entry(out["record"], "t:y")["value"] == "live"
    assert out["pinned"] == ["t:x"]
    assert out["stale"] == ["t:y"]
    assert warned(out["record"], "s") == [
        f"answered from record {base.record_hash}",
        "'t:x': pinned",
    ]


# ---------------------------------------------------------------------------------------------
# The record, the refusals and the twin


def test_the_returned_record_replays_clean_under_the_design_it_ran_under() -> None:
    base = go().record
    rerun = slice_design(name="changed")
    out = scenario_sync(
        base,
        LOADED,
        [{"path": "t:state.subject", "value": "bye"}],
        live(ask=returns({"q:spam": NO, "t:body": "b"}), look=returns({"t:size": 2})),
        rerun_design=rerun,
    )
    assert out["record"].design_hash == rerun.design_hash
    _, found = replay_sync(out["record"], rerun)
    assert [f for f in found if f["kind"] != "helpers-mismatch"] == []


def test_bindings_are_checked_as_decide_checks_them_before_the_record_is_read() -> None:
    with pytest.raises(EntryRefusal, match="bindings is a"):
        scenario_sync(b"x", LOADED, [{"ask": "ask"}], [])  # type: ignore[arg-type]
    with pytest.raises(EntryRefusal, match="no binding is given for the oracle"):
        scenario_sync(b"x", LOADED, [{"ask": "ask"}], {"default": Scripted({})})


def test_a_scenario_refuses_as_intervene_does() -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        scenario_sync(go().record, LOADED, [], live())
    assert raised.value.reason is R.no_edit
    with pytest.raises(EntryRefusal, match="rerun_design is a"):
        scenario_sync(go().record, LOADED, [], live(), rerun_design=object())  # type: ignore[arg-type]


def test_scenario_is_awaitable_and_its_twin_refuses_inside_a_loop() -> None:
    base = go().record

    async def main() -> dict[str, Any]:
        with pytest.raises(RuntimeError, match="scenario_sync was called inside a running"):
            scenario_sync(base, LOADED, [{"ask": "ask"}], live())
        return await scenario(base, LOADED, [{"ask": "ask"}], live(ask=returns({"q:spam": NO})))

    out = asyncio.run(main())
    assert out["stale"] == ["q:spam"]


def test_a_live_binding_is_given_the_calls_seconds() -> None:
    async def slow(batch: Any, *, seconds: float) -> object:
        assert seconds > 0
        return CallResult({"q:spam": NO})

    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=slow))
    assert finished(out["record"].log)


def test_a_bindings_key_the_design_does_not_declare_is_refused() -> None:
    loaded: LoadedDesign = CHAIN
    assert all(s.oracle is None for s in loaded._design.sources.values())
    with pytest.raises(EntryRefusal, match="not an oracle name the design declares"):
        scenario_sync(
            chained(7), CHAIN, [{"ask": "a"}], {"default": Scripted({}), "x": Scripted({})}
        )


# ---------------------------------------------------------------------------------------------
# Round 1 of review: the clock, the pins, what a live binding may return


def test_a_base_cut_by_its_deadline_runs_past_the_cut_live() -> None:
    base = go(binds=bindings(), settings=RunSettings(deadline=0.1), codec=Slow(0.2)).record
    assert rounds(base)[-1]["deadline"] is True
    given = live(ask=returns(SPAM), look=returns({"t:size": 4}))
    out = scenario_sync(base, LOADED, [], given, rerun_design=LOADED)
    assert all(r["deadline"] is False for r in rounds(out["record"]))
    assert out["stale"] == ["q:spam", "t:body", "t:size"]
    assert finished(out["record"].log)


def test_a_live_scenario_enforces_the_bases_deadline() -> None:
    base = go(settings=RunSettings(deadline=0.2)).record
    given = live(ask=sleeping(5.0, returns({"q:spam": NO})))
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], given)
    assert rounds(out["record"])[-1]["deadline"] is True
    assert call_of(out["record"], "ask")["outcome"]["failed"]["code"] == "timed-out"


def test_a_live_binding_is_given_the_calls_seconds_under_the_bases_timeout() -> None:
    base = go(settings=RunSettings(timeout=7)).record
    seen: list[float] = []

    async def note(batch: Any, *, seconds: float) -> object:
        seen.append(seconds)
        return CallResult({"q:spam": NO})

    scenario_sync(base, LOADED, [{"ask": "ask"}], live(ask=note))
    assert seen == [7.0]


def test_a_recorded_failure_given_back_whole_is_pinned() -> None:
    base = go(binds=bindings(look=Failed(Cause(CauseCode.raised, ["t:size"])))).record
    out = scenario_sync(base, LOADED, [{"ask": "ask"}], live(ask=returns({"q:spam": NO})))
    assert out["pinned"] == ["t:body", "t:size"]
    assert call_of(out["record"], "look")["outcome"]["failed"]["code"] == "raised"


@pytest.mark.parametrize(
    "reply",
    ["not a result", CallResult(["q:spam"])],  # pyright: ignore[reportArgumentType]
    ids=["str", "list"],
)
def test_a_live_result_that_is_not_a_call_result_of_a_dict_is_invalid_return(
    reply: object,
) -> None:
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=reply))
    assert call_of(out["record"], "ask")["outcome"]["failed"]["code"] == "invalid-return"
    assert out["pinned"] == ["t:size"]


class Key(str):
    """A ``str`` subclass, which interpret reads by its string data."""


def test_a_str_subclass_key_for_a_batchmate_is_unexpected_ref() -> None:
    answer = returns({"q:spam": NO, Key("t:body"): "sneaked"})
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=answer))
    assert call_of(out["record"], "ask")["outcome"]["failed"] == {
        "code": "unexpected-ref",
        "refs": ["q:spam", "t:body"],
    }


@pytest.mark.parametrize("lines", [["a list"], ("  ",), (5,)], ids=["list", "blank", "int"])
def test_warnings_that_do_not_conform_are_left_for_interpret_to_judge(lines: Any) -> None:
    answer = CallResult({"q:spam": NO}, warnings=lines)
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=answer))
    record = out["record"]
    assert warned(record, "ask") == []
    index = next(
        n for n, i in enumerate(record.log) if i["item"] == "call" and "ask" in i["sources"]
    )
    assert "warnings" in record.unhashed["calls"][str(index)]["invalid"]


def test_truncated_naming_a_pinned_answer_is_dropped_with_a_line() -> None:
    base = grounded.decided()
    ana = 'q:ask/out["Ana"]'

    async def answer(batch: Any, *, seconds: float) -> CallResult:
        refs = [ref for s in batch for ref in s.questions]
        return CallResult(
            {ref: {"yes": 0.2, "no": 0.8} for ref in refs},
            truncated=(ana,),
            warnings=("live line",),
        )

    people = [grounded.BO, grounded.ANA, {"name": "Cy", "age": 5}]
    out = scenario_sync(
        base,
        grounded.LOADED,
        [{"path": "t:state.people", "value": people}, {"ref": ana, "value": NO}],
        {"default": answer},
    )
    record = out["record"]
    assert entry(record, ana)["value"]["probabilities"] == NO
    assert out["pinned"] == [ana]
    assert "truncated" not in unhashed_call(record, "ask")
    assert warned(record, "ask") == [
        f"answered from record {base.record_hash}",
        f"'{ana}': pinned",
        f"'{ana}': named truncated, answered from record",
        "live line",
    ]


def test_truncated_naming_a_live_answer_is_kept() -> None:
    answer = returns({"q:spam": NO}, truncated=("q:spam",))
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=answer))
    record = out["record"]
    index = next(
        n for n, i in enumerate(record.log) if i["item"] == "call" and "ask" in i["sources"]
    )
    assert record.unhashed["calls"][str(index)]["truncated"] == ["q:spam"]


def test_bindings_are_checked_against_the_design_run() -> None:
    rerun = slice_design(sources={"ask": ASK, "body": BODY})  # mail alone: no default
    with pytest.raises(EvaluationRefusal) as raised:
        scenario_sync(b"x", LOADED, [], {"mail": Scripted({})}, rerun_design=rerun)
    assert raised.value.reason is R.invalid_record
    with pytest.raises(EntryRefusal, match="default"):
        scenario_sync(b"x", LOADED, [{"ask": "ask"}], {"mail": Scripted({})})


def test_a_source_new_in_the_rerun_design_is_asked_live_whole() -> None:
    extra = Source({"t:x": t.string}, needs=["t:state"])
    rerun = slice_design(sources={"ask": ASK, "body": BODY, "look": LOOK, "extra": extra})
    given = live(extra=returns({"t:x": "new"}))
    out = scenario_sync(go().record, LOADED, [], given, rerun_design=rerun)
    assert [[s.id for s in b] for b in given["default"].batches] == [["extra"]]
    found = {f["ref"]: f["clause"] for f in out["findings"] if f["kind"] == "stale-question"}
    assert found == {"t:x": "no-answer"}
    assert out["pinned"] == ["q:spam", "t:body", "t:size"]


def test_a_grounded_source_with_one_stale_candidate_is_asked_whole_and_stale_whole() -> None:
    base = grounded.decided()
    seen: list[list[str]] = []

    async def answer(batch: Any, *, seconds: float) -> CallResult:
        seen.append([ref for s in batch for ref in s.questions])
        return CallResult({ref: {"yes": 0.2, "no": 0.8} for s in batch for ref in s.questions})

    people = [grounded.BO, {"name": "Cy", "age": 5}]
    out = scenario_sync(
        base, grounded.LOADED, [{"path": "t:state.people", "value": people}], {"default": answer}
    )
    assert seen == [['q:ask/out["Bo"]', 'q:ask/out["Cy"]']]
    assert out["stale"] == ['q:ask/out["Bo"]', 'q:ask/out["Cy"]']
    assert out["pinned"] == []
    clauses = {f["ref"]: f["clause"] for f in out["findings"] if f["kind"] == "stale-question"}
    assert clauses == {'q:ask/out["Bo"]': "no-answer", 'q:ask/out["Cy"]': "no-answer"}


# ---------------------------------------------------------------------------------------------
# Round 2 of review: what stands as pinned, moved pins, truncated left to interpret


@pytest.mark.parametrize(
    "answer",
    [returns({"q:spam": NO, "q:zzz": NO}), returns({"q:spam": NO}, provider=5)],
    ids=["an unknown key", "a bad provider"],
)
def test_a_live_result_interpret_fails_counts_no_ref_of_its_call_pinned(answer: object) -> None:
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=answer))
    assert "failed" in call_of(out["record"], "ask")["outcome"]
    assert "marker" in entry(out["record"], "t:body")
    assert out["pinned"] == ["t:size"]


def test_a_live_raise_or_timeout_counts_no_ref_of_its_call_pinned() -> None:
    async def boom(batch: Any, *, seconds: float) -> object:
        raise RuntimeError("down")

    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=boom))
    assert out["pinned"] == ["t:size"]


def test_truncated_naming_a_pinned_ref_of_a_batchmate_is_left_for_interpret() -> None:
    answer = returns({"q:spam": NO}, truncated=("t:body",))
    out = scenario_sync(go().record, LOADED, [{"ask": "ask"}], live(ask=answer))
    calls = unhashed_call(out["record"], "ask")
    assert "truncated" in calls["invalid"]
    assert not any("named truncated" in w for w in warned(out["record"], "ask"))


def test_a_pinned_ref_moved_to_a_source_with_no_recorded_call_is_not_answered_live() -> None:
    replies = Scripted({"s": returns({"t:x": "a", "t:y": "b"})})
    base = decide_sync(PAIR, {"n": 1}, {"default": replies}).record
    moved = load(
        minimal(
            state=t.record(n=t.integer()),
            sources={
                "s": Source({"t:y": t.string}, needs=["t:state"]),
                "s2": Source({"t:x": t.string, "t:z": t.string}, needs=["t:state"]),
            },
        )
    )
    given = Scripted({"s2": returns({"t:x": "live", "t:z": "live"})})
    out = scenario_sync(
        base, PAIR, [{"ref": "t:x", "value": "set"}], {"default": given}, rerun_design=moved
    )
    assert entry(out["record"], "t:x")["marker"]["cause"]["code"] == "not-returned"
    assert out["stale"] == ["t:z"]
    assert "t:x" not in out["pinned"]
