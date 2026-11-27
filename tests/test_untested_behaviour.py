# pyright: reportPrivateUsage=false
"""Behaviour no other file checks: a call's cost charged once over its attempts, the lint's
grid order, ``lookup`` over bytes ``write`` would not have written, an ``EntryRefusal`` from
replay through every replaying job, and the round's appends in source id order after its
calls all end."""

import asyncio
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from designs import minimal
from ledger_logs import rehashed
from ommat import (
    CallResult,
    EntryRefusal,
    Parameter,
    Question,
    Record,
    Rule,
    RunSettings,
    Source,
    Text,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import hash_value
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    _store,
    deciding_inputs_sync,
    gold_record_sync,
    intervene_sync,
    question_reach_sync,
    rescore_grid_sync,
    rescore_sync,
    scenario_sync,
    switching_values_batch_sync,
    switching_values_sync,
    use_answers_sync,
)
from ommat.helpers import dead_questions, points
from scripted import Batch, Scripted, returns

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")


def body(record: Record, n: int) -> list[dict[str, Any]]:
    """Round ``n``'s items, in log order."""
    return [i for i in record.log if i.get("round") == n]


# ---------------------------------------------------------------------------------------------
# A call is charged its cost once, however many attempts it takes (§cost-tag)


def test_a_call_is_charged_its_cost_once_however_many_attempts_the_binding_made() -> None:
    def second(r: Any) -> bool:
        return r.require("t:a") == "a"

    sources = {
        "first": Source({"t:a": t.string}, cost=2),
        "then": Source({"t:b": t.string}, needs=["d:ready"], cost=1),
    }
    rules = {"ready": Rule("d:ready", t.boolean, second, version="1", reads=["t:a"])}
    loaded = load(minimal(state=t.record(), sources=sources, rules=rules))
    binding = Scripted(
        {"first": returns({"t:a": "a"}, attempts=3), "then": returns({"t:b": "b"}, attempts=4)}
    )
    # Charged per attempt, round 1 would spend 6 and cut round 2 under a budget of 3.
    record = decide_sync(loaded, {}, {"default": binding}, RunSettings(budget=3)).record
    assert [b[0].id for b in binding.batches] == ["first", "then"]
    assert [i["cost"] for i in record.log if i["item"] == "round"] == [2, 1]
    assert [c["attempts"] for c in record.unhashed["calls"].values()] == [3, 4]


# ---------------------------------------------------------------------------------------------
# The lint runs its points in grid order: refs in code-point order, the last varying fastest


ran_at: list[tuple[object, ...]] = []
# In code-point order ("/" < "_" < "b"), and declared in the reverse order.
GRID_REFS = ("t:a/b", "t:a_b", "t:ab")


def noting(r: Any) -> bool:
    ran_at.append(tuple(r.read(ref) for ref in GRID_REFS))
    return True


def test_the_dead_question_lint_runs_its_points_in_grid_order() -> None:
    provides = {"t:ab": t.label("x", "y"), "t:a_b": t.label("p", "q", "r"), "t:a/b": t.boolean}
    policy = Rule("d:decision", t.boolean, noting, version="1", reads=list(GRID_REFS))
    loaded = load(
        minimal(state=t.record(), sources={"s": Source(provides)}, rules={"decision": policy})
    )
    ran_at.clear()
    report = dead_questions(loaded, ["d:decision"])
    expected = list(itertools.product(*(points(loaded, ref) for ref in GRID_REFS)))
    assert report["points"] == len(expected) == 12
    assert ran_at == expected
    assert ran_at[:3] == [(False, "p", "x"), (False, "p", "y"), (False, "q", "x")]


# ---------------------------------------------------------------------------------------------
# lookup: an entry in bytes write would not write, and its kind by its key members

KEY = {"record": "sha256:1", "design": "sha256:2", "parameter": "lim"}


def filed(root: Path, data: bytes) -> None:
    """``data`` held where ``lookup("switching-value", KEY)`` reads."""
    path = root / "switching-value" / "sha256" / hash_value(KEY).removeprefix("sha256:")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_lookup_accepts_an_entry_held_in_bytes_that_are_not_canonical(tmp_path: Path) -> None:
    held = {"value": {"b": [1.0, None], "a": "é"}, "key": dict(reversed(KEY.items()))}
    held |= {"format": 1, "kind": "switching-value"}
    data = json.dumps(held, indent=2, ensure_ascii=True).encode()
    filed(tmp_path, data)
    found = EvaluationStore(tmp_path).lookup("switching-value", KEY)
    assert found == {**held, "key": KEY}


def test_lookup_tells_an_entry_of_another_kind_by_its_key_members(tmp_path: Path) -> None:
    # Two kinds sharing key members would let one kind's entry read back as the other's.
    members = [frozenset(m) for m in _store._KEYS.values()]
    assert len(set(members)) == len(members)
    other = {"kind": "gold-answer", "format": 1, "key": KEY, "value": 0}
    filed(tmp_path, json.dumps(other).encode())
    with pytest.raises(EvaluationRefusal) as info:
        EvaluationStore(tmp_path).lookup("switching-value", KEY)
    assert info.value.reason is RefusalReason.misfiled_entry


# ---------------------------------------------------------------------------------------------
# An EntryRefusal from replay passes through every replaying job unwrapped


def policy(r: Any) -> str:
    r.read("q:spam")
    return "high" if r.read("t:state.n") > r.param("lim") else "low"


SPAM = {"yes": 0.9, "no": 0.1}
REPLAYED = load(
    minimal(
        state=t.record(n=t.number()),
        sources={"ask": Source({"q:spam": Question("noul", Text("Spam?"))})},
        rules={
            "act": Rule(
                "d:act",
                t.label("high", "low"),
                policy,
                version="1",
                reads=["q:spam", "t:state.n"],
                params=["lim"],
            )
        },
        alone=True,
        parameters={"lim": Parameter(t.number(), 1.0)},
    )
)


def timeless() -> Record:
    """A finished turn whose recorded timeout replay refuses: the read takes any number there,
    and decide's bounds check takes only a positive one."""
    binding = Scripted({"ask": returns({"q:spam": SPAM})})
    record = decide_sync(REPLAYED, {"n": 2.0}, {"default": binding}).record
    doc = json.loads(record.document)
    doc["log"][1]["settings"]["timeout"] = 0
    return read_record(rehashed(doc))


async def answering(batch: Batch, *, seconds: float) -> object:
    return CallResult({"q:spam": SPAM})


JOBS: dict[str, Callable[[Record, Path], object]] = {
    "replay": lambda r, _: replay_sync(r, REPLAYED),
    "use_answers": lambda r, _: use_answers_sync(r, REPLAYED, r),
    "intervene": lambda r, _: intervene_sync(r, REPLAYED, [], rerun_design=REPLAYED),
    "scenario": lambda r, _: scenario_sync(
        r, REPLAYED, [], {"default": answering}, rerun_design=REPLAYED
    ),
    "deciding_inputs": lambda r, _: deciding_inputs_sync(r, REPLAYED, "d:act"),
    "question_reach": lambda r, _: question_reach_sync([r], REPLAYED, refs=["d:act"]),
    "gold_record": lambda r, root: gold_record_sync(
        r, REPLAYED, "g1", results=EvaluationStore(root)
    ),
    "switching_values": lambda r, root: switching_values_sync(
        r, REPLAYED, "lim", results=EvaluationStore(root)
    ),
    "switching_values_batch": lambda r, root: switching_values_batch_sync(
        [r], REPLAYED, ["lim"], results=EvaluationStore(root)
    ),
    "rescore": lambda r, _: rescore_sync([r], REPLAYED, [REPLAYED], refs=["d:act"]),
    "rescore_grid": lambda r, _: rescore_grid_sync(
        [r], REPLAYED, REPLAYED, [{"lim": 3.0}], refs=["d:act"]
    ),
}


@pytest.mark.parametrize("job", list(JOBS))
def test_an_entry_refusal_from_replay_passes_through_each_replaying_job_unwrapped(
    job: str, tmp_path: Path
) -> None:
    with pytest.raises(EntryRefusal, match="the timeout 0 is not") as info:
        JOBS[job](timeless(), tmp_path)
    assert type(info.value) is EntryRefusal
    assert info.value.__cause__ is None
    assert not any(tmp_path.iterdir())  # refused before any store write


def test_the_timeless_record_replays_cleanly_once_its_timeout_is_positive() -> None:
    """The refusal above is the recorded setting's, not the turn's."""
    doc = json.loads(timeless().document)
    doc["log"][1]["settings"]["timeout"] = 60
    rerun, found = replay_sync(read_record(rehashed(doc)), REPLAYED)
    assert rerun is not None
    assert [f["kind"] for f in found] == []


# ---------------------------------------------------------------------------------------------
# A round's items are appended in source id order once its calls have all ended


def test_a_rounds_items_are_in_source_id_order_though_its_calls_end_in_reverse() -> None:
    ids = ("c", "b", "a")  # declared out of order too
    finished: list[str] = []
    when_appended: list[tuple[str, int]] = []

    # "a" ends last and "c" first: each call waits for the done event of the one before it
    # ("c" waits for none), so the order holds under any load and any start order.
    done = {sid: asyncio.Event() for sid in ids}
    waits_for = {"a": "b", "b": "c", "c": None}

    def reply(sid: str) -> Callable[..., Any]:
        async def answer(batch: Batch, *, seconds: float) -> object:
            if (before := waits_for[sid]) is not None:
                await done[before].wait()
            finished.append(sid)
            done[sid].set()
            return CallResult({f"t:{sid}": sid})

        return answer

    def observer(index: int, item: dict[str, Any]) -> None:
        if item.get("round") == 1:
            when_appended.append((item["item"], len(finished)))

    sources = {sid: Source({f"t:{sid}": t.string}, oracle=f"o_{sid}") for sid in ids}
    loaded = load(minimal(state=t.record(), sources=sources))
    bindings = {f"o_{sid}": reply(sid) for sid in ids}
    settings = RunSettings(timeout=5)  # a call order that deadlocks fails fast
    record = decide_sync(loaded, {}, bindings, settings, observer=observer).record
    assert finished == ["c", "b", "a"]
    shown = [(i["item"], i.get("sources", i.get("ref"))) for i in body(record, 1)]
    assert shown == [
        ("call", ["a"]),
        ("call", ["b"]),
        ("call", ["c"]),
        ("entry", "t:a"),
        ("entry", "t:b"),
        ("entry", "t:c"),
        ("round", None),
    ]
    calls = [i for i in record.log if i["item"] == "call"]
    assert [e["call"] for e in body(record, 1) if e["item"] == "entry"] == [
        record.log.index(c) for c in calls
    ]
    assert {n for _, n in when_appended} == {3}  # every append after the last call ended
