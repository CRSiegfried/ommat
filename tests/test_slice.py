"""The engine end to end, through public names only: a minimal design with one function
rule and one function policy rule loads; decide runs one turn against a test fixture binding;
the record closes with its hash and reads back; and replay reproduces it with no finding."""

from typing import Any

from ommat import (
    CallResult,
    Design,
    Question,
    Rule,
    Source,
    SourceCall,
    Text,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_with_margin


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


def act(r: Any) -> str:
    return "archive" if r.read("d:route") == "yes" else "keep"


DESIGN = Design(
    state=t.record(subject=t.string),
    policy="d:act",
    sources={"ask": Source({"q:spam": Question("noul", Text("Is it spam?"))}, needs=["t:state"])},
    rules={
        "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:spam"]),
        "act": Rule("d:act", t.label("archive", "keep"), act, version="1", reads=["d:route"]),
    },
)


class Answering:
    """A test fixture binding: every batch's spam question answered with ``spam``."""

    def __init__(self, spam: float) -> None:
        self.spam = spam
        self.asked: list[tuple[SourceCall, ...]] = []

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        self.asked.append(batch)
        return CallResult({"q:spam": {"yes": self.spam, "no": 1 - self.spam}})


def test_one_turn_loads_decides_closes_and_replays() -> None:
    # Load: one function rule and one function policy rule.
    loaded = load(DESIGN)
    assert loaded.design_hash.startswith("sha256:")

    # Decide: one turn against the fixture binding; the decision is the policy's estimate.
    binding = Answering(0.9)
    decision = decide_sync(loaded, {"subject": "win a prize"}, {"default": binding})
    assert len(binding.asked) == 1
    assert decision.estimate == "archive"

    # The record closes with its hash and reads back with the same one.
    record = decision.record
    turn = record.record_hash
    assert turn.startswith("sha256:")
    assert len(turn) == len("sha256:") + 64
    assert record.design_hash == loaded.design_hash == decision.design_hash
    assert read_record(record.document).record_hash == turn

    # Replay reproduces it with no finding.
    rerun, findings = replay_sync(record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == turn


def test_a_keep_turn_decides_keep_and_replays() -> None:
    loaded = load(DESIGN)
    decision = decide_sync(loaded, {"subject": "lunch?"}, {"default": Answering(0.1)})
    assert decision.estimate == "keep"
    rerun, findings = replay_sync(decision.record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == decision.record.record_hash
