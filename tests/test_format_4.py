# pyright: reportPrivateUsage=false
"""Format 4: each binding call receives its ``seconds``; a binding's ``timed-out`` or ``raised``
on a ref under a returned call, the other refs standing, replayed as recorded and refused at the
read of an earlier format; and a stored format-3 design and record: the record read under
format 3's rules to its own hash and refused at replay, the design refused at load."""

import asyncio
import copy
import json
import time
from pathlib import Path
from typing import Any

import pytest

from ledger_logs import before_profile, rehashed
from ommat import (
    CallResult,
    Cause,
    Design,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    Question,
    ReadView,
    Record,
    Rule,
    RunSettings,
    Source,
    SourceCall,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    decide_sync,
    from_record,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import decode
from ommat._provenance import DETAILS
from ommat.bindings.conformance import check
from ommat.helpers import pick_with_margin

Batch = tuple[SourceCall, ...]
FORMAT_3 = Path(__file__).parent / "format_3"
FORMAT_3_DESIGN = "sha256:46db78a006d63c3b5e3acae929bf6f931d2918c72a5f63e5fb6c5d2824d84943"
FORMAT_3_RECORD = "sha256:b423e73193918ccf47a6ef34a8bdd2371de090a3b8f7a08392fb519b8b604c54"
MARKERS = (Unsure, Unavailable, Unasked)


# ---------------------------------------------------------------------------------------------
# A design of three sources on one oracle: two questions and a tool value, one call


def kept(r: ReadView) -> object:
    return pick_with_margin(r, "q:a/kept", 0.5, 0.1)


def lost(r: ReadView) -> object:
    return pick_with_margin(r, "q:b/lost", 0.5, 0.1)


def report(r: ReadView) -> str:
    held = [r.read(ref) for ref in ("d:kept", "d:lost", "t:c/size")]
    shown = ["marker" if isinstance(v, MARKERS) else "value" for v in held]
    return ", ".join(shown)


DESIGN = Design(
    name="per-ref",
    state=t.record(text=t.string),
    sources={
        "a": Source({"q:a/kept": Question("noul", Text("Is it kept?"))}, oracle="o"),
        "b": Source({"q:b/lost": Question("noul", Text("Is it lost?"))}, oracle="o"),
        "c": Source({"t:c/size": t.integer()}, oracle="o"),
    },
    rules={
        "kept": Rule("d:kept", t.label("no", "yes"), kept, version="1", reads=["q:a/kept"]),
        "lost": Rule("d:lost", t.label("no", "yes"), lost, version="1", reads=["q:b/lost"]),
        "report": Rule(
            "d:report", t.string, report, version="1", reads=["d:kept", "d:lost", "t:c/size"]
        ),
    },
)
LOADED = load(DESIGN)
YES = {"yes": 0.8, "no": 0.2}


async def split(batch: Batch, *, seconds: float) -> object:
    """Each source sent as its own request: ``a`` answered, ``b``'s request out of time and
    ``c``'s failed, each said why in ``warnings``."""
    return CallResult(
        {
            "q:a/kept": YES,
            "q:b/lost": Unavailable(Cause("timed-out")),
            "t:c/size": Unavailable(Cause("raised")),
        },
        warnings=("b: the request ran out of time", "c: HTTP 503 Service Unavailable"),
        attempts=3,
        missing=dict.fromkeys(sorted(DETAILS), "not-reported"),
    )


def per_ref_turn() -> Record:
    return decide_sync(LOADED, {"text": "a report"}, {"o": split}).record


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


# ---------------------------------------------------------------------------------------------
# seconds


def seen_seconds(settings: RunSettings | None = None, **source: Any) -> list[float]:
    seen: list[float] = []

    async def binding(batch: Batch, *, seconds: float) -> object:
        seen.append(seconds)
        return CallResult({"q:x": YES})

    design = Design(
        state=t.record(),
        sources={"s": Source({"q:x": Question("noul", Text("Is it?"))}, oracle="o", **source)},
        rules={
            "x": Rule("d:x", t.label("no", "yes"), lambda r: "yes", version="1", reads=[]),
            "ok": Rule("d:ok", t.boolean, lambda r: True, version="1", reads=["d:x"]),
        },
    )
    decide_sync(load(design), {}, {"o": binding}, settings)
    return seen


def test_a_binding_receives_the_batchs_timeout_as_its_seconds() -> None:
    (seconds,) = seen_seconds(RunSettings(timeout=7.5))
    assert seconds == 7.5
    (declared,) = seen_seconds(RunSettings(timeout=60), timeout=2.0)
    assert declared == 2.0
    (whole,) = seen_seconds(RunSettings(timeout=60))
    assert (type(whole), whole) == (float, 60.0)


def test_a_binding_receives_the_time_left_to_the_deadline_when_that_is_smaller() -> None:
    (seconds,) = seen_seconds(RunSettings(timeout=60, deadline=3.0))
    assert type(seconds) is float
    assert 2.0 < seconds <= 3.0
    # Some time has passed since the deadline started, unless the clock is too coarse to see it:
    # Windows' monotonic clock before Python 3.13 ticks every 15.6 ms.
    if time.get_clock_info("monotonic").resolution < 1e-3:
        assert seconds < 3.0


def test_a_binding_without_the_seconds_keyword_is_raised_on_every_ref() -> None:
    async def old(batch: Batch) -> object:
        return CallResult({})

    record = decide_sync(LOADED, {"text": "a report"}, {"o": old}).record  # pyright: ignore[reportArgumentType]
    call = next(i for i in record.log if i["item"] == "call")
    assert call["outcome"] == {
        "failed": {"code": "raised", "refs": ["q:a/kept", "q:b/lost", "t:c/size"]}
    }
    (index,) = (str(i) for i, item in enumerate(record.log) if item["item"] == "call")
    raised = json.loads(record.document)["unhashed"]["exceptions"][index]
    assert raised["type"] == "TypeError"
    assert "seconds" in raised["message"]


# ---------------------------------------------------------------------------------------------
# The binding's per-ref timed-out and raised


def test_a_per_ref_timed_out_and_raised_stand_beside_an_answered_ref() -> None:
    record = per_ref_turn()
    (call,) = (i for i in record.log if i["item"] == "call")
    assert call["outcome"] == "returned"
    assert entry(record, "q:a/kept")["value"]["probabilities"] == YES
    assert entry(record, "q:b/lost")["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "timed-out", "refs": ["q:b/lost"]},
    }
    assert entry(record, "t:c/size")["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "raised", "refs": ["t:c/size"]},
    }
    # The rule that read the marker provides it unchanged, naming the answer.
    assert entry(record, "d:lost")["marker"]["cause"] == {"code": "timed-out", "refs": ["q:b/lost"]}
    unhashed = json.loads(record.document)["unhashed"]["calls"]
    (calls,) = unhashed.values()
    assert calls["warnings"] == [
        "b: the request ran out of time",
        "c: HTTP 503 Service Unavailable",
    ]
    assert calls["attempts"] == 3
    decision = decide_sync(LOADED, {"text": "a report"}, {"o": split})
    assert decision.record.entries["d:report"]["value"] == "value, marker, marker"


def test_a_per_ref_record_replays_with_no_finding_and_its_markers_as_recorded() -> None:
    record = per_ref_turn()
    rerun, findings = replay_sync(record, LOADED)
    assert findings == []
    assert rerun is not None
    for ref in ("q:b/lost", "t:c/size", "d:lost"):
        assert entry(rerun, ref) == entry(record, ref)


def test_from_record_answers_a_per_ref_marker_as_unsupported() -> None:
    # from_record keeps only invalid-answer and invalid-return as stored.
    binding = from_record(per_ref_turn(), LOADED)
    record = decide_sync(LOADED, {"text": "a report"}, {"o": binding}).record
    assert entry(record, "q:a/kept")["value"]["probabilities"] == YES
    assert entry(record, "q:b/lost")["marker"]["cause"]["code"] == "unsupported"
    assert entry(record, "t:c/size")["marker"]["cause"]["code"] == "unsupported"
    (lines,) = (c["warnings"] for c in json.loads(record.document)["unhashed"]["calls"].values())
    assert lines[0].startswith("answered from record ")
    assert any("q:b/lost" in line for line in lines)
    assert any("t:c/size" in line for line in lines)


GOLDEN_DESIGN = "sha256:58f570c379b3f7de7b88d99e6f7545f2259d7acbe30884d51ff9e53e9cafa229"
GOLDEN_RECORD = "sha256:a9d2ef3cc5631400e1551aba7e7c9c098f5a1ea6c82d839026f8230634c5db3f"


def test_the_golden_per_ref_record() -> None:
    record = per_ref_turn()
    assert record.log[0] == {"item": "design", "hash": GOLDEN_DESIGN, "format": 11, "helpers": 4}
    assert "policy" not in record.log[1]
    assert entry(record, "d:report")["value"] == "value, marker, marker"
    assert (record.design_hash, record.record_hash) == (GOLDEN_DESIGN, GOLDEN_RECORD)
    assert decode(record.document)["hash"] == GOLDEN_RECORD


def earlier(doc: dict[str, Any], fmt: int) -> dict[str, Any]:
    """A current record document made format ``fmt`` by ``before_profile``. Its ``hash`` is left
    to recompute."""
    return before_profile(doc, fmt)


def test_a_per_ref_marker_reads_back_at_format_4_and_is_refused_at_format_3() -> None:
    doc = json.loads(per_ref_turn().document)
    assert read_record(json.dumps(doc).encode()).log == per_ref_turn().log
    for fmt in (1, 2, 3):
        old = copy.deepcopy(doc)
        earlier(old, fmt)
        if fmt < 3:
            for item in old["log"]:
                if item["item"] == "call":
                    item.pop("precision", None)
            del old["hash"]
        with pytest.raises(InvalidRecord, match="timed-out marker under returned call item"):
            read_record(rehashed(old))


async def raised_only(batch: Batch, *, seconds: float) -> object:
    return CallResult(
        {"q:a/kept": YES, "q:b/lost": YES, "t:c/size": Unavailable(Cause("raised"))},
        warnings=("c: HTTP 503 Service Unavailable",),
    )


def test_a_raised_marker_alone_is_refused_at_format_3() -> None:
    doc = json.loads(decide_sync(LOADED, {"text": "a"}, {"o": raised_only}).record.document)
    earlier(doc, 3)
    with pytest.raises(InvalidRecord, match="raised marker under returned call item"):
        read_record(rehashed(doc))


def test_an_engine_timeout_under_a_failed_call_is_read_at_every_format() -> None:
    # The engine's own timed-out, the call Failed, is format 1's as much as format 4's.
    async def slow(batch: Batch, *, seconds: float) -> object:
        await asyncio.sleep(1)
        return CallResult({})

    record = decide_sync(LOADED, {"text": "a"}, {"o": slow}, RunSettings(timeout=0.01)).record
    (call,) = (i for i in record.log if i["item"] == "call")
    assert call["outcome"]["failed"]["code"] == "timed-out"
    doc = json.loads(record.document)
    earlier(doc, 3)
    assert read_record(rehashed(doc)).log[0]["format"] == 3


# ---------------------------------------------------------------------------------------------
# The stored format-3 design and record


def format_3_record() -> Record:
    return read_record((FORMAT_3 / "record.json").read_bytes())


def test_check_11_refuses_the_stored_format_3_design_naming_its_format() -> None:
    with pytest.raises(InvalidDesign) as info:
        load((FORMAT_3 / "design.json").read_bytes())
    assert [f.check for f in info.value.failures] == [11]
    assert "the document's format 3 is not 12" in str(info.value)


def test_a_format_3_record_reads_back_to_its_own_hash() -> None:
    data = (FORMAT_3 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_3_RECORD
    assert record.log[0] == {"item": "design", "hash": FORMAT_3_DESIGN, "format": 3, "helpers": 2}


def test_a_format_3_record_is_wrong_design_at_replay() -> None:
    from test_decide import LOADED as SLICE

    assert replay_sync(format_3_record(), SLICE) == (None, [{"kind": FindingKind.wrong_design}])


# ---------------------------------------------------------------------------------------------
# The bindings ommat ships, under the suite


def test_the_fixture_binding_returns_per_ref_markers_and_passes_the_suite() -> None:
    record = per_ref_turn()
    seen: list[Batch] = []

    async def capturing(batch: Batch, *, seconds: float) -> object:
        seen.append(batch)
        return await split(batch, seconds=seconds)

    decide_sync(LOADED, {"text": "a report"}, {"o": capturing})
    from ommat._fixture import Base, FixtureBinding, RecordedCalls

    fixture = FixtureBinding(RecordedCalls(record.log), LOADED, Base(record.log, LOADED))
    # The fixture returns warnings empty and null provenance, so it owes no per-ref line.
    assert check(fixture, batches=seen, deterministic=True) == []
    assert check(split, batches=seen, deterministic=True) == []
