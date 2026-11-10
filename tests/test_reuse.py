"""A record's runs under designs differing only in leaf values, one evaluation job's, fire each
rule from the job's first run when nothing it reads, declares or calls changed
(``_interpret.Reuse``): each run's log and stale refs are a full rerun's, byte for byte, and so
are the rows and entries of rescore, the grid and the switching-value job. The design reaches a
changed leaf through a gated source, a wording slot, a table's ``Param`` row, a candidate set
whose members change under the same candidate values, a returned marker naming a grounded ref,
the budget and the policy's own leaf, beneath a costly rule no leaf reaches."""

import asyncio
import itertools
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from designs import slot
from ommat import (
    Answer,
    CallResult,
    Cause,
    CauseCode,
    DefectWarning,
    Design,
    Helper,
    LoadedDesign,
    Param,
    Parameter,
    Question,
    Record,
    Rule,
    RunSettings,
    Source,
    SourceRequirement,
    Table,
    Text,
    Unasked,
    Undetermined,
    Unsure,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._canonical import encode
from ommat._interpret import Reuse, _same  # pyright: ignore[reportPrivateUsage]
from ommat._load import with_parameters
from ommat.evaluation import (
    EvaluationStore,
    _grid,  # pyright: ignore[reportPrivateUsage]
    _rescore,  # pyright: ignore[reportPrivateUsage]
    _switching,  # pyright: ignore[reportPrivateUsage]
    rescore_grid_sync,
    rescore_sync,
    switching_values_batch_sync,
)
from ommat.evaluation._switching import _rerun as rerun  # pyright: ignore[reportPrivateUsage]
from ommat.helpers import mass, pick_with_margin

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")

STATE_T = t.record(text=t.string, n=t.number())
DAY = t.record(name=t.string, slot=t.string)
HOURS = t.list_(t.record(day=t.string, hours=t.number()))
SET = "d:days"
CALLS = [0]
"""Each call of the costly rule."""


def calendar(r: Any) -> object:
    CALLS[0] += 1
    n = r.read("t:state.n")
    return [{"day": d, "hours": n + i} for i, d in enumerate(("mon", "tue", "wed"))]


def base(r: Any) -> object:
    r.read("d:cal")
    return pick_with_margin(r, "q:intent", 0.5, 0.0)


def days(r: Any) -> object:
    held = r.read("d:cal")[: int(r.param("g/count"))]
    return [{"name": c["day"], "slot": r.param("g/mode")} for c in held]


def fit(r: Any) -> object:
    return f"{r.read('candidate')}:{r.read('candidate.slot')}:{len(r.read('d:cal'))}"


def span(r: Any) -> object:
    return float(len(r.read("d:cal")))


def flag(r: Any) -> object:
    r.read("d:span")  # two rules past the calendar, so it fires in the wave after the set's
    return Unsure(Cause(CauseCode.required, ['d:fit["mon"]']))  # kept only when on the board


def act(r: Any) -> str:
    said = [str(r.read("d:gate")), str(r.read("d:tone")), repr(r.read("d:fit"))]
    said.append(repr(r.read("d:flag")))
    for ref in ("q:amount", "q:late"):
        found = mass(r, ref, ["yes"])
        said.append("-" if isinstance(found, Undetermined) else str(found >= r.param("g/stated")))
    return " ".join(said)


LEAVES: dict[str, list[Any]] = {
    "g/bar": [0.5, 0.7],
    "g/mode": ["am", "pm"],
    "g/count": [2, 0],
    "g/word": ["hello", "bye"],
    "g/floor": ["ten", "five"],
    "g/stated": [0.5, 0.2],
}


def swept(version: str = "1") -> LoadedDesign:
    return load(
        Design(
            name="swept",
            state=STATE_T,
            sources={
                "ask": Source(
                    {"q:intent": Question("choice", Text("Intent?"), labels=("refund", "other"))},
                    needs=["t:state"],
                    cost=1,
                ),
                "more": Source(
                    {
                        "q:amount": Question(
                            "noul",
                            Text("Amount over {floor}?", [slot("floor", param="g/floor")]),
                        )
                    },
                    needs=["t:state"],
                    requirement=SourceRequirement("d:gate", ["open"]),
                    cost=1,
                ),
                "late": Source(
                    {"q:late": Question("noul", Text("Late?"))}, needs=["d:base"], cost=1
                ),
            },
            rules={
                "cal": Rule("d:cal", HOURS, calendar, version=version, reads=["t:state.n"]),
                "base": Rule(
                    "d:base",
                    t.label("refund", "other"),
                    base,
                    version="1",
                    reads=["q:intent", "d:cal"],
                ),
                "gate": Table(
                    "d:gate",
                    t.label("open", "shut"),
                    "F",
                    [Helper("pick_with_margin", "q:intent", bar="g/bar", margin="g/m")],
                    [["refund", "open"], ["-", "shut"]],
                ),
                "days": Rule(
                    SET,
                    t.list_(DAY),
                    days,
                    version="1",
                    reads=["d:cal"],
                    params=["g/count", "g/mode"],
                    candidates="name",
                ),
                "fit": Rule(
                    "d:fit",
                    t.string,
                    fit,
                    version="1",
                    reads=["candidate", "candidate.slot", "d:cal"],
                    grounded=SET,
                ),
                "span": Rule("d:span", t.number(), span, version="1", reads=["d:cal"]),
                "flag": Rule("d:flag", t.string, flag, version="1", reads=["d:span"]),
                "tone": Table(
                    "d:tone",
                    t.string,
                    "F",
                    ["d:base"],
                    [["refund", Param("g/word")], ["-", "none"]],
                ),
                "act": Rule(
                    "d:act",
                    t.string,
                    act,
                    version="1",
                    reads=["d:gate", "d:tone", "d:fit", "d:flag", "q:amount", "q:late"],
                    params=["g/stated"],
                ),
            },
            parameters={
                "g/bar": Parameter(t.number(0, 1), 0.5),
                "g/m": Parameter(t.number(0, 1), 0.0),
                "g/mode": Parameter(t.label("am", "pm"), "am"),
                "g/count": Parameter(t.integer(0, 3), 2),
                "g/word": Parameter(t.label("hello", "bye"), "hello"),
                "g/floor": Parameter(t.label("ten", "five"), "ten"),
                "g/stated": Parameter(t.number(0, 1), 0.5),
            },
            policy="d:act",
            classes={SET: {"name": "identifier"}},
        )
    )


LOADED = swept()


def turn(refund: float, budget: float | None) -> Record:
    given = {
        "q:intent": {"refund": refund, "other": round(1 - refund, 6)},
        "q:amount": {"yes": 0.4, "no": 0.6},
        "q:late": {"yes": 0.3, "no": 0.7},
    }

    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({ref: given[ref] for call in batch for ref in call.questions})

    settings = RunSettings(budget=budget)
    return decide_sync(LOADED, {"text": "hi", "n": 1.0}, {"default": answer}, settings).record


# The gate open and shut, each with no budget and with one that cuts round 2 only when the
# gated source joins the late one.
RECORDS = [turn(refund, budget) for refund in (0.6, 0.2) for budget in (None, 2.5)]


def designs() -> Iterator[LoadedDesign]:
    for values in itertools.product(*LEAVES.values()):
        yield with_parameters(LOADED, dict(zip(LEAVES, values, strict=True)))


DESIGNS = list(designs())


def _rerun(log: Any, loaded: LoadedDesign, answers: Any, **given: Any) -> Any:
    return asyncio.run(rerun(log, loaded, answers, **given))


class _Never(Reuse):
    """A memo that never keeps its reference, so that every run is a full one."""

    def kept(self, loaded: LoadedDesign, board: dict[str, object]) -> None:
        pass


def test_each_run_writes_what_a_full_rerun_writes_and_calls_no_unreached_rule() -> None:
    # Every design is the reference's but for some of six leaves; the first leaves it alone.
    for record in RECORDS:
        log = record.log
        reuse = Reuse()
        for i, loaded in enumerate(DESIGNS):
            full, stale = _rerun(log, loaded, log, design=LOADED)
            CALLS[0] = 0
            mine, mine_stale = _rerun(log, loaded, log, design=LOADED, reuse=reuse)
            assert [encode(item) for item in mine.log] == [encode(item) for item in full.log]
            assert mine_stale.stale == stale.stale
            assert CALLS[0] == (1 if i == 0 else 0)  # the costly rule runs in the reference alone


def test_the_runs_reach_every_way_a_leaf_changes_a_turn() -> None:
    # The sweep is not idle: each leaf changes some record's log under the reference.
    log = RECORDS[0].log
    reference = _rerun(log, LOADED, log, design=LOADED)[0].log
    entries = {i["ref"]: i for i in reference if i["item"] == "entry"}
    for path, values in LEAVES.items():
        changed = with_parameters(LOADED, {path: values[1]})
        rerun, fixture = _rerun(log, changed, log, design=LOADED)
        theirs = {i["ref"]: i for i in rerun.log if i["item"] == "entry"}
        differ = {r for r in {*entries, *theirs} if entries.get(r) != theirs.get(r)}
        assert differ or fixture.stale, path
    # The budget cuts the late source only when the gated one is asked beside it.
    cut = _rerun(RECORDS[1].log, LOADED, RECORDS[1].log, design=LOADED)[0].log
    late = next(i for i in cut if i["item"] == "entry" and i["ref"] == "q:late")
    assert late["marker"]["cause"]["code"] == "over-budget"
    shut = with_parameters(LOADED, {"g/bar": 0.7})
    late = next(
        i
        for i in _rerun(RECORDS[1].log, shut, RECORDS[1].log, design=LOADED)[0].log
        if i["item"] == "entry" and i["ref"] == "q:late"
    )
    assert late["marker"]["cause"]["code"] != "over-budget"  # asked, the record unanswering


def test_rescore_gives_a_full_rerun_s_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    CALLS[0] = 0
    rows = rescore_sync(RECORDS, LOADED, DESIGNS)
    assert CALLS[0] == len(RECORDS)  # once per record, not per design
    monkeypatch.setattr(_rescore, "Reuse", _Never)
    theirs = rescore_sync(RECORDS, LOADED, DESIGNS)
    assert [encode(row) for row in rows] == [encode(row) for row in theirs]
    assert {encode(row["decision"]) for row in rows} != {encode(rows[0]["decision"])}


def test_the_grid_gives_a_full_rerun_s_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    points = [
        {"g/bar": bar, "g/word": word, "g/floor": floor, "g/mode": mode}
        for bar in (0.1, 0.55, 0.65, 0.9)
        for word in LEAVES["g/word"]
        for floor in LEAVES["g/floor"]
        for mode in LEAVES["g/mode"]
    ]
    rows = rescore_grid_sync(RECORDS, LOADED, LOADED, points)
    monkeypatch.setattr(_grid, "Reuse", _Never)
    theirs = rescore_grid_sync(RECORDS, LOADED, LOADED, points)
    assert [encode(row) for row in rows] == [encode(row) for row in theirs]


def test_the_switching_job_writes_a_full_rerun_s_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = ["g/bar", "g/stated", "g/count", "g/mode", "g/word"]
    CALLS[0] = 0
    entries = switching_values_batch_sync(
        RECORDS, LOADED, paths, results=EvaluationStore(tmp_path / "a")
    )
    reused = CALLS[0]
    monkeypatch.setattr(_switching, "Reuse", _Never)
    CALLS[0] = 0
    theirs = switching_values_batch_sync(
        RECORDS, LOADED, paths, results=EvaluationStore(tmp_path / "b")
    )
    assert [encode(e) for e in entries] == [encode(e) for e in theirs]
    # One reference run per record serves every path; each baseline replay calls it once more.
    assert reused == 2 * len(RECORDS) < CALLS[0]


def test_a_design_differing_beyond_its_leaves_calls_what_differs() -> None:
    log = RECORDS[0].log
    reuse = Reuse()
    _rerun(log, LOADED, log, design=LOADED, reuse=reuse)
    # Another version string, which the document holds; the same document with another callable.
    twin = types.FunctionType(calendar.__code__, calendar.__globals__, calendar.__name__)
    twin.__qualname__, twin.__module__ = calendar.__qualname__, calendar.__module__
    other_code = {**LOADED._callables, "cal": twin}  # pyright: ignore[reportPrivateUsage]
    for other in (swept(version="2"), load(LOADED.document, functions=other_code)):
        CALLS[0] = 0
        mine = _rerun(log, other, log, design=LOADED, reuse=reuse)[0].log
        assert CALLS[0] == 1
        assert [encode(i) for i in mine] == [
            encode(i) for i in _rerun(log, other, log, design=LOADED)[0].log
        ]
    # A design loaded afresh from a document differing in a leaf's value, the callables the
    # reference's: nothing shared but the code, yet the documents match, so it reuses.
    document = with_parameters(LOADED, {"g/word": "bye"}).document
    fresh = load(document, functions=LOADED._callables)  # pyright: ignore[reportPrivateUsage]
    CALLS[0] = 0
    mine = _rerun(log, fresh, log, design=LOADED, reuse=reuse)[0].log
    assert CALLS[0] == 0
    assert [encode(i) for i in mine] == [
        encode(i) for i in _rerun(log, fresh, log, design=LOADED)[0].log
    ]


def test_alike_means_alike_in_all_a_rule_can_tell() -> None:
    nan = float("nan")
    alike: list[tuple[object, object]] = [
        ({"a": 1.0, "b": [b"x", None]}, {"a": 1.0, "b": [b"x", None]}),
        (nan, nan),
        (
            Answer("noul", ("yes", "no"), {"yes": 0.4, "no": 0.6}),
            Answer("noul", ("yes", "no"), {"yes": 0.4, "no": 0.6}),
        ),
        (Unsure(Cause(CauseCode.required, ["d:x"])), Unsure(Cause(CauseCode.required, ["d:x"]))),
    ]
    unlike: list[tuple[object, object]] = [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),  # key order
        (0.0, -0.0),
        (1, 1.0),
        (True, 1),
        ([1.0], (1.0,)),
        ("x", b"x"),
        (Unsure(Cause(CauseCode.required, ["d:x"])), Unasked(Cause(CauseCode.required, ["d:x"]))),
        (
            Answer("noul", ("yes", "no"), {"yes": 0.4, "no": 0.6}),
            Answer("noul", ("yes", "no"), {"yes": 0.5, "no": 0.5}),
        ),
        (object(), object()),  # a type of no known shape is alike only to itself
    ]
    assert all(_same(*pair) for pair in alike)
    assert not any(_same(*pair) for pair in unlike)


class Abort(BaseException):
    """Ends a run as a cancellation would: no rule's outcome, past ``call``'s guard."""


ABORT = [False]


def pre(r: Any) -> object:
    CALLS[0] += 1
    if ABORT[0]:
        raise Abort
    return r.read("t:state.n")


def abstain(r: Any) -> object:
    return Unsure(Cause(CauseCode.abstained))


def stopped(r: Any) -> object:
    r.read("d:pre")
    return r.require("d:gone")


def sneaking(r: Any) -> object:
    r.read("d:pre")
    return r.read("d:boom")  # not declared


def boom(r: Any) -> object:
    raise ValueError(r.read("d:pre"))


def branches() -> LoadedDesign:
    def rule(ref: str, fn: Any, reads: list[str]) -> Rule:
        return Rule(ref, t.number(), fn, version="1", reads=reads)

    return load(
        Design(
            name="branches",
            state=STATE_T,
            sources={},
            rules={
                "pre": rule("d:pre", pre, ["t:state.n"]),
                "gone": rule("d:gone", abstain, []),
                "stop": rule("d:stop", stopped, ["d:pre", "d:gone"]),
                "sneak": rule("d:sneak", sneaking, ["d:pre"]),
                "boom": rule("d:boom", boom, ["d:pre"]),
                "act": Rule(
                    "d:act",
                    t.string,
                    lambda r: f"{r.read('d:stop')} {r.read('d:sneak')} {r.param('g/k')}",
                    version="1",
                    reads=["d:stop", "d:sneak", "d:boom"],
                    params=["g/k"],
                ),
            },
            parameters={"g/k": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )


def test_a_kept_stop_raise_or_undeclared_read_is_judged_as_calling_it_would() -> None:
    loaded = branches()
    with pytest.warns(DefectWarning):
        log = decide_sync(loaded, {"text": "", "n": 2.0}, {}).record.log
    reuse = Reuse()
    for k in (0.5, 0.1, 0.9):
        design = with_parameters(loaded, {"g/k": k})
        full = _rerun(log, design, log)[0].log
        CALLS[0] = 0
        mine = _rerun(log, design, log, reuse=reuse)[0].log
        assert [encode(i) for i in mine] == [encode(i) for i in full]
        assert CALLS[0] == (1 if k == 0.5 else 0)


def test_a_reference_run_that_raises_keeps_nothing() -> None:
    loaded = branches()
    with pytest.warns(DefectWarning):
        log = decide_sync(loaded, {"text": "", "n": 2.0}, {}).record.log
    reuse = Reuse()
    ABORT[0] = True
    try:
        with pytest.raises(Abort):
            _rerun(log, loaded, log, reuse=reuse)
    finally:
        ABORT[0] = False
    assert reuse.reference is None
    other = with_parameters(loaded, {"g/k": 0.1})
    CALLS[0] = 0
    mine = _rerun(log, other, log, reuse=reuse)[0].log  # the reference now
    assert CALLS[0] == 1
    assert reuse.reference is other
    assert [encode(i) for i in mine] == [encode(i) for i in _rerun(log, other, log)[0].log]


def test_every_entry_a_view_reads_is_one_the_reuse_test_compares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each read view's board is wrapped to note every key it reads; each rule's notes must lie
    # within what ``reached`` gives for that firing, over every design and record.
    from ommat import _interpret

    read: list[str] = []
    real_view = _interpret.ReadView

    class Noting(dict[str, object]):
        def __getitem__(self, key: str) -> object:
            read.append(key)
            return super().__getitem__(key)

        def get(self, key: str, default: object = None) -> object:  # type: ignore[override]
            read.append(key)
            return super().get(key, default)

    def view(role: Any, reads: Any, params: Any, board: Any, *a: Any, **k: Any) -> Any:
        noted = Noting(board)
        return real_view(role, reads, params, noted, *a, **k)

    monkeypatch.setattr(_interpret, "ReadView", view)
    real_outcome = _interpret._Turn.outcome  # pyright: ignore[reportPrivateUsage]
    firings = [0]

    def outcome(self: Any, rid: str, r: Any, current: Any) -> Any:
        read.clear()
        found = real_outcome(self, rid, r, current)
        if read:  # called, not reused
            firings[0] += 1
            reached = set(self.reached(rid, r, current))
            assert set(read) <= reached, (rid, set(read) - reached)
        return found

    monkeypatch.setattr(_interpret._Turn, "outcome", outcome)  # pyright: ignore[reportPrivateUsage]
    for record in RECORDS:
        for loaded in DESIGNS[::7]:
            _rerun(record.log, loaded, record.log, design=LOADED)
    assert firings[0] > 0


# ---------------------------------------------------------------------------------------------
# A design's shape, the document but for its leaves' values, kept once per design hash


def test_a_job_computes_each_design_s_shape_once_across_its_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ommat import _interpret

    class Counting(dict[str, bytes]):
        stored = 0

        def __setitem__(self, key: str, value: bytes) -> None:
            Counting.stored += 1
            super().__setitem__(key, value)

    monkeypatch.setattr(_interpret, "_SHAPES", Counting())
    # Each design loaded afresh from its document, so no two share their parts and the shape
    # alone tells that they differ only in their leaves' values.
    fresh = [
        load(d.document, functions=LOADED._callables)  # pyright: ignore[reportPrivateUsage]
        for d in DESIGNS[:6]
    ]
    rows = rescore_sync(RECORDS, LOADED, fresh)
    assert Counting.stored == len({d.design_hash for d in fresh})  # not once per record
    monkeypatch.setattr(_rescore, "Reuse", _Never)
    theirs = rescore_sync(RECORDS, LOADED, fresh)
    assert [encode(row) for row in rows] == [encode(row) for row in theirs]


def test_two_loads_of_one_document_share_one_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    from ommat import _interpret

    monkeypatch.setattr(_interpret, "_SHAPES", {})
    again = load(LOADED.document, functions=LOADED._callables)  # pyright: ignore[reportPrivateUsage]
    assert again is not LOADED
    assert again.design_hash == LOADED.design_hash
    assert _interpret.design_shape(again) is _interpret.design_shape(LOADED)
    # A leaf's value is dropped from the shape; its type and every other part are kept.
    valued = load(with_parameters(LOADED, {"g/word": "bye"}).document, functions=again._callables)  # pyright: ignore[reportPrivateUsage]
    assert valued.design_hash != LOADED.design_hash
    assert _interpret.design_shape(valued) == _interpret.design_shape(LOADED)
    assert _interpret.design_shape(swept(version="2")) != _interpret.design_shape(LOADED)
    assert len(_interpret._SHAPES) == 3  # pyright: ignore[reportPrivateUsage]


def test_the_shape_cache_drops_its_oldest_entry_past_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ommat import _interpret

    monkeypatch.setattr(_interpret, "_SHAPES", {})
    monkeypatch.setattr(_interpret, "_SHAPES_HELD", 2)
    for d in DESIGNS[:4]:
        _interpret.design_shape(d)
    held = list(_interpret._SHAPES)  # pyright: ignore[reportPrivateUsage]
    assert held == [d.design_hash for d in DESIGNS[2:4]]
