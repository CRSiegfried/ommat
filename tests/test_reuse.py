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
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast

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
    When,
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


# ---------------------------------------------------------------------------------------------
# A probe's rerun keeps its entries alone


def _varied(loaded: LoadedDesign) -> list[str]:
    """Each leaf of ``loaded`` the switching-value job varies."""
    out: list[str] = []
    for path in loaded._leaf_types:  # pyright: ignore[reportPrivateUsage]
        try:
            _switching._leaf_type(path, loaded)  # pyright: ignore[reportPrivateUsage]
        except _switching.EvaluationRefusal:
            continue
        out.append(path)
    return out


def test_a_probe_keeping_its_entries_alone_writes_what_a_whole_log_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import test_switching_values as sv
    from ommat._ledger import decision_bytes

    jobs: list[tuple[list[Record], LoadedDesign, list[str]]] = [
        (RECORDS, LOADED, ["g/bar", "g/stated", "g/count", "g/mode", "g/word"]),
        ([sv.mood_record()], sv.MOOD, _varied(sv.MOOD)),
        ([sv.tables_record()], sv.TABLES, _varied(sv.TABLES)),
    ]
    kept = [
        switching_values_batch_sync(
            records, loaded, paths, results=EvaluationStore(tmp_path / str(i))
        )
        for i, (records, loaded, paths) in enumerate(jobs)
    ]
    real = _switching._rerun  # pyright: ignore[reportPrivateUsage]
    probes = [0]

    async def whole(*a: Any, **k: Any) -> Any:
        k["keep"] = None
        out = await real(*a, **k)
        policy = a[1]._design.policy  # pyright: ignore[reportPrivateUsage]
        mine = _switching._decision_bytes(out[0].log, policy)  # pyright: ignore[reportPrivateUsage]
        assert mine == decision_bytes(out[0].log)
        probes[0] += 1
        return out

    monkeypatch.setattr(_switching, "_rerun", whole)
    theirs = [
        switching_values_batch_sync(
            records, loaded, paths, results=EvaluationStore(tmp_path / f"w{i}")
        )
        for i, (records, loaded, paths) in enumerate(jobs)
    ]
    assert probes[0] > 0
    assert [[encode(e) for e in job] for job in kept] == [
        [encode(e) for e in job] for job in theirs
    ]


def test_a_rerun_writing_no_unhashed_entry_skips_the_sum_check_and_writes_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ommat import _interpret

    checked = [0]
    real = _interpret.off_sum

    def counting(answer: Any) -> Any:
        checked[0] += 1
        return real(answer)

    monkeypatch.setattr(_interpret, "off_sum", counting)
    points = [{"g/bar": bar, "g/word": word} for bar in (0.1, 0.65) for word in LEAVES["g/word"]]
    paths = ["g/bar", "g/stated", "g/mode"]

    def jobs(where: str) -> list[bytes]:
        out = [encode(r) for r in rescore_sync(RECORDS, LOADED, DESIGNS[:9])]
        out += [encode(r) for r in rescore_grid_sync(RECORDS, LOADED, LOADED, points)]
        store = EvaluationStore(tmp_path / where)
        out += [
            encode(e) for e in switching_values_batch_sync(RECORDS, LOADED, paths, results=store)
        ]
        return out

    skipping = jobs("a")
    skipped = checked[0]
    monkeypatch.setattr(_switching._Light, "drops_unhashed", False)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(_rescore._Kept, "drops_unhashed", False)  # pyright: ignore[reportPrivateUsage]
    checked[0] = 0
    assert jobs("b") == skipping
    assert checked[0] > skipped  # the baseline replays' real ledger checks alone when skipping


# ---------------------------------------------------------------------------------------------
# A record's entries encoded once per record, and a probe's entry encoded only when it differs


def test_the_record_side_is_encoded_once_per_record_and_alike_entries_not_at_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = ["g/bar", "g/stated", "g/count", "g/mode", "g/word"]
    logs = [record.log for record in RECORDS]  # read once: each read decodes afresh
    held = {id(i) for log in logs for i in log}  # each record's own items
    encoded = {"record": 0, "probe": 0}
    real = _switching.outcome

    def counting(entry: Any) -> bytes:
        encoded["record" if id(entry) in held else "probe"] += 1
        return real(entry)

    monkeypatch.setattr(_switching, "outcome", counting)

    def checked(record: Record, loaded: LoadedDesign) -> tuple[Record, Any]:
        return record, logs[RECORDS.index(record)]

    def never(a: object, b: object) -> bool:
        return False

    monkeypatch.setattr(_switching, "_checked", checked)
    entries = switching_values_batch_sync(
        RECORDS, LOADED, paths, results=EvaluationStore(tmp_path / "a")
    )
    count = sum(1 for log in logs for i in log if i["item"] == "entry")
    assert encoded["record"] == count  # once per record, not once per path or probe
    alike = encoded["probe"]
    monkeypatch.setattr(_switching, "_alike_outcome", never)
    encoded["probe"] = 0
    theirs = switching_values_batch_sync(
        RECORDS, LOADED, paths, results=EvaluationStore(tmp_path / "b")
    )
    assert [encode(e) for e in entries] == [encode(e) for e in theirs]
    assert alike < encoded["probe"]


def test_alike_values_share_their_canonical_bytes() -> None:
    from ommat._canonical import decode

    values: list[object] = [
        None,
        True,
        False,
        0,
        1,
        -1,
        1.5,
        0.1,
        "",
        "x",
        "é\n",
        [],
        [1, 1.5],
        [1.5, 1],
        {},
        {"a": 1, "b": [True, None]},
        {"b": [True, None], "a": 1},
        {"a": 1.0},
        {"a": 1},
        {"a": True},
        [[{"x": "y"}]],
    ]
    for x in values:
        for y in values:
            a, b = decode(encode(x)), decode(encode(y))  # read back, as a log holds them
            if _switching._alike(a, b):  # pyright: ignore[reportPrivateUsage]
                assert encode(a) == encode(b), (x, y)
            if encode(a) == encode(b):
                assert _switching._alike(a, b), (x, y)  # pyright: ignore[reportPrivateUsage]
    # Alike decides nothing it cannot see as alike: a type read-back data never holds.
    one = [1]
    assert not _switching._alike(tuple(one), tuple(one))  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------------------------
# A probe's design built from its parameter object alone, its sources and shape the base's


def _checked_build(loaded: LoadedDesign, values: Any, seen: dict[str, int]) -> LoadedDesign:
    """``with_parameters``, each design it builds checked against the whole-document build and
    a fresh load of its document, its kept sources and shape against its document's."""
    from ommat import _interpret
    from ommat import _load as load_module
    from ommat._canonical import decode

    built = with_parameters(loaded, values)
    fresh = load(built.document, functions=loaded._callables)  # pyright: ignore[reportPrivateUsage]
    assert built.design_hash == fresh.design_hash
    written = {
        p: load_module.document_value(v, loaded._leaf_types[p])  # pyright: ignore[reportPrivateUsage]
        for p, v in values.items()
    }
    if not load_module._takes_a_load(loaded, values):  # pyright: ignore[reportPrivateUsage]
        spliced = load_module._spliced(loaded, values, written)  # pyright: ignore[reportPrivateUsage]
        whole = load_module._rebuilt(loaded, values, written)  # pyright: ignore[reportPrivateUsage]
        seen["spliced" if spliced is not None else "rebuilt"] += 1
        for name in LoadedDesign.__slots__:
            mine, theirs = getattr(built, name), getattr(whole, name)
            if name == "_design":
                assert mine.parameters == theirs.parameters
            elif isinstance(mine, Mapping):
                assert dict(cast("Mapping[str, Any]", mine)) == dict(theirs), name
            else:
                assert mine == theirs, name
    else:
        seen["loaded"] += 1
    doc = decode(built.document)
    assert _interpret.design_sources(built) == doc["sources"]
    doc["parameters"] = {p: leaf["type"] for p, leaf in doc["parameters"].items()}
    assert _interpret.design_shape(built) == encode(doc, 128)
    return built


def test_every_probe_design_hashes_as_a_fresh_load_of_its_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import test_switching_values as sv

    seen = {"spliced": 0, "rebuilt": 0, "loaded": 0}

    def checking(loaded: LoadedDesign, values: Any) -> LoadedDesign:
        return _checked_build(loaded, values, seen)

    monkeypatch.setattr(_switching, "with_parameters", checking)
    monkeypatch.setattr(_grid, "with_parameters", checking)
    jobs: list[tuple[list[Record], LoadedDesign, list[str]]] = [
        (RECORDS, LOADED, ["g/bar", "g/stated", "g/count", "g/mode", "g/word", "g/floor"]),
        ([sv.mood_record()], sv.MOOD, _varied(sv.MOOD)),
        ([sv.tables_record()], sv.TABLES, _varied(sv.TABLES)),
    ]
    for i, (records, loaded, paths) in enumerate(jobs):
        store = EvaluationStore(tmp_path / str(i))
        switching_values_batch_sync(records, loaded, paths, results=store)
    points = [
        {"g/bar": bar, "g/floor": floor} for bar in (0.1, 0.65) for floor in LEAVES["g/floor"]
    ]
    rescore_grid_sync(RECORDS, LOADED, LOADED, points)
    for values in itertools.product(*LEAVES.values()):
        _checked_build(LOADED, dict(zip(LEAVES, values, strict=True)), seen)
    assert seen["spliced"] > 0
    assert seen["rebuilt"] == 0  # every document here cuts around its parameters


# ---------------------------------------------------------------------------------------------
# A source's call taken from the reference run


WHENS = [0]
"""Each call of the counted ``when``."""


def counted_when(r: Any) -> bool:
    WHENS[0] += 1
    return r.read("t:state.n") > 0


def upstream(r: Any) -> float:
    return float(r.read("t:state.n") * r.param("g/k"))


def downstream_act(r: Any) -> str:
    said = [repr(r.read("q:first")), repr(r.read("q:down")), str(r.read("d:up"))]
    return " ".join(said) + f" {r.param('g/other')}"


def chained() -> LoadedDesign:
    return load(
        Design(
            name="chained",
            state=STATE_T,
            sources={
                "first": Source(
                    {"q:first": Question("noul", Text("First?"))},
                    needs=["t:state"],
                    when=When(counted_when, version="1"),
                    oracle="side",  # a batch of its own beside the down source's
                    cost=1,
                ),
                "down": Source({"q:down": Question("noul", Text("Down?"))}, needs=["d:up"], cost=1),
            },
            rules={
                "up": Rule(
                    "d:up",
                    t.number(),
                    upstream,
                    version="1",
                    reads=["t:state.n"],
                    params=["g/k"],
                ),
                "act": Rule(
                    "d:act",
                    t.string,
                    downstream_act,
                    version="1",
                    reads=["q:first", "q:down", "d:up"],
                    params=["g/other"],
                ),
            },
            parameters={
                "g/k": Parameter(t.number(0, 10), 1.0),
                "g/other": Parameter(t.number(0, 1), 0.5),
            },
            policy="d:act",
        )
    )


CHAINED = chained()


def chained_record() -> Record:
    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({ref: {"yes": 0.7, "no": 0.3} for c in batch for ref in c.questions})

    bindings = {"default": answer, "side": answer}
    return decide_sync(CHAINED, {"text": "hi", "n": 2.0}, bindings).record


def _asked_sources(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Each fixture call's sources, noted as the binding is called."""
    from ommat._fixture import FixtureBinding

    asked: list[list[str]] = []
    real = FixtureBinding.call

    async def noting(self: Any, batch: Any, seconds: float, live: Any) -> object:
        asked.append([s.id for s in batch])
        return await real(self, batch, seconds, live)

    monkeypatch.setattr(FixtureBinding, "call", noting)
    return asked


def test_a_source_reaching_what_the_reference_reached_takes_its_call_and_its_when_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = chained_record().log
    asked = _asked_sources(monkeypatch)
    reuse = Reuse()
    reference = _rerun(log, CHAINED, log, reuse=reuse)
    assert asked == [["down"], ["first"]]
    other = with_parameters(CHAINED, {"g/other": 0.2})  # read by the policy alone
    asked.clear()
    WHENS[0] = 0
    mine, fixture = _rerun(log, other, log, reuse=reuse)
    assert asked == []  # both calls taken from the reference
    assert WHENS[0] == 1  # the when is called afresh
    full, theirs = _rerun(log, other, log)
    assert [encode(i) for i in mine.log] == [encode(i) for i in full.log]
    assert fixture.stale == theirs.stale == {}
    assert reference[1].stale == {}


def test_a_leaf_changing_an_upstream_rule_asks_its_downstream_source_again_and_it_goes_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = chained_record().log
    asked = _asked_sources(monkeypatch)
    reuse = Reuse()
    _rerun(log, CHAINED, log, reuse=reuse)
    changed = with_parameters(CHAINED, {"g/k": 2.0})  # d:up, the down source's need, changes
    asked.clear()
    mine, fixture = _rerun(log, changed, log, reuse=reuse)
    assert asked == [["down"]]  # the first source's call is the reference's; down is asked
    assert set(fixture.stale) == {"q:down"}
    assert fixture.stale["q:down"][1].clause == "needs"
    full, theirs = _rerun(log, changed, log)
    assert [encode(i) for i in mine.log] == [encode(i) for i in full.log]
    assert fixture.stale == theirs.stale


def test_the_jobs_write_the_same_with_and_without_a_source_s_call_from_the_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ommat import _interpret

    asked = _asked_sources(monkeypatch)
    points = [
        {"g/bar": bar, "g/word": word, "g/floor": floor}
        for bar in (0.1, 0.65)
        for word in LEAVES["g/word"]
        for floor in LEAVES["g/floor"]
    ]
    paths = ["g/bar", "g/stated", "g/count", "g/mode", "g/word", "g/floor"]
    chained_log = [chained_record()]

    def jobs(where: str) -> list[bytes]:
        out = [encode(r) for r in rescore_sync(RECORDS, LOADED, DESIGNS)]
        out += [encode(r) for r in rescore_grid_sync(RECORDS, LOADED, LOADED, points)]
        store = EvaluationStore(tmp_path / where)
        out += [
            encode(e) for e in switching_values_batch_sync(RECORDS, LOADED, paths, results=store)
        ]
        store = EvaluationStore(tmp_path / f"{where}-chained")
        chained_paths = ["g/k", "g/other"]
        out += [
            encode(e)
            for e in switching_values_batch_sync(chained_log, CHAINED, chained_paths, results=store)
        ]
        return out

    reusing = jobs("a")
    calls = len(asked)

    def never(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(_interpret._Turn, "reusable", never)  # pyright: ignore[reportPrivateUsage]
    asked.clear()
    assert jobs("b") == reusing
    assert calls < len(asked)


# ---------------------------------------------------------------------------------------------
# A rebuilt design is a fresh load of its document


def _leafy() -> LoadedDesign:
    """A design holding leaves no check reads: mappings, lists and a scalar."""
    from designs import minimal

    return load(
        minimal(
            parameters={
                "z/map": Parameter(t.mapping(t.number(0, 1)), {"a": 0.1, "b": 0.2}),
                "a/map": Parameter(t.mapping(t.integer(0, 9)), {"": 1, "\U0001f600": 2}),
                "m/list": Parameter(t.list_(t.string), ["x"]),
                "k/n": Parameter(t.number(0, 1), 0.5),
            }
        )
    )


def _is_a_fresh_load(built: LoadedDesign, loaded: LoadedDesign) -> None:
    fresh = load(built.document, functions=loaded._callables)  # pyright: ignore[reportPrivateUsage]
    assert built.design_hash == fresh.design_hash
    assert built.document == fresh.document
    assert built._leaf_bytes == fresh._leaf_bytes  # pyright: ignore[reportPrivateUsage]
    assert built._grid == fresh._grid  # pyright: ignore[reportPrivateUsage]


def test_a_mass_labels_leaf_takes_a_load_and_is_a_fresh_load() -> None:
    import test_switching_values as sv

    path = "mood/labels"
    assert path in sv.MOOD._read_leaves  # pyright: ignore[reportPrivateUsage]
    built = with_parameters(sv.MOOD, {path: ["wild"]})
    _is_a_fresh_load(built, sv.MOOD)
    fresh = load(
        sv.MOOD.document.replace(b'"value":["calm"]', b'"value":["wild"]'),
        functions=sv.MOOD._callables,  # pyright: ignore[reportPrivateUsage]
    )
    assert built._grid == fresh._grid  # pyright: ignore[reportPrivateUsage]
    assert built.design_hash == fresh.design_hash
    assert built.document == fresh.document


def test_a_leaf_of_each_shape_splices_as_a_fresh_load_does() -> None:
    from ommat import _load as load_module

    base = _leafy()
    changes: list[dict[str, Any]] = [
        {"z/map": {"b": 0.9, "a": 0.3}},  # keys not in canonical order
        {"a/map": {"\U0001f600": 5, "": 4}},  # keys sorting differently in UTF-16
        {"m/list": ["y", "x", "y"]},
        {"k/n": 0.7, "m/list": []},
    ]
    for values in changes:
        assert not load_module._takes_a_load(base, values)  # pyright: ignore[reportPrivateUsage]
        written = {p: load_module.document_value(v, base._leaf_types[p]) for p, v in values.items()}  # pyright: ignore[reportPrivateUsage]
        assert load_module._spliced(base, values, written) is not None  # pyright: ignore[reportPrivateUsage]
        _is_a_fresh_load(with_parameters(base, values), base)


def test_chained_builds_are_a_fresh_load() -> None:
    base = _leafy()
    once = with_parameters(base, {"z/map": {"b": 0.9, "a": 0.3}, "m/list": ["q"]})
    twice = with_parameters(once, {"m/list": ["r", "s"], "k/n": 0.1})
    _is_a_fresh_load(twice, base)
    again = with_parameters(twice, {"z/map": {"a": 0.0, "b": 0.0}})
    _is_a_fresh_load(again, base)


@pytest.mark.parametrize("how", ["cut", "template", "spliced"])
def test_a_build_with_no_splice_is_the_whole_document_build(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    from ommat import _load as load_module

    base = _leafy()
    if how == "cut":
        monkeypatch.setattr(load_module, "_cut_document", lambda loaded: None)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    elif how == "template":
        monkeypatch.setitem(load_module._TEMPLATES, base.design_hash, None)  # pyright: ignore[reportPrivateUsage]
    else:
        monkeypatch.setattr(load_module, "_spliced", lambda *a: None)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    values = {"z/map": {"b": 0.9, "a": 0.3}, "m/list": ["y"], "k/n": 0.2}
    rebuilt: list[bool] = []
    real = load_module._rebuilt  # pyright: ignore[reportPrivateUsage]

    def watching(*a: Any) -> LoadedDesign:
        rebuilt.append(True)
        return real(*a)

    monkeypatch.setattr(load_module, "_rebuilt", watching)
    built = with_parameters(base, values)
    assert rebuilt
    _is_a_fresh_load(built, base)
