# pyright: reportPrivateUsage=false
"""The switch screen: a record is clear exactly when no value it compares a leaf against parts
the window, the leaf reaches nothing unseen, fills no wording and loads across the window; it
replays nothing; at every value of a dense sweep of a clear record's window, ends and the
floats about each compared value included, a rerun writes the record's own entries; the load
scan, the window's clipping, and the refusals in their order."""

import asyncio
import math
import time
from fractions import Fraction
from typing import Any

import pytest

import test_grounded_rules as grounded
from ommat import (
    EntryRefusal,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    Record,
    Table,
    Text,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._canonical import decode
from ommat._load import with_parameters
from ommat._types import _numeric
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _diff,
    _grid,
    _switching,
    screen_switches,
)
from ommat.evaluation import _screen as screen_module
from ommat.helpers._numbers import exact
from test_switching_values import (
    BIG_SMALL,
    MOOD,
    STATE,
    STATE_T,
    TABLES,
    designed,
    mood_design,
    mood_record,
    policy_rule,
    slot,
    tables_record,
    three_way,
)
from test_use_answers import format_7

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")

R = RefusalReason


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any replay or rerun fails the test: the screen reads records alone."""

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)
    monkeypatch.setattr(_switching, "_rerun", replayed)
    monkeypatch.setattr(_grid, "_rerun", replayed)


def row(record: Record, loaded: LoadedDesign, path: str, width: float) -> dict[str, Any]:
    (only,) = screen_switches([record], loaded, {path: width})
    return only


def passed(r: Any) -> object:
    return r.read("d:tab")


def deciding(
    table: Table, state: Any = None, **parameters: Parameter
) -> tuple[LoadedDesign, Record]:
    """A design whose policy decides ``d:tab``, ``table`` alone over ``STATE_T``, and its
    record over ``state``, ``STATE`` when ``None``."""
    loaded = load(
        designed(
            state=STATE_T,
            rules={"tab": table},
            parameters=parameters,
            policy=policy_rule(passed, "d:tab", outputs=BIG_SMALL),
        )
    )
    return loaded, decide_sync(loaded, STATE if state is None else state, {}).record


def cell(op: str, own: float) -> tuple[LoadedDesign, Record]:
    """A table comparing ``t:state.n``, 7.0 in ``STATE``, against ``lim/c`` by ``op``."""
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [[f"{op} lim/c", "big"], ["-", "small"]])
    return deciding(table, **{"lim/c": Parameter(t.number(), own)})


# ---------------------------------------------------------------------------------------------
# The rows


def test_a_record_with_no_compared_value_in_the_window_is_clear(no_replay: None) -> None:
    loaded, record = cell(">=", 5.0)
    got = row(record, loaded, "lim/c", 1.5)
    assert got == {
        "record": record.record_hash,
        "parameter": "lim/c",
        "within": 1.5,
        "clear": True,
        "near": [],
        "unseen": [],
        "worded": [],
        "not_loading": [],
    }


def test_a_compared_value_inside_the_window_is_near(no_replay: None) -> None:
    loaded, record = cell(">=", 5.0)
    got = row(record, loaded, "lim/c", 2.5)
    assert not got["clear"]
    assert got["near"] == [{"key": None, "value": 7.0, "components": ["tab"]}]
    assert type(got["near"][0]["value"]) is float  # a number leaf's, though integral


def test_the_width_comes_back_as_given() -> None:
    loaded, record = cell(">=", 5.0)
    got = row(record, loaded, "lim/c", 1)
    assert got["within"] == 1
    assert type(got["within"]) is int


@pytest.mark.parametrize(("op", "clear"), [(">=", True), ("<", True), (">", False), ("<=", False)])
def test_a_compared_value_at_the_windows_high_end_is_near_unless_its_tie_joins_below(
    op: str, clear: bool
) -> None:
    # The window is [3, 7]: at 7, >= and < give what they give below 7, > and <= do not.
    loaded, record = cell(op, 5.0)
    assert row(record, loaded, "lim/c", 2.0)["clear"] is clear


@pytest.mark.parametrize(("op", "clear"), [(">=", False), ("<", False), (">", True), ("<=", True)])
def test_a_compared_value_at_the_windows_low_end_is_near_unless_its_tie_joins_above(
    op: str, clear: bool
) -> None:
    loaded, record = cell(op, 9.0)  # the window is [7, 11]
    assert row(record, loaded, "lim/c", 2.0)["clear"] is clear


@pytest.mark.parametrize("own", [5.0, 9.0])
def test_a_value_compared_on_both_sides_is_near_at_either_end(own: float) -> None:
    # d:cmp compares 7 by > and by >=, so 7 parts the window at its high end and its low end.
    loaded = three_way(own)
    record = decide_sync(loaded, STATE, {}).record
    got = row(record, loaded, "lim/n", 2.0)
    assert got["near"] == [{"key": None, "value": 7.0, "components": ["cmp"]}]


def test_a_zero_window_is_clear_even_at_a_value_compared_on_both_sides() -> None:
    loaded = three_way(7.0)
    record = decide_sync(loaded, STATE, {}).record
    assert row(record, loaded, "lim/n", 0)["clear"]


def test_an_integer_window_holds_the_integers_within_it() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.k"], [["> lim/i", "big"], ["-", "small"]])
    loaded, record = deciding(table, **{"lim/i": Parameter(t.integer(), 2)})  # k is 4
    assert row(record, loaded, "lim/i", 1.9)["clear"]  # [1, 3]
    got = row(record, loaded, "lim/i", 2.0)  # [0, 4]: at 4, 4 > 4 fails
    assert got["near"] == [{"key": None, "value": 4, "components": ["tab"]}]
    assert type(got["near"][0]["value"]) is int


def test_an_integer_window_leaves_out_a_fractional_value_past_its_last_integer() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [["> lim/i", "big"], ["-", "small"]])
    loaded, record = deciding(table, {**STATE, "n": 6.5}, **{"lim/i": Parameter(t.integer(), 2)})
    assert row(record, loaded, "lim/i", 4.9)["clear"]  # [-2, 6]: 6.5 lies past 6
    assert [n["value"] for n in row(record, loaded, "lim/i", 5)["near"]] == [6.5]


def test_the_window_is_clipped_to_the_leafs_interval() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [["> lim/u", "big"], ["-", "small"]])
    loaded, record = deciding(table, **{"lim/u": Parameter(t.number(0, 6.5), 6.0)})
    assert row(record, loaded, "lim/u", 2.0)["clear"]  # 7 is past the interval's 6.5


def test_an_unbounded_integer_window_stops_at_the_largest_integer_a_leaf_holds() -> None:
    biggest = 2**53 - 1
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.k"], [["> lim/i", "big"], ["-", "small"]])
    loaded, record = deciding(table, **{"lim/i": Parameter(t.integer(), biggest - 1)})
    got = row(record, loaded, "lim/i", 5)
    assert got["clear"]
    assert got["not_loading"] == []
    with pytest.raises(InvalidDesign):
        with_parameters(loaded, {"lim/i": biggest + 1})  # past the window's end


def test_a_leaf_reached_otherwise_than_by_a_comparison_is_unseen(no_replay: None) -> None:
    got = row(tables_record(), TABLES, "lim/n", 0.1)
    assert not got["clear"]
    assert got["near"] == []
    # hid declares it with no boundaries; echo and part output Param of it.
    assert got["unseen"] == ["echo", "hid", "part"]


def test_a_mapping_leaf_screens_each_key_and_a_key_path_one(no_replay: None) -> None:
    # Each threshold, 0.6, meets each label's probability, calm's 0.7 and wild's 0.3.
    assert row(mood_record(), MOOD, "mood/th", 0.1)["near"] == []  # a tie at 0.7 holds
    whole = row(mood_record(), MOOD, "mood/th", 0.11)
    assert whole["near"] == [
        {"key": "calm", "value": 0.7, "components": ["th"]},
        {"key": "wild", "value": 0.7, "components": ["th"]},
    ]
    assert whole["unseen"] == ["thf"]  # it declares the mapping with no boundaries
    for key in ("calm", "wild"):
        one = row(mood_record(), MOOD, f"mood/th.{key}", 0.11)
        assert one["near"] == [{"key": key, "value": 0.7, "components": ["th"]}]
        assert one["unseen"] == ["thf"]


def test_a_pick_bar_is_near_at_a_labels_probability() -> None:
    assert row(mood_record(), MOOD, "mood/bar", 0.1)["clear"]  # 0.5: 0.3 and 0.7 lie outside
    # In [0.3, 0.7]: at 0.7 calm passes as inside; at 0.3 wild passes, as it does not inside.
    assert [n["value"] for n in row(mood_record(), MOOD, "mood/bar", 0.2)["near"]] == [0.3]
    got = row(mood_record(), MOOD, "mood/bar", 0.21)
    assert [n["value"] for n in got["near"]] == [0.3, 0.7]


def test_a_leaf_filling_a_wording_slot_names_its_source() -> None:
    loaded = mood_design(Text("Mood over {bar}?", [slot("bar", param="mood/bar")]))
    got = row(mood_record(loaded), loaded, "mood/bar", 0.1)
    assert not got["clear"]
    assert got["worded"] == ["ask"]


def test_a_candidate_table_whose_set_holds_no_value_is_unseen() -> None:
    rules = dict(grounded.design().rules)
    rules["level"] = Table(
        grounded.LEVEL,
        t.label("adult", "minor"),
        "F",
        ["candidate.age"],
        [[">= lim/age", "adult"], ["-", "minor"]],
        grounded=grounded.SET,
    )
    loaded = load(grounded.design(rules=rules, parameters={"lim/age": Parameter(t.integer(), 18)}))
    people = {"people": [grounded.BO, grounded.ANA]}
    held = decide_sync(loaded, people, {"default": grounded.answer}).record
    assert row(held, loaded, "lim/age", 1)["clear"]
    raising = {"people": [{"name": "raise", "age": 1}]}  # the set's rule raises
    marked = decide_sync(loaded, raising, {"default": grounded.answer}).record
    got = row(marked, loaded, "lim/age", 1)
    assert not got["clear"]
    assert got["unseen"] == ["level"]


def test_rows_are_record_major_with_paths_in_code_point_order() -> None:
    loaded, first = band()
    second = decide_sync(loaded, {"k": 5, "j": 9}, {}).record
    assert first.record_hash != second.record_hash
    rows = screen_switches([first, second, first], loaded, {"lim/y": 1, "lim/x": 1})
    assert [(r["record"], r["parameter"]) for r in rows] == [
        (h, p)
        for h in (first.record_hash, second.record_hash, first.record_hash)
        for p in ("lim/x", "lim/y")
    ]
    assert screen_switches([], loaded, {"lim/y": 1}) == []
    assert screen_switches([first], loaded, {}) == []


# ---------------------------------------------------------------------------------------------
# The load scan


def band() -> tuple[LoadedDesign, Record]:
    rules = {
        "tab": Table(
            "d:tab", BIG_SMALL, "F", ["t:state.k"], [["[lim/y, lim/x]", "big"], ["-", "small"]]
        ),
        "hi": Table("d:hi", BIG_SMALL, "F", ["t:state.j"], [[">= lim/y", "big"], ["-", "small"]]),
    }
    leaves = {"lim/x": Parameter(t.integer(), 7), "lim/y": Parameter(t.integer(), 2)}
    loaded = load(
        designed(state=t.record(k=t.integer(), j=t.integer()), rules=rules, parameters=leaves)
    )
    return loaded, decide_sync(loaded, {"k": 4, "j": 9}, {}).record


def test_a_value_in_the_window_that_does_not_load_is_named() -> None:
    loaded, record = band()
    got = row(record, loaded, "lim/y", 1)  # [1, 3]: every value loads
    assert got["clear"]
    got = row(record, loaded, "lim/y", 6)  # [-4, 8]: 8 is above lim/x
    assert got["not_loading"] == [{"key": None, "value": 8}]
    assert [n["value"] for n in got["near"]] == [4]


def test_a_number_leaf_past_its_between_bound_does_not_load() -> None:
    table = Table(
        "d:tab", BIG_SMALL, "F", ["t:state.n"], [["[lim/lo, lim/hi]", "big"], ["-", "small"]]
    )
    loaded, record = deciding(
        table,
        **{"lim/lo": Parameter(t.number(), 6.0), "lim/hi": Parameter(t.number(), 7.5)},
    )
    got = row(record, loaded, "lim/lo", 2.0)  # [4, 8]: above 7.5 the band is empty
    values = [n["value"] for n in got["not_loading"]]
    assert values  # the window's end, and a value between 7.5 and the integers past it
    assert all(v > 7.5 for v in values)
    assert 8.0 in values


@pytest.mark.parametrize("width", [0.2, 1000.0, 1e300, 10**400])
def test_a_table_no_load_check_reads_the_leaf_in_is_never_loaded(
    width: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("loaded")

    monkeypatch.setattr(screen_module, "with_parameters", loads)
    loaded, record = cell(">=", 5.3)
    assert (
        "lim/c" in loaded._read_leaves
    )  # check 9 reads a cell's leaf, but only for U, A and between
    assert row(record, loaded, "lim/c", width)["clear"] is (width < 1.7)


def test_the_load_scan_is_bounded_whatever_the_width() -> None:
    loaded, record = band()
    began = time.perf_counter()
    got = row(record, loaded, "lim/y", 10**30)
    assert time.perf_counter() - began < 5
    assert got["not_loading"]


def test_a_huge_number_in_a_wording_is_no_cut() -> None:
    wording = Text("Mood? ref 5e20000000")
    loaded = mood_design(wording)
    began = time.perf_counter()
    row(mood_record(loaded), loaded, "mood/bar", 0.1)
    assert time.perf_counter() - began < 5


def test_one_value_is_tried_in_each_gap_between_cuts() -> None:
    cuts = [Fraction(0), Fraction(1, 2), Fraction(3)]
    floats = screen_module._tried(cuts, integer=False)
    assert floats[0] == 0
    assert floats[-1] == 3
    assert any(0 < x < Fraction(1, 2) for x in floats)
    assert any(Fraction(1, 2) < x < 3 for x in floats)
    ints = screen_module._tried([Fraction(1, 2), Fraction(5, 2), Fraction(3)], integer=True)
    assert ints == [Fraction(1), Fraction(3)]  # 1 or 2 in the first gap; the 0.5 cut holds no int


# ---------------------------------------------------------------------------------------------
# A clear record decides as recorded across the window


def ulps(x: float, n: int = 2) -> set[float]:
    out, a, b = {x}, x, x
    for _ in range(n):
        a, b = math.nextafter(a, -math.inf), math.nextafter(b, math.inf)
        out |= {a, b}
    return out


def outcomes(log: list[dict[str, Any]]) -> dict[str, bytes]:
    return {e["ref"]: _diff.outcome(e) for e in log if e["item"] == "entry"}


def differing(record: Record, loaded: LoadedDesign, path: str, width: float) -> list[object]:
    """Each value tried in the window at which a rerun does not write the record's own
    entries, does not load, or asks a stale question: 41 evenly spaced values, the window's
    ends, and the floats within two of each value the record compares, on an integer leaf the
    integers."""
    leaf, _, key = path.partition(".")
    held = loaded._leaf_value(leaf)
    own = exact(held[key] if key else held)
    integer = _numeric(_grid._grid_type(path, loaded))[0] == "integer"  # type: ignore[index]
    lo, hi = own - exact(width), own + exact(width)
    found, _ = _switching._comparisons(
        decode(loaded.document), path, _switching._Entries(record.log)
    )
    values = {float(lo + (hi - lo) * i / 40) for i in range(41)} | {float(lo), float(hi)}
    for _, _, x in found:
        values |= ulps(float(x))
    if integer:
        values = {math.floor(v) + d for v in values for d in (0, 1)}
    mine = outcomes(record.log)
    out: list[object] = []
    for v in sorted(v for v in values if lo <= exact(v) <= hi):
        value = {**held, key: v} if key else v
        try:
            design = with_parameters(loaded, {leaf: value})
        except InvalidDesign:
            out.append(v)
            continue
        rerun, fixture = asyncio.run(
            _switching._rerun(record.log, design, record.log, design=loaded)
        )
        if fixture.stale or outcomes(rerun.log) != mine:
            out.append(v)
    return out


CASES: list[tuple[str, Any, str, float]] = [
    ("ge", lambda: cell(">=", 5.0), "lim/c", 2.0),
    ("ge-near", lambda: cell(">=", 5.0), "lim/c", 2.5),
    ("gt", lambda: cell(">", 5.0), "lim/c", 2.0),
    ("le", lambda: cell("<=", 9.0), "lim/c", 2.0),
    ("lt", lambda: cell("<", 9.0), "lim/c", 2.0),
    (
        "three-way",
        lambda: (three_way(5.0), decide_sync(three_way(5.0), STATE, {}).record),
        "lim/n",
        1.9,
    ),
    ("band", band, "lim/y", 1),
    ("band-wide", band, "lim/y", 6),
    ("bar", lambda: (MOOD, mood_record()), "mood/bar", 0.15),
    ("bar-at", lambda: (MOOD, mood_record()), "mood/bar", 0.2),
    ("margin", lambda: (MOOD, mood_record()), "mood/m", 0.15),
    ("margin-at", lambda: (MOOD, mood_record()), "mood/m", 0.2),
    ("mass", lambda: (MOOD, mood_record()), "mood/mbar", 0.15),
    ("mass-at", lambda: (MOOD, mood_record()), "mood/mbar", 0.2),
    ("threshold", lambda: (MOOD, mood_record()), "mood/th.calm", 0.1),
    ("threshold-near", lambda: (MOOD, mood_record()), "mood/th.calm", 0.11),
]


@pytest.mark.parametrize(("name", "make", "path", "width"), CASES, ids=[c[0] for c in CASES])
def test_a_clear_record_writes_its_own_entries_at_every_value_tried_in_the_window(
    name: str, make: Any, path: str, width: float
) -> None:
    loaded, record = make()
    if row(record, loaded, path, width)["clear"]:
        assert differing(record, loaded, path, width) == [], name


@pytest.mark.parametrize(
    ("make", "path", "width"),
    [
        (lambda: cell(">=", 5.0), "lim/c", 2.5),
        (lambda: (MOOD, mood_record()), "mood/th.calm", 0.11),
        (band, "lim/y", 6),
    ],
)
def test_the_sweep_sees_a_switch_where_the_screen_is_not_clear(
    make: Any, path: str, width: float
) -> None:
    loaded, record = make()
    assert not row(record, loaded, path, width)["clear"]
    assert differing(record, loaded, path, width)  # the sweep can fail the test above


# ---------------------------------------------------------------------------------------------
# Refusals, in order


def test_loaded_must_be_a_loaded_design() -> None:
    with pytest.raises(EntryRefusal):
        screen_switches("abc", object(), 3)  # type: ignore[arg-type]


@pytest.mark.parametrize("records", ["abc", b"abc", 3, (r for r in ())])
def test_records_must_be_a_sequence_before_within_is_checked(records: Any) -> None:
    loaded, _ = cell(">=", 5.0)
    with pytest.raises(TypeError, match="records"):
        screen_switches(records, loaded, 3)  # type: ignore[arg-type]


def test_within_must_be_a_mapping() -> None:
    loaded, record = cell(">=", 5.0)
    with pytest.raises(TypeError, match="within"):
        screen_switches([record], loaded, [("lim/c", 1.0)])  # type: ignore[arg-type]


def test_a_record_of_another_design_is_wrong_design() -> None:
    loaded, _ = cell(">=", 5.0)
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([tables_record()], loaded, {"lim/c": 1.0})
    assert raised.value.reason is R.wrong_design


@pytest.mark.parametrize("within", [{"lim/n": 1.0}, {}, {"nope": "x"}])
def test_records_are_refused_before_any_path(within: dict[str, Any]) -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([format_7()], TABLES, within)
    assert raised.value.reason is R.wrong_design


@pytest.mark.parametrize(
    ("path", "named"),
    [
        ("nope", "nope"),
        ("lim", "lim"),
        (3, None),
        ("lim/n.x", "lim/n.x"),
        ("mood/th.nope", "mood/th.nope"),
        ("mood/bar.x", "mood/bar.x"),
    ],
)
def test_a_path_naming_no_leaf_is_unknown_path(path: Any, named: str | None) -> None:
    record, loaded = (
        (mood_record(), MOOD) if str(path).startswith("mood") else (tables_record(), TABLES)
    )
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([record], loaded, {path: 1.0})
    assert raised.value.reason is R.unknown_path
    assert raised.value.path == named


@pytest.mark.parametrize("path", ["sw/on", "sw/mode", "text", "tags", "names"])
def test_a_leaf_with_no_window_is_unsupported(path: str) -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([tables_record()], TABLES, {path: 1.0})
    assert raised.value.reason is R.unsupported_leaf
    assert raised.value.path == path


@pytest.mark.parametrize(
    ("within", "error", "reason"),
    [
        ({"lim/n": "x", 3: 1}, EvaluationRefusal, R.unknown_path),  # a non-string path first
        ({"sw/on": "x"}, EvaluationRefusal, R.unsupported_leaf),  # the leaf before its width
        ({"lim/n": "x", "sw/on": 1}, TypeError, None),  # code-point order: lim/n first
        ({"sw/on": 1, "nope": 1}, EvaluationRefusal, R.unknown_path),  # nope before sw/on
    ],
)
def test_paths_and_widths_are_refused_in_order(
    within: dict[Any, Any], error: type[Exception], reason: RefusalReason | None
) -> None:
    with pytest.raises(error) as raised:
        screen_switches([tables_record()], TABLES, within)
    if reason is not None:
        assert isinstance(raised.value, EvaluationRefusal)
        assert raised.value.reason is reason


@pytest.mark.parametrize(
    ("width", "error"),
    [
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (Fraction(1), TypeError),
        (-0.1, ValueError),
        (-1, ValueError),
        (-(10**400), ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_a_width_is_a_finite_number_of_at_least_0(width: Any, error: type[Exception]) -> None:
    loaded, record = cell(">=", 5.0)
    with pytest.raises(error):
        screen_switches([record], loaded, {"lim/c": width})
