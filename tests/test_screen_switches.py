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
    Rule,
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


def test_the_window_is_clipped_to_the_leafs_interval_below() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [[">= lim/c", "big"], ["-", "small"]])
    loaded, record = deciding(table, **{"lim/c": Parameter(t.number(8.5, 100), 9.0)})
    assert row(record, loaded, "lim/c", 3.0)["clear"]  # [8.5, 12]: 7 lies below 8.5


def both(r: Any) -> object:
    r.read("d:b")
    return r.read("d:a")


@pytest.mark.parametrize("own", [5.0, 9.0])
def test_one_value_two_components_compare_on_two_sides_is_near_at_either_end(own: float) -> None:
    # 7 at an end: d:a's >= joins one side, d:b's > the other, so 7 parts the window.
    a = Table("d:a", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "small"]])
    b = Table("d:b", BIG_SMALL, "F", ["t:state.n"], [["> lim/n", "big"], ["-", "small"]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"b": b, "a": a},
            parameters={"lim/n": Parameter(t.number(), own)},
            policy=Rule("d:act", BIG_SMALL, both, version="1", reads=["d:a", "d:b"]),
        )
    )
    record = decide_sync(loaded, STATE, {}).record
    got = row(record, loaded, "lim/n", 2.0)
    assert got["near"] == [{"key": None, "value": 7.0, "components": ["a", "b"]}]


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


def test_each_gap_between_cuts_is_tried_at_an_integer_and_a_value_that_is_none() -> None:
    floats = screen_module._tried([Fraction(0), Fraction(1, 2), Fraction(3)], integer=False)
    assert floats[0] == 0
    assert floats[-1] == 3
    assert Fraction(1) in floats  # the integer in (1/2, 3)
    assert any(0 < x < Fraction(1, 2) for x in floats)
    assert any(Fraction(1, 2) < x < 3 and x.denominator != 1 for x in floats)
    for a, b in ((0, 5), (0, 2), (4, 2000)):  # (0, 2) and (4, 2000) have integral midpoints
        tried = screen_module._tried([Fraction(a), Fraction(b)], integer=False)
        assert any(a < x < b and x.denominator == 1 for x in tried)
        assert any(a < x < b and x.denominator != 1 for x in tried)
    assert Fraction(1, 3) not in screen_module._tried([Fraction(1, 3), Fraction(2)], integer=False)
    ints = screen_module._tried([Fraction(1, 2), Fraction(5, 2), Fraction(3)], integer=True)
    assert ints == [Fraction(1), Fraction(3)]  # the 1/2 and 5/2 cuts hold no integer


def table_of(
    hit: str, rows: list[Any], read: str, **parameters: Parameter
) -> tuple[LoadedDesign, Record]:
    return deciding(Table("d:tab", BIG_SMALL, hit, [read], rows), **parameters)


@pytest.mark.parametrize("hit", ["U", "A"])
def test_rows_of_a_u_or_a_table_that_come_to_overlap_do_not_load(hit: str) -> None:
    # "< lim/a" and ">= lim/b" overlap once lim/a passes lim/b, 6.
    loaded, record = table_of(
        hit,
        [["< lim/a", "big"], [">= lim/b", "small"]],
        "t:state.n",
        **{"lim/a": Parameter(t.number(), 5.0), "lim/b": Parameter(t.number(), 6.0)},
    )
    got = row(record, loaded, "lim/a", 4.0)  # [1, 9]
    values = [n["value"] for n in got["not_loading"]]
    assert 9.0 in values
    assert all(v > 6 for v in values)
    assert not got["clear"]


@pytest.mark.parametrize("cell", [">= 60", 60, {"one_of": [59, 60]}], ids=["ge", "is", "one-of"])
def test_a_literal_a_u_table_compares_the_leaf_with_is_a_cut(cell: Any) -> None:
    loaded, record = table_of(
        "U",
        [["< lim/i", "big"], [cell, "small"]],
        "t:state.k",
        **{"lim/i": Parameter(t.integer(), 5)},
    )
    values = {n["value"] for n in row(record, loaded, "lim/i", 100)["not_loading"]}  # [-95, 105]
    assert 61 in values  # from 61 the rows share 60
    assert 55 not in values


def test_a_type_bound_is_a_cut() -> None:
    # n is in [0, 4.25]: "> lim/a" and "< lim/b" share a value while lim/a is below 4.25.
    table = Table(
        "d:tab", BIG_SMALL, "U", ["t:state.n"], [["> lim/a", "big"], ["< lim/b", "small"]]
    )
    loaded = load(
        designed(
            state=t.record(n=t.number(0, 4.25)),
            rules={"tab": table},
            parameters={
                "lim/a": Parameter(t.number(), 10.0),
                "lim/b": Parameter(t.number(), 100.0),
            },
            policy=policy_rule(passed, "d:tab", outputs=BIG_SMALL),
        )
    )
    record = decide_sync(loaded, {"n": 4.0}, {}).record
    values = [n["value"] for n in row(record, loaded, "lim/a", 8.0)["not_loading"]]  # [2, 18]
    assert any(4 < v < 4.25 for v in values)
    assert all(v < 4.25 for v in values)


def u_over(state_type: Any, state: Any, **values: float) -> tuple[LoadedDesign, Record]:
    """A ``U`` table of ``> lim/a`` and ``< lim/b`` over ``t:state.n`` of ``state_type``."""
    table = Table(
        "d:tab", BIG_SMALL, "U", ["t:state.n"], [["> lim/a", "big"], ["< lim/b", "small"]]
    )
    loaded = load(
        designed(
            state=state_type,
            rules={"tab": table},
            parameters={k: Parameter(t.number(), v) for k, v in values.items()},
            policy=policy_rule(passed, "d:tab", outputs=BIG_SMALL),
        )
    )
    return loaded, decide_sync(loaded, state, {}).record


def test_a_nullable_inputs_bound_is_a_cut() -> None:
    loaded, record = u_over(
        t.record(n=t.nullable(t.number(0, 4.25))), {"n": 4.0}, **{"lim/a": 10.0, "lim/b": 100.0}
    )
    values = [n["value"] for n in row(record, loaded, "lim/a", 8.0)["not_loading"]]  # [2, 18]
    assert any(4 < v < 4.25 for v in values)


def test_a_types_low_bound_is_a_cut() -> None:
    # n is in [2.5, 9]: "> 1" and "< lim/b" share a value while lim/b is above 2.5.
    loaded, record = u_over(
        t.record(n=t.number(2.5, 9)), {"n": 4.0}, **{"lim/a": 1.0, "lim/b": 2.0}
    )
    values = [n["value"] for n in row(record, loaded, "lim/b", 3.0)["not_loading"]]  # [-1, 5]
    assert any(2.5 < v < 3 for v in values)


def test_a_huge_window_on_a_number_axis_stays_within_the_floats() -> None:
    loaded, record = u_over(STATE_T, STATE, **{"lim/a": 10.0, "lim/b": 1.0})
    for width in (1e308, 10**400):
        values = [n["value"] for n in row(record, loaded, "lim/a", width)["not_loading"]]
        assert values
        assert all(math.isfinite(v) for v in values)


def test_a_u_table_not_naming_the_leaf_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    u = Table("d:tab", BIG_SMALL, "U", ["t:state.n"], [["> lim/a", "big"], ["< lim/b", "small"]])
    other = Table("d:o", BIG_SMALL, "F", ["t:state.n"], [[">= lim/z", "big"], ["-", "small"]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"tab": u, "o": other},
            parameters={
                "lim/a": Parameter(t.number(), 10.0),
                "lim/b": Parameter(t.number(), 1.0),
                "lim/z": Parameter(t.number(), 50.0),
            },
            policy=policy_rule(passed, "d:tab", outputs=BIG_SMALL),
        )
    )
    record = decide_sync(loaded, STATE, {}).record

    def loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("loaded")

    monkeypatch.setattr(screen_module, "with_parameters", loads)
    assert row(record, loaded, "lim/z", 1000.0)["not_loading"] == []


def test_an_integer_input_overlap_changes_two_from_the_other_bound() -> None:
    # "> lim/a" and "< lim/b" share an integer while lim/a is at most 48, two below lim/b.
    loaded, record = table_of(
        "U",
        [["> lim/a", "big"], ["< lim/b", "small"]],
        "t:state.k",
        **{"lim/a": Parameter(t.integer(), 100), "lim/b": Parameter(t.integer(), 50)},
    )
    values = {n["value"] for n in row(record, loaded, "lim/a", 100)["not_loading"]}  # [0, 200]
    assert 48 in values
    assert 49 not in values


@pytest.mark.parametrize(
    "rows",
    [[["[lim/c, lim/c]", "big"], ["-", "small"]], [[">= lim/c", "big"], ["<= lim/c", "small"]]],
)
@pytest.mark.parametrize("hit", ["U", "A"])
def test_a_number_leaf_meeting_itself_over_an_integer_input_fails_at_an_integer(
    hit: str, rows: list[Any]
) -> None:
    # The leaf meets itself only at an integer over k, so 1000 fails though no number is near.
    loaded, record = table_of(hit, rows, "t:state.k", **{"lim/c": Parameter(t.number(), 1000.5)})
    got = row(record, loaded, "lim/c", 0.7)
    assert not got["clear"]
    assert {"key": None, "value": 1000.0} in got["not_loading"]
    with pytest.raises(InvalidDesign):
        with_parameters(loaded, {"lim/c": 1000.0})


def test_values_between_a_cut_and_the_window_ends_are_tried() -> None:
    table = Table(
        "d:tab", BIG_SMALL, "F", ["t:state.n"], [["[lim/lo, lim/hi]", "big"], ["-", "small"]]
    )
    loaded, record = deciding(
        table, **{"lim/lo": Parameter(t.number(), 6.0), "lim/hi": Parameter(t.number(), 7.5)}
    )
    values = [n["value"] for n in row(record, loaded, "lim/lo", 10.0)["not_loading"]]  # [-4, 16]
    assert 16.0 in values
    assert any(10 < v < 16 for v in values)
    values = [
        n["value"] for n in row(record, loaded, "lim/hi", 10.0)["not_loading"]
    ]  # [-2.5, 17.5]
    assert -2.5 in values
    assert any(-2.5 < v < 4 for v in values)


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


def u_cell() -> tuple[LoadedDesign, Record]:
    return table_of(
        "U",
        [["< lim/a", "big"], [">= lim/b", "small"]],
        "t:state.n",
        **{"lim/a": Parameter(t.number(), 5.0), "lim/b": Parameter(t.number(), 7.5)},
    )


CASES: list[tuple[str, Any, str, float, bool]] = [
    ("ge", lambda: cell(">=", 5.0), "lim/c", 2.0, True),
    ("ge-near", lambda: cell(">=", 5.0), "lim/c", 2.5, False),
    ("gt", lambda: cell(">", 5.0), "lim/c", 2.0, False),
    ("le", lambda: cell("<=", 9.0), "lim/c", 2.0, True),
    ("lt", lambda: cell("<", 9.0), "lim/c", 2.0, False),
    (
        "three-way",
        lambda: (three_way(5.0), decide_sync(three_way(5.0), STATE, {}).record),
        "lim/n",
        1.9,
        True,
    ),
    ("band", band, "lim/y", 1, True),
    ("band-wide", band, "lim/y", 6, False),
    ("u-table", u_cell, "lim/a", 1.5, True),
    ("u-table-wide", u_cell, "lim/a", 3.0, False),
    ("size", lambda: (TABLES, tables_record()), "lim/s", 1, True),
    ("bar", lambda: (MOOD, mood_record()), "mood/bar", 0.15, True),
    ("bar-at", lambda: (MOOD, mood_record()), "mood/bar", 0.2, False),
    ("margin", lambda: (MOOD, mood_record()), "mood/m", 0.15, True),
    ("margin-at", lambda: (MOOD, mood_record()), "mood/m", 0.2, True),
    ("mass", lambda: (MOOD, mood_record()), "mood/mbar", 0.15, True),
    ("mass-at", lambda: (MOOD, mood_record()), "mood/mbar", 0.2, True),
]


@pytest.mark.parametrize(
    ("name", "make", "path", "width", "clear"), CASES, ids=[c[0] for c in CASES]
)
def test_a_clear_record_writes_its_own_entries_at_every_value_tried_in_the_window(
    name: str, make: Any, path: str, width: float, clear: bool
) -> None:
    loaded, record = make()
    assert row(record, loaded, path, width)["clear"] is clear, name
    if clear:
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
