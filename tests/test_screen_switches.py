# pyright: reportPrivateUsage=false
"""The switch screen: a record is clear exactly when no value it compares a leaf against parts
the window, the leaf reaches nothing unseen, fills no wording and loads across the window; it
replays nothing; a clear record decides as recorded at every value of a dense sweep of the
window; and its refusals."""

from fractions import Fraction
from typing import Any

import pytest

from ommat import (
    EntryRefusal,
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
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _diff,
    _grid,
    _switching,
    rescore_sync,
    screen_switches,
)
from ommat.evaluation import _screen as screen_module
from test_switching_values import (
    BIG_SMALL,
    MOOD,
    STATE,
    TABLES,
    designed,
    mood_design,
    mood_record,
    one_table,
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


def cell(op: str, own: float) -> tuple[LoadedDesign, Record]:
    """A table comparing ``t:state.n``, 7.0 in ``STATE``, against ``lim/c`` by ``op``."""
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [[f"{op} lim/c", "big"], ["-", "small"]])
    return one_table(table, **{"lim/c": Parameter(t.number(), own)})


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


def test_a_zero_window_is_clear_at_a_compared_value() -> None:
    loaded, record = cell(">", 7.0)
    assert row(record, loaded, "lim/c", 0)["clear"]


def test_an_integer_window_holds_the_integers_within_it() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.k"], [["> lim/i", "big"], ["-", "small"]])
    loaded, record = one_table(table, **{"lim/i": Parameter(t.integer(), 2)})  # k is 4
    assert row(record, loaded, "lim/i", 1.9)["clear"]  # [1, 3]
    got = row(record, loaded, "lim/i", 2.0)  # [0, 4]: at 4, 4 > 4 fails
    assert got["near"] == [{"key": None, "value": 4, "components": ["tab"]}]
    assert type(got["near"][0]["value"]) is int


def test_the_window_is_clipped_to_the_leafs_interval() -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [["> lim/u", "big"], ["-", "small"]])
    loaded, record = one_table(table, **{"lim/u": Parameter(t.number(0, 6.5), 6.0)})
    assert row(record, loaded, "lim/u", 2.0)["clear"]  # 7 is past the interval's 6.5


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
    wild = row(mood_record(), MOOD, "mood/th.wild", 0.11)
    assert wild["near"] == [{"key": "wild", "value": 0.7, "components": ["th"]}]
    assert wild["unseen"] == ["thf"]


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


def test_a_leaf_no_check_reads_is_never_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    def loads(*args: object, **kwargs: object) -> object:
        raise AssertionError("loaded")

    monkeypatch.setattr(screen_module, "with_parameters", loads)
    loaded, record = cell(">=", 5.3)
    assert "lim/c" in loaded._read_leaves  # check 9 reads a cell's leaf
    assert row(record, loaded, "lim/c", 0.2)["clear"]  # no other number, no integer, in [5.1, 5.5]
    loaded3 = three_way(5.3)
    record3 = decide_sync(loaded3, STATE, {}).record
    assert row(record3, loaded3, "lim/n", 0.2)["clear"]


def test_rows_are_record_major_with_paths_in_code_point_order() -> None:
    loaded, record = band()
    rows = screen_switches([record, record], loaded, {"lim/y": 1, "lim/x": 1})
    assert [(r["record"], r["parameter"]) for r in rows] == [
        (record.record_hash, "lim/x"),
        (record.record_hash, "lim/y"),
    ] * 2
    assert screen_switches([], loaded, {"lim/y": 1}) == []
    assert screen_switches([record], loaded, {}) == []


# ---------------------------------------------------------------------------------------------
# A clear record decides as recorded across the window


def sweep(loaded: LoadedDesign, path: str, width: float) -> list[LoadedDesign]:
    """``loaded`` at 41 evenly spaced values of the leaf across the window, ends included."""
    own = Fraction(loaded._leaf_value(path))
    out: list[LoadedDesign] = []
    for i in range(41):
        document = decode(loaded.document)
        value = float(own - Fraction(width) + Fraction(width) * i / 20)
        document["parameters"][path]["value"] = value
        out.append(load(document))
    return out


CASES: list[tuple[str, Any, str, float]] = [
    ("ge", lambda: cell(">=", 5.0), "lim/c", 2.0),
    ("gt", lambda: cell(">", 5.0), "lim/c", 2.0),
    ("le", lambda: cell("<=", 9.0), "lim/c", 2.0),
    ("lt-far", lambda: cell("<", 9.0), "lim/c", 1.5),
    ("bar", lambda: (MOOD, mood_record()), "mood/bar", 0.15),
    ("bar-wide", lambda: (MOOD, mood_record()), "mood/bar", 0.25),
    ("margin", lambda: (MOOD, mood_record()), "mood/m", 0.15),
    ("margin-at", lambda: (MOOD, mood_record()), "mood/m", 0.2),
    ("mass", lambda: (MOOD, mood_record()), "mood/mbar", 0.15),
]


@pytest.mark.parametrize(("name", "make", "path", "width"), CASES, ids=[c[0] for c in CASES])
def test_a_clear_record_decides_as_recorded_at_every_value_of_the_window(
    name: str, make: Any, path: str, width: float
) -> None:
    loaded, record = make()
    got = row(record, loaded, path, width)
    designs = sweep(loaded, path, width)
    decisions = {repr(r["decision"]) for r in rescore_sync([record], loaded, designs)}
    if got["clear"]:
        assert len(decisions) == 1, name
    elif len(decisions) > 1:
        assert got["near"] or got["unseen"] or got["not_loading"], name


def test_the_sweep_finds_a_switch_where_the_screen_is_near() -> None:
    # The screen's converse is no promise, but here the near value is a switch.
    loaded, record = cell(">", 5.0)
    designs = sweep(loaded, "lim/c", 2.5)
    decisions = {repr(r["decision"]) for r in rescore_sync([record], loaded, designs)}
    assert len(decisions) == 1  # the policy is constant: d:tab's entry switches, not d:act
    assert not row(record, loaded, "lim/c", 2.5)["clear"]


# ---------------------------------------------------------------------------------------------
# Refusals


def test_loaded_must_be_a_loaded_design() -> None:
    with pytest.raises(EntryRefusal):
        screen_switches([], object(), {})  # type: ignore[arg-type]


@pytest.mark.parametrize("records", ["abc", b"abc", 3])
def test_records_must_be_a_sequence(records: Any) -> None:
    loaded, _ = cell(">=", 5.0)
    with pytest.raises(TypeError):
        screen_switches(records, loaded, {})


def test_within_must_be_a_mapping() -> None:
    loaded, record = cell(">=", 5.0)
    with pytest.raises(TypeError):
        screen_switches([record], loaded, [("lim/c", 1.0)])  # type: ignore[arg-type]


def test_a_record_of_another_design_is_wrong_design() -> None:
    loaded, _ = cell(">=", 5.0)
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([tables_record()], loaded, {"lim/c": 1.0})
    assert raised.value.reason is R.wrong_design


def test_a_format_7_record_is_wrong_design() -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([format_7()], TABLES, {"lim/n": 1.0})
    assert raised.value.reason is R.wrong_design


@pytest.mark.parametrize("path", ["nope", "lim", 3, "lim/n.x"])
def test_a_path_naming_no_leaf_is_unknown_path(path: Any) -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([tables_record()], TABLES, {path: 1.0})
    assert raised.value.reason is R.unknown_path


@pytest.mark.parametrize("path", ["sw/on", "sw/mode", "text", "tags", "names"])
def test_a_leaf_with_no_window_is_unsupported(path: str) -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        screen_switches([tables_record()], TABLES, {path: 1.0})
    assert raised.value.reason is R.unsupported_leaf
    assert raised.value.path == path


@pytest.mark.parametrize(
    ("width", "error"),
    [
        (True, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (-0.1, ValueError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
    ],
)
def test_a_width_is_a_finite_number_of_at_least_0(width: Any, error: type[Exception]) -> None:
    loaded, record = cell(">=", 5.0)
    with pytest.raises(error):
        screen_switches([record], loaded, {"lim/c": width})
