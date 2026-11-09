"""The grid form of the rescoring job: its rows equal plain ``rescore`` over the points' designs,
row for row, while each record is rerun once per region of points, not once per point; a leaf a
run could reach other than by a visible comparison is refused before any run."""

import json
from collections.abc import Sequence
from typing import Any

import pytest

import extraction as x
import test_rescore as tr
import test_switching_values as sv
from designs import parts, roles, slot
from ommat import (
    CallResult,
    Design,
    EntryRefusal,
    Helper,
    InvalidDesign,
    LoadedDesign,
    Param,
    Parameter,
    Question,
    Record,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Text,
    Undetermined,
    When,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._canonical import encode
from ommat._load import with_parameters
from ommat.evaluation import (
    EvaluationRefusal,
    RefusalReason,
    _grid,
    rescore_grid_sync,
    rescore_sync,
)
from ommat.helpers import mass

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")

YES_NO = 'label{"no", "yes"}'
_R = RefusalReason
Point = dict[str, Any]


def gated(bar: float, margin: float = 0.0, stated: float = 0.5) -> LoadedDesign:
    """``test_rescore``'s gated design with its rules' comparisons declared in ``boundaries``."""
    return load(
        Design(
            name="gated",
            state=tr.STATE_T,
            sources={
                "ask": Source(
                    {"q:intent": Question("choice", Text("Intent?"), labels=tr.INTENTS)},
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
                    t.label(*tr.INTENTS),
                    tr.intent,
                    version="1",
                    reads=["q:intent"],
                    params=["g/bar", "g/margin"],
                    boundaries={"q:intent": roles(bar=["g/bar"], margin=["g/margin"])},
                ),
                "act": Rule(
                    "d:act",
                    t.label("refund", "ask", "pass"),
                    tr.act,
                    version="1",
                    reads=["d:intent", "q:amount"],
                    params=["g/stated"],
                    boundaries={
                        "q:amount": roles(mass=[{"labels": ["yes"], "bounds": ["g/stated"]}])
                    },
                ),
            },
            parameters={
                "g/bar": Parameter(t.number(0, 1), bar),
                "g/margin": Parameter(t.number(0, 1), margin),
                "g/stated": Parameter(t.number(0, 1), stated),
            },
            policy="d:act",
        )
    )


def counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the grid's reruns in the list's one member."""
    count = [0]
    real = _grid._rerun  # pyright: ignore[reportPrivateUsage]

    async def rerun(*args: Any, **kwargs: Any) -> Any:
        count[0] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(_grid, "_rerun", rerun)
    return count


def plain(
    records: Sequence[Record], recorded: LoadedDesign, loaded: LoadedDesign, points: list[Point]
) -> list[dict[str, Any]]:
    """``rescore``'s rows over the points' designs, a key path's value merged into its leaf's."""
    return rescore_sync(records, recorded, [designed(loaded, p) for p in points])


def designed(loaded: LoadedDesign, point: Point) -> LoadedDesign:
    return with_parameters(loaded, _grid._leaf_values(loaded, point))  # pyright: ignore[reportPrivateUsage]


def same(
    records: Sequence[Record], recorded: LoadedDesign, loaded: LoadedDesign, points: list[Point]
) -> list[dict[str, Any]]:
    """The grid's rows, asserted equal to ``rescore``'s row for row: as plain data, by
    canonical bytes, and in type."""
    rows = rescore_grid_sync(records, recorded, loaded, points)
    theirs = plain(records, recorded, loaded, points)
    assert rows == theirs
    assert [encode(row) for row in rows] == [encode(row) for row in theirs]
    assert repr(rows) == repr(theirs)
    assert [row["design"] for row in rows] == [
        designed(loaded, p).design_hash for p in points for _ in records
    ]
    return rows


# ---------------------------------------------------------------------------------------------
# Parity with the plain job


def test_the_gated_grid_equals_plain_rescore_with_fewer_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = gated(0.6)
    loaded = gated(0.5, stated=0.7)  # the grid's base need not be the recorded design
    records = [tr.turn(recorded, *split) for split in tr.SPLITS]
    # Bars and margins at the records' own probabilities and leads, so ties are exercised.
    points = [
        {"g/bar": bar, "g/margin": margin}
        for bar in (*tr.BARS, 0.9, 0.2, 0.62)
        for margin in (0.0, 0.2, 0.8, 0.24)
    ]
    count = counted(monkeypatch)
    rows = same(records, recorded, loaded, points)
    assert len(rows) == len(points) * len(records)
    assert count[0] < len(points) * len(records)
    assert 0 < sum(row["decision"] != "not-determined" for row in rows) < len(rows)


def test_a_three_leaf_product_equals_plain_rescore(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = gated(0.5)
    records = [tr.turn(recorded, *split) for split in tr.SPLITS]
    points = [
        {"g/bar": bar, "g/margin": margin, "g/stated": stated}
        for bar in (0.3, 0.55, 0.9)
        for margin in (0.0, 0.3)
        for stated in (0.2, 0.6, 0.9)  # 0.2, 0.9 equal recorded masses: a mass tie is its own
    ]
    count = counted(monkeypatch)
    rows = same(records, recorded, recorded, points)
    assert count[0] < len(points) * len(records)
    assert len({encode(row["decision"]) for row in rows}) > 2


def mood_without_thf() -> LoadedDesign:
    """The mood table design without ``thf``, the function reading its mapping leaf unseen."""
    document = json.loads(sv.MOOD.document)
    del document["rules"]["thf"]
    return load(document, functions=sv.MOOD._callables)  # pyright: ignore[reportPrivateUsage]


def test_the_mood_tables_equal_plain_rescore_over_a_mapping_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = mood_without_thf()
    records = [sv.mood_record(loaded)]
    # The split is calm 0.7, wild 0.3, so its lead is 0.4: a margin at 0.4 is a tie the helper
    # holds at, and the thresholds at 0.7 and 0.3 are ties too.
    points = [
        {"mood/th": {"calm": calm, "wild": wild}, "mood/m": margin}
        for calm in (0.3, 0.6, 0.7, 0.8)
        for wild in (0.2, 0.3, 0.5)
        for margin in (0.2, 0.4, 0.5)
    ]
    count = counted(monkeypatch)
    rows = same(records, loaded, loaded, points)
    assert count[0] < len(points)
    assert len({encode(row["decision"]) for row in rows}) > 1


# ---------------------------------------------------------------------------------------------
# Two runs the record alone cannot predict


def pick_number(r: Any) -> float:
    """``r1`` when leaf A's table says yes, else ``r2``: which number meets leaf B's table
    depends on leaf A."""
    chosen = r.read("d:la")
    if not isinstance(chosen, Undetermined) and chosen == "yes":
        return r.read("t:state.r1")
    return r.read("t:state.r2")


def switcher() -> LoadedDesign:
    return load(
        Design(
            name="switcher",
            state=t.record(r1=t.number(), r2=t.number()),
            sources={
                "ask": Source(
                    {"q:x": Question("choice", Text("Which?"), labels=("hi", "lo"))},
                    needs=["t:state"],
                )
            },
            rules={
                "la": Table(
                    "d:la",
                    YES_NO,
                    "F",
                    [Helper("pick_with_margin", "q:x", bar="a/bar", margin="a/m")],
                    [["hi", "yes"], ["-", "no"]],
                ),
                "n": Rule(
                    "d:n",
                    t.number(),
                    pick_number,
                    version="1",
                    reads=["d:la", "t:state.r1", "t:state.r2"],
                ),
                "act": Table("d:act", YES_NO, "F", ["d:n"], [[">= b/bar", "yes"], ["-", "no"]]),
            },
            parameters={
                "a/bar": Parameter(t.number(0, 1), 0.5),
                "a/m": Parameter(t.number(0, 1), 0.0),
                "b/bar": Parameter(t.number(0, 1), 0.5),
            },
            policy="d:act",
        )
    )


def answered(loaded: LoadedDesign, state: dict[str, Any], given: dict[str, Any]) -> Record:
    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({ref: given[ref] for call in batch for ref in call.questions})

    return decide_sync(loaded, state, {"default": answer}).record


def test_a_label_switching_which_number_meets_the_other_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = switcher()
    record = answered(loaded, {"r1": 0.3, "r2": 0.8}, {"q:x": {"hi": 0.6, "lo": 0.4}})
    # Under leaf A at 0.4 or 0.6 the table says yes and 0.3 meets leaf B; at 0.7 it says no and
    # 0.8 does: the values compared against leaf B differ between A's regions.
    points = [{"a/bar": a, "b/bar": b} for a in (0.4, 0.6, 0.7) for b in (0.2, 0.3, 0.5, 0.8, 0.9)]
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, points)
    # Leaf A is compared against both labels' probabilities, 0.6 and 0.4, so each of its three
    # points is a region of its own, 0.4 and 0.6 ties on the low side; leaf B has two within each.
    assert count[0] == 6
    decisions = [row["decision"]["value"] for row in rows]
    assert decisions[:5] == ["yes", "yes", "no", "no", "no"]  # a 0.4: 0.3 against b
    assert decisions[10:] == ["yes", "yes", "yes", "yes", "no"]  # a 0.7: 0.8 against b


def act_when_both(r: Any) -> str:
    if r.read("d:both") != "yes":
        return "pass"
    found = mass(r, "q:c", ["yes"])
    if isinstance(found, Undetermined):
        return "ask"
    return "go" if found >= 0.5 else "ask"


def double_gate(a: float, b: float) -> LoadedDesign:
    """A source asked only when the gates on leaves A and B are both open."""

    def gate(ref: str, leaf: str, margin: str) -> Table:
        return Table(
            f"d:g{ref[-1]}",
            YES_NO,
            "F",
            [Helper("pick_with_margin", ref, bar=leaf, margin=margin)],
            [["yes", "yes"], ["-", "no"]],
        )

    return load(
        Design(
            name="double",
            state=tr.STATE_T,
            sources={
                "ask": Source(
                    {
                        "q:a": Question("choice", Text("A?"), labels=("yes", "no")),
                        "q:b": Question("choice", Text("B?"), labels=("yes", "no")),
                    },
                    needs=["t:state"],
                ),
                "more": Source(
                    {"q:c": Question("noul", Text("C?"))},
                    needs=["t:state"],
                    requirement=SourceRequirement("d:both", ["yes"]),
                ),
            },
            rules={
                "ga": gate("q:a", "a/bar", "a/m"),
                "gb": gate("q:b", "b/bar", "b/m"),
                "both": Table(
                    "d:both",
                    YES_NO,
                    "F",
                    ["d:ga", "d:gb"],
                    [["yes", "yes", "yes"], ["-", "-", "no"]],
                ),
                "act": Rule(
                    "d:act",
                    t.label("pass", "ask", "go"),
                    act_when_both,
                    version="1",
                    reads=["d:both", "q:c"],
                ),
            },
            parameters={
                "a/bar": Parameter(t.number(0, 1), a),
                "a/m": Parameter(t.number(0, 1), 0.0),
                "b/bar": Parameter(t.number(0, 1), b),
                "b/m": Parameter(t.number(0, 1), 0.0),
            },
            policy="d:act",
        )
    )


def test_a_source_behind_two_gates_is_not_determined_at_the_joint_point_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = double_gate(0.9, 0.9)  # both gates closed: q:c was never asked
    record = answered(
        recorded,
        tr.STATE,
        {"q:a": {"yes": 0.7, "no": 0.3}, "q:b": {"yes": 0.7, "no": 0.3}, "q:c": {"yes": 1.0}},
    )
    points = [{"a/bar": a, "b/bar": b} for a in (0.5, 0.6, 0.9) for b in (0.5, 0.6, 0.9)]
    count = counted(monkeypatch)
    rows = same([record], recorded, recorded, points)
    assert count[0] == 4  # one run per side of 0.7 on each leaf
    joint: tuple[Any, list[str]] = ("not-determined", ["q:c"])
    closed: tuple[Any, list[str]] = ({"value": "pass"}, [])
    assert [(row["decision"], row["stale"]) for row in rows] == [
        *([joint, joint, closed] * 2),
        *([closed] * 3),
    ]


# ---------------------------------------------------------------------------------------------
# The refusals, before any run


def refused(
    monkeypatch: pytest.MonkeyPatch,
    records: list[Any],
    recorded: Any,
    loaded: Any,
    points: Any,
) -> EvaluationRefusal:
    count = counted(monkeypatch)
    with pytest.raises(EvaluationRefusal) as caught:
        rescore_grid_sync(records, recorded, loaded, points)
    assert count[0] == 0
    return caught.value


def test_a_function_declaring_the_leaf_without_boundaries_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = tr.gated(0.6)  # its intent rule declares g/bar and has no boundaries field
    record = tr.turn(recorded, 0.9, 0.9)
    refusal = refused(monkeypatch, [record], recorded, recorded, [{"g/bar": 0.5}])
    assert refusal.reason is RefusalReason.hidden_leaf
    assert refusal.path == "g/bar"
    assert "intent" in str(refusal)


LIMIT_STATE = t.record(n=t.number())


def limit_design(policy: Any) -> LoadedDesign:
    return load(
        Design(
            name="limit",
            state=LIMIT_STATE,
            sources={},
            rules={"part": policy},
            parameters={"lim/n": Parameter(t.number(), 5.0)},
            policy="d:part",
        )
    )


def test_a_table_row_outputting_param_of_the_leaf_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = limit_design(
        Table("d:part", "number", "F", ["t:state.n"], [[">= lim/n", Param("lim/n")], ["-", 0.0]])
    )
    record = decide_sync(loaded, {"n": 7.0}, {}).record
    refusal = refused(monkeypatch, [record], loaded, loaded, [{"lim/n": 3.0}])
    assert refusal.reason is RefusalReason.hidden_leaf
    assert refusal.path == "lim/n"
    assert "part" in str(refusal)


def clipped(r: Any) -> float:
    return min(r.read("t:state.n"), r.param("lim/n"))


def test_a_function_of_a_numeric_type_declaring_the_leaf_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = limit_design(
        Rule(
            "d:part",
            t.number(),
            clipped,
            version="1",
            reads=["t:state.n"],
            params=["lim/n"],
            boundaries={"t:state.n": ["lim/n"]},  # named, yet its output can hold the leaf
        )
    )
    record = decide_sync(loaded, {"n": 7.0}, {}).record
    refusal = refused(monkeypatch, [record], loaded, loaded, [{"lim/n": 3.0}])
    assert refusal.reason is RefusalReason.hidden_leaf
    assert "part" in str(refusal)


# ---------------------------------------------------------------------------------------------
# A wording-slot leaf, dedupe and the arguments


def test_a_wording_slot_leaf_gives_each_value_its_own_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = sv.mood_design(Text("Mood over {bar}?", [slot("bar", param="mood/bar")]))
    records = [sv.mood_record(loaded)]
    points = [{"mood/bar": bar} for bar in (0.3, 0.5, 0.5, 0.7)]
    count = counted(monkeypatch)
    rows = same(records, loaded, loaded, points)
    assert count[0] == 3  # one per distinct value, the repeated 0.5 sharing its run
    assert [row["stale"] for row in rows] == [["q:mood"], [], [], ["q:mood"]]


def test_records_of_one_hash_are_rerun_once(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = gated(0.6)
    first, second = tr.turn(recorded, 0.9, 0.9), tr.turn(recorded, 0.55, 0.9)
    points = [{"g/bar": bar} for bar in (0.3, 0.5, 0.95)]
    count = counted(monkeypatch)
    distinct = same([first, second], recorded, recorded, points)
    once = count[0]
    rows = same([first, second, first], recorded, recorded, points)
    assert count[0] == 2 * once
    assert [row for row in rows if row["record"] == first.record_hash][::2] == [
        row for row in distinct if row["record"] == first.record_hash
    ]


def test_a_state_that_does_not_type_under_the_grid_gives_the_broken_entry() -> None:
    recorded = gated(0.5)
    record = tr.turn(recorded, 0.9, 0.9)
    document = json.loads(recorded.document)
    document["state"] = "record{text: integer}"
    loaded = load(document, functions=recorded._callables)  # pyright: ignore[reportPrivateUsage]
    rows = same([record], recorded, loaded, [{"g/bar": 0.3}, {"g/bar": 0.7}])
    assert all("marker" in row["decision"] for row in rows)


def test_the_arguments_are_refused_as_rescores_are() -> None:
    recorded = gated(0.5)
    record = tr.turn(recorded, 0.9, 0.9)
    with pytest.raises(EntryRefusal):
        rescore_grid_sync([record], object(), recorded, [{"g/bar": 0.3}])  # type: ignore[arg-type]
    with pytest.raises(EntryRefusal):
        rescore_grid_sync([record], recorded, object(), [{"g/bar": 0.3}])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rescore_grid_sync([record], recorded, recorded, "points")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rescore_grid_sync([record], recorded, recorded, [("g/bar", 0.3)])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        rescore_grid_sync("records", recorded, recorded, [{"g/bar": 0.3}])  # type: ignore[arg-type]


def test_the_records_are_checked_before_the_points(monkeypatch: pytest.MonkeyPatch) -> None:
    other = tr.turn(gated(0.7), 0.9, 0.9)
    refusal = refused(monkeypatch, [other], gated(0.5), gated(0.5), [{}])
    assert refusal.reason is RefusalReason.wrong_design


def test_the_points_are_refused_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = gated(0.5)
    record = tr.turn(recorded, 0.9, 0.9)
    assert rescore_grid_sync([record], recorded, recorded, []) == []
    uneven: list[list[Point]] = [[{}], [{"g/bar": 0.3}, {"g/margin": 0.1}], [{"g/bar": 0.3}, {}]]
    for points in uneven:
        refusal = refused(monkeypatch, [record], recorded, recorded, points)
        assert refusal.reason is RefusalReason.uneven_grid
    refusal = refused(monkeypatch, [record], recorded, recorded, [{"g/nope": 0.3}])
    assert refusal.reason is RefusalReason.unknown_path
    assert refusal.path == "g/nope"
    refusal = refused(monkeypatch, [record], recorded, recorded, [{"g/bar": 0.3}, {"g/bar": 2}])
    assert refusal.reason is RefusalReason.invalid_design
    assert isinstance(refusal.error, InvalidDesign)
    assert "points[1]" in str(refusal)


# ---------------------------------------------------------------------------------------------
# Ties: a grid point at a compared value, the first rerun elsewhere

TIE_STATE = t.record(n=t.number())
TIES = (0.4, 0.6, 0.8)  # the record's number is 0.6; the first rerun is at 0.4


def tie_table(cell: str) -> LoadedDesign:
    """A policy table with one cell comparing the state's number against the leaf; ``k/lo``
    and ``k/hi`` are the other bound of a ``between``."""
    return load(
        Design(
            name="tie",
            state=TIE_STATE,
            sources={},
            rules={"act": Table("d:act", YES_NO, "F", ["t:state.n"], [[cell, "yes"], ["-", "no"]])},
            parameters={
                "k/bar": Parameter(t.number(0, 1), 0.5),
                "k/lo": Parameter(t.number(0, 1), 0.0),
                "k/hi": Parameter(t.number(0, 1), 1.0),
            },
            policy="d:act",
        )
    )


def tie_rows(
    monkeypatch: pytest.MonkeyPatch, loaded: LoadedDesign, record: Record, reruns: int
) -> list[str]:
    """The decisions over ``TIES``, the grid asserted equal to ``rescore`` and rerun exactly
    ``reruns`` times."""
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, [{"k/bar": v} for v in TIES])
    assert count[0] == reruns
    return [row["decision"]["value"] for row in rows]


@pytest.mark.parametrize(
    ("cell", "decisions", "reruns"),
    [
        (">= k/bar", ["yes", "yes", "no"], 2),  # holds at the tie, as below it
        ("> k/bar", ["yes", "no", "no"], 2),  # fails at the tie, as above it
        ("<= k/bar", ["no", "yes", "yes"], 2),
        ("< k/bar", ["no", "no", "yes"], 2),
        ("[k/lo, k/bar]", ["no", "yes", "yes"], 2),  # the high bound holds at the tie
        ("[k/bar, k/hi]", ["yes", "yes", "no"], 2),  # and the low bound
    ],
)
def test_a_table_cells_tie_joins_the_side_its_comparison_holds_on(
    monkeypatch: pytest.MonkeyPatch, cell: str, decisions: list[str], reruns: int
) -> None:
    loaded = tie_table(cell)
    record = decide_sync(loaded, {"n": 0.6}, {}).record
    assert tie_rows(monkeypatch, loaded, record, reruns) == decisions


def over_mass(r: Any) -> str:
    found = mass(r, "q:m", ["yes"])
    if isinstance(found, Undetermined):
        return "no"
    return "yes" if found > r.param("k/bar") else "no"


M_SOURCE = Source({"q:m": Question("noul", Text("M?"))}, needs=["t:state"])


def test_a_function_rules_mass_tie_is_a_bucket_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load(
        Design(
            name="over",
            state=TIE_STATE,
            sources={"ask": M_SOURCE},
            rules={
                "act": Rule(
                    "d:act",
                    t.label("yes", "no"),
                    over_mass,
                    version="1",
                    reads=["q:m"],
                    params=["k/bar"],
                    boundaries={"q:m": roles(mass=[{"labels": ["yes"], "bounds": ["k/bar"]}])},
                )
            },
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )
    record = answered(loaded, {"n": 0.0}, {"q:m": {"yes": 0.6, "no": 0.4}})
    # The rule compares with >, which the document does not hold, so the tie is rerun alone.
    assert tie_rows(monkeypatch, loaded, record, 3) == ["yes", "no", "no"]


def test_a_function_rules_null_tie_is_a_bucket_of_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = x.loaded()  # extract nulls at or above amount/null; the document does not say so
    record = answered(loaded, x.STATE, {"q:amount": x.AMOUNT, "q:dates": x.DATES})
    points = [{"amount/null": v} for v in (0.05, 0.1, 0.3)]  # p_null is 0.1
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, points)
    assert count[0] == 3
    assert rows[0]["decision"] != rows[2]["decision"]


def test_a_function_rules_pick_helper_role_tie_is_a_bucket_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = gated(0.6)
    record = tr.turn(recorded, 0.9, 0.9)  # refund at 0.9, picked at any bar up to 0.9
    points = [{"g/bar": bar} for bar in (0.5, 0.9, 0.95)]
    count = counted(monkeypatch)
    rows = same([record], recorded, recorded, points)
    # The rule's code compares the bar; the document does not say how, so the tie is rerun.
    assert count[0] == 3
    assert [row["decision"]["value"] for row in rows] == ["refund", "refund", "ask"]


def above_bar(r: Any) -> str:
    return "yes" if r.read("q:x").probabilities["yes"] > r.param("k/bar") else "no"


def above_margin(r: Any) -> str:
    probabilities = r.read("q:x").probabilities
    return "yes" if probabilities["yes"] - probabilities["no"] > r.param("k/bar") else "no"


@pytest.mark.parametrize(
    ("role", "function", "points", "decisions"),
    [
        ("bar", above_bar, (0.5, 0.6, 0.7), ["yes", "no", "no"]),
        ("margin", above_margin, (0.1, 0.2, 0.3), ["yes", "no", "no"]),
    ],
)
def test_a_declared_pick_helper_role_compared_strictly_in_code_is_rerun_at_the_tie(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    function: Any,
    points: tuple[float, ...],
    decisions: list[str],
) -> None:
    """A rule declaring ``bar`` or ``margin`` for a read and comparing with ``>``: were the tie
    trusted to the helper's side, the middle point would take the first run's row."""
    loaded = load(
        Design(
            name="strict",
            state=tr.STATE_T,
            sources={
                "ask": Source(
                    {"q:x": Question("choice", Text("?"), labels=("yes", "no"))},
                    needs=["t:state"],
                )
            },
            rules={
                "p": Rule(
                    "d:p",
                    t.label("yes", "no"),
                    function,
                    version="1",
                    reads=["q:x"],
                    params=["k/bar"],
                    boundaries={"q:x": roles(**{role: ["k/bar"]})},
                )
            },
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:p",
        )
    )
    record = answered(loaded, tr.STATE, {"q:x": {"yes": 0.6, "no": 0.4}})
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, [{"k/bar": v} for v in points])
    assert count[0] == 3
    assert [row["decision"]["value"] for row in rows] == decisions


# ---------------------------------------------------------------------------------------------
# Ties a table fixes two ways, an integer leaf, signed zero, repeats and the points' order


def two_row_table(first: str, second: str) -> LoadedDesign:
    """``tie_table`` with two cells over the one input, the first row's taken first."""
    return load(
        Design(
            name="tie",
            state=TIE_STATE,
            sources={},
            rules={
                "act": Table(
                    "d:act",
                    'label{"maybe", "no", "yes"}',
                    "F",
                    ["t:state.n"],
                    [[first, "yes"], [second, "maybe"], ["-", "no"]],
                )
            },
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )


def test_a_tie_two_cells_put_on_different_sides_is_a_bucket_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = two_row_table("> k/bar", ">= k/bar")  # the tie fails the first and holds the second
    record = decide_sync(loaded, {"n": 0.6}, {}).record
    assert tie_rows(monkeypatch, loaded, record, 3) == ["yes", "maybe", "no"]


def test_a_tie_two_cells_put_on_one_side_shares_that_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = two_row_table("> k/bar", "<= k/bar")  # both read the tie as the leaf above
    record = decide_sync(loaded, {"n": 0.6}, {}).record
    assert tie_rows(monkeypatch, loaded, record, 2) == ["yes", "maybe", "maybe"]


def test_a_between_with_the_leaf_at_both_bounds_holds_at_the_tie_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = tie_table("[k/bar, k/bar]")
    record = decide_sync(loaded, {"n": 0.6}, {}).record
    assert tie_rows(monkeypatch, loaded, record, 3) == ["no", "yes", "no"]


def test_the_leaf_in_two_inputs_on_different_sides_is_rerun_at_the_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load(
        Design(
            name="two",
            state=t.record(n=t.number(), m=t.number()),
            sources={},
            rules={
                "act": Table(
                    "d:act",
                    'label{"maybe", "no", "yes"}',
                    "F",
                    ["t:state.n", "t:state.m"],
                    [[">= k/bar", "-", "yes"], ["-", "<= k/bar", "maybe"], ["-", "-", "no"]],
                )
            },
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )
    record = decide_sync(loaded, {"n": 0.6, "m": 0.6}, {}).record
    assert tie_rows(monkeypatch, loaded, record, 3) == ["yes", "yes", "maybe"]
    # With the inputs apart, each tie joins its own cell's side: 0.6 low, 0.3 high.
    record = decide_sync(loaded, {"n": 0.6, "m": 0.3}, {}).record
    count = counted(monkeypatch)
    points = [{"k/bar": v} for v in (0.2, 0.3, 0.4, 0.6, 0.8)]
    rows = same([record], loaded, loaded, points)
    assert count[0] == 3
    assert [row["decision"]["value"] for row in rows] == ["yes", "yes", "yes", "yes", "maybe"]


def test_an_integer_leaf_is_bucketed_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load(
        Design(
            name="int",
            state=t.record(n=t.integer()),
            sources={},
            rules={
                "act": Table("d:act", YES_NO, "F", ["t:state.n"], [[">= k/n", "yes"], ["-", "no"]])
            },
            parameters={"k/n": Parameter(t.integer(), 5)},
            policy="d:act",
        )
    )
    record = decide_sync(loaded, {"n": 5}, {}).record
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, [{"k/n": v} for v in (3, 5, 6, 2**53 - 1, -(2**53 - 1))])
    assert count[0] == 2
    assert [row["decision"]["value"] for row in rows] == ["yes", "yes", "no", "no", "yes"]


def test_signed_zero_shares_zeros_bucket_and_design(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = tie_table(">= k/bar")
    record = decide_sync(loaded, {"n": 0.0}, {}).record
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, [{"k/bar": v} for v in (-0.0, 0.0, 0.5)])
    assert count[0] == 2
    assert rows[0]["design"] == rows[1]["design"]  # the canonical document writes both as 0
    assert [row["decision"]["value"] for row in rows] == ["yes", "yes", "no"]


def test_repeated_points_and_points_out_of_axis_order_keep_their_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = tie_table(">= k/bar")
    record = decide_sync(loaded, {"n": 0.6}, {}).record
    count = counted(monkeypatch)
    values = (0.8, 0.4, 0.6, 0.8, 0.4, 0.7)  # descending first, each repeated once
    rows = same([record], loaded, loaded, [{"k/bar": v} for v in values])
    assert count[0] == 2
    assert [row["decision"]["value"] for row in rows] == ["no", "yes", "yes", "no", "yes", "no"]
    assert rows[0]["design"] == rows[3]["design"]


# ---------------------------------------------------------------------------------------------
# What a source can see of the leaf


def opens(r: Any) -> bool:
    return r.param("k/bar") < 0.5


def act_m(r: Any) -> str:
    return "no" if isinstance(mass(r, "q:m", ["yes"]), Undetermined) else "yes"


def test_a_when_reading_the_leaf_is_an_undeclared_read_at_every_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load(
        Design(
            name="when",
            state=TIE_STATE,
            sources={
                "ask": Source(
                    {"q:m": Question("noul", Text("M?"))},
                    needs=["t:state"],
                    when=When(opens, version="1"),
                )
            },
            rules={"act": Rule("d:act", t.label("yes", "no"), act_m, version="1", reads=["q:m"])},
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )
    record = answered(loaded, {"n": 0.0}, {"q:m": {"yes": 1.0, "no": 0.0}})
    count = counted(monkeypatch)
    rows = same([record], loaded, loaded, [{"k/bar": v} for v in TIES])
    assert count[0] == 1  # a when declares no leaf, so no point differs from another
    assert len({encode(row["decision"]) for row in rows}) == 1


def test_a_source_needing_a_param_valued_entry_of_the_leaf_is_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load(
        Design(
            name="needs",
            state=TIE_STATE,
            sources={"ask": Source({"q:m": Question("noul", Text("M?"))}, needs=["d:lim"])},
            rules={
                "lim": Table("d:lim", "number", "F", [], [[Param("k/bar")]]),
                "act": Rule("d:act", t.label("yes", "no"), act_m, version="1", reads=["q:m"]),
            },
            parameters={"k/bar": Parameter(t.number(0, 1), 0.5)},
            policy="d:act",
        )
    )
    record = answered(loaded, {"n": 0.0}, {"q:m": {"yes": 1.0, "no": 0.0}})
    refusal = refused(monkeypatch, [record], loaded, loaded, [{"k/bar": 0.3}])
    assert refusal.reason is RefusalReason.hidden_leaf
    assert "lim" in str(refusal)


# ---------------------------------------------------------------------------------------------
# A key of a numeric mapping leaf: named in boundaries and varied by a point

BARS = (0.1, 0.3, 0.5, 0.7, 0.9)


def part_records(loaded: LoadedDesign) -> list[Record]:
    return [
        decide_sync(loaded, {"email": e, "phone": p}, {}).record
        for e, p in ((0.4, 0.6), (0.3, 0.8))
    ]


def test_key_paths_vary_a_mapping_leaf_as_number_leaves_would(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The workaround, a number leaf per part, and the mapping form by key path give the same
    # decisions with the same reruns.
    by_leaves, by_keys = parts(leaves=True), parts()
    leaf_points = [{"part/email": e, "part/phone": p} for e in BARS for p in BARS]
    key_points = [{"part/at.email": e, "part/at.phone": p} for e in BARS for p in BARS]
    count = counted(monkeypatch)
    leaf_rows = same(part_records(by_leaves), by_leaves, by_leaves, leaf_points)
    leaf_runs = count[0]
    key_rows = same(part_records(by_keys), by_keys, by_keys, key_points)
    assert count[0] - leaf_runs == leaf_runs < len(leaf_points) * 2
    assert [r["decision"] for r in key_rows] == [r["decision"] for r in leaf_rows]
    point = key_points[7]
    assert (
        key_rows[14]["design"]
        == with_parameters(
            by_keys, {"part/at": {"email": point["part/at.email"], "phone": point["part/at.phone"]}}
        ).design_hash
    )


def test_a_whole_mapping_leaf_buckets_each_key_on_its_own_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = parts()
    records = part_records(loaded)
    points = [{"part/at": {"email": e, "phone": p}} for e in BARS for p in BARS]
    count = counted(monkeypatch)
    same(records, loaded, loaded, points)
    # Each key's axis holds that key's compared value alone. The first record's email 0.4 and
    # phone 0.6 split the bars in two each, 4 regions; the second's email 0.3, a tie in a bucket
    # of its own, in three and its phone 0.8 in two, 6. One axis pooling both keys' values
    # would give 9 and 16.
    assert count[0] == 4 + 6


def test_key_paths_mix_with_whole_leaves_and_take_any_key_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = parts()
    records = part_records(loaded)
    points: list[Point] = [{"part/at.phone": p, "part/at.email": e} for e in BARS for p in BARS]
    same(records, loaded, loaded, points)
    only_email = [{"part/at.email": e} for e in BARS]
    same(records, loaded, loaded, only_email)


def test_a_function_sees_only_the_keys_its_boundaries_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = parts({"t:state.email": ["part/at.email"], "t:state.phone": []})
    records = part_records(loaded)
    refusal = refused(monkeypatch, records, loaded, loaded, [{"part/at.phone": 0.3}])
    assert refusal.reason is RefusalReason.hidden_leaf
    assert refusal.path == "part/at.phone"
    assert "fired" in str(refusal)
    varied = [{"part/at": {"email": 0.3, "phone": 0.3}}, {"part/at": {"email": 0.3, "phone": 0.7}}]
    refusal = refused(monkeypatch, records, loaded, loaded, varied)
    assert refusal.reason is RefusalReason.hidden_leaf
    assert refusal.path == "part/at"
    # A key no point varies gives every run its one value, so it needs no boundary.
    same(records, loaded, loaded, [{"part/at.email": e} for e in BARS])
    same(records, loaded, loaded, [{"part/at": {"email": e, "phone": 0.3}} for e in BARS])


def test_a_key_path_is_refused_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = parts()
    records = part_records(loaded)
    cases: list[tuple[Point, RefusalReason, str]] = [
        ({"part/at": {"email": 0.1}, "part/at.email": 0.2}, _R.overlapping_paths, "part/at.email"),
        ({"part/at.fax": 0.2, "part/at.email": 0.1}, _R.unknown_path, "part/at.fax"),
        ({"part/no.email": 0.2}, _R.unknown_path, "part/no.email"),
        # Overlap is read from the names alone, before any path is looked up.
        (
            {"part/no.x": 0.2, "part/at.email": 0.1, "part/at": {}},
            _R.overlapping_paths,
            "part/at.email",
        ),
    ]
    for point, reason, path in cases:
        refusal = refused(monkeypatch, records, loaded, loaded, [point])
        assert refusal.reason is reason
        assert refusal.path == path
    refusal = refused(monkeypatch, records, loaded, loaded, [{"part/at.email": "high"}])
    assert refusal.reason is RefusalReason.invalid_design


def test_paths_are_checked_in_code_point_order(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = parts()
    records = part_records(loaded)
    for point in ({"part/zz": 1.0, "part/aa": 1.0}, {"part/aa": 1.0, "part/zz": 1.0}):
        refusal = refused(monkeypatch, records, loaded, loaded, [point])
        assert refusal.path == "part/aa"


def test_a_key_path_into_a_mapping_the_job_does_not_vary_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = parts()
    document = json.loads(base.document)
    document["parameters"]["part/names"] = {
        **document["parameters"]["part/at"],
        "type": "mapping<string, string>",
        "value": {"email": "x", "phone": "y"},
    }
    loaded = load(document, functions=base._callables)  # pyright: ignore[reportPrivateUsage]
    records = part_records(loaded)
    refusal = refused(monkeypatch, records, loaded, loaded, [{"part/names.email": "y"}])
    assert refusal.reason is RefusalReason.unsupported_leaf
    assert refusal.path == "part/names.email"


def test_a_key_of_a_wording_slot_leaf_gives_each_value_its_own_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worded = sv.mood_design(Text("Mood over {th}?", [slot("th", param="mood/th")]))
    document = json.loads(worded.document)
    del document["rules"]["thf"]
    loaded = load(document, functions=worded._callables)  # pyright: ignore[reportPrivateUsage]
    records = [sv.mood_record(loaded)]
    points = [{"mood/th.calm": calm} for calm in (0.3, 0.6, 0.6, 0.7)]
    count = counted(monkeypatch)
    rows = same(records, loaded, loaded, points)
    assert count[0] == 3  # one per distinct value, as for a whole leaf that fills a slot
    assert [row["stale"] for row in rows] == [["q:mood"], [], [], ["q:mood"]]
