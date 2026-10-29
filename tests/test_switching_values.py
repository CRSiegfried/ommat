# pyright: reportPrivateUsage=false
"""The switching-value job: each leaf type varied, each source of a compared value probed,
``unseen``, ``carries``, not-determined and not-loading ranges, the refusals in their order,
and the entry written to the store, looked up from it and returned as plain data reaching no
``Record``."""

import asyncio
import copy
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import extraction as x
import test_grounded_rules as grounded
from designs import minimal, slot
from ommat import (
    CallResult,
    Cause,
    Context,
    ContextEntry,
    Failed,
    Helper,
    Input,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Origin,
    Param,
    Parameter,
    Plan,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Record,
    Requirement,
    Rule,
    Source,
    Step,
    Table,
    Text,
    decide_sync,
    load,
    read_record,
)
from ommat import types as t
from ommat._canonical import decode
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    _diff,
    _switching,
    deciding_inputs_sync,
    switching_values,
    switching_values_sync,
)
from scripted import Scripted
from test_decide import LOADED
from test_use_answers import broken, unfinished

R = RefusalReason
KIND = "switching-value"
YES_NO = 'label{"no", "yes"}'
BIG_SMALL = 'label{"big", "small"}'
FLIP = [False]

# ---------------------------------------------------------------------------------------------
# A design of tables and function rules over the state


def hot(r: Any) -> bool:
    return r.read("t:state.n") > r.param("lim/f")


def hid(r: Any) -> bool:
    return r.param("lim/n") > 1000


def on(r: Any) -> bool:
    return r.param("sw/on") and r.read("t:state.flag")


def mode(r: Any) -> str:
    return r.param("sw/mode")


def kfn(r: Any) -> bool:
    return r.param("lim/k") > 100


def flip(r: Any) -> bool:
    return FLIP[0]


def told(r: Any) -> Plan:
    return Plan("inform", reply=r.read("d:big"))


def plan_of(r: Any) -> Plan:
    return Plan("inform", reply="ok")


STATE_T = t.record(n=t.number(), k=t.integer(), flag=t.boolean, items=t.list_(t.integer()))
STATE: dict[str, Any] = {"n": 7.0, "k": 4, "flag": True, "items": [1, 2, 3]}
RULES: dict[str, Any] = {
    "big": Table("d:big", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "small"]]),
    "many": Table("d:many", BIG_SMALL, "F", ["t:state.k"], [[">= lim/k", "big"], ["-", "small"]]),
    "echo": Table("d:echo", "number", "F", [], [[Param("lim/n")]]),
    "part": Table(
        "d:part", "number", "F", ["t:state.n"], [[">= lim/n", Param("lim/n")], ["-", 0.0]]
    ),
    "sz": Table(
        "d:sz",
        BIG_SMALL,
        "F",
        [Helper("size", "t:state.items")],
        [[">= lim/s", "big"], ["-", "small"]],
    ),
    "kfn": Rule("d:kfn", t.boolean, kfn, version="1", reads=["t:state.flag"], params=["lim/k"]),
    "band": Table(
        "d:band",
        'label{"in", "out"}',
        "F",
        ["t:state.n"],
        [["[band/lo, band/hi]", "in"], ["-", "out"]],
    ),
    "hot": Rule(
        "d:hot",
        t.boolean,
        hot,
        version="1",
        reads=["t:state.n"],
        params=["lim/f"],
        boundaries={"t:state.n": ["lim/f"]},
    ),
    "hid": Rule("d:hid", t.boolean, hid, version="1", reads=["t:state.n"], params=["lim/n"]),
    "on": Rule("d:on", t.boolean, on, version="1", reads=["t:state.flag"], params=["sw/on"]),
    "mode": Rule(
        "d:mode",
        t.label("a", "b", "c"),
        mode,
        version="1",
        reads=["t:state.flag"],
        params=["sw/mode"],
    ),
    "flip": Rule("d:flip", t.boolean, flip, version="1", reads=["t:state.flag"]),
}
PARAMETERS: dict[str, Any] = {
    "lim/n": Parameter(t.number(), 5.0),
    "lim/k": Parameter(t.integer(0, 10), 2),
    "lim/f": Parameter(t.number(), 1.0),
    "lim/s": Parameter(t.integer(), 1),
    "sw/on": Parameter(t.boolean, False),
    "sw/mode": Parameter(t.label("a", "b", "c"), "a"),
    "band/lo": Parameter(t.number(), 1.0),
    "band/hi": Parameter(t.number(), 7.5),
    "text": Parameter(t.string, "x"),
    "tags": Parameter(t.list_(t.string), ["a"]),
    "names": Parameter(t.mapping(t.string), {"a": "b"}),
}
TABLES = load(
    minimal(
        state=STATE_T,
        rules=RULES,
        parameters=PARAMETERS,
        policy=Policy(told, version="1", reads=["d:big"]),
    )
)


def tables_record() -> Record:
    return decide_sync(TABLES, STATE, {}).record


def run(
    record: object, loaded: LoadedDesign, path: Any, root: Path, functions: Any = None
) -> dict[str, Any]:
    return switching_values_sync(
        record, loaded, path, functions=functions, results=EvaluationStore(root)
    )


def value_of(result: Any) -> Any:
    """A result's entry or plan value, or the result itself when it is not one."""
    if type(result) is not dict:
        return result
    held = cast("dict[str, Any]", result)
    if "entry" in held:
        entry = cast("dict[str, Any] | None", held["entry"])
        return None if entry is None else entry.get("value", entry.get("marker"))
    if "plan" in held:
        return held["plan"]["reply"]
    return held


def shape(summary: Any) -> Any:
    """A summary with each class's result reduced to its value."""
    if summary == "carries":
        return summary
    out = dict(summary)
    out["classes"] = [{**c, "result": value_of(c["result"])} for c in summary["classes"]]
    return out


@pytest.fixture
def no_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)
    monkeypatch.setattr(_switching, "_replay_with", replayed)


def refusal(*args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        run(*args, **kwargs)
    return raised.value


# ---------------------------------------------------------------------------------------------
# Numeric leaves


def test_a_table_cell_is_probed_at_its_input_and_the_gap_and_end_values(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "lim/n", tmp_path)["value"]
    assert value["unseen"] == ["hid"]  # it declares the leaf and has no boundaries field
    assert list(value["entries"]) == ["d:big", "d:echo", "d:part"]
    assert shape(value["entries"]["d:big"]) == {
        "switching": [7.0],
        "slacks": [None, 2.0],
        "classes": [
            {"least": 5.0, "greatest": 7.0, "result": "big"},
            {"least": 8.0, "greatest": 8.0, "result": "small"},
        ],
    }
    assert value["entries"]["d:echo"] == "carries"  # a Param output
    assert value["entries"]["d:part"] == "carries"  # a Param row firing below 7 alone
    assert shape(value["plan"]) == shape(value["entries"]["d:big"])  # the reply is d:big


def test_an_integer_leaf_takes_integer_points_within_its_interval(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "lim/k", tmp_path)["value"]
    assert value["unseen"] == ["kfn"]  # no integer interval is enumerated
    summary = value["entries"]["d:many"]
    assert type(summary["slacks"][1]) is int
    assert shape(summary) == {
        "switching": [5],  # 4 and 5 are tried either side of k: the upper class's least
        "slacks": [None, 3],
        "classes": [
            {"least": 0, "greatest": 4, "result": "big"},
            {"least": 5, "greatest": 10, "result": "small"},
        ],
    }


def test_a_function_rule_with_boundaries_is_probed_at_its_reads(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "lim/f", tmp_path)["value"]
    assert value["unseen"] == []
    # n > lim/f: true up to 6, false from 7, the compared value, on.
    assert shape(value["entries"]["d:hot"]) == {
        "switching": [7.0],
        "slacks": [None, 6.0],
        "classes": [
            {"least": 1.0, "greatest": 6.0, "result": True},
            {"least": 7.0, "greatest": 8.0, "result": False},
        ],
    }


def test_a_value_whose_design_fails_load_marks_its_range_not_loading(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "band/lo", tmp_path)["value"]
    assert value["entries"] == {}  # d:band holds wherever the design loads
    plan = value["plan"]
    assert plan["switching"] == [7.0]  # beside a not-loading class, as beside any
    assert plan["slacks"] == [None, 6.0]
    (first, last) = plan["classes"]
    assert (first["least"], first["greatest"], last["least"]) == (1.0, 7.0, 8.0)
    (failure,) = last["result"]["not-loading"]
    assert failure["check"] == 9
    assert failure["names"] == ["band"]
    assert "band" in failure["message"]


# ---------------------------------------------------------------------------------------------
# Finite leaves


def test_a_boolean_leaf_is_partitioned_false_first(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "sw/on", tmp_path)["value"]
    assert value["unseen"] == []  # every value was tried
    assert shape(value["entries"]["d:on"]) == {
        "classes": [{"values": [False], "result": False}, {"values": [True], "result": True}]
    }
    assert shape(value["plan"]) == {"classes": [{"values": [False, True], "result": "big"}]}


def test_a_label_leaf_is_partitioned_in_declared_order(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "sw/mode", tmp_path)["value"]
    assert shape(value["entries"]["d:mode"]) == {
        "classes": [
            {"values": ["a"], "result": "a"},
            {"values": ["b"], "result": "b"},
            {"values": ["c"], "result": "c"},
        ]
    }  # a lookup giving every label its own result is no carrying


# ---------------------------------------------------------------------------------------------
# The pick helpers and a numeric mapping


def calm(r: Any) -> Plan:
    return Plan("inform", reply=r.read("d:th"))


def thf(r: Any) -> bool:
    return r.param("mood/th")["calm"] > 2


def mood_design(wording: Text | None = None) -> LoadedDesign:
    question = Question("choice", wording or Text("Mood?"), labels=("calm", "wild"))
    return load(
        minimal(
            state=STATE_T,
            sources={"ask": Source({"q:mood": question}, needs=["t:state"])},
            rules={
                "th": Table(
                    "d:th",
                    YES_NO,
                    "F",
                    [
                        Helper(
                            "pick_with_thresholds", "q:mood", thresholds="mood/th", margin="mood/m"
                        )
                    ],
                    [["calm", "yes"], ["-", "no"]],
                ),
                "bar": Table(
                    "d:bar",
                    YES_NO,
                    "F",
                    [Helper("pick_with_margin", "q:mood", bar="mood/bar", margin="mood/m2")],
                    [["calm", "yes"], ["-", "no"]],
                ),
                "ms": Table(
                    "d:ms",
                    YES_NO,
                    "F",
                    [Helper("mass", "q:mood", labels="mood/labels")],
                    [[">= mood/mbar", "yes"], ["-", "no"]],
                ),
                "thf": Rule(
                    "d:thf", t.boolean, thf, version="1", reads=["q:mood"], params=["mood/th"]
                ),
            },
            parameters={
                "mood/th": Parameter(t.mapping(t.number(0, 1)), {"calm": 0.6, "wild": 0.6}),
                "mood/m": Parameter(t.number(0, 1), 0.2),
                "mood/bar": Parameter(t.number(0, 1), 0.5),
                "mood/m2": Parameter(t.number(0, 1), 0.0),
                "mood/labels": Parameter(t.list_(t.string), ["calm"]),
                "mood/mbar": Parameter(t.number(0, 1), 0.5),
            },
            policy=Policy(calm, version="1", reads=["d:th"]),
        )
    )


MOOD = mood_design()
SPLIT = {"calm": 0.7, "wild": 0.3}


def mood_record(loaded: LoadedDesign = MOOD) -> Record:
    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({"q:mood": SPLIT})

    return decide_sync(loaded, STATE, {"default": answer}).record


def test_a_mapping_leaf_varies_each_key_alone(tmp_path: Path) -> None:
    value = run(mood_record(), MOOD, "mood/th", tmp_path)["value"]
    assert set(value) == {"unseen", "keys"}
    assert value["unseen"] == ["thf"]  # it declares the mapping and has no boundaries field
    assert list(value["keys"]) == ["calm", "wild"]
    calm_key = value["keys"]["calm"]
    # Probed at each label's probability: calm passes its threshold up to 0.7.
    assert shape(calm_key["entries"]["d:th"]) == {
        "switching": [0.7],
        "slacks": [None, 0.1],
        "classes": [
            {"least": 0.0, "greatest": 0.7, "result": "yes"},
            {"least": 1.0, "greatest": 1.0, "result": "no"},
        ],
    }
    assert value["keys"]["wild"]["entries"] == {}  # calm leads whatever wild's threshold


def test_a_pick_bar_is_probed_at_each_labels_probability(tmp_path: Path) -> None:
    value = run(mood_record(), MOOD, "mood/bar", tmp_path)["value"]
    summary = value["entries"]["d:bar"]
    assert summary["switching"] == [0.7]
    assert summary["slacks"] == [None, 0.2]
    tried = [v for c in summary["classes"] for v in (c["least"], c["greatest"])]
    assert tried == [0.0, 0.7, 1.0, 1.0]  # 0, 0.3, 0.5, 0.7 and 1 tried


def test_a_margin_is_probed_at_the_lead(tmp_path: Path) -> None:
    value = run(mood_record(), MOOD, "mood/m", tmp_path)["value"]
    assert shape(value["entries"]["d:th"]) == {
        "switching": [0.4],  # 0.7 leads 0.3 by exactly 0.4
        "slacks": [None, 0.2],
        "classes": [
            {"least": 0.0, "greatest": 0.4, "result": "yes"},
            {"least": 1.0, "greatest": 1.0, "result": "no"},
        ],
    }


def test_the_margin_probe_is_the_greatest_float_not_above_the_lead() -> None:
    assert _switching._margin_probe({"a": 0.6, "b": 0.4}) == 0.2
    assert _switching._margin_probe({"a": 0.9, "b": 0.8, "c": 0.0}) == 0.1
    assert _switching._margin_probe({"a": 1.0}) == 1.0


def test_a_question_reading_the_leaf_leaves_other_values_not_determined(tmp_path: Path) -> None:
    wording = Text("Mood over {bar}?", [slot("bar", param="mood/bar")])
    loaded = mood_design(wording)
    value = run(mood_record(loaded), loaded, "mood/bar", tmp_path)["value"]
    bar = value["entries"]["d:bar"]  # it changes in a run outside the record, so it is listed
    assert [value_of(c["result"]) for c in bar["classes"]] == [
        "not-determined",
        "yes",
        "not-determined",
    ]
    classes = value["plan"]["classes"]
    assert [value_of(c["result"]) for c in classes] == ["not-determined", "yes", "not-determined"]
    assert classes[1]["least"] == classes[1]["greatest"] == 0.5
    assert value["plan"]["switching"] == [0.3, 0.7]  # each facing value compared
    assert value["plan"]["slacks"] == [-0.2, 0.2]


# ---------------------------------------------------------------------------------------------
# Extraction


def test_a_function_boundaries_null_role_is_probed_at_p_null(tmp_path: Path) -> None:
    loaded = x.loaded()
    record = decide_sync(loaded, x.STATE, {"default": x_answer}).record
    value = run(record, loaded, "amount/null", tmp_path)["value"]
    summary = value["entries"]["d:amount"]
    assert summary["switching"] == [0.1]
    assert summary["slacks"] == [-0.4, None]
    assert [value_of(c["result"])["kind"] for c in summary["classes"][:1]] == ["Unsure"]


async def x_answer(batch: Any, *, seconds: float) -> object:
    return CallResult({"q:amount": x.AMOUNT, "q:dates": x.DATES})


def test_extracts_null_bar_and_a_cell_over_its_value_are_probed(tmp_path: Path) -> None:
    table = Table(
        "d:amount",
        BIG_SMALL,
        "F",
        [Helper("extract", "q:amount", null_bar="amount/null")],
        [[">= amount/big", "big"], ["-", "small"]],
    )
    d = x.design()
    loaded = load(x.replaced(d, rules={**d.rules, "amount": table}))
    record = decide_sync(loaded, x.STATE, {"default": x_answer}).record
    null = run(record, loaded, "amount/null", tmp_path / "a")["value"]["entries"]["d:amount"]
    assert shape(null)["classes"][0] == {"least": 0.0, "greatest": 0.1, "result": "small"}
    assert null["switching"] == [0.1]
    big = run(record, loaded, "amount/big", tmp_path / "b")["value"]["entries"]["d:amount"]
    assert big["switching"] == [120.0]
    assert big["slacks"] == [None, 20.0]


# ---------------------------------------------------------------------------------------------
# The store and the return


def test_the_entry_is_written_looked_up_and_returned_as_plain_data(tmp_path: Path) -> None:
    record = tables_record()
    entry = run(record, TABLES, "lim/n", tmp_path)
    key = {"record": record.record_hash, "design": TABLES.design_hash, "parameter": "lim/n"}
    assert {k: entry[k] for k in ("kind", "format", "key")} == {
        "kind": KIND,
        "format": 1,
        "key": key,
    }
    assert EvaluationStore(tmp_path).lookup(KIND, key) == entry
    assert json.loads(json.dumps(entry)) == entry
    assert copy.deepcopy(entry) == entry
    stack: list[Any] = [entry]
    while stack:
        item = stack.pop()
        assert type(item) is not Record
        if isinstance(item, dict):
            stack += item.values()
        elif isinstance(item, list):
            stack += item


def test_a_stored_document_is_read_as_the_record(tmp_path: Path) -> None:
    record = tables_record()
    assert run(record.document, TABLES, "lim/k", tmp_path) == run(
        record, TABLES, "lim/k", tmp_path / "again"
    )


def test_the_sync_twin_refuses_a_running_loop_and_the_job_awaits(tmp_path: Path) -> None:
    record = tables_record()

    async def inside() -> dict[str, Any]:
        with pytest.raises(RuntimeError, match="await switching_values"):
            switching_values_sync(record, TABLES, "lim/k", functions=None, results=store)
        return await switching_values(record, TABLES, "lim/k", functions=None, results=store)

    store = EvaluationStore(tmp_path)
    assert asyncio.run(inside()) == run(record, TABLES, "lim/k", tmp_path / "sync")


# ---------------------------------------------------------------------------------------------
# The refusals, in order


def test_invalid_record_comes_first(tmp_path: Path) -> None:
    exc = refusal({"not": "a record"}, TABLES, "nope", tmp_path)
    assert exc.reason is R.invalid_record
    assert isinstance(exc.error, InvalidRecord)


def test_unfinished_then_nothing_to_replay_then_wrong_design(tmp_path: Path) -> None:
    assert refusal(unfinished(), LOADED, "nope", tmp_path).reason is R.unfinished
    assert refusal(broken(), LOADED, "nope", tmp_path).reason is R.nothing_to_replay
    record = tables_record()
    exc = refusal(record, LOADED, "nope", tmp_path)
    assert exc.reason is R.wrong_design
    assert exc.turn == record.record_hash


@pytest.mark.parametrize("path", ["lim", "nope", "lim//n", "Lim/n", 3, ["lim"]])
def test_unknown_path_names_no_leaf(path: Any, tmp_path: Path, no_replay: None) -> None:
    exc = refusal(tables_record(), TABLES, path, tmp_path, functions={"policy": 1})
    assert exc.reason is R.unknown_path
    assert exc.path == (path if isinstance(path, str) else None)


@pytest.mark.parametrize(
    ("path", "type_"), [("text", "string"), ("tags", "list<string>"), ("names", "mapping")]
)
def test_unsupported_leaf_names_the_type(
    path: str, type_: str, tmp_path: Path, no_replay: None
) -> None:
    exc = refusal(tables_record(), TABLES, path, tmp_path, functions={"policy": 1})
    assert exc.reason is R.unsupported_leaf
    assert exc.path == path
    assert type_ in str(exc)


def test_invalid_design_when_the_document_does_not_load_with_functions(
    tmp_path: Path, no_replay: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def local(r: Any) -> Plan:
        return Plan("inform", reply="ok")

    loaded = load(
        minimal(
            state=STATE_T,
            rules={"many": RULES["many"]},
            parameters={"lim/k": PARAMETERS["lim/k"]},
            policy=Policy(local, version="1", reads=["d:many"]),
        )
    )
    record = decide_sync(loaded, STATE, {}).record
    exc = refusal(record, loaded, "lim/k", tmp_path)
    assert exc.reason is R.invalid_design
    assert isinstance(exc.error, InvalidDesign)
    assert exc.__cause__ is exc.error
    monkeypatch.undo()  # replay again: with the function given, the document loads
    assert run(record, loaded, "lim/k", tmp_path, functions={"policy": local})["value"]["entries"]


def test_a_held_key_is_refused_before_any_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tables_record()
    first = run(record, TABLES, "lim/k", tmp_path)

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)
    monkeypatch.setattr(_switching, "_replay_with", replayed)
    exc = refusal(record, TABLES, "lim/k", tmp_path)
    assert exc.reason is R.held_key
    assert exc.key == [first["key"]]


def test_baseline_differs_when_replay_under_loaded_finds_a_change(tmp_path: Path) -> None:
    record = tables_record()
    FLIP[0] = True
    try:
        exc = refusal(record, TABLES, "lim/k", tmp_path)
    finally:
        FLIP[0] = False
    assert exc.reason is R.baseline_differs
    assert "entry" in str(exc)
    first = run(record, TABLES, "lim/k", tmp_path)
    FLIP[0] = True
    try:  # the held key is refused before the baseline replays
        assert refusal(record, TABLES, "lim/k", tmp_path).reason is R.held_key
    finally:
        FLIP[0] = False
    assert first["key"]["parameter"] == "lim/k"


class Blind(EvaluationStore):
    """A store whose lookup finds nothing, as when another run writes between."""

    def lookup(self, kind: str, key: dict[str, str]) -> dict[str, Any] | None:
        return None


def test_a_held_key_at_the_write_is_refused_and_the_first_entry_stands(tmp_path: Path) -> None:
    record = tables_record()
    first = run(record, TABLES, "lim/k", tmp_path)
    with pytest.raises(EvaluationRefusal) as raised:
        switching_values_sync(record, TABLES, "lim/k", functions=None, results=Blind(tmp_path))
    assert raised.value.reason is R.held_key
    assert raised.value.key == [first["key"]]
    assert EvaluationStore(tmp_path).lookup(KIND, first["key"]) == first


# ---------------------------------------------------------------------------------------------
# No switching value between two adjacent probes, against a dense grid


def at(loaded: LoadedDesign, path: str, value: float) -> LoadedDesign:
    """``loaded`` with the leaf at ``path`` set to ``value``."""
    document = decode(loaded.document)
    document["parameters"][path]["value"] = value
    return load(document)


def decided_entry(record: Record, ref: str) -> object:
    return next(i["value"] for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def test_a_table_switches_nowhere_but_its_reported_switching_value(tmp_path: Path) -> None:
    (switch,) = run(tables_record(), TABLES, "lim/n", tmp_path)["value"]["entries"]["d:big"][
        "switching"
    ]
    for v in [4 + i / 16 for i in range(81)]:
        got = decided_entry(decide_sync(at(TABLES, "lim/n", v), STATE, {}).record, "d:big")
        assert got == ("big" if v <= switch else "small"), v


def test_a_pick_helper_switches_nowhere_but_its_reported_switching_value(tmp_path: Path) -> None:
    (switch,) = run(mood_record(), MOOD, "mood/m", tmp_path)["value"]["entries"]["d:th"][
        "switching"
    ]

    async def answer(batch: Any, *, seconds: float) -> object:
        return CallResult({"q:mood": SPLIT})

    for v in [i / 100 for i in range(101)]:
        record = decide_sync(at(MOOD, "mood/m", v), STATE, {"default": answer}).record
        assert decided_entry(record, "d:th") == ("yes" if v <= switch else "no"), v


# ---------------------------------------------------------------------------------------------
# Two rounds of probes, and what they leave unseen


def constant(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def twice(r: Any) -> float:
    return 2 * r.param("lim/f") + 1


def over(r: Any) -> bool:
    return r.read("d:y") > r.param("lim/f")


def counting_replays(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen = [0]
    real = _switching._replay_with

    async def counted(*args: Any, **kwargs: Any) -> Any:
        seen[0] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(_switching, "_replay_with", counted)
    return seen


def test_a_value_each_round_recomputes_ends_at_round_1_and_lists_its_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load(
        minimal(
            state=STATE_T,
            rules={
                "cap": Table("d:cap", "number", "F", [], [[Param("lim/n")]]),
                "chk": Table(
                    "d:chk", BIG_SMALL, "F", ["d:cap"], [[">= lim/n", "big"], ["-", "small"]]
                ),
            },
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(constant, version="1", reads=["d:chk"]),
        )
    )
    seen = counting_replays(monkeypatch)
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    assert value["unseen"] == ["chk"]  # d:cap holds every value tried, so chk compares anew
    assert value["entries"]["d:cap"] == "carries"
    # Round 0 over 5: 4 and 6; round 1 over 4, 5 and 6: 3, 4.5, 5.5 and 7.
    assert seen[0] == 2 + 4


def test_a_function_computing_from_the_leaf_ends_and_is_unseen(tmp_path: Path) -> None:
    loaded = load(
        minimal(
            state=STATE_T,
            rules={
                "y": Rule("d:y", t.number(), twice, version="1", reads=[], params=["lim/f"]),
                "over": Rule(
                    "d:over",
                    t.boolean,
                    over,
                    version="1",
                    reads=["d:y"],
                    params=["lim/f"],
                    boundaries={"d:y": ["lim/f"]},
                ),
            },
            parameters={"lim/f": PARAMETERS["lim/f"]},
            policy=Policy(constant, version="1", reads=["d:over"]),
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/f", tmp_path)["value"]
    # y has no boundaries; over reads d:y, which holds the leaf, as the policy reads d:over,
    # whose function declares it.
    assert value["unseen"] == ["over", "policy", "y"]


def three_way(own: float) -> LoadedDesign:
    return load(
        minimal(
            state=STATE_T,
            rules={
                "cmp": Table(
                    "d:cmp",
                    'label{"eq", "gt", "lt"}',
                    "F",
                    ["t:state.n"],
                    [["> lim/n", "gt"], [">= lim/n", "eq"], ["-", "lt"]],
                )
            },
            parameters={"lim/n": Parameter(t.number(), own)},
            policy=Policy(constant, version="1", reads=["d:cmp"]),
        )
    )


@pytest.mark.parametrize(
    ("own", "slacks"), [(5.0, [None, 2.0]), (7.0, [0.0, 0.0]), (8.0, [-1.0, None])]
)
def test_a_three_way_table_gives_classes_whatever_the_own_value(
    own: float, slacks: list[float | None], tmp_path: Path
) -> None:
    loaded = three_way(own)
    summary = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"][
        "entries"
    ]["d:cmp"]
    assert [value_of(c["result"]) for c in summary["classes"]] == ["gt", "eq", "lt"]
    assert summary["switching"] == [7.0, 7.0]
    assert summary["slacks"] == slacks


def counted(own: float) -> LoadedDesign:
    fields = ("a", "b", "c", "d")
    return load(
        minimal(
            state=t.record(**dict.fromkeys(fields, t.number())),
            rules={
                "count": Table(
                    "d:count",
                    "integer",
                    "C#",
                    [f"t:state.{f}" for f in fields],
                    [["-"] * i + [">= lim/c"] + ["-"] * (3 - i) for i in range(4)],
                )
            },
            parameters={"lim/c": Parameter(t.number(), own)},
            policy=Policy(constant, version="1", reads=["d:count"]),
        )
    )


@pytest.mark.parametrize(("own", "slacks"), [(5.5, [-1.5, 1.5]), (7.0, [-3.0, 0.0])])
def test_the_slacks_are_the_switching_values_bounding_the_own_class(
    own: float, slacks: list[float], tmp_path: Path
) -> None:
    loaded = counted(own)
    state = {"a": 1.0, "b": 4.0, "c": 7.0, "d": 10.0}
    summary = run(decide_sync(loaded, state, {}).record, loaded, "lim/c", tmp_path)["value"][
        "entries"
    ]["d:count"]
    assert summary["switching"] == [1.0, 4.0, 7.0, 10.0]
    assert summary["slacks"] == slacks


def one_table(table: Table, **parameters: Parameter) -> tuple[LoadedDesign, Record]:
    """A design of ``table`` alone over ``STATE``, its leaves ``parameters``, and its record."""
    loaded = load(
        minimal(
            state=STATE_T,
            rules={table.provides.removeprefix("d:"): table},
            parameters=parameters,
            policy=Policy(constant, version="1", reads=[table.provides]),
        )
    )
    return loaded, decide_sync(loaded, STATE, {}).record


@pytest.mark.parametrize(("cell", "least_big"), [("<= lim/lt", 7.0), ("< lim/lt", 8.0)])
def test_le_and_lt_cells_switch_at_their_input(cell: str, least_big: float, tmp_path: Path) -> None:
    table = Table("d:tab", BIG_SMALL, "F", ["t:state.n"], [[cell, "big"], ["-", "small"]])
    loaded, record = one_table(table, **{"lim/lt": Parameter(t.number(), 9.0)})
    summary = run(record, loaded, "lim/lt", tmp_path)["value"]["entries"]["d:tab"]
    assert summary["classes"][1]["least"] == least_big  # 7 is big under <=, small under <
    assert summary["switching"] == [7.0]  # the compared value, on either side
    assert summary["slacks"] == [-2.0, None]


def test_a_size_input_is_probed_on_a_plain_integer_leaf(tmp_path: Path) -> None:
    summary = run(tables_record(), TABLES, "lim/s", tmp_path)["value"]["entries"]["d:sz"]
    assert shape(summary) == {
        "switching": [4],
        "slacks": [None, 3],
        "classes": [
            {"least": 1, "greatest": 3, "result": "big"},
            {"least": 4, "greatest": 4, "result": "small"},
        ],
    }
    assert type(summary["slacks"][1]) is int


def test_a_mass_input_is_probed_at_the_mass(tmp_path: Path) -> None:
    summary = run(mood_record(), MOOD, "mood/mbar", tmp_path)["value"]["entries"]["d:ms"]
    assert summary["switching"] == [0.7]
    assert summary["slacks"] == [None, 0.2]


def test_a_slack_no_float_holds_is_refused_before_the_write(tmp_path: Path) -> None:
    big = 1.7976931348623157e308
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": Parameter(t.number(), big)},
            policy=Policy(constant, version="1", reads=["d:big"]),
        )
    )
    record = decide_sync(loaded, {**STATE, "n": -big}, {}).record
    exc = refusal(record, loaded, "lim/n", tmp_path)
    assert exc.reason is R.unrepresentable_slack
    assert exc.path == "lim/n"
    key = {"record": record.record_hash, "design": loaded.design_hash, "parameter": "lim/n"}
    assert EvaluationStore(tmp_path).lookup(KIND, key) is None


# ---------------------------------------------------------------------------------------------
# The plan and the verdict


def act(r: Any) -> Plan:
    return Plan("execute", [Step("archive", {"what": Origin(literal="x")})])


def test_a_verdict_change_alone_splits_the_plan(tmp_path: Path) -> None:
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(act, version="1", reads=["d:big"]),
            requirements=[Requirement("d:big", "big")],
            tools=["archive"],
        )
    )
    plan = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]["plan"]
    verdicts = [c["result"]["verdict"]["failures"] for c in plan["classes"]]
    assert len(verdicts[1]) == len(verdicts[0]) + 1  # d:big's plan-level requirement fails
    assert plan["classes"][0]["result"]["plan"] == plan["classes"][1]["result"]["plan"]
    assert plan["switching"] == [7.0]


def test_a_change_of_context_alone_is_no_plan_change(tmp_path: Path) -> None:
    entry = ContextEntry(
        "d:big", Text("{b} over {lim}", [slot("b", ref="d:big"), slot("lim", param="lim/n")])
    )
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(constant, version="1", reads=["d:big"]),
            context=Context(entries=[entry]),
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    assert len(value["entries"]["d:big"]["classes"]) == 2
    assert len(value["plan"]["classes"]) == 1


# ---------------------------------------------------------------------------------------------
# A record of an older format, the helper version its only difference

FORMAT_5 = Path(__file__).parent / "format_5"


def test_both_jobs_take_a_record_whose_only_finding_is_the_helper_version(tmp_path: Path) -> None:
    loaded = load((FORMAT_5 / "design.json").read_bytes())
    record = read_record((FORMAT_5 / "record.json").read_bytes())
    assert deciding_inputs_sync(record, loaded, "d:weights") is not None
    value = run(record, loaded, "order/weights", tmp_path)["value"]
    assert value["unseen"] == ["weights"]
    assert value["keys"]["alpha"]["entries"] == {}  # the rule reads zeta alone
    classes = value["keys"]["zeta"]["entries"]["d:weights"]["classes"]
    # No value compares the key, so it is tried at the axis's two ends and its own.
    assert [c["least"] for c in classes] == [-(2**53 - 1), 2, 2**53 - 1]
    assert len(value["keys"]["zeta"]["plan"]["classes"]) == 1  # the plan's context alone moves


# ---------------------------------------------------------------------------------------------
# The parts


def test_the_margin_probe_rounds_down_when_the_nearest_float_is_above_the_lead() -> None:
    probe = _switching._margin_probe({"a": 0.123456789012345, "b": 1e-18})
    lead = Fraction("0.123456789012345") - Fraction("1e-18")
    assert Fraction(repr(probe)) <= lead
    assert Fraction(repr(math.nextafter(probe, math.inf))) > lead


def test_a_not_returned_ref_of_a_failed_call_is_not_determined() -> None:
    cause = {"code": "not-returned", "refs": ["t:x"]}
    rerun = [{"item": "entry", "ref": "t:x", "marker": {"kind": "Unavailable", "cause": cause}}]

    def log(outcome: object) -> list[dict[str, Any]]:
        return [
            {"item": "design"},
            {"item": "call", "outcome": outcome},
            {"item": "entry", "ref": "t:x", "call": 1},
        ]

    def unreturned(outcome: object, rerun: list[dict[str, Any]]) -> bool:
        held = log(outcome)
        mine = {i["ref"]: i for i in held if i["item"] == "entry"}
        return _switching._unreturned(mine, held, rerun)

    assert unreturned({"failed": {"code": "raised", "refs": []}}, rerun)
    assert not unreturned("returned", rerun)
    other = [{**rerun[0], "marker": {"kind": "Unavailable", "cause": {**cause, "code": "raised"}}}]
    assert not unreturned({"failed": {}}, other)
    unsure = [{**rerun[0], "marker": {"kind": "Unsure", "cause": cause}}]
    assert not unreturned({"failed": {}}, unsure)
    assert not unreturned({"failed": {}}, [{**rerun[0], "ref": "t:other"}])


def test_a_marker_a_candidate_read_and_a_missing_field_give_no_value() -> None:
    marker: dict[str, Any] = {"item": "entry", "ref": "t:x", "marker": {"kind": "Unsure"}}
    held = {"item": "entry", "ref": "t:y", "value": {"f": 2}}
    other = {"item": "entry", "ref": "t:z", "value": {"f": 3}}
    assert _switching._read_values("t:x", [marker], None) == []
    assert _switching._read_values("t:y.g", [held], None) == []
    assert _switching._read_values("t:y.f", [held, other], None) == [2]  # t:z is not read
    # A candidate read takes the members of its table's set, or nothing for a marked set.
    members = {"item": "entry", "ref": "d:set", "value": [{"f": 4}, {"f": 5}, {"g": 6}]}
    assert _switching._read_values("candidate.f", [held, members], "d:set") == [4, 5]
    assert _switching._read_values("candidate.f", [{**marker, "ref": "d:set"}], "d:set") is None
    assert _switching._read_values("candidate.f", [held], "d:set") is None


# ---------------------------------------------------------------------------------------------
# Round 2: candidate reads, integer sides, what counts as a change, carrying


def test_a_candidate_read_is_probed_at_each_members_field(tmp_path: Path) -> None:
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
    state = {"people": [grounded.BO, grounded.ANA]}
    record = decide_sync(loaded, state, {"default": grounded.answer}).record
    value = run(record, loaded, "lim/age", tmp_path)["value"]
    assert value["unseen"] == []
    assert list(value["entries"]) == ['d:level["Ana"]', 'd:level["Bo"]']  # code-point order
    assert value["entries"]['d:level["Ana"]']["switching"] == [31]
    assert value["entries"]['d:level["Ana"]']["slacks"] == [None, 13]
    assert value["entries"]['d:level["Bo"]']["switching"] == [42]
    people = {"people": [{"name": "raise", "age": 1}]}  # the set's rule raises
    marked = decide_sync(loaded, people, {"default": grounded.answer}).record
    assert run(marked, loaded, "lim/age", tmp_path / "m")["value"]["unseen"] == ["level"]


def test_a_change_of_rows_alone_is_no_change(tmp_path: Path) -> None:
    rows = Table("d:same", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "big"]])
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"same": rows, "big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(told, version="1", reads=["d:big"]),
        )
    )
    entries = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"][
        "entries"
    ]
    assert list(entries) == ["d:big"]  # d:same's rows change past 7, its value never


def test_a_run_writing_not_returned_for_a_failed_call_is_not_determined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load(
        minimal(
            state=STATE_T,
            sources={"a": Source({"t:a": t.string}, oracle="o")},
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(constant, version="1", reads=["d:big"]),
        )
    )
    binding = Scripted({"a": Failed(Cause("raised", ["t:a"]))})
    record = decide_sync(loaded, STATE, {"o": binding, "default": binding}).record
    real = _switching._replay_with

    async def unreturned(*args: Any, **kwargs: Any) -> Any:
        rerun, found = await real(*args, **kwargs)

        marker = {"kind": "Unavailable", "cause": {"code": "not-returned"}}
        log = [
            {**i, "marker": marker} if i["item"] == "entry" and i["ref"] == "t:a" else i
            for i in rerun.log
        ]
        return SimpleNamespace(log=log), found

    monkeypatch.setattr(_switching, "_replay_with", unreturned)
    plan = run(record, loaded, "lim/n", tmp_path)["value"]["plan"]
    assert [value_of(c["result"]) for c in plan["classes"]] == ["ok", "not-determined"]


def test_under_f_a_later_param_row_matched_but_not_first_does_not_carry(tmp_path: Path) -> None:
    table = Table(
        "d:f",
        "number",
        "F",
        ["t:state.n"],
        [[">= lim/n", 1.0], [">= lim/n", Param("lim/n")], ["-", 2.0]],
    )
    literal = Table("d:lit", "string", "F", ["t:state.n"], [[">= lim/n", "lim/n"], ["-", "x"]])
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"f": table, "lit": literal},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(constant, version="1", reads=["d:f"]),
        )
    )
    entries = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"][
        "entries"
    ]
    assert entries["d:f"]["switching"] == [7.0]
    assert entries["d:lit"]["switching"] == [7.0]  # a literal equal to the path is no Param


def test_the_plan_carries_when_its_policy_row_names_a_plan_holding_the_leaf(
    tmp_path: Path,
) -> None:
    loud = PlanDeclaration("inform", reply=Text("At {p}.", [slot("p", param="lim/n")]))
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=PolicyTable(["d:big"], [["big", "loud"], ["-", "quiet"]]),
            plans={"loud": loud, "quiet": PlanDeclaration("inform", reply=Text("Quiet."))},
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    assert value["plan"] == "carries"


def integer_leaf(own: int, k: int) -> tuple[LoadedDesign, Record]:
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"many": RULES["many"]},
            parameters={"lim/k": Parameter(t.integer(), own)},
            policy=Policy(constant, version="1", reads=["d:many"]),
        )
    )
    return loaded, decide_sync(loaded, {**STATE, "k": k}, {}).record


def test_an_integer_slack_at_the_bound_is_kept_and_past_it_refused(tmp_path: Path) -> None:
    top = 2**53 - 1
    loaded, record = integer_leaf(0, top - 1)
    summary = run(record, loaded, "lim/k", tmp_path)["value"]["entries"]["d:many"]
    assert summary["switching"] == [top]
    assert summary["slacks"] == [None, top]
    loaded, record = integer_leaf(-1, top - 1)
    exc = refusal(record, loaded, "lim/k", tmp_path / "past")
    assert exc.reason is R.unrepresentable_slack


def test_between_two_compared_values_the_upper_is_the_switching_value(tmp_path: Path) -> None:
    loaded = counted(5.5)
    state = {"a": 1.0, "b": 7.0, "c": math.nextafter(7.0, math.inf), "d": 10.0}
    summary = run(decide_sync(loaded, state, {}).record, loaded, "lim/c", tmp_path)["value"][
        "entries"
    ]["d:count"]
    # Between 7.0 and its next float no point lies: both are compared, so the upper is taken.
    assert summary["switching"] == [1.0, state["c"], state["c"], 10.0]


def test_only_a_number_is_a_compared_value() -> None:
    document = decode(TABLES.document)
    values = [True, "x", None, 3]
    log = [{"item": "entry", "ref": "t:state", "value": {"n": v}} for v in values]
    found, blind = _switching._compared(document, "lim/n", log)
    assert found == {Fraction(3): {"big", "part"}}
    assert blind == set()


def test_a_boundaries_member_not_naming_the_leaf_compares_nothing() -> None:
    roles = {"bar": ["other"], "thresholds": [], "margin": ["other"]}
    mass = [{"labels": ["calm"], "bounds": ["other"]}]
    rule: dict[str, Any] = {"form": "function", "boundaries": {"q:mood": {**roles, "mass": mass}}}
    document = {"rules": {"r": rule}, "policy": {"form": "function", "boundaries": None}}
    answer = {"kind": "choice", "labels": ["calm"], "probabilities": {"calm": 0.7}}
    log = [{"item": "entry", "ref": "q:mood", "value": answer}]
    assert _switching._compared(document, "lim", log) == ({}, set())
    rule["boundaries"]["q:mood"]["mass"][0]["bounds"] = ["lim"]
    assert _switching._compared(document, "lim", log) == ({Fraction("0.7"): {"r"}}, set())


def test_the_input_and_role_measures_guard_their_values() -> None:
    document = {"parameters": {"labels": {"value": ["a", "b"]}}}
    assert _switching._input(document, {"helper": "size", "params": {}}, "abc") == []
    assert _switching._input(document, {"helper": "size", "params": {}}, [1, 2]) == [2]
    over = {"probabilities": {"a": 0.7, "b": 0.6}}
    assert _switching._input(
        document, {"helper": "mass", "params": {"labels": "labels"}}, over
    ) == [1.0]
    assert _switching._measure("null", over, ()) == []  # no p_null in a probability answer
    assert _switching._measure("null", {"p_null": 0.2, "value": 3}, ()) == [0.2]
    assert _switching._measure("bar", 0.5, ()) == []


# ---------------------------------------------------------------------------------------------
# Round 3: what unseen lists


def test_a_reader_of_an_entry_holding_the_leaf_is_unseen(tmp_path: Path) -> None:
    cap = Table("d:cap", "number", "F", [], [[Param("lim/x")]])
    leaves = {"lim/x": Parameter(t.number(), 1.0), "lim/y": Parameter(t.number(), 4.0)}
    for reader in (
        Table("d:chk", BIG_SMALL, "F", ["d:cap"], [[">= lim/y", "big"], ["-", "small"]]),
        Table("d:chk", "number", "F", ["d:cap"], [["-", Input("d:cap")]]),
    ):
        loaded = load(
            minimal(
                state=STATE_T,
                rules={"cap": cap, "chk": reader},
                parameters=leaves,
                policy=Policy(constant, version="1", reads=["d:chk"]),
            )
        )
        record = decide_sync(loaded, STATE, {}).record
        value = run(record, loaded, "lim/x", tmp_path / reader.hit / str(len(reader.rows)))
        assert value["value"]["unseen"] == ["chk"]  # it compares d:cap against another leaf


@pytest.mark.parametrize("hit", ["C+", "C>"])
def test_an_aggregate_mixing_the_leaf_with_other_values_is_unseen(hit: str, tmp_path: Path) -> None:
    table = Table("d:tab", "number", hit, ["t:state.n"], [[">= lim/n", Param("lim/n")], ["-", 1.0]])
    loaded, record = one_table(table, **{"lim/n": PARAMETERS["lim/n"]})
    assert run(record, loaded, "lim/n", tmp_path)["value"]["unseen"] == ["policy", "tab"]


def test_the_components_a_load_failure_names_are_unseen(tmp_path: Path) -> None:
    table = Table(
        "d:tab", BIG_SMALL, "F", ["t:state.k"], [["[lim/y, lim/x]", "big"], ["-", "small"]]
    )
    leaves = {"lim/x": Parameter(t.integer(), 7), "lim/y": Parameter(t.integer(), 5)}
    loaded, record = one_table(table, **leaves)
    value = run(record, loaded, "lim/x", tmp_path)["value"]
    assert value["unseen"] == ["tab"]  # below lim/y the band does not load


def test_a_candidate_table_not_reading_the_leaf_is_never_unseen(tmp_path: Path) -> None:
    rules = {
        **grounded.design().rules,
        "other": Table(
            "d:other", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "small"]]
        ),
    }
    state_t = t.record(people=t.list_(grounded.PERSON), n=t.number())
    loaded = load(
        grounded.design(state=state_t, rules=rules, parameters={"lim/n": PARAMETERS["lim/n"]})
    )
    for n, people in enumerate(([grounded.BO, grounded.ANA], [{"name": "raise", "age": 1}])):
        record = decide_sync(
            loaded, {"people": people, "n": 7.0}, {"default": grounded.answer}
        ).record
        assert run(record, loaded, "lim/n", tmp_path / str(n))["value"]["unseen"] == []


def test_an_integer_side_past_the_interval_is_clipped_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded, record = one_table(RULES["many"], **{"lim/k": PARAMETERS["lim/k"]})
    record = decide_sync(loaded, {**STATE, "k": 10}, {}).record
    seen = counting_replays(monkeypatch)
    plan = run(record, loaded, "lim/k", tmp_path)["value"]["plan"]
    # 10 and 11 beside k: 11 clipped to 10, so no value outside the interval fails load.
    assert [(c["least"], c["greatest"]) for c in plan["classes"]] == [(0, 10)]
    assert seen[0] == 2  # 0 and 10 beside the own 2; 11 is never tried


def test_a_round_1_value_already_compared_leaves_nothing_unseen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Table("d:v", "number", "F", ["t:state.n"], [[">= lim/n", 3.0], ["-", 9.0]])
    check = Table("d:chk", BIG_SMALL, "F", ["d:v"], [[">= lim/n", "big"], ["-", "small"]])
    loaded = load(
        minimal(
            state=STATE_T,
            rules={"v": step, "chk": check},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=Policy(constant, version="1", reads=["d:chk"]),
        )
    )
    seen = counting_replays(monkeypatch)
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    # Round 0 over 3 and 7 finds 9 past 7; round 1 over 3, 7 and 9 recomputes 3 and 9 alone.
    assert seen[0] > 5
    assert value["unseen"] == []
