# pyright: reportPrivateUsage=false
"""The switching-value job: each leaf type varied, each source of a compared value probed,
``unseen``, ``carries``, not-determined and not-loading ranges, the refusals in their order,
and the entry written to the store, looked up from it and returned as plain data reaching no
``Record``."""

import asyncio
import copy
import json
from pathlib import Path
from typing import Any, cast

import pytest

import extraction as x
from designs import minimal, slot
from ommat import (
    CallResult,
    Helper,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Param,
    Parameter,
    Plan,
    Policy,
    Question,
    Record,
    Rule,
    Source,
    Table,
    Text,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._canonical import decode
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    _diff,
    _switching,
    switching_values,
    switching_values_sync,
)
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


def mode(r: Any) -> bool:
    return r.param("sw/mode") == "a"


def flip(r: Any) -> bool:
    return FLIP[0]


def told(r: Any) -> Plan:
    return Plan("inform", reply=r.read("d:big"))


def plan_of(r: Any) -> Plan:
    return Plan("inform", reply="ok")


STATE_T = t.record(n=t.number(), k=t.integer(), flag=t.boolean)
STATE: dict[str, Any] = {"n": 7.0, "k": 4, "flag": True}
RULES: dict[str, Any] = {
    "big": Table("d:big", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "small"]]),
    "many": Table("d:many", BIG_SMALL, "F", ["t:state.k"], [[">= lim/k", "big"], ["-", "small"]]),
    "echo": Table("d:echo", "number", "F", [], [[Param("lim/n")]]),
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
        "d:mode", t.boolean, mode, version="1", reads=["t:state.flag"], params=["sw/mode"]
    ),
    "flip": Rule("d:flip", t.boolean, flip, version="1", reads=["t:state.flag"]),
}
PARAMETERS: dict[str, Any] = {
    "lim/n": Parameter(t.number(), 5.0),
    "lim/k": Parameter(t.integer(0, 10), 2),
    "lim/f": Parameter(t.number(), 1.0),
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
    assert list(value["entries"]) == ["d:big", "d:echo"]
    assert shape(value["entries"]["d:big"]) == {
        "switching": [7.0],
        "slacks": [None, 2.0],
        "classes": [
            {"least": 5.0, "greatest": 7.0, "result": "big"},
            {"least": 8.0, "greatest": 8.0, "result": "small"},
        ],
    }
    assert value["entries"]["d:echo"] == "carries"  # a Param output, changing at every value
    assert shape(value["plan"]) == shape(value["entries"]["d:big"])  # the reply is d:big


def test_an_integer_leaf_takes_integer_points_within_its_interval(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "lim/k", tmp_path)["value"]
    assert value["unseen"] == []
    assert shape(value["entries"]["d:many"]) == {
        "switching": [4],
        "slacks": [None, 2],
        "classes": [
            {"least": 0, "greatest": 4, "result": "big"},
            {"least": 10, "greatest": 10, "result": "small"},
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
    assert plan["switching"] == []
    assert plan["slacks"] == [None, None]
    (first, last) = plan["classes"]
    assert (first["least"], first["greatest"], last["least"]) == (1.0, 7.0, 8.0)
    (failure,) = last["result"]["not-loading"]
    assert failure["check"] == 9
    assert "band" in failure["message"]
    assert set(failure) == {"check", "names", "message"}


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
        "classes": [{"values": ["a"], "result": True}, {"values": ["b", "c"], "result": False}]
    }


# ---------------------------------------------------------------------------------------------
# The pick helpers and a numeric mapping


def calm(r: Any) -> Plan:
    return Plan("inform", reply=r.read("d:th"))


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
            },
            parameters={
                "mood/th": Parameter(t.mapping(t.number(0, 1)), {"calm": 0.6, "wild": 0.6}),
                "mood/m": Parameter(t.number(0, 1), 0.2),
                "mood/bar": Parameter(t.number(0, 1), 0.5),
                "mood/m2": Parameter(t.number(0, 1), 0.0),
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
    assert value["unseen"] == []
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
    assert value["entries"] == {}
    classes = value["plan"]["classes"]
    assert [value_of(c["result"]) for c in classes] == ["not-determined", "yes", "not-determined"]
    assert classes[1]["least"] == classes[1]["greatest"] == 0.5
    assert value["plan"]["switching"] == []


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


@pytest.mark.parametrize("path", ["lim", "nope", "lim//n", "Lim/n", 3])
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
