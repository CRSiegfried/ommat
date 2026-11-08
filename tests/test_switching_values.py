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
from typing import Any, cast

import pytest

import extraction as x
import test_grounded_rules as grounded
from designs import minimal, roles, slot
from ledger_logs import rehashed
from ommat import (
    CallResult,
    Cause,
    Helper,
    Input,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Param,
    Parameter,
    Question,
    Record,
    Rule,
    Source,
    Table,
    Text,
    Unsure,
    decide_sync,
    load,
    read_record,
)
from ommat import types as t
from ommat._canonical import decode, encode
from ommat._replay import FindingKind
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
from ommat.helpers import pick_with_thresholds
from scripted import Scripted, returns
from test_decide import LOADED
from test_use_answers import broken, format_7, unfinished

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")  # defects are the point here

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


def told(r: Any) -> object:
    return r.read("d:big")


def policy_rule(function: Any, *reads: str, outputs: Any = t.string) -> Rule:
    """The policy's rule, ``d:act``, of ``function`` over ``reads``."""
    return Rule("d:act", outputs, function, version="1", reads=list(reads))


def designed(policy: Rule | Table | None = None, **fields: Any) -> Any:
    """``minimal`` with ``policy`` as the policy ``d:act`` beside the rules given, or
    ``minimal``'s own policy when ``None``."""
    if policy is None:
        return minimal(**fields)
    return minimal(
        **{**fields, "rules": {**fields.get("rules", {}), "act": policy}}, policy="d:act"
    )


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
    designed(
        state=STATE_T,
        rules=RULES,
        parameters=PARAMETERS,
        policy=policy_rule(told, "d:big", outputs=BIG_SMALL),
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
    """A result's entry value, or the result itself when it is not one."""
    if type(result) is not dict:
        return result
    held = cast("dict[str, Any]", result)
    if "entry" in held:
        entry = cast("dict[str, Any] | None", held["entry"])
        return None if entry is None else entry.get("value", entry.get("marker"))
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
    monkeypatch.setattr(_switching, "_rerun", replayed)


def refusal(*args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        run(*args, **kwargs)
    return raised.value


# ---------------------------------------------------------------------------------------------
# Numeric leaves


def test_a_table_cell_is_probed_at_its_input_and_the_gap_and_end_values(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "lim/n", tmp_path)["value"]
    assert value["unseen"] == ["hid"]  # it declares the leaf and has no boundaries field
    assert list(value["entries"]) == ["d:act", "d:big", "d:echo", "d:part"]
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
    assert shape(value["entries"]["d:act"]) == shape(value["entries"]["d:big"])  # d:act is d:big


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
    record = decide_sync(loaded, {"k": 4, "j": 9}, {}).record
    value = run(record, loaded, "lim/y", tmp_path)["value"]
    assert value["unseen"] == ["tab"]  # past lim/x the band does not load
    # d:hi holds wherever the design loads; the decision is listed always.
    assert list(value["entries"]) == ["d:decision", "d:tab"]
    decision = value["entries"]["d:decision"]
    assert [list(c["result"]) for c in decision["classes"]] == [["entry"], ["not-loading"]]
    tab = value["entries"]["d:tab"]
    assert tab["switching"] == [5, 9]  # beside a not-loading class, as beside any
    assert tab["slacks"] == [None, 3]
    (big, small, last) = tab["classes"]
    assert [(c["least"], c["greatest"]) for c in (big, small, last)] == [(2, 4), (5, 5), (9, 10)]
    (failure,) = last["result"]["not-loading"]
    assert failure["check"] == 9
    assert failure["names"] == ["tab"]
    assert "tab" in failure["message"]


# ---------------------------------------------------------------------------------------------
# Finite leaves


def test_a_boolean_leaf_is_partitioned_false_first(tmp_path: Path) -> None:
    value = run(tables_record(), TABLES, "sw/on", tmp_path)["value"]
    assert value["unseen"] == []  # every value was tried
    assert shape(value["entries"]["d:on"]) == {
        "classes": [{"values": [False], "result": False}, {"values": [True], "result": True}]
    }
    # The policy's entry is always listed: d:big's "big" at both values, one class.
    assert shape(value["entries"]["d:act"]) == {
        "classes": [{"values": [False, True], "result": "big"}]
    }


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


def calm(r: Any) -> object:
    return r.read("d:th")


def thf(r: Any) -> bool:
    return r.param("mood/th")["calm"] > 2


def mood_design(wording: Text | None = None) -> LoadedDesign:
    question = Question("choice", wording or Text("Mood?"), labels=("calm", "wild"))
    return load(
        designed(
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
            policy=policy_rule(calm, "d:th", outputs=YES_NO),
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
    # calm leads whatever wild's threshold: the decision alone is listed, in one class.
    assert list(value["keys"]["wild"]["entries"]) == ["d:act"]
    assert len(value["keys"]["wild"]["entries"]["d:act"]["classes"]) == 1


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
    act = value["entries"]["d:act"]  # the policy's entry, listed as any
    classes = act["classes"]
    assert [value_of(c["result"]) for c in classes] == ["not-determined", "yes", "not-determined"]
    assert classes[1]["least"] == classes[1]["greatest"] == 0.5
    assert act["switching"] == [0.3, 0.7]  # each facing value compared
    assert act["slacks"] == [-0.2, 0.2]


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
        "format": 2,
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


def test_wrong_design_then_unfinished_then_nothing_to_replay_then_wrong_design(
    tmp_path: Path,
) -> None:
    assert refusal(format_7(), LOADED, "nope", tmp_path).reason is R.wrong_design
    assert refusal(unfinished(), LOADED, "nope", tmp_path).reason is R.unfinished
    assert refusal(broken(), LOADED, "nope", tmp_path).reason is R.nothing_to_replay
    record = tables_record()
    exc = refusal(record, LOADED, "nope", tmp_path)
    assert exc.reason is R.wrong_design
    assert exc.turn == record.record_hash


@pytest.mark.parametrize("path", ["lim", "nope", "lim//n", "Lim/n", 3, ["lim"]])
def test_unknown_path_names_no_leaf(path: Any, tmp_path: Path, no_replay: None) -> None:
    exc = refusal(tables_record(), TABLES, path, tmp_path, functions={"act": 1})
    assert exc.reason is R.unknown_path
    assert exc.path == (path if isinstance(path, str) else None)


@pytest.mark.parametrize(
    ("path", "type_"), [("text", "string"), ("tags", "list<string>"), ("names", "mapping")]
)
def test_unsupported_leaf_names_the_type(
    path: str, type_: str, tmp_path: Path, no_replay: None
) -> None:
    exc = refusal(tables_record(), TABLES, path, tmp_path, functions={"act": 1})
    assert exc.reason is R.unsupported_leaf
    assert exc.path == path
    assert type_ in str(exc)


def test_invalid_design_when_the_document_does_not_load_with_functions(
    tmp_path: Path, no_replay: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def local(r: Any) -> str:
        return "ok"

    loaded = load(
        designed(
            state=STATE_T,
            rules={"many": RULES["many"]},
            parameters={"lim/k": PARAMETERS["lim/k"]},
            policy=policy_rule(local, "d:many"),
        )
    )
    record = decide_sync(loaded, STATE, {}).record
    exc = refusal(record, loaded, "lim/k", tmp_path, functions={})
    assert exc.reason is R.invalid_design
    assert isinstance(exc.error, InvalidDesign)
    assert exc.__cause__ is exc.error
    monkeypatch.undo()  # replay again: with the function given, or none, the document loads
    assert run(record, loaded, "lim/k", tmp_path, functions={"act": local})["value"]["entries"]
    assert run(record, loaded, "lim/k", tmp_path / "own")["value"]["entries"]


def test_a_held_key_is_refused_before_any_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = tables_record()
    first = run(record, TABLES, "lim/k", tmp_path)

    async def replayed(*args: object, **kwargs: object) -> object:
        raise AssertionError("replayed")

    monkeypatch.setattr(_diff, "_replay_with", replayed)
    monkeypatch.setattr(_switching, "_rerun", replayed)
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


def constant(r: Any) -> str:
    return "ok"


def twice(r: Any) -> float:
    return 2 * r.param("lim/f") + 1


def over(r: Any) -> bool:
    return r.read("d:y") > r.param("lim/f")


def counting_replays(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen = [0]
    real = _switching._rerun

    async def counted(*args: Any, **kwargs: Any) -> Any:
        seen[0] += 1
        return await real(*args, **kwargs)

    monkeypatch.setattr(_switching, "_rerun", counted)
    return seen


def test_a_value_each_round_recomputes_ends_at_round_1_and_lists_its_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load(
        designed(
            state=STATE_T,
            rules={
                "cap": Table("d:cap", "number", "F", [], [[Param("lim/n")]]),
                "chk": Table(
                    "d:chk", BIG_SMALL, "F", ["d:cap"], [[">= lim/n", "big"], ["-", "small"]]
                ),
            },
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(constant, "d:chk"),
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
        designed(
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
            policy=policy_rule(constant, "d:over"),
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/f", tmp_path)["value"]
    # y has no boundaries, and over reads d:y, a number that holds the leaf. The policy reads
    # d:over, a boolean, which sees the leaf only where d:over switches.
    assert value["unseen"] == ["over", "y"]


def mood_pick(r: Any) -> object:
    return pick_with_thresholds(r, "q:mood", r.param("mood/th"), 0)


def told_tool(r: Any) -> object:
    return r.read("d:pick")


def test_a_reader_of_a_label_a_named_bar_decides_is_not_unseen(tmp_path: Path) -> None:
    # The turn route's shape: a label rule naming its thresholds in boundaries, and a rule
    # declaring no parameter reading it.
    loaded = load(
        designed(
            state=STATE_T,
            sources={
                "ask": Source(
                    {"q:mood": Question("choice", Text("Mood?"), labels=("calm", "wild"))},
                    needs=["t:state"],
                )
            },
            rules={
                "pick": Rule(
                    "d:pick",
                    t.label("calm", "wild"),
                    mood_pick,
                    version="1",
                    reads=["q:mood"],
                    params=["mood/th"],
                    boundaries={"q:mood": roles(thresholds=["mood/th"])},
                ),
            },
            parameters={
                "mood/th": Parameter(t.mapping(t.number(0, 1)), {"calm": 0.6, "wild": 0.6}),
            },
            policy=policy_rule(told_tool, "d:pick", outputs=t.label("calm", "wild")),
        )
    )
    value = run(mood_record(loaded), loaded, "mood/th", tmp_path)["value"]
    assert value["unseen"] == []
    calm = value["keys"]["calm"]["entries"]
    assert calm["d:pick"]["switching"] == [0.7]
    assert calm["d:act"]["switching"] == [0.7]  # the reader is still listed where it switches


def bar_label(r: Any) -> str:
    return "hi" if r.read("t:state.n") > r.param("lim/f") else "lo"


def bar_maybe(r: Any) -> object:
    return None if r.read("t:state.n") > r.param("lim/f") else "lo"


def bar_pair(r: Any) -> dict[str, object]:
    hot = r.read("t:state.n") > r.param("lim/f")
    return {"band": "hi" if hot else "lo", "hot": hot}


def bar_text(r: Any) -> str:
    return str(r.param("lim/f"))


def bar_flags(r: Any) -> list[bool]:
    return [True] * int(r.param("lim/f"))


def bar_figure(r: Any) -> dict[str, object]:
    return {"band": "lo", "at": r.param("lim/f")}


HI_LO = t.label("hi", "lo")


@pytest.mark.parametrize(
    ("function", "outputs", "unseen"),
    [
        (bar_label, HI_LO, []),
        (bar_maybe, t.nullable(HI_LO), []),
        (bar_pair, t.record(band=HI_LO, hot=t.boolean), []),
        (bar_text, t.string, ["act"]),  # a string can spell the leaf
        (bar_flags, t.list_(t.boolean), ["act"]),  # a list's length can count it
        (bar_figure, t.record(band=HI_LO, at=t.number()), ["act"]),
    ],
)
def test_a_reader_is_unseen_only_when_the_entry_can_hold_the_leaf(
    tmp_path: Path, function: Any, outputs: Any, unseen: list[str]
) -> None:
    loaded = load(
        designed(
            state=STATE_T,
            rules={
                "p": Rule(
                    "d:p",
                    outputs,
                    function,
                    version="1",
                    reads=["t:state.n"],
                    params=["lim/f"],
                    boundaries={"t:state.n": ["lim/f"]},
                ),
            },
            parameters={"lim/f": PARAMETERS["lim/f"]},
            policy=policy_rule(constant, "d:p"),
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/f", tmp_path)["value"]
    assert value["unseen"] == unseen


def test_a_held_key_refusal_carries_the_key_the_store_looks_up(tmp_path: Path) -> None:
    record = mood_record()
    results = EvaluationStore(tmp_path)
    first = switching_values_sync(record, MOOD, "mood/bar", results=results)
    with pytest.raises(EvaluationRefusal) as caught:
        switching_values_sync(record, MOOD, "mood/bar", results=results)
    assert caught.value.reason is R.held_key
    assert caught.value.key is not None
    assert results.lookup(KIND, caught.value.key[0]) == first


def three_way(own: float) -> LoadedDesign:
    return load(
        designed(
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
            policy=policy_rule(constant, "d:cmp"),
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
    assert summary["switching"] == [7.0]  # once, though it bounds both sides of eq
    assert summary["slacks"] == slacks


def counted(own: float) -> LoadedDesign:
    fields = ("a", "b", "c", "d")
    return load(
        designed(
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
            policy=policy_rule(constant, "d:count"),
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
        designed(
            state=STATE_T,
            rules={table.provides.removeprefix("d:"): table},
            parameters=parameters,
            policy=policy_rule(constant, table.provides),
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
        designed(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": Parameter(t.number(), big)},
            policy=policy_rule(constant, "d:big"),
        )
    )
    record = decide_sync(loaded, {**STATE, "n": -big}, {}).record
    exc = refusal(record, loaded, "lim/n", tmp_path)
    assert exc.reason is R.unrepresentable_slack
    assert exc.path == "lim/n"
    key = {"record": record.record_hash, "design": loaded.design_hash, "parameter": "lim/n"}
    assert EvaluationStore(tmp_path).lookup(KIND, key) is None


# ---------------------------------------------------------------------------------------------
# A record whose only difference is the helper version


def test_both_jobs_take_a_record_whose_only_finding_is_the_helper_version(tmp_path: Path) -> None:
    doc = json.loads(tables_record().document)
    doc["log"][0]["helpers"] -= 1
    record = read_record(rehashed(doc))
    assert deciding_inputs_sync(record, TABLES, "d:on") == {"t:state.flag": "not-deciding"}
    value = run(record, TABLES, "lim/n", tmp_path)["value"]
    assert value == run(tables_record(), TABLES, "lim/n", tmp_path / "own")["value"]


# ---------------------------------------------------------------------------------------------
# The parts


def test_the_margin_probe_rounds_down_when_the_nearest_float_is_above_the_lead() -> None:
    probe = _switching._margin_probe({"a": 0.123456789012345, "b": 1e-18})
    lead = Fraction("0.123456789012345") - Fraction("1e-18")
    assert Fraction(repr(probe)) <= lead
    assert Fraction(repr(math.nextafter(probe, math.inf))) > lead


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
    assert list(value["entries"]) == ["d:act", 'd:level["Ana"]', 'd:level["Bo"]']  # code-point
    assert value["entries"]['d:level["Ana"]']["switching"] == [31]
    assert value["entries"]['d:level["Ana"]']["slacks"] == [None, 13]
    assert value["entries"]['d:level["Bo"]']["switching"] == [42]
    people = {"people": [{"name": "raise", "age": 1}]}  # the set's rule raises
    marked = decide_sync(loaded, people, {"default": grounded.answer}).record
    assert run(marked, loaded, "lim/age", tmp_path / "m")["value"]["unseen"] == ["level"]


def test_a_change_of_rows_alone_is_no_change(tmp_path: Path) -> None:
    rows = Table("d:same", BIG_SMALL, "F", ["t:state.n"], [[">= lim/n", "big"], ["-", "big"]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"same": rows, "big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(told, "d:big", outputs=BIG_SMALL),
        )
    )
    entries = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"][
        "entries"
    ]
    assert list(entries) == ["d:act", "d:big"]  # d:same's rows change past 7, its value never


def refs_by_big(r: Any) -> object:
    return Unsure(Cause("abstained", ["d:big"] if r.read("d:big") == "big" else []))


def test_a_change_of_the_policys_marker_refs_alone_switches_the_decision(tmp_path: Path) -> None:
    loaded = load(
        designed(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(refs_by_big, "d:big", outputs=BIG_SMALL),
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    act = value["entries"]["d:act"]
    assert act["switching"] == [7.0]
    assert [c["result"]["entry"]["marker"]["cause"]["refs"] for c in act["classes"]] == [
        ["d:big"],
        [],
    ]


def test_a_run_whose_replay_asks_a_stale_question_is_not_determined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load(
        designed(
            state=STATE_T,
            sources={"a": Source({"t:a": t.string}, oracle="o")},
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(constant, "d:big"),
        )
    )
    binding = Scripted({"a": returns({"t:a": "x"})})
    record = decide_sync(loaded, STATE, {"o": binding, "default": binding}).record
    real = _switching._rerun
    stale = {"kind": FindingKind.stale_question, "ref": "t:a", "source": "a", "clause": "needs"}

    async def staled(*args: Any, **kwargs: Any) -> Any:
        rerun, found, fixture = await real(*args, **kwargs)
        return rerun, [*found, stale], fixture

    monkeypatch.setattr(_switching, "_rerun", staled)
    big = run(record, loaded, "lim/n", tmp_path)["value"]["entries"]["d:big"]
    # Its own value's result is the record's, not replayed; every other value's is undetermined.
    assert [value_of(c["result"]) for c in big["classes"]] == ["big", "not-determined"]


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
        designed(
            state=STATE_T,
            rules={"f": table, "lit": literal},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(constant, "d:f"),
        )
    )
    entries = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"][
        "entries"
    ]
    assert entries["d:f"]["switching"] == [7.0]
    assert entries["d:lit"]["switching"] == [7.0]  # a literal equal to the path is no Param


def test_a_policy_table_whose_row_outputs_the_leaf_carries(tmp_path: Path) -> None:
    table = Table("d:act", "number", "F", ["d:big"], [["big", Param("lim/n")], ["-", 0.0]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=table,
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    assert value["entries"]["d:act"] == "carries"


def test_a_policy_change_of_rows_alone_is_no_switch(tmp_path: Path) -> None:
    table = Table("d:act", "string", "F", ["d:big"], [["big", "quiet"], ["-", "quiet"]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"big": RULES["big"]},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=table,
        )
    )
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    assert value["entries"]["d:big"]["switching"] == [7.0]  # the winning row moves there
    assert list(value["entries"]) == ["d:act", "d:big"]
    assert shape(value["entries"]["d:act"])["classes"] == [
        {"least": 5.0, "greatest": 8.0, "result": "quiet"}
    ]


def integer_leaf(own: int, k: int) -> tuple[LoadedDesign, Record]:
    loaded = load(
        designed(
            state=STATE_T,
            rules={"many": RULES["many"]},
            parameters={"lim/k": Parameter(t.integer(), own)},
            policy=policy_rule(constant, "d:many"),
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
    assert summary["switching"] == [1.0, state["c"], 10.0]


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
    document = {"rules": {"r": rule}}
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
            designed(
                state=STATE_T,
                rules={"cap": cap, "chk": reader},
                parameters=leaves,
                policy=policy_rule(constant, "d:chk"),
            )
        )
        record = decide_sync(loaded, STATE, {}).record
        value = run(record, loaded, "lim/x", tmp_path / reader.hit / str(len(reader.rows)))
        assert value["value"]["unseen"] == ["chk"]  # it compares d:cap against another leaf


@pytest.mark.parametrize("hit", ["C+", "C>"])
def test_an_aggregate_mixing_the_leaf_with_other_values_is_unseen(hit: str, tmp_path: Path) -> None:
    table = Table("d:tab", "number", hit, ["t:state.n"], [[">= lim/n", Param("lim/n")], ["-", 1.0]])
    loaded, record = one_table(table, **{"lim/n": PARAMETERS["lim/n"]})
    assert run(record, loaded, "lim/n", tmp_path)["value"]["unseen"] == ["act", "tab"]


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
    value = run(record, loaded, "lim/k", tmp_path)["value"]
    # 10 and 11 beside k: 11 clipped to 10, so no value outside the interval fails load.
    assert value["unseen"] == []
    assert list(value["entries"]) == ["d:act"]  # the decision, listed always, never switches
    assert value["entries"]["d:act"]["switching"] == []
    assert [(c["least"], c["greatest"]) for c in value["entries"]["d:act"]["classes"]] == [(0, 10)]
    assert seen[0] == 2  # 0 and 10 beside the own 2; 11 is never tried


def test_a_round_1_value_already_compared_leaves_nothing_unseen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = Table("d:v", "number", "F", ["t:state.n"], [[">= lim/n", 3.0], ["-", 9.0]])
    check = Table("d:chk", BIG_SMALL, "F", ["d:v"], [[">= lim/n", "big"], ["-", "small"]])
    loaded = load(
        designed(
            state=STATE_T,
            rules={"v": step, "chk": check},
            parameters={"lim/n": PARAMETERS["lim/n"]},
            policy=policy_rule(constant, "d:chk"),
        )
    )
    seen = counting_replays(monkeypatch)
    value = run(decide_sync(loaded, STATE, {}).record, loaded, "lim/n", tmp_path)["value"]
    # Round 0 over 3 and 7 finds 9 past 7; round 1 over 3, 7 and 9 recomputes 3 and 9 alone.
    assert seen[0] > 5
    assert value["unseen"] == []


# ---------------------------------------------------------------------------------------------
# Round 4: carrying read from every run, an aggregate alone with its Param row


def test_a_param_row_giving_the_recorded_value_still_carries(tmp_path: Path) -> None:
    table = Table(
        "d:tab",
        "number",
        "F",
        ["t:state.n"],
        [["[lim/x, lim/y]", 2.0], ["> lim/z", Param("lim/y")], ["-", 1.0]],
    )
    leaves = {
        "lim/x": Parameter(t.number(), -2.0),
        "lim/y": Parameter(t.number(), 7.5),
        "lim/z": Parameter(t.number(), 2.5),
    }
    loaded = load(
        designed(
            state=STATE_T,
            rules={"tab": table},
            parameters=leaves,
            policy=policy_rule(constant, "d:tab"),
        )
    )
    record = decide_sync(loaded, {**STATE, "n": 3.0}, {}).record
    # Below n the Param row gives d:tab the leaf; no value tried there need differ from 2.0.
    value = run(record, loaded, "lim/y", tmp_path)["value"]
    assert value["entries"]["d:tab"] == "carries"


@pytest.mark.parametrize("hit", ["F", "C+", "C<", "C>"])
def test_an_aggregate_whose_one_row_is_param_is_not_unseen(hit: str, tmp_path: Path) -> None:
    loaded, record = one_table(
        Table("d:tab", "number", hit, [], [[Param("lim/n")]]), **{"lim/n": PARAMETERS["lim/n"]}
    )
    value = run(record, loaded, "lim/n", tmp_path)["value"]
    assert value["entries"]["d:tab"] == "carries"
    assert value["unseen"] == ["act"]  # the policy reads d:tab, which holds the leaf


# ---------------------------------------------------------------------------------------------
# The batch form: many records and leaves on one loop, the single job's entries


def test_the_batch_gives_the_single_jobs_entries_in_record_then_path_order(tmp_path: Path) -> None:
    for record, ld, ps in (
        (tables_record(), TABLES, ["lim/n", "lim/k"]),
        (mood_record(), MOOD, ["mood/th"]),
    ):
        singles = [run(record, ld, p, tmp_path / "single" / p.replace("/", "_")) for p in ps]
        batch = _switching.switching_values_batch_sync(
            [record, record],
            ld,
            ps,
            results=EvaluationStore(tmp_path / "batch" / ps[0].replace("/", "_")),
        )
        # The second record is the first again: its entries are held, and come back as held.
        assert [encode(e) for e in batch] == [encode(s) for s in [*singles, *singles]]


def test_the_batch_returns_a_held_entry_rather_than_refusing(tmp_path: Path) -> None:
    record = tables_record()
    store = EvaluationStore(tmp_path)
    first = run(record, TABLES, "lim/n", tmp_path)
    with pytest.raises(EvaluationRefusal) as raised:
        run(record, TABLES, "lim/n", tmp_path)
    assert raised.value.reason is R.held_key
    [again] = _switching.switching_values_batch_sync([record], TABLES, ["lim/n"], results=store)
    assert again == first


def test_the_batch_checks_every_record_and_path_before_any_replay(
    tmp_path: Path, no_replay: None
) -> None:
    store = EvaluationStore(tmp_path)
    with pytest.raises(EvaluationRefusal) as raised:
        _switching.switching_values_batch_sync(
            [tables_record(), {"not": "a record"}], TABLES, ["lim/n"], results=store
        )
    assert raised.value.reason is R.invalid_record
    with pytest.raises(EvaluationRefusal) as raised:
        _switching.switching_values_batch_sync(
            [tables_record()], TABLES, ["lim/n", "text"], results=store
        )
    assert raised.value.reason is R.unsupported_leaf
    assert (
        store.lookup(
            KIND,
            {
                "record": tables_record().record_hash,
                "design": TABLES.design_hash,
                "parameter": "lim/n",
            },
        )
        is None
    )


def test_the_batch_refuses_what_is_not_a_sequence(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    with pytest.raises(TypeError, match="records are a"):
        _switching.switching_values_batch_sync(
            cast(Any, tables_record()), TABLES, ["lim/n"], results=store
        )
    with pytest.raises(TypeError, match="paths are a"):
        _switching.switching_values_batch_sync(
            [tables_record()], TABLES, cast(Any, "lim/n"), results=store
        )


def test_the_batch_is_exported_and_its_sync_twin_refuses_a_running_loop(tmp_path: Path) -> None:
    from ommat.evaluation import switching_values_batch, switching_values_batch_sync

    store = EvaluationStore(tmp_path)
    record = tables_record()

    async def inside() -> None:
        with pytest.raises(RuntimeError, match="switching_values_batch_sync was called inside"):
            switching_values_batch_sync([record], TABLES, ["lim/n"], results=store)
        [entry] = await switching_values_batch([record], TABLES, ["lim/n"], results=store)
        assert entry["key"]["parameter"] == "lim/n"

    asyncio.run(inside())
