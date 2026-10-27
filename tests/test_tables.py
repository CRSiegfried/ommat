# pyright: reportPrivateUsage=false
"""Decision tables (§decision-tables, §table-notation): a table's notation and tagged form,
the load checks it adds, its evaluation under each hit policy and the marker contract, the
policy table and its plan declarations, replay over a table turn, the grid and the lints."""

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    Binding,
    CallResult,
    Cause,
    CauseCode,
    Design,
    FindingKind,
    Helper,
    Input,
    InvalidDesign,
    LoadedDesign,
    Origin,
    Param,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Record,
    Rule,
    Source,
    Step,
    Table,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import encode, sha256_ref
from ommat.helpers import dead_questions, unsure_to_execute

STATE = t.record(
    n=t.integer(),
    m=t.integer(),
    x=t.number(),
    y=t.number(),
    word=t.label("a", "b", "c"),
    flag=t.boolean,
    maybe=t.nullable(t.integer()),
    items=t.list_(t.integer()),
    name=t.string,
)
BASE = {
    "n": 3,
    "m": 5,
    "x": 0.5,
    "y": 0.25,
    "word": "a",
    "flag": True,
    "maybe": None,
    "items": [1, 2],
    "name": "Ada",
}
LEAVES = {
    "lo": Parameter(t.number(), 0.25),
    "hi": Parameter(t.number(), 0.75),
    "bar": Parameter(t.number(0, 1), 0.6),
    "margin": Parameter(t.number(0, 1), 0.2),
    "five": Parameter(t.integer(), 5),
    "fallback": Parameter(t.integer(), 7),
}


def show(r: Any) -> Any:
    from ommat import Plan

    return Plan("inform", reply="done")


def design(*tables: tuple[str, Any], reads: tuple[str, ...] = (), **changes: Any) -> Design:
    fields: dict[str, Any] = {
        "state": STATE,
        "rules": dict(tables),
        "parameters": LEAVES,
        "policy": Policy(show, version="1", reads=list(reads)),
    }
    fields.update(changes)
    return Design(**fields)


def one(
    table: Table, state: Mapping[str, Any] | None = None, rules: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The entry ``table``, under rule id ``tab``, writes over ``state``."""
    loaded = load(design(("tab", table), *(rules or {}).items()))
    record = decide_sync(loaded, {**BASE, **(state or {})}, {}).record
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == "d:tab")


def value(
    table: Table, state: Mapping[str, Any] | None = None, rules: Mapping[str, Any] | None = None
) -> Any:
    entry = one(table, state, rules)
    return entry.get("value", entry.get("marker"))


def failures(given: Design | bytes | dict[str, Any]) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as raised:
        load(given)
    return [(f.check, f.message) for f in raised.value.failures]


def refused(table: Table, check: int, says: str, **changes: Any) -> None:
    found = failures(design(("tab", table), **changes))
    assert any(c == check and says in m for c, m in found), found


def tab(
    hit: str, inputs: Sequence[Any], rows: list[Any], outputs: str = "integer", **kw: Any
) -> Table:
    return Table("d:tab", outputs, hit, inputs, rows, **kw)


# ---------------------------------------------------------------------------------------------
# The notation and the tagged form


NOTATION = tab(
    "F",
    ["t:state.n", "t:state.word", "t:state.x", "t:state.items", "t:state.m"],
    [
        ["-", "a", "-", "-", "-", 1],
        ["-", "{a, b}", "-", "-", "-", 2],
        [">= 1", "-", "> lo", "-", "<= five", 3],
        ["< 0", "-", "[lo, hi]", "-", "-", 4],
        ["-", "-", "-", "-", ">= t:state.n", 5],
        ["in t:state.items", "-", "-", "-", "-", 6],
        [Unsure, "Unavailable", "-", "-", "-", 7],
        ["[0, 10]", "-", "-", "-", "-", Param("fallback")],
        ["-", "-", "-", "-", "-", Input("t:state.n")],
    ],
)
TAGGED = tab(
    "F",
    NOTATION.inputs,
    [
        ["-", {"is": "a"}, "-", "-", "-", 1],
        ["-", {"one_of": ["b", "a"]}, "-", "-", "-", 2],
        [
            {">=": {"literal": 1}},
            "-",
            {">": {"param": "lo"}},
            "-",
            {"<=": {"param": "five"}},
            3,
        ],
        [{"<": {"literal": 0}}, "-", {"between": [{"param": "lo"}, {"param": "hi"}]}, "-", "-", 4],
        ["-", "-", "-", "-", {">=": {"ref": "t:state.n"}}, 5],
        [{"in": "t:state.items"}, "-", "-", "-", "-", 6],
        [{"marker": "Unsure"}, {"marker": "Unavailable"}, "-", "-", "-", 7],
        [{"between": [{"literal": 0}, {"literal": 10}]}, "-", "-", "-", "-", Param("fallback")],
        ["-", "-", "-", "-", "-", Input("t:state.n")],
    ],
)


def test_notation_and_the_tagged_form_load_to_one_hash_and_round_trip() -> None:
    a, b = load(design(("tab", NOTATION))), load(design(("tab", TAGGED)))
    assert a.design_hash == b.design_hash
    again = load(a.document)
    assert again.document == a.document
    assert again.design_hash == a.design_hash
    rows = json.loads(a.document)["rules"]["tab"]["rows"]
    assert rows[1]["cells"][1] == {"one_of": ["a", "b"]}  # an unordered list, in byte order
    assert rows[7]["output"] == {"param": "fallback"}
    assert rows[8]["output"] == {"ref": "t:state.n"}
    assert json.loads(a.document)["rules"]["tab"]["inputs"][0] == {
        "read": "t:state.n",
        "helper": None,
        "params": {},
    }


@pytest.mark.parametrize(
    ("cell", "tagged"),
    [
        ("-", "-"),
        ("  -  ", "-"),
        ("Unasked", {"marker": "Unasked"}),
        (">=lo", {">=": {"param": "lo"}}),
        (">  -3", {">": {"literal": -3}}),
        ("< t:state.n", {"<": {"ref": "t:state.n"}}),
        ("[ lo , 10 ]", {"between": [{"param": "lo"}, {"literal": 10}]}),
        ("{ a ,b }", {"one_of": ["a", "b"]}),
        ("in   t:state.items", {"in": "t:state.items"}),
        ("in progress", {"is": "in progress"}),
        ("-5", {"is": "-5"}),
        ("1", {"is": "1"}),
        (1, {"is": 1}),
        (True, {"is": True}),
        (Unavailable, {"marker": "Unavailable"}),
    ],
)
def test_a_string_cell_is_read_by_how_it_begins(cell: Any, tagged: Any) -> None:
    from ommat._table import cell_form

    assert cell_form(cell) == tagged


@pytest.mark.parametrize("cell", [">=", "> ", "[1, 2", "[1]", "{a}", "{a,,b}", "[t:state.n, 1]"])
def test_a_string_cell_that_begins_a_form_and_does_not_complete_it_fails_check_10(
    cell: str,
) -> None:
    refused(tab("F", ["t:state.n"], [[cell, 1]]), 10, "row 1")


@pytest.mark.parametrize("cell", [1.5, None, object()])
def test_a_cell_of_another_python_type_fails_check_10(cell: Any) -> None:
    refused(tab("F", ["t:state.n"], [[cell, 1]]), 10, "is not a cell")


def test_a_label_the_notation_would_read_as_a_form_is_written_tagged() -> None:
    labels = t.label("-", "Unsure", "x")
    table = Table("d:tab", "integer", "F", ["t:state.w"], [[{"is": "-"}, 1], ["-", 2]])
    for label, expected in (("-", 1), ("Unsure", 2), ("x", 2)):
        loaded = load(design(("tab", table), state=t.record(w=labels)))
        record = decide_sync(loaded, {"w": label}, {}).record
        assert next(i for i in record.log if i.get("ref") == "d:tab")["value"] == expected


def test_a_table_and_its_members_are_snapshot_when_built() -> None:
    rows: list[Any] = [["-", 1]]
    table = tab("F", ["t:state.n"], rows)
    rows.append(["-", 2])
    assert len(table.rows) == 1


def test_a_format_6_document_holding_a_table_loads_to_its_own_hash() -> None:
    doc = json.loads((Path(__file__).parent / "format_6" / "design.json").read_bytes())
    doc["rules"]["flag"] = {
        "form": "table",
        "provides": "d:flag",
        "outputs": "boolean",
        "hit": "F",
        "order": None,
        "inputs": [{"read": "t:state.request", "helper": None, "params": {}}],
        "rows": [{"cells": [{"is": "x"}], "output": {"literal": True}}],
        "candidates": False,
        "grounded": None,
    }
    data = encode(doc)
    found = failures(data)
    # The request is a string: a value cell over it is refused, under format 6 as under 7.
    assert [c for c, _ in found] == [9]
    doc["rules"]["flag"]["rows"] = [{"cells": ["-"], "output": {"literal": True}}]
    data = encode(doc)
    loaded = load(data)
    assert loaded.design_hash == sha256_ref(data)
    assert loaded.document == data


def test_a_document_rows_member_set_differing_fails_load() -> None:
    doc = json.loads(load(design(("tab", NOTATION))).document)
    doc["rules"]["tab"]["rows"][0]["extra"] = 1
    assert failures(doc)


# ---------------------------------------------------------------------------------------------
# Load


def test_check_1_a_one_of_repeat_and_a_plan_name_clash() -> None:
    refused(tab("F", ["t:state.word"], [[{"one_of": ["a", "a"]}, 1]]), 1, "twice in a one_of")
    found = failures(
        dataclasses.replace(
            policy_design(), tools=["page", "notify"], plans={**PLANS, "page": PLANS["tell"]}
        )
    )
    assert any(c == 1 and "declared more than once" in m for c, m in found)


def test_check_3_an_input_origin_and_slot_need_a_provider() -> None:
    refused(tab("F", ["d:none"], [["-", 1]]), 3, "no provider")
    bad = {
        **PLANS,
        "tell": PlanDeclaration(
            "inform", reply=Text("Hi {n}.", [{"name": "n", "origin": {"ref": "d:none"}}])
        ),
    }
    found = failures(dataclasses.replace(policy_design(), plans=bad))
    assert any(c == 3 and "d:none" in m for c, m in found)


@pytest.mark.parametrize(
    ("table", "says"),
    [
        (Table("d:other", "integer", "F", [], []), "provides"),
        (
            tab(
                "F",
                [Helper("pick_with_margin", "t:state.n", bar="bar", margin="margin")],
                [["-", 1]],
            ),
            "not a question",
        ),
        (tab("F", [Helper("size", "q:urgent")], [["-", 1]]), "no list"),
        (
            tab("F", [Helper("extract", "q:urgent", null_bar="bar")], [["-", 1]]),
            "not an extraction",
        ),
    ],
)
def test_check_4_a_tables_provides_and_helper_reads(table: Table, says: str) -> None:
    found = failures(asked(("tab", table)))
    assert any(c == 4 and says in m for c, m in found), found


def test_check_4_the_policy_table_reads_no_question() -> None:
    found = failures(asked(policy=PolicyTable(["q:urgent"], [["-", "ask"]]), plans=PLANS))
    assert any(c == 4 and "not at level d: or t:" in m for c, m in found), found


@pytest.mark.parametrize("helper", ["determined", "agree", "pick_each_with_margin", "nope"])
def test_check_6_a_helper_outside_the_tables_library(helper: str) -> None:
    refused(tab("F", [Helper(helper, "t:state.items")], [["-", 1]]), 6, "not a helper a table")


def test_check_7_a_cell_or_pass_through_ref_out_of_the_grammar() -> None:
    refused(tab("F", ["t:state.n"], [["in d:Bad", 1]]), 7, "not a ref")
    refused(tab("F", ["t:state.n"], [["-", Input("d:")]]), 7, "not a ref")


def test_check_8_a_leaf_named_and_its_type() -> None:
    refused(tab("F", ["t:state.x"], [["> nope", 1]]), 8, "not a leaf")
    refused(tab("F", ["t:state.x"], [["-", Param("nope")]]), 8, "not a leaf")
    leaves = {**LEAVES, "word": Parameter(t.string, "a")}
    refused(tab("F", ["t:state.x"], [["> word", 1]]), 8, "not numeric", parameters=leaves)
    found = failures(
        asked(
            (
                "tab",
                tab(
                    "F",
                    [Helper("pick_with_margin", "q:urgent", bar="five", margin="margin")],
                    [["-", 1]],
                ),
            ),
        )
    )
    assert any(c == 8 and "does not conform to number [0, 1]" in m for c, m in found), found


@pytest.mark.parametrize(
    ("table", "says"),
    [
        (tab("F", ["t:state.n"], [["-", "-", 1]]), "2 cells for its 1 inputs"),
        (tab("F", ["t:state.word"], [["d", 1]]), "not a label of the input"),
        (tab("F", ["t:state.x"], [[{"is": 1}, 1]]), "only over a label, boolean or integer"),
        (tab("F", ["t:state.flag"], [[{"is": "yes"}, 1]]), "not a boolean"),
        (tab("F", ["t:state.word"], [["> lo", 1]]), "which is not numeric"),
        (tab("F", ["t:state.x"], [[">= 1", 1]]), "integer literal over"),
        (tab("F", ["t:state.n", "t:state.x"], [["in t:state.x", "-", 1]]), "is not a list"),
        (tab("F", ["t:state.n"], [["in t:state.items", 1]]), "no input of it reads"),
        (tab("F", ["t:state.n", "t:state.n"], [["-", "-", Input("t:state.n")]]), "two inputs"),
        (tab("F", ["t:state.x"], [[">= t:state.word", 1]]), "no input of it reads"),
        (tab("F", ["t:state.x"], [["-", Input("t:state.x")]]), "does not conform to integer"),
        (tab("F", ["t:state.x"], [["-", "many"]]), "is not a value of integer"),
        (tab("F", ["t:state.x"], [["-", Param("lo")]]), "does not conform to integer"),
        (tab("F", ["t:state.x"], [["[hi, lo]", 1]]), "lower bound above its upper"),
        (tab("P", ["t:state.x"], [["-", 1]]), "a P table with no order"),
        (tab("P", ["t:state.x"], [["-", 1], ["-", Unsure]], order=[1]), "cannot rank"),
        (tab("P", ["t:state.x"], [["-", 1], ["-", 2]], order=[1]), "does not list"),
        (tab("P", ["t:state.x"], [["-", 1]], order=[1, 1]), "lists a value twice"),
        (tab("F", ["t:state.x"], [["-", 1]], order=[1]), "only a P table has"),
        (tab("C#", ["t:state.x"], [["-"]], outputs="number"), "is not integer"),
        (
            tab("C+", ["t:state.x"], [["-", "a"]], outputs='label{"a"}'),
            "does not conform to number",
        ),
        (tab("C+", ["t:state.x"], [["-", Unsure]], outputs="integer"), "cannot rank or aggregate"),
        (tab("U", ["t:state.n"], [[">= 1", 1], ["<= 3", 2]]), "overlap, which a U table refuses"),
        (tab("U", ["t:state.n"], [["Unsure", 1], ["-", 2]]), "overlap, which a U table refuses"),
        (tab("A", ["t:state.n"], [[">= 1", 1], ["<= 3", 2]]), "give different outputs"),
        (
            tab("F", [Helper("size", "t:state.items", bar="bar")], [["-", 1]]),
            "size takes no argument",
        ),
    ],
)
def test_check_9_a_tables_clauses(table: Table, says: str) -> None:
    refused(table, 9, says)


@pytest.mark.parametrize(
    "table",
    [
        tab("U", ["t:state.n"], [[">= 2", 1], ["< 2", 2]]),
        tab("U", ["t:state.n"], [[">= 1", 1], ["<= 0", 2], ["Unsure", 3]]),
        tab("U", ["t:state.n", "t:state.m"], [[">= t:state.m", "-", 1], ["< t:state.m", "-", 2]]),
        tab("U", ["t:state.n"], [["-", Input("t:state.n")], ["Unsure", 2]]),
        tab("U", ["t:state.n"], [["> 1", 1], ["< 2", 2]]),  # no integer between
        tab("A", ["t:state.n"], [[">= 1", 1], ["<= 3", 1]]),
    ],
)
def test_rows_that_never_both_match_pass_the_overlap_test(table: Table) -> None:
    load(design(("tab", table)))


def test_check_9_a_mass_inputs_labels_and_a_bare_question() -> None:
    leaves = {"labels": Parameter(t.list_(t.string), ["yes", "maybe"])}
    found = failures(
        asked(
            ("tab", tab("F", [Helper("mass", "q:urgent", labels="labels")], [["-", 1]])),
            parameters={**LEAVES, **leaves},
        )
    )
    assert any(c == 9 and "not a label of" in m for c, m in found), found
    found = failures(asked(("tab", tab("F", ["q:urgent"], [["yes", 1]]))))
    assert any(c == 9 and "only - and marker cells" in m for c, m in found), found
    found = failures(asked(("tab", tab("F", ["q:urgent"], [["-", Input("q:urgent")]]))))
    assert any(c == 9 and "cannot be passed through" in m for c, m in found), found


def test_check_9_a_boundaries_field_a_table_already_covers() -> None:
    reader = Rule(
        "d:r",
        "integer",
        lambda r: 1,
        version="1",
        reads=["t:state.x"],
        boundaries={"t:state.x": ["lo"]},
    )
    refused(
        tab("F", ["t:state.x"], [["> lo", 1], ["-", 2]]),
        9,
        "already compares",
        rules={"tab": tab("F", ["t:state.x"], [["> lo", 1], ["-", 2]]), "r": reader},
    )


@pytest.mark.parametrize(
    ("plans", "rows", "says"),
    [
        ({"x": PlanDeclaration("broken")}, [["-", "x"]], "kind is broken"),
        ({"x": PlanDeclaration("inform")}, [["-", "x"]], "which declares a reply"),
        ({"x": PlanDeclaration("escalate")}, [["-", "x"]], "which declares a reason"),
        ({"x": PlanDeclaration("execute", reply=Text("Hi."))}, [["-", "x"]], "declares no reply"),
        (
            {"x": PlanDeclaration("clarify", [Step("page")], reply=Text("Hi?"))},
            [["-", "x"]],
            "holds no steps",
        ),
        ({"x": PlanDeclaration("execute", [Step("gone")])}, [["-", "x"]], "not declared"),
        (
            {
                "x": PlanDeclaration(
                    "execute", [Step("page", {"idempotency_key": Origin(literal=1)})]
                )
            },
            [["-", "x"]],
            "idempotency_key",
        ),
        (
            {"x": PlanDeclaration("execute", [Step("page", {"a": Binding(1)})])},
            [["-", "x"]],
            "binds no earlier step",
        ),
        ({"x": PlanDeclaration("execute")}, [["-", "y"]], "not a declared plan"),
        ({"x": PlanDeclaration("execute")}, [["yes", "x"]], "last row holds a cell"),
        ({"x": PlanDeclaration("execute")}, [], "has no row"),
        (
            {
                "x": PlanDeclaration(
                    "inform", reply=Text("{a} {b}", [{"name": "a", "origin": {"param": "bar"}}])
                )
            },
            [["-", "x"]],
            "names no slot",
        ),
    ],
)
def test_check_9_a_policy_tables_and_a_plan_declarations_clauses(
    plans: dict[str, PlanDeclaration], rows: list[Any], says: str
) -> None:
    found = failures(asked(policy=PolicyTable(["d:urgent"], rows), plans=plans, tools=["page"]))
    assert any(c == 9 and says in m for c, m in found), found


def test_check_9_the_last_rows_plan_reads_no_board() -> None:
    found = failures(
        dataclasses.replace(policy_design(), policy=PolicyTable(["d:urgent"], [["-", "tell"]]))
    )
    assert any(c == 9 and "whose plan reads the board" in m for c, m in found), found


def test_check_10_a_tables_outputs_and_order() -> None:
    refused(tab("F", ["t:state.n"], [["-", 1]], outputs="nope<"), 10, "outputs")
    refused(tab("P", ["t:state.n"], [["-", 1]], order=[float("nan")]), 10, "order")
    refused(tab("F", ["t:state.n"], [["-", float("nan")]]), 10, "value domain")


# ---------------------------------------------------------------------------------------------
# Evaluation


@pytest.mark.parametrize(
    ("cell", "state", "matched"),
    [
        ("a", {}, True),
        ("b", {}, False),
        ("{b, a}", {}, True),
        ("{b, c}", {}, False),
        (">= 3", {}, True),
        ("> 3", {}, False),
        ("<= 3", {}, True),
        ("< 3", {}, False),
        ("[3, five]", {}, True),
        ("[4, five]", {}, False),
        (">= t:state.m", {}, False),
        ("< t:state.m", {}, True),
        ("in t:state.items", {"n": 2}, True),
        ("in t:state.items", {}, False),
        (True, {}, True),
        (False, {}, False),
    ],
)
def test_each_cell_form_matches_and_does_not(
    cell: Any, state: dict[str, Any], matched: bool
) -> None:
    read = (
        "t:state.word"
        if cell in ("a", "b") or (isinstance(cell, str) and cell.startswith("{"))
        else "t:state.flag"
        if type(cell) is bool
        else "t:state.n"
    )
    inputs = [read, "t:state.m", "t:state.items"]
    table = tab("F", inputs, [[cell, "-", "-", 1], ["-", "-", "-", 0]])
    assert value(table, state) == (1 if matched else 0)


def test_null_is_matched_by_minus_alone() -> None:
    table = tab("F", ["t:state.maybe"], [[">= 0", 1], ["< 0", 2], ["-", 3]])
    assert value(table) == 3
    assert value(table, {"maybe": 4}) == 1


def test_rows_lists_every_matching_row_and_f_takes_the_first() -> None:
    entry = one(tab("F", ["t:state.n"], [[">= 1", 1], ["-", 2], ["< 0", 3], [">= 3", 4]]))
    assert entry["value"] == 1
    assert entry["rows"] == [1, 2, 4]


def marker_rule(kind: str) -> Rule:
    """A function rule writing ``d:src`` as a marker of ``kind``, or 1."""

    def fn(r: Any) -> Any:
        if kind == "Unavailable":
            raise RuntimeError("down")
        if kind == "Unsure":
            return Unsure(Cause(CauseCode.abstained, ["t:state"]))
        if kind == "Unasked":
            return Unasked(Cause(CauseCode.abstained, ["t:state"]))
        return 1

    return Rule("d:src", "integer", fn, version="1", reads=["t:state"])


def srcs(*kinds: str) -> dict[str, Rule]:
    rules: dict[str, Rule] = {}
    for n, kind in enumerate(kinds):
        rule = marker_rule(kind)
        rules[f"s{n}"] = dataclasses.replace(rule, provides=f"d:s{n}")
    return rules


def test_rule_1_no_row_matching_names_the_highest_marker_then_abstains() -> None:
    reads = ["d:s0", "d:s1", "d:s2"]
    none = tab("F", reads, [[2, 2, 2, 1]])
    entry = one(none, rules=srcs("Unasked", "Unsure", "Unavailable"))
    assert entry["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "required", "refs": ["d:s2"]},
    }
    entry = one(none, rules=srcs("Unasked", "Unsure", "value"))
    assert entry["marker"] == {"kind": "Unsure", "cause": {"code": "required", "refs": ["d:s1"]}}
    entry = one(none, rules=srcs("Unasked", "value", "Unasked"))
    assert entry["marker"] == {"kind": "Unasked", "cause": {"code": "required", "refs": ["d:s0"]}}
    entry = one(tab("F", ["t:state.n", "t:state.m", "t:state.n"][:2], [[1, 1, 1]]))
    assert entry["marker"] == {
        "kind": "Unsure",
        "cause": {"code": "abstained", "refs": ["t:state.n", "t:state.m"]},
    }
    assert entry["rows"] == []


def test_an_unsure_output_abstains_naming_each_input_once() -> None:
    entry = one(tab("F", ["t:state.n", "t:state.m"], [["-", "-", Unsure]]))
    assert entry["marker"]["cause"] == {"code": "abstained", "refs": ["t:state.n", "t:state.m"]}


def test_rule_2_under_f_from_a_cell_and_from_a_pass_through() -> None:
    rules = srcs("Unavailable")
    entry = one(tab("F", ["d:s0"], [[1, 1], ["-", 2]]), rules=rules)
    assert entry["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "required", "refs": ["d:s0"]},
    }
    assert entry["rows"] == [2]
    entry = one(tab("F", ["d:s0"], [["-", Input("d:s0")], ["-", 2]]), rules=rules)
    assert entry["marker"]["cause"] == {"code": "required", "refs": ["d:s0"]}
    # A row failing on a value cell as well does not propagate: the design chose there.
    entry = one(tab("F", ["d:s0", "t:state.n"], [[1, 9, 1], ["-", "-", 2]]), rules=rules)
    assert entry["value"] == 2


def test_rule_2_does_not_propagate_from_unsure_or_unasked() -> None:
    for kind in ("Unsure", "Unasked"):
        assert value(tab("F", ["d:s0"], [[1, 1], ["-", 2]]), rules=srcs(kind)) == 2


def test_rule_2_under_p_from_a_row_ranking_above_the_winner() -> None:
    rules = srcs("Unavailable")
    table = tab("P", ["d:s0", "t:state.n"], [["-", "-", 1], [1, "-", 2]], order=[2, 1])
    assert value(table, rules=rules)["cause"]["refs"] == ["d:s0"]
    table = tab("P", ["d:s0", "t:state.n"], [["-", "-", 2], [1, "-", 1]], order=[2, 1])
    assert value(table, rules=rules) == 2


def test_p_takes_the_highest_ranked_output() -> None:
    table = tab("P", ["t:state.n"], [[">= 1", 1], [">= 2", 3], [">= 9", 9]], order=[9, 3, 1])
    entry = one(table)
    assert entry["value"] == 3
    assert entry["rows"] == [1, 2]


def test_a_and_u_give_the_matching_output() -> None:
    assert value(tab("A", ["t:state.n"], [[">= 1", 1], ["[1, 3]", 1], ["< 1", 2]])) == 1
    assert value(tab("U", ["t:state.n"], [[">= 4", 1], ["< 4", 2]])) == 2


@pytest.mark.parametrize(
    ("hit", "outputs", "rows", "expected"),
    [
        ("C+", "integer", [[">= 1", 2], [">= 2", 3], [">= 9", 9]], 5),
        ("C<", "integer", [[">= 1", 2], [">= 2", 3]], 2),
        ("C>", "integer", [[">= 1", 2], [">= 2", Param("five")]], 5),
        ("C#", "integer", [[">= 1"], [">= 2"], [">= 9"]], 2),
        ("C+", "integer", [[">= 9", 2]], 0),
        ("C#", "integer", [[">= 9"]], 0),
        ("C+", "number", [[">= 1", Input("t:state.n")], [">= 1", 0.5]], 3.5),
    ],
)
def test_each_collect_aggregator(hit: str, outputs: str, rows: list[Any], expected: Any) -> None:
    assert value(tab(hit, ["t:state.n"], rows, outputs=outputs)) == expected


def test_min_and_max_with_no_match_follow_rule_1() -> None:
    entry = one(tab("C<", ["t:state.n"], [[">= 9", 1]]))
    assert entry["marker"]["cause"]["code"] == "abstained"


def test_collect_propagates_an_unavailable_from_any_row() -> None:
    rules = srcs("Unavailable")
    table = tab("C+", ["t:state.n", "d:s0"], [[">= 1", "-", 1], ["-", 1, 2]])
    assert value(table, rules=rules)["cause"] == {"code": "required", "refs": ["d:s0"]}


def test_a_value_outputs_does_not_allow_is_invalid_return() -> None:
    table = Table("d:tab", t.integer(0, 4), "C+", ["t:state.n"], [["-", 3], [">= 1", 3]])
    assert value(table)["cause"] == {"code": "invalid-return", "refs": ["tab"]}
    assert value(table, {"n": 0}) == 3


def test_a_raising_helper_is_unavailable_with_no_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from ommat.helpers import _choice

    def boom(r: Any, ref: str) -> int:
        raise RuntimeError("no")

    monkeypatch.setattr(_choice, "size", boom)
    loaded = load(design(("tab", tab("F", [Helper("size", "t:state.items")], [["-", 1]]))))
    record = decide_sync(loaded, BASE, {}).record
    entry = next(i for i in record.log if i.get("ref") == "d:tab")
    assert entry["marker"]["cause"] == {"code": "raised", "refs": ["tab"]}
    assert entry["rows"] == []
    assert record.unhashed["exceptions"]


def test_a_table_entry_holds_rows_and_no_derivation() -> None:
    entry = one(tab("F", ["t:state.n"], [["-", 1]]))
    assert entry["rows"] == [1]
    assert "derivation" not in entry


# ---------------------------------------------------------------------------------------------
# A question, a data rule and the policy table


ASK = Source({"q:urgent": Question("noul", Text("Urgent?"))}, needs=["t:state"])
URGENT = Table(
    "d:urgent",
    t.label("no", "yes"),
    "F",
    [Helper("pick_with_margin", "q:urgent", bar="bar", margin="margin")],
    [["-", Input("q:urgent")]],
)
PLANS = {
    "page": PlanDeclaration("execute", [Step("notify", {"who": Origin(ref="t:state.name")})]),
    "tell": PlanDeclaration(
        "inform",
        reply=Text(
            "Nothing urgent for {n}; bar {b}.",
            [
                {"name": "n", "origin": {"ref": "t:state.name"}},
                {"name": "b", "origin": {"param": "bar"}},
            ],
        ),
    ),
    "ask": PlanDeclaration("escalate", reason=Text("Unsure if urgent.")),
}


def asked(*tables: tuple[str, Any], **changes: Any) -> Design:
    rules = dict(tables) if tables else {}
    return design(*({"urgent": URGENT, **rules}).items(), sources={"s": ASK}, **changes)


def policy_design() -> Design:
    return asked(
        policy=PolicyTable(["d:urgent"], [["yes", "page"], ["no", "tell"], ["-", "ask"]]),
        plans=PLANS,
        tools=["notify"],
    )


def answering(p_yes: float | None) -> Callable[..., Any]:
    async def binding(batch: Any, *, seconds: float) -> object:
        if p_yes is None:
            raise RuntimeError("down")
        return CallResult({"q:urgent": {"yes": p_yes, "no": round(1 - p_yes, 10)}})

    return binding


def run(loaded: LoadedDesign, p_yes: float | None) -> Record:
    return decide_sync(loaded, BASE, {"default": answering(p_yes)}).record


def item(record: Record, kind: str, ref: str | None = None) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == kind and (ref is None or i["ref"] == ref))


def test_a_data_rule_picks_and_passes_the_pick_through() -> None:
    loaded = load(asked())
    assert item(run(loaded, 0.9), "entry", "d:urgent")["value"] == "yes"
    assert item(run(loaded, 0.1), "entry", "d:urgent")["value"] == "no"
    # The pick abstains; its pass-through row needs a value, so rule 1 passes the marker on.
    marker = item(run(loaded, 0.55), "entry", "d:urgent")["marker"]
    assert marker == {"kind": "Unsure", "cause": {"code": "required", "refs": ["q:urgent"]}}


def test_the_policy_table_gives_each_plan_its_rows_and_slots() -> None:
    loaded = load(policy_design())
    executed = run(loaded, 0.9)
    plan = item(executed, "plan")
    assert plan["kind"] == "execute"
    assert plan["rows"] == [1, 3]
    assert plan["steps"] == [
        {"tool": "notify", "arguments": {"who": {"ref": "t:state.name"}}, "requirements": []}
    ]
    assert "slots" not in plan
    assert "derivation" not in plan
    told = item(run(loaded, 0.1), "plan")
    assert told["kind"] == "inform"
    assert told["reply"] == "Nothing urgent for Ada; bar 0.6."
    assert told["slots"] == {
        "reply": [
            {"name": "n", "origin": {"ref": "t:state.name"}},
            {"name": "b", "origin": {"param": "bar"}},
        ]
    }
    asked_ = item(run(loaded, 0.55), "plan")
    assert asked_["kind"] == "escalate"
    assert asked_["reason"] == "Unsure if urgent."
    assert asked_["rows"] == [3]
    assert asked_["cause"] is None
    decision = decide_sync(loaded, BASE, {"default": answering(0.1)})
    assert decision.plan.slots["reply"][0]["name"] == "n"


def test_rule_2_in_the_policy_table_escalates_naming_its_input() -> None:
    plan = item(run(load(policy_design()), None), "plan")
    assert plan["kind"] == "escalate"
    assert plan["rows"] == [3]
    assert plan["cause"] == {"code": "required", "refs": ["d:urgent"]}


def test_a_row_whose_plan_reads_a_marker_is_skipped_and_names_it() -> None:
    plans = {
        **PLANS,
        "page": PlanDeclaration("execute", [Step("notify", {"who": Origin(ref="d:s0")})]),
    }
    design_ = dataclasses.replace(
        policy_design(),
        rules={**policy_design().rules, **srcs("Unavailable")},
        plans=plans,
    )
    plan = item(run(load(design_), 0.9), "plan")
    assert plan["kind"] == "escalate"
    assert plan["cause"]["refs"] == ["d:s0"]
    design_ = dataclasses.replace(design_, rules={**policy_design().rules, **srcs("Unsure")})
    plan = item(run(load(design_), 0.9), "plan")
    assert plan["kind"] == "escalate"
    assert plan["cause"] is None
    assert plan["rows"] == [3]


def test_a_raising_helper_in_the_policy_table_is_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    from ommat.helpers import _choice

    def boom(r: Any, ref: str) -> int:
        raise RuntimeError("no")

    monkeypatch.setattr(_choice, "size", boom)
    policy = PolicyTable([Helper("size", "t:state.items")], [[">= 1", "page"], ["-", "ask"]])
    design_ = dataclasses.replace(policy_design(), policy=policy)
    record = run(load(design_), 0.9)
    plan = item(record, "plan")
    assert plan["kind"] == "broken"
    assert plan["cause"] == {"code": "raised", "refs": ["policy"]}
    assert plan["rows"] == []


def test_a_declared_plan_breaking_a_plan_rule_is_broken_with_its_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Load holds a declaration to every plan rule it can; a break here is past load's reach.
    from ommat import _compile

    def never(step: object, n: int) -> bool:
        return False

    monkeypatch.setattr(_compile, "_rule_2", never)
    record = run(load(policy_design()), 0.9)
    plan = item(record, "plan")
    assert plan["kind"] == "broken"
    assert plan["cause"]["refs"] == ["rule:2", "step:1"]
    assert plan["rows"] == [1, 3]


def test_replay_reproduces_a_table_turn_and_finds_a_changed_rows() -> None:
    loaded = load(policy_design())
    for p in (0.9, 0.55, 0.1, None):
        record = run(loaded, p)
        assert replay_sync(record, loaded)[1] == []
    record = run(loaded, 0.1)

    def bump(doc: dict[str, Any]) -> None:
        next(i for i in doc["log"] if i["item"] == "entry" and i["ref"] == "d:urgent")["rows"] = []

    copy = read_record(rehashed(edited(record, bump)))
    found = replay_sync(copy, loaded)[1]
    assert [(f["kind"], f.get("ref")) for f in found] == [(FindingKind.entry, "d:urgent")]

    def plan_rows(doc: dict[str, Any]) -> None:
        next(i for i in doc["log"] if i["item"] == "plan")["rows"] = [3]

    found = replay_sync(read_record(rehashed(edited(record, plan_rows))), loaded)[1]
    assert [f["kind"] for f in found] == [FindingKind.plan]


def edited(record: Record, change: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    doc = json.loads(record.document)
    change(doc)
    return doc


def test_redact_digests_a_plans_rendered_text_and_keeps_its_slots() -> None:
    record = run(load(policy_design()), 0.1)
    view = json.loads(redact(record, b"k" * 32))
    plan = next(i for i in view["log"] if i["item"] == "plan")
    assert plan["reply"] != "Nothing urgent for Ada; bar 0.6."
    assert plan["slots"]["reply"][0] == {"name": "n", "origin": {"ref": "t:state.name"}}
    assert plan["rows"] == [2, 3]


GOLDEN_DESIGN = "sha256:026f125597b88dd0af9a8a011d791c765e50c0e71c728c82d0580656d5da3388"
GOLDEN_RECORD = "sha256:b3fef95440fa90531d9de7fbb80249cb3c976c898ac9043c47be537dc01a8c88"


def test_a_golden_table_turn() -> None:
    loaded = load(policy_design())
    assert loaded.design_hash == GOLDEN_DESIGN
    record = run(loaded, 0.1)
    assert record.record_hash == GOLDEN_RECORD
    assert read_record(record.document).record_hash == GOLDEN_RECORD


# ---------------------------------------------------------------------------------------------
# The grid and the lints


def test_the_grid_takes_a_tables_cells_and_bindings() -> None:
    leaves = {**LEAVES, "labels": Parameter(t.list_(t.string), ["yes"])}
    tables = (
        (
            "big",
            Table(
                "d:big",
                "boolean",
                "F",
                ["t:state.x", "t:state.n"],
                [["> lo", "-", True], ["-", ">= 2", True], ["-", "-", False]],
            ),
        ),
        (
            "share",
            Table(
                "d:share",
                "boolean",
                "F",
                [Helper("mass", "q:urgent", labels="labels")],
                [[">= hi", True], ["-", False]],
            ),
        ),
    )
    loaded = load(asked(*tables, parameters=leaves))
    grid = loaded._grid
    assert grid["t:state.x"] == ("lo",)
    assert grid["t:state.n"] == (2,)
    assert grid["q:urgent"]["bar"] == ("bar",)
    assert grid["q:urgent"]["margin"] == ("margin",)
    assert [dict(m) for m in grid["q:urgent"]["mass"]] == [{"labels": ("yes",), "bounds": ("hi",)}]


def test_an_integer_literal_is_probed_on_both_sides() -> None:
    table = Table("d:tab", "boolean", "F", ["t:state.n"], [[">= 2", True], ["-", False]])

    def pol(r: Any) -> Any:
        from ommat import Plan

        return Plan("inform", reply=str(r.read("d:tab")))

    loaded = load(
        design(("tab", table), policy=Policy(pol, version="1", reads=["d:tab", "t:state.n"]))
    )
    report = unsure_to_execute(loaded, values={})
    assert report["execute"] == []
    from ommat.helpers._points import fill

    points = [p.live for p in fill(loaded, ["t:state.n"], {})["t:state.n"]]
    assert points == [1, 2, 3]


def test_the_abstention_search_reports_a_pick_inside_a_data_rule() -> None:
    def pol(r: Any) -> Any:
        from ommat import Plan, Step

        return Plan("execute", [Step("notify")])

    def keep(r: Any) -> Any:
        urgent = r.read("d:urgent")
        return "yes" if isinstance(urgent, Unsure) else urgent

    keeper = Rule("d:keep", t.label("no", "yes"), keep, version="1", reads=["d:urgent"])
    loaded = load(
        asked(
            ("keep", keeper),
            policy=Policy(pol, version="1", reads=["d:keep"]),
            tools=["notify"],
            state=t.record(name=t.string),
        )
    )
    report = unsure_to_execute(loaded, values={"t:state.name": ["Ada"]})
    assert [i["read"] for i in report["execute"]] == ["d:keep", "q:urgent"]


def test_the_lints_run_over_a_policy_table() -> None:
    loaded = load(dataclasses.replace(policy_design(), state=t.record(name=t.string)))
    report = unsure_to_execute(loaded, values={"t:state.name": ["Ada"]})
    assert report["execute"] == []
    dead = dead_questions(loaded, values={"t:state.name": ["Ada"]})
    assert dead["points"] > 0


def test_a_count_table_is_probed_over_the_values_given() -> None:
    table = Table(
        "d:tab", "boolean", "F", [Helper("size", "t:state.items")], [[">= 2", True], ["-", False]]
    )

    def pol(r: Any) -> Any:
        from ommat import Plan

        return Plan("inform", reply="x")

    loaded = load(
        design(
            ("tab", table),
            policy=Policy(pol, version="1", reads=["d:tab"]),
            state=t.record(items=t.list_(t.integer())),
        )
    )
    assert "t:state.items" not in loaded._grid
    report = dead_questions(loaded, values={"t:state.items": [[], [1, 2]]})
    assert report["points"] > 0
