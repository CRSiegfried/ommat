# pyright: reportPrivateUsage=false
"""Decision tables (§decision-tables, §table-notation): a table's notation and tagged form,
the load checks it adds, its evaluation under each hit policy and the marker contract, the
policy table, replay over a table turn, the grid and the lint."""

import contextlib
import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    CallResult,
    Cause,
    CauseCode,
    Design,
    FindingKind,
    Helper,
    Input,
    InvalidDesign,
    LoadedDesign,
    Param,
    Parameter,
    Question,
    Record,
    Rule,
    Source,
    Table,
    Text,
    Unasked,
    Unavailable,
    Undetermined,
    Unsure,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import encode
from ommat.helpers import dead_questions

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")  # defects are the point here

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
    return "done"


def design(*tables: tuple[str, Any], reads: tuple[str, ...] = (), **changes: Any) -> Design:
    """The tables as rules; unless ``changes`` names a ``policy``, the policy is ``d:show``, a
    function rule reading ``reads``."""
    rules: dict[str, Any] = {**dict(tables), **changes.pop("rules", {})}
    fields: dict[str, Any] = {"state": STATE, "parameters": LEAVES}
    if "policy" not in changes:
        rules["show"] = Rule("d:show", t.string, show, version="1", reads=list(reads))
        fields["policy"] = "d:show"
    fields["rules"] = rules
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


def test_a_tables_row_list_is_copied_when_built() -> None:
    rows: list[Any] = [["-", 1]]
    table = tab("F", ["t:state.n"], rows)
    rows.append(["-", 2])
    assert len(table.rows) == 1


def test_a_stored_earlier_format_document_holding_a_table_is_refused_naming_its_format() -> None:
    doc = json.loads((Path(__file__).parent / "format_6" / "design.json").read_bytes())
    doc["rules"]["flag"] = {
        "form": "table",
        "provides": "d:flag",
        "outputs": "boolean",
        "hit": "F",
        "order": None,
        "inputs": [{"read": "t:state.request", "helper": None, "params": {}}],
        "rows": [{"cells": ["-"], "output": {"literal": True}}],
        "candidates": False,
        "grounded": None,
    }
    found = failures(encode(doc))
    assert [c for c, _ in found] == [11]
    assert "format 6 is not 10" in found[0][1]


def test_a_document_rows_member_set_differing_fails_load() -> None:
    doc = json.loads(load(design(("tab", NOTATION))).document)
    doc["rules"]["tab"]["rows"][0]["extra"] = 1
    assert failures(doc)


# ---------------------------------------------------------------------------------------------
# Load


def test_check_1_a_one_of_repeat() -> None:
    refused(tab("F", ["t:state.word"], [[{"one_of": ["a", "a"]}, 1]]), 1, "twice in a one_of")


def test_check_3_an_input_needs_a_provider() -> None:
    refused(tab("F", ["d:none"], [["-", 1]]), 3, "no provider")


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


@pytest.mark.parametrize("helper", ["determined", "agree", "pick_each_with_margin", "nope"])
def test_check_6_a_helper_outside_the_tables_library(helper: str) -> None:
    refused(tab("F", [Helper(helper, "t:state.items")], [["-", 1]]), 6, "not a helper a table")


def test_check_7_a_cell_or_pass_through_ref_out_of_the_grammar() -> None:
    refused(tab("F", ["t:state.n"], [[{"in": "d:Bad"}, 1]]), 7, "not a ref")
    refused(tab("F", ["t:state.n"], [["in d:Bad", 1]]), 10, "names no ref")
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
    assert any(c == 9 and "only -, Known and marker cells" in m for c, m in found), found
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
    assert entry["marker"]["cause"] == {"code": "abstained", "refs": ["t:state.n"]}


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
ACT = Table(
    "d:act",
    t.label("page", "tell", "ask"),
    "F",
    ["d:urgent"],
    [["yes", "page"], ["no", "tell"], ["-", "ask"]],
)


def asked(*tables: tuple[str, Any], **changes: Any) -> Design:
    rules = dict(tables) if tables else {}
    fields: dict[str, Any] = {"sources": {"s": ASK}, **changes}
    return design(*({"urgent": URGENT, **rules}).items(), **fields)


def policy_design(act: Table = ACT) -> Design:
    """The design whose policy, ``d:act``, is a decision table over ``d:urgent``."""
    return asked(("act", act), policy="d:act")


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


def test_a_policy_table_is_a_rule_whose_entry_is_the_decision() -> None:
    loaded = load(policy_design())
    for p_yes, expected, rows in ((0.9, "page", [1, 3]), (0.1, "tell", [2, 3]), (0.55, "ask", [3])):
        decision = decide_sync(loaded, BASE, {"default": answering(p_yes)})
        entry = item(decision.record, "entry", "d:act")
        assert entry["value"] == expected
        assert entry["rows"] == rows
        assert decision.estimate == expected
        assert "derivation" not in entry


def test_the_policy_table_names_the_marker_its_input_gave() -> None:
    decision = decide_sync(load(policy_design()), BASE, {"default": answering(None)})
    entry = item(decision.record, "entry", "d:act")
    assert entry["marker"]["cause"] == {"code": "required", "refs": ["d:urgent"]}
    assert isinstance(decision.estimate, Unavailable)


def test_a_raising_helper_in_the_policy_table_is_the_policys_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ommat.helpers import _choice

    def boom(r: Any, ref: str) -> int:
        raise RuntimeError("no")

    monkeypatch.setattr(_choice, "size", boom)
    act = Table(
        "d:act",
        t.label("page", "ask"),
        "F",
        [Helper("size", "t:state.items")],
        [[">= 1", "page"], ["-", "ask"]],
    )
    decision = decide_sync(load(policy_design(act)), BASE, {"default": answering(0.9)})
    entry = item(decision.record, "entry", "d:act")
    assert entry["marker"]["cause"] == {"code": "raised", "refs": ["act"]}
    assert entry["rows"] == []
    assert isinstance(decision.estimate, Unavailable)


def test_replay_reproduces_a_table_turn_and_finds_a_changed_rows() -> None:
    loaded = load(policy_design())
    for p in (0.9, 0.55, 0.1, None):
        record = run(loaded, p)
        assert replay_sync(record, loaded)[1] == []
    record = run(loaded, 0.1)
    for ref in ("d:urgent", "d:act"):

        def bump(doc: dict[str, Any], ref: str = ref) -> None:
            next(i for i in doc["log"] if i["item"] == "entry" and i["ref"] == ref)["rows"] = []

        copy = read_record(rehashed(edited(record, bump)))
        found = replay_sync(copy, loaded)[1]
        assert [(f["kind"], f.get("ref")) for f in found] == [(FindingKind.entry, ref)]


def edited(record: Record, change: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
    doc = json.loads(record.document)
    change(doc)
    return doc


GOLDEN_DESIGN = "sha256:2ea189ff88ad7dbdbcb28ecd9fab4d4bcfdc0e7c8c54037cb4ee95b92b479515"
GOLDEN_RECORD = "sha256:bedd268b9cbb68e17e1b8f16dd082bd036abbf61b8d41acf8f3deb401c8c3592"


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

    loaded = load(design(("tab", table), reads=("d:tab", "t:state.n")))
    from ommat.helpers._points import fill

    points = [p.live for p in fill(loaded, ["t:state.n"], {})["t:state.n"]]
    assert points == [1, 2, 3]


def test_a_count_table_is_probed_over_the_values_given() -> None:
    table = Table(
        "d:tab", "boolean", "F", [Helper("size", "t:state.items")], [[">= 2", True], ["-", False]]
    )

    loaded = load(
        design(("tab", table), reads=("d:tab",), state=t.record(items=t.list_(t.integer())))
    )
    assert "t:state.items" not in loaded._grid
    with pytest.raises(ValueError, match="unfilled"):
        dead_questions(loaded)
    assert dead_questions(loaded, values={"t:state.items": [[], [1, 2]]})["points"] == 2


# ---------------------------------------------------------------------------------------------
# Further cases


def test_a_design_and_its_document_give_one_record() -> None:
    built = load(policy_design())
    stored = load(built.document)
    assert stored.design_hash == built.design_hash
    a, b = run(built, 0.9), run(stored, 0.9)
    assert a.record_hash == b.record_hash
    assert replay_sync(a, stored)[1] == []


CHOOSE = Source(
    {
        "q:desk": Question("choice", Text("Which desk?"), labels=("hotel", "flight", "other")),
        "q:ref": Question(
            "extraction",
            Text("The booking reference"),
            value_type="text",
            text={"ref": "t:state.name"},
        ),
    },
    needs=["t:state"],
)


def choose(*tables: tuple[str, Table], **changes: Any) -> Design:
    leaves = {
        **LEAVES,
        "desk/bars": Parameter(t.mapping(t.number(0, 1)), {"hotel": 0.5, "flight": 0.5}),
        "null": Parameter(t.number(0, 1), 0.5),
    }
    fields: dict[str, Any] = {"sources": {"s": CHOOSE}, "parameters": leaves, **changes}
    return design(*tables, **fields)


def answering_choice(desk: dict[str, float], value: str | None) -> Callable[..., Any]:
    async def binding(batch: Any, *, seconds: float) -> object:
        extraction = {
            "value": value,
            "guess": value,
            "p_null": 0.1,
            "confidence": 0.9,
            "span": None,
        }
        return CallResult({"q:desk": desk, "q:ref": extraction})

    return binding


def test_a_yes_no_condition_from_a_choice_and_an_extract_table() -> None:
    hotel = Table(
        "d:hotel",
        t.label("no", "yes"),
        "F",
        [Helper("pick_with_thresholds", "q:desk", thresholds="desk/bars", margin="margin")],
        [["hotel", "yes"], ["{flight, other}", "no"]],
    )
    booking = Table(
        "d:booking",
        "string",
        "F",
        [Helper("extract", "q:ref", null_bar="null")],
        [["-", Input("q:ref")]],
    )
    loaded = load(choose(("hotel", hotel), ("booking", booking)))
    record = decide_sync(
        loaded,
        BASE,
        {"default": answering_choice({"hotel": 0.8, "flight": 0.1, "other": 0.1}, "AB12")},
    ).record
    assert item(record, "entry", "d:hotel")["value"] == "yes"
    assert item(record, "entry", "d:booking")["value"] == "AB12"
    record = decide_sync(
        loaded,
        BASE,
        {"default": answering_choice({"hotel": 0.1, "flight": 0.8, "other": 0.1}, None)},
    ).record
    assert item(record, "entry", "d:hotel")["value"] == "no"
    assert item(record, "entry", "d:booking")["marker"]["cause"]["refs"] == ["q:ref"]
    grid = loaded._grid
    assert grid["q:desk"]["thresholds"] == ("desk/bars",)
    assert grid["q:ref"]["null"] == ("null",)


def test_each_marker_cell_matches_its_own_kind() -> None:
    table = tab("F", ["d:s0"], [["Unsure", 1], ["Unavailable", 2], ["Unasked", 3], ["-", 4]])
    for kind, expected in (("Unsure", 1), ("Unavailable", 2), ("Unasked", 3), ("value", 4)):
        assert value(table, rules=srcs(kind)) == expected


def test_a_label_spelt_unsure_is_matched_in_tagged_form() -> None:
    labels = t.label("Unsure", "x")
    table = Table("d:tab", "integer", "F", ["t:state.w"], [[{"is": "Unsure"}, 1], [{"is": "x"}, 2]])
    loaded = load(design(("tab", table), state=t.record(w=labels)))
    for label, expected in (("Unsure", 1), ("x", 2)):
        record = decide_sync(loaded, {"w": label}, {}).record
        assert item(record, "entry", "d:tab")["value"] == expected


def test_rule_2_ignores_a_row_after_the_winner_under_f() -> None:
    assert value(tab("F", ["d:s0"], [["-", 2], [1, 1]]), rules=srcs("Unavailable")) == 2


def test_collect_propagates_from_a_row_before_a_match_and_with_no_match() -> None:
    rules = srcs("Unavailable")
    before = tab("C+", ["d:s0", "t:state.n"], [[1, "-", 1], ["-", ">= 1", 2]])
    assert value(before, rules=rules)["cause"] == {"code": "required", "refs": ["d:s0"]}
    alone = tab("C#", ["d:s0"], [[1]])
    assert value(alone, rules=rules)["cause"] == {"code": "required", "refs": ["d:s0"]}


def test_a_collect_sum_is_exact_over_the_decimals() -> None:
    leaves = {**LEAVES, "a": Parameter(t.number(), 0.1), "b": Parameter(t.number(), 0.2)}
    table = tab("C+", ["t:state.n"], [["-", Param("a")], ["-", Param("b")]], outputs="number")
    loaded = load(design(("tab", table), parameters=leaves))
    record = decide_sync(loaded, BASE, {}).record
    assert item(record, "entry", "d:tab")["value"] == 0.3


def test_rule_2_names_the_first_input_of_a_cell() -> None:
    listed = tab("F", ["d:s0"], [[{"one_of": [1, 2]}, 1], ["-", 2]])
    assert value(listed, rules=srcs("Unavailable"))["cause"]["refs"] == ["d:s0"]
    both = tab("F", ["d:s0", "d:s1"], [["-", ">= d:s0", 1], ["-", "-", 2]])
    assert value(both, rules=srcs("Unavailable", "Unavailable"))["cause"]["refs"] == ["d:s0"]
    one_side = tab("F", ["d:s0", "d:s1"], [[">= d:s1", "-", 1], ["-", "-", 2]])
    # With its other input Unsure, the cell fails whatever the outage withheld.
    assert value(one_side, rules=srcs("Unsure", "Unavailable")) == 2
    assert value(one_side, rules=srcs("value", "Unavailable"))["cause"]["refs"] == ["d:s1"]


def test_a_comparison_against_a_marker_and_an_in_over_a_marker_do_not_match() -> None:
    table = tab("F", ["t:state.n", "d:s0"], [[">= d:s0", "-", 1], ["-", "-", 2]])
    assert value(table, rules=srcs("Unsure")) == 2
    listed = Rule(
        "d:s0",
        t.list_(t.integer()),
        lambda r: Unsure(Cause(CauseCode.abstained, ["t:state"])),
        version="1",
        reads=["t:state"],
    )
    table = tab("F", ["t:state.n", "d:s0"], [["in d:s0", "-", 1], ["-", "-", 2]])
    assert value(table, rules={"s0": listed}) == 2


def test_the_dead_question_lint_finds_a_question_a_policy_table_rests_on_live() -> None:
    loaded = load(dataclasses.replace(policy_design(), state=t.record(name=t.string)))
    dead = dead_questions(loaded, values={"t:state.name": ["Ada"]})
    assert dead["questions"]["q:urgent"]["status"] == "live"


def test_an_integer_value_cell_is_a_grid_boundary() -> None:
    table = tab("F", ["t:state.n"], [[5, 1], [{"one_of": [7, 8]}, 2], ["-", 0]])
    loaded = load(design(("tab", table)))
    assert loaded._grid["t:state.n"] == (5, 7, 8)


def test_a_mass_set_from_a_table_and_a_boundaries_field_share_one_entry() -> None:
    leaves = {
        **LEAVES,
        "labels": Parameter(t.list_(t.string), ["yes"]),
        "m2": Parameter(t.number(0, 1), 0.4),
    }
    share = Table(
        "d:share",
        "boolean",
        "F",
        [Helper("mass", "q:urgent", labels="labels")],
        [[">= hi", True], ["-", False]],
    )
    boundary = {
        "bar": [],
        "thresholds": [],
        "margin": [],
        "mass": [{"labels": ["yes"], "bounds": ["m2"]}],
    }
    reader = Rule(
        "d:r",
        "boolean",
        lambda r: True,
        version="1",
        reads=["q:urgent"],
        boundaries={"q:urgent": boundary},
    )
    loaded = load(asked(("share", share), ("r", reader), parameters=leaves))
    assert [dict(m) for m in loaded._grid["q:urgent"]["mass"]] == [
        {"labels": ("yes",), "bounds": ("hi", "m2")}
    ]


@pytest.mark.parametrize(
    "table",
    [
        tab("U", ["t:state.word"], [["a", 1], ["{b, c}", 2]]),
        tab("U", ["t:state.flag"], [[True, 1], [False, 2]]),
        tab("U", ["t:state.n"], [["[0, 2]", 1], ["> 2", 2], ["< 0", 3]]),
        tab(
            "U", ["t:state.n", "t:state.items"], [["in t:state.items", "-", 1], ["Unsure", "-", 2]]
        ),
    ],
)
def test_more_rows_that_never_both_match(table: Table) -> None:
    load(design(("tab", table)))


@pytest.mark.parametrize(
    "table",
    [
        tab("U", ["t:state.word"], [["a", 1], ["{a, c}", 2]]),
        tab("U", ["t:state.n"], [["[0, 2]", 1], [">= 2", 2]]),
        tab("U", ["t:state.n", "t:state.items"], [["in t:state.items", "-", 1], ["> 1", "-", 2]]),
        tab("U", ["t:state.maybe"], [["-", 1], ["-", 2]]),
    ],
)
def test_more_rows_that_can_both_match(table: Table) -> None:
    refused(table, 9, "overlap")


@pytest.mark.parametrize(
    ("table", "check", "says"),
    [
        (tab("F", [Helper("first", "t:state.items")], [["-", 1]]), 6, "not a helper a table"),
        (tab("F", [Helper("unique", "t:state.items")], [["-", 1]]), 6, "not a helper a table"),
        (tab("F", [Helper("size", "t:state.n")], [["-", 1]]), 9, "which is not a list"),
        (tab("F", [Helper("size", "t:state.items")], [[-1, 1]]), 9, "outside the input's interval"),
        (tab("P", ["t:state.n"], [["-", 1]], order=["x"]), 9, "is not a value of"),
        (
            Table("d:tab", "integer or null", "P", ["t:state.n"], [["-", None]], order=[1]),
            9,
            "does not list",
        ),
        (tab("F", ["t:state.n"], [["-", Input(3)]]), 10, "not a string"),  # type: ignore[arg-type]
        (tab("F", [b"ab"], [["-", 1]]), 10, "not a ref or a Helper"),
        (tab("F", ["t:state.n"], [b"\x01\x02"]), 10, "not a list"),
    ],
)
def test_further_clauses(table: Table, check: int, says: str) -> None:
    refused(table, check, says)


def test_a_tables_input_outside_the_domain_fails_check_10() -> None:
    refused(tab("F", ["\ud800"], [["-", 1]]), 10, "value domain")


def test_a_mass_or_thresholds_leaf_names_labels_of_its_read_once() -> None:
    leaves = {**LEAVES, "labels": Parameter(t.list_(t.string), ["yes", "yes"])}
    found = failures(
        asked(
            ("tab", tab("F", [Helper("mass", "q:urgent", labels="labels")], [["-", 1]])),
            parameters=leaves,
        )
    )
    assert any(c == 9 and "lists a label twice" in m for c, m in found)
    leaves = {**LEAVES, "th": Parameter(t.mapping(t.number(0, 1)), {"zzz": 0.5})}
    picked = Helper("pick_with_thresholds", "q:desk", thresholds="th", margin="margin")
    leaves["desk/bars"] = Parameter(t.mapping(t.number(0, 1)), {"hotel": 0.5})
    leaves["null"] = Parameter(t.number(0, 1), 0.5)
    found = failures(choose(("tab", tab("F", [picked], [["-", 1]])), parameters=leaves))
    assert any(c == 9 and "not a label of" in m for c, m in found)


def table_doc() -> dict[str, Any]:
    return json.loads(load(design(("tab", tab("F", ["t:state.n"], [["-", 1]])))).document)


def row_of(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["rules"]["tab"]["rows"][0]


EDITS: dict[str, tuple[dict[str, Any], int]] = {
    "an extra row member": ({"extra": 1}, 11),
    "a bare integer cell": ({"cells": [5]}, 10),
    "a bare label cell": ({"cells": ["yes"]}, 10),
    "a null output outside C#": ({"output": None}, 10),
    "an Unavailable output": ({"output": {"marker": "Unavailable"}}, 10),
    "a ref output that is no string": ({"output": {"ref": 5}}, 10),
}


@pytest.mark.parametrize("edit", list(EDITS))
def test_a_hand_edited_tables_members_fail_their_check(edit: str) -> None:
    change, check = EDITS[edit]
    doc = table_doc()
    row_of(doc).update(change)
    assert {c for c, _ in failures(doc)} == {check}


def test_a_hand_edited_input_missing_a_member_fails_check_11() -> None:
    doc = table_doc()
    del doc["rules"]["tab"]["inputs"][0]["params"]
    assert {c for c, _ in failures(doc)} == {11}


@pytest.mark.parametrize(
    ("cell", "n", "expected"),
    [(3, 3, 1), (3, 4, 0), ({"one_of": [2, 3]}, 3, 1), ({"one_of": [2, 4]}, 3, 0)],
)
def test_an_integer_value_cell_matches_its_value(cell: Any, n: int, expected: int) -> None:
    assert value(tab("F", ["t:state.n"], [[cell, 1], ["-", 0]]), {"n": n}) == expected


def test_an_integer_value_cell_is_probed_on_both_sides() -> None:
    from ommat.helpers._points import fill

    loaded = load(design(("tab", tab("F", ["t:state.n"], [[5, 1], ["-", 0]]))))
    assert [p.live for p in fill(loaded, ["t:state.n"], {})["t:state.n"]] == [4, 5, 6]


def test_the_grid_takes_intervals_extract_values_and_the_policy_tables_cells() -> None:
    table = tab("F", ["t:state.n", "t:state.x"], [["[2, 5]", "[lo, hi]", 1], ["-", "-", 0]])
    loaded = load(design(("tab", table)))
    assert loaded._grid["t:state.n"] == (2, 5)
    assert loaded._grid["t:state.x"] == ("hi", "lo")
    act = Table(
        "d:act", t.label("page", "ask"), "F", ["t:state.n"], [[">= 2", "page"], ["-", "ask"]]
    )
    loaded = load(design(("act", act), policy="d:act"))
    assert loaded._grid["t:state.n"] == (2,)
    leaves = {**LEAVES, "null": Parameter(t.number(0, 1), 0.5), "big": Parameter(t.number(), 9)}
    number_ref = Source(
        {
            "q:amount": Question(
                "extraction", Text("The amount"), value_type="number", text={"ref": "t:state.name"}
            )
        },
        needs=["t:state"],
    )
    amount = tab("F", [Helper("extract", "q:amount", null_bar="null")], [[">= big", 1], ["-", 0]])
    loaded = load(design(("tab", amount), sources={"s": number_ref}, parameters=leaves))
    assert loaded._grid["q:amount"]["value"] == ("big",)
    assert loaded._grid["q:amount"]["null"] == ("null",)


def test_a_boolean_cell_and_a_value_cell_over_a_size_input_are_no_boundary() -> None:
    flag = tab("F", ["t:state.flag"], [[{"is": True}, 1], ["-", 0]])
    assert "t:state.flag" not in load(design(("tab", flag)))._grid
    count = tab("F", [Helper("size", "t:state.items")], [[2, 1], ["-", 0]])
    assert "t:state.items" not in load(design(("tab", count)))._grid


def test_a_marker_row_under_u_stands_over_an_outage() -> None:
    table = tab("U", ["d:s0"], [[1, 1], ["Unavailable", 2]])
    assert value(table, rules=srcs("Unavailable")) == 2


def test_hostile_and_deep_rows_fail_load_cleanly() -> None:
    class Broken(dict[str, Any]):
        def __len__(self) -> int:
            raise ZeroDivisionError

    refused(tab("F", ["t:state.n"], [[Broken(), 1]]), 10, "cannot be read")
    refused(tab("F", ["t:state.n"], [["in t:state.items extra", 1]]), 10, "not in and one ref")
    for depth in (58, 60, 62, 64, 70):
        deep: Any = 1
        for _ in range(depth):
            deep = [deep]
        doc = table_doc()
        row_of(doc)["output"] = deep
        with contextlib.suppress(InvalidDesign):  # loads, or is refused; never raises else
            load(doc)
    doc = table_doc()
    doc["rules"]["tab"]["rows"][0] = ["-", 1]
    assert {c for c, _ in failures(doc)} == {10}


def test_a_float_input_compares_and_an_interval_holds_both_ends() -> None:
    assert value(tab("F", ["t:state.x"], [["> lo", 1], ["-", 2]])) == 1
    assert value(tab("F", ["t:state.x"], [["[lo, hi]", 1], ["-", 2]]), {"x": 0.75}) == 1
    assert value(tab("F", ["t:state.n"], [["[2, 5]", 1], ["-", 2]]), {"n": 5}) == 1


def raising_list() -> Rule:
    def fn(r: Any) -> Any:
        raise RuntimeError("down")

    return Rule("d:s0", t.list_(t.integer()), fn, version="1", reads=["t:state"])


def test_an_in_over_an_unavailable_list_meets_rule_2() -> None:
    table = tab("F", ["t:state.n", "d:s0"], [["in d:s0", "-", 1], ["-", "-", 2]])
    assert value(table, rules={"s0": raising_list()})["cause"] == {
        "code": "required",
        "refs": ["d:s0"],
    }


def test_p_does_not_propagate_from_a_row_ranked_equal_to_the_winner() -> None:
    table = tab("P", ["d:s0", "t:state.n"], [[1, "-", 1], ["-", "-", 1]], order=[1])
    assert value(table, rules=srcs("Unavailable")) == 1


def test_a_sum_past_every_float_is_invalid_return() -> None:
    leaves = {**LEAVES, "huge": Parameter(t.number(), 1e308)}
    table = tab("C+", ["t:state.n"], [["-", Param("huge")], ["-", Param("huge")]], outputs="number")
    loaded = load(design(("tab", table), parameters=leaves))
    record = decide_sync(loaded, BASE, {}).record
    assert item(record, "entry", "d:tab")["marker"]["cause"] == {
        "code": "invalid-return",
        "refs": ["tab"],
    }


@pytest.mark.parametrize(
    ("cell", "inputs"),
    [
        ({"between": [{"ref": "t:state.m"}, {"literal": 9}]}, ["t:state.n", "t:state.m"]),
        ({">=": {"literal": True}}, ["t:state.n", "t:state.m"]),
        (">= 007", ["t:state.n", "t:state.m"]),
    ],
)
def test_a_tagged_or_notation_operand_of_another_form_fails_check_10(
    cell: Any, inputs: list[str]
) -> None:
    refused(tab("F", inputs, [[cell, "-", 1]]), 10, "row 1")


@pytest.mark.parametrize(
    "table",
    [
        tab(
            "U",
            ["t:state.n", "t:state.m", "t:state.x"],
            [["< t:state.m", "-", "-", 1], [">= t:state.x", "-", "-", 2]],
        ),
        tab("U", ["t:state.x"], [[">= p", 1], ["-", 2]]),
    ],
)
def test_overlaps_against_different_refs_and_a_decimal_bound(table: Table) -> None:
    leaves = {**LEAVES, "p": Parameter(t.number(0, 0.1), 0.1)}
    state = t.record(n=t.integer(), m=t.integer(), x=t.number(0, 0.1))
    found = failures(design(("tab", table), parameters=leaves, state=state))
    assert any(c == 9 and "overlap" in m for c, m in found), found


@pytest.mark.parametrize(
    "table",
    [
        tab("U", ["t:state.x"], [["> lo", 1], ["<= lo", 2]]),
        tab("U", ["t:state.n"], [[3, 1], ["< 3", 2], ["> 3", 3]]),
    ],
)
def test_rows_split_at_a_shared_bound_never_both_match(table: Table) -> None:
    load(design(("tab", table)))


def test_a_cell_comparing_an_input_with_itself_fails_check_9() -> None:
    refused(tab("F", ["t:state.n"], [[">= t:state.n", 1]]), 9, "with itself")


def test_pick_with_thresholds_over_a_noul_question_fails_check_9() -> None:
    leaves = {**LEAVES, "th": Parameter(t.mapping(t.number(0, 1)), {"yes": 0.6})}
    picked = Helper("pick_with_thresholds", "q:urgent", thresholds="th", margin="margin")
    found = failures(asked(("tab", tab("F", [picked], [["-", 1]])), parameters=leaves))
    assert any(c == 9 and "takes pick_with_margin" in m for c, m in found), found


# ---------------------------------------------------------------------------------------------
# A cell listing marker kinds


@pytest.mark.parametrize(
    ("cell", "tagged"),
    [
        ("Undetermined", {"markers": ["Unsure", "Unavailable", "Unasked"]}),
        ("Unsure or Unavailable", {"markers": ["Unsure", "Unavailable"]}),
        ("Unasked or Unsure", {"markers": ["Unasked", "Unsure"]}),
        (Undetermined, {"markers": ["Unsure", "Unavailable", "Unasked"]}),
        ((Unsure, Unavailable), {"markers": ["Unsure", "Unavailable"]}),
        ({"markers": ["Unavailable", "Unsure"]}, {"markers": ["Unavailable", "Unsure"]}),
        ("Unsure or b", {"is": "Unsure or b"}),
    ],
)
def test_a_cell_listing_marker_kinds_is_read(cell: Any, tagged: Any) -> None:
    from ommat._table import cell_form

    assert cell_form(cell) == tagged


def test_a_markers_cell_matches_each_kind_it_lists_and_no_value() -> None:
    table = tab("F", ["d:s0"], [["Unsure or Unavailable", 1], ["Unasked", 2], ["-", 3]])
    for kind, expected in (("Unsure", 1), ("Unavailable", 1), ("Unasked", 2), ("value", 3)):
        assert value(table, rules=srcs(kind)) == expected
    either = tab("F", ["d:s0"], [["Undetermined", 1], ["-", 2]])
    for kind, expected in (("Unsure", 1), ("Unavailable", 1), ("Unasked", 1), ("value", 2)):
        assert value(either, rules=srcs(kind)) == expected


def test_a_markers_cell_routes_an_outage_before_a_value_row() -> None:
    # The outage row first keeps rule 2 from escalating a later value row's failure.
    table = tab("F", ["d:s0"], [["Unsure or Unavailable", 0], [1, 1], ["-", 2]])
    assert value(table, rules=srcs("Unavailable")) == 0


def test_a_markers_cell_is_written_in_canonical_order_and_round_trips() -> None:
    loaded = load(
        design(("tab", tab("F", ["t:state.n"], [["Unsure or Unavailable", 1], ["-", 0]])))
    )
    cell = json.loads(loaded.document)["rules"]["tab"]["rows"][0]["cells"][0]
    assert cell == {"markers": ["Unavailable", "Unsure"]}
    assert load(loaded.document).design_hash == loaded.design_hash


def test_a_markers_cell_of_one_kind_or_a_repeat_fails_load() -> None:
    refused(tab("F", ["t:state.n"], [[{"markers": ["Unsure"]}, 1]]), 10, "fewer than two")
    refused(tab("F", ["t:state.n"], [[{"markers": ["Unsure", "x"]}, 1]]), 10, "not a marker kind")
    refused(
        tab("F", ["t:state.n"], [[{"markers": ["Unsure", "Unsure"]}, 1]]), 1, "twice in a markers"
    )


def test_markers_cells_overlap_by_their_kinds() -> None:
    load(
        design(
            (
                "tab",
                tab(
                    "U",
                    ["t:state.n"],
                    [["Unsure or Unavailable", 1], ["Unasked", 2], [">= 0", 3], ["< 0", 4]],
                ),
            )
        )
    )
    refused(tab("U", ["t:state.n"], [["Unsure or Unavailable", 1], ["Unsure", 2]]), 9, "overlap")


# ---------------------------------------------------------------------------------------------
# The Known cell


@pytest.mark.parametrize(
    ("cell", "tagged"),
    [
        ("Known", {"known": True}),
        (" Known ", {"known": True}),
        ({"known": True}, {"known": True}),
        ({"is": "Known"}, {"is": "Known"}),
        ("known", {"is": "known"}),
    ],
)
def test_a_known_cell_is_read(cell: Any, tagged: Any) -> None:
    from ommat._table import cell_form

    assert cell_form(cell) == tagged


@pytest.mark.parametrize("member", [False, 1, "true", None])
def test_a_known_cell_holding_anything_but_true_fails_check_10(member: Any) -> None:
    refused(tab("F", ["t:state.n"], [[{"known": member}, 1], ["-", 0]]), 10, "not true")


def test_a_known_cell_matches_any_value_and_no_marker() -> None:
    table = tab("F", ["d:s0"], [["Known", 1], ["-", 2]])
    for kind, expected in (("value", 1), ("Unsure", 2), ("Unavailable", 2), ("Unasked", 2)):
        assert value(table, rules=srcs(kind)) == expected


def test_a_last_row_of_known_keeps_the_table_unasked_where_a_row_of_dash_gives_a_value() -> None:
    # d:s0 holds 1, as an earlier question's answer; d:s1 was never asked.
    rules = srcs("value", "Unasked")
    catch_all = tab("F", ["d:s0", "d:s1"], [[1, 2, 10], [1, "-", 20]])
    assert value(catch_all, rules=rules) == 20
    known = tab("F", ["d:s0", "d:s1"], [[1, 2, 10], [1, "Known", 20]])
    assert value(known, rules=rules) == {
        "kind": "Unasked",
        "cause": {"code": "required", "refs": ["d:s1"]},
    }


def test_a_known_cell_matches_null() -> None:
    table = tab("F", ["t:state.maybe"], [["Known", 1], ["-", 2]])
    assert value(table) == 1


def test_a_value_row_after_a_known_cell_over_an_outage_does_not_meet_rule_2() -> None:
    # Over the same outage a value cell escalates and a Known cell gives way to the next row.
    rules = srcs("Unavailable")
    assert value(tab("F", ["d:s0"], [[1, 1], ["-", 2]]), rules=rules)["kind"] == "Unavailable"
    assert value(tab("F", ["d:s0"], [["Known", 1], ["-", 2]]), rules=rules) == 2


def test_a_known_cell_lets_a_policy_row_over_an_outage_give_way() -> None:
    def decided(rows: list[Any]) -> tuple[Any, Record]:
        act = Table("d:act", t.label("page", "tell", "ask"), "F", ["d:urgent", "d:s0"], rows)
        design_ = dataclasses.replace(
            policy_design(act), rules={**policy_design(act).rules, **srcs("Unavailable")}
        )
        decision = decide_sync(load(design_), BASE, {"default": answering(0.9)})
        return decision, decision.record

    rows = [["yes", "Known", "page"], ["yes", "-", "tell"], ["-", "-", "ask"]]
    decision, record = decided(rows)
    assert decision.estimate == "tell"
    assert item(record, "entry", "d:act")["rows"] == [2, 3]
    assert item(record, "entry", "d:s0")["marker"]["kind"] == "Unavailable"
    # With a value cell over the same outage the policy is Unavailable, naming the input.
    rows = [["yes", 1, "page"], ["yes", "-", "tell"], ["-", "-", "ask"]]
    decision, record = decided(rows)
    assert isinstance(decision.estimate, Unavailable)
    cause = item(record, "entry", "d:act")["marker"]["cause"]
    assert cause == {"code": "required", "refs": ["d:s0"]}


def test_a_known_cell_stands_over_a_bare_question_and_a_record() -> None:
    record_input = tab("F", ["t:state"], [["Known", 1], ["-", 0]])
    load(design(("tab", record_input)))
    bare = Table("d:tab", "integer", "F", ["q:urgent"], [["Known", 1], ["-", 0]])
    load(design(("tab", bare), sources={"s": ASK}))


def test_a_known_cell_is_written_tagged_and_round_trips() -> None:
    loaded = load(design(("tab", tab("F", ["t:state.n"], [["Known", 1], ["-", 0]]))))
    doc = json.loads(loaded.document)
    assert doc["format"] == 10
    assert doc["rules"]["tab"]["rows"][0]["cells"][0] == {"known": True}
    assert load(loaded.document).design_hash == loaded.design_hash


def test_a_known_cell_overlaps_a_value_cell_and_no_marker_cell() -> None:
    load(design(("tab", tab("U", ["t:state.n"], [["Known", 1], ["Undetermined", 2]]))))
    refused(tab("U", ["t:state.n"], [["Known", 1], [">= 0", 2]]), 9, "overlap")


def test_a_collect_leaves_out_a_row_failing_on_known_and_propagates_from_a_value_cell() -> None:
    rules = srcs("Unavailable")
    known = tab("C+", ["d:s0", "t:state.n"], [["Known", "-", 1], ["-", "-", 3]])
    assert value(known, rules=rules) == 3
    valued = tab("C+", ["d:s0", "t:state.n"], [[1, "-", 1], ["-", "-", 3]])
    assert value(valued, rules=rules)["cause"] == {"code": "required", "refs": ["d:s0"]}
