"""``ommat.authoring``: a parsed document read as the design or slice it says.

The fixtures are ``authoring_slice.toml`` and ``authoring_table.toml``; each has a Python twin
here, and a design built from the document has the twin's hash. The rule functions the slice
names are in ``authoring_rules``.
"""

import copy
import re
import sys
import tomllib
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import authoring_rules as rules
from ommat import (
    Design,
    Helper,
    InvalidDesign,
    Parameter,
    Question,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Target,
    Text,
    load,
)
from ommat._table import cell_form
from ommat.authoring import AuthoringError, Slice, design_from, slice_from

HERE = Path(__file__).parent
SLICE_DOC: dict[str, Any] = tomllib.loads((HERE / "authoring_slice.toml").read_text())
TABLE_DOC: dict[str, Any] = tomllib.loads((HERE / "authoring_table.toml").read_text())
VALUES = {"bars/move_at": 0.7}
UNIT = "number [0, 1]"
OP = 'label{"keep", "move"}'
YES_NO = [{"name": "yes", "description": "It does"}, {"name": "no", "description": "It does not"}]


def slice_doc() -> dict[str, Any]:
    return copy.deepcopy(SLICE_DOC)


def table_doc() -> dict[str, Any]:
    return copy.deepcopy(TABLE_DOC)


def at(rules: Mapping[str, Any], name: str) -> Any:
    """A rule or table of a mapping, typed loosely for its members."""
    return rules[name]


def hash_of(design: Design) -> str:
    return load(design).design_hash


# ---------------------------------------------------------------------------------------------
# The Python twins


def twin_parts(version: str, move_at: float) -> Slice:
    return Slice(
        sources={
            "list_folders": Source(
                {"t:folders": "list<record{id: string, name: string}>"}, oracle="read", timeout=10.0
            ),
            "ask_move": Source(
                {
                    "q:move": Question(
                        "noul",
                        Text("Does the request ask to move the message?"),
                        YES_NO,
                        target="tg:op",
                    ),
                    "q:where": Question(
                        "choice",
                        Text("Where should it go?"),
                        ["inbox", "archive"],
                        latent_targets=[{"target": "tg:op", "role": "reflective"}],
                    ),
                },
                needs=["t:state"],
                oracle="classifier",
                requirement=SourceRequirement("d:kind", ["move", "copy"]),
            ),
        },
        rules={
            "kind": Rule(
                "d:kind",
                'label{"copy", "move", "none"}',
                rules.folder_kind,
                version=version,
                reads=["t:state.text"],
            ),
            "op": Rule(
                "d:op",
                OP,
                rules.folder_op,
                version=version,
                reads=["q:move", "t:folders"],
                target="tg:op",
            ),
            "plan": Rule(
                "d:plan",
                UNIT,
                rules.folder_plan,
                version=version,
                reads=["d:op"],
                params=["bars/move_at", "bars/keep_at"],
            ),
            "act": Rule("d:act", OP, rules.folder_act, version="2", reads=["d:op", "d:plan"]),
        },
        parameters={
            "bars/move_at": Parameter(UNIT, move_at),
            "bars/keep_at": Parameter(UNIT, 0.25),
        },
    )


def slice_twin(version: str = "1", move_at: float = 0.7) -> Design:
    part = twin_parts(version, move_at)
    return Design(
        "record{text: string}",
        sources=part.sources,
        rules=part.rules,
        parameters=part.parameters,
        targets={"tg:op": Target("Whether the request moves a message", values=OP)},
    )


GRADE = 'label{"high", "low", "none"}'
STATE = "record{n: integer, flag: boolean, items: list<string>, word: string, m: integer}"
ROWS = [
    [">= bars/high", {"one_of": [True, False]}, "-", "high"],
    ["-", "-", ">= 2", "low"],
    ["-", {"is": True}, "-", "low"],
    ["-", "-", "-", "none"],
]


def table_twin() -> Design:
    count = Helper("size", "t:state.items")
    return Design(
        STATE,
        rules={
            "grade": Table("d:grade", GRADE, "F", ["t:state.n", "t:state.flag", count], ROWS),
            "grade_again": Table(
                "d:grade_again", GRADE, "F", ["t:state.m", "t:state.flag", count], ROWS
            ),
        },
        parameters={"bars/high": Parameter("integer", 5)},
    )


def test_a_slice_document_has_its_twins_hash() -> None:
    built = design_from(slice_doc(), values=VALUES)
    assert hash_of(built) == hash_of(slice_twin())


def test_a_table_document_has_its_twins_hash() -> None:
    assert hash_of(design_from(table_doc())) == hash_of(table_twin())


def test_a_document_read_from_json_builds_the_same_design() -> None:
    import json

    again = json.loads(json.dumps(SLICE_DOC))
    assert hash_of(design_from(again, values=VALUES)) == hash_of(slice_twin())


def test_slice_from_gives_a_frozen_slice_that_equals_its_twins() -> None:
    got = slice_from(as_slice(slice_doc()), values=VALUES)
    assert isinstance(got, Slice)
    assert got.sources.keys() == {"list_folders", "ask_move"}
    want = twin_parts("1", 0.7)
    assert got.sources == want.sources
    assert got.rules.keys() == want.rules.keys()
    assert got.parameters == want.parameters
    with pytest.raises(AttributeError):
        got.sources = {}  # pyright: ignore[reportAttributeAccessIssue]


def test_the_caller_version_and_value_win() -> None:
    doc = slice_doc()
    del doc["version"]
    got = slice_from(as_slice(doc), version="W", values={"bars/move_at": 0.9})
    want = twin_parts("W", 0.9)
    assert at(got.rules, "op").version == "W"
    assert at(got.rules, "act").version == "2"
    assert got.rules == want.rules
    assert got.parameters == want.parameters
    whole = design_from(doc, version="W", values={"bars/move_at": 0.9})
    assert hash_of(whole) == hash_of(slice_twin("W", 0.9))


def test_authoring_error_is_a_value_error_naming_a_path() -> None:
    assert issubclass(AuthoringError, ValueError)
    with pytest.raises(ValueError, match=r"^document: unknown key 'nope'"):
        slice_from({"nope": 1})


# ---------------------------------------------------------------------------------------------
# Refusals, each with its path


def as_slice(document: Mapping[str, Any]) -> dict[str, Any]:
    """A whole document without the members only a design holds."""
    return {k: v for k, v in document.items() if k not in ("state", "targets")}


def refused(document: Any, expect: str, **kw: Any) -> None:
    # A whole document is read as a design, so its design keys are checked too.
    read = design_from if isinstance(document, dict) and "state" in document else slice_from
    with pytest.raises(AuthoringError) as caught:
        read(cast("dict[str, Any]", document), **kw)
    assert expect in str(caught.value), str(caught.value)


def slip(edit: Callable[[dict[str, Any]], object], expect: str) -> None:
    doc = slice_doc()
    del doc["version"]  # the caller's `version="V"` is the one owner
    edit(doc)
    refused(doc, expect, version="V", values=VALUES)


def rename(
    table: Callable[[dict[str, Any]], dict[str, Any]], old: str, new: str
) -> Callable[..., None]:
    def go(doc: dict[str, Any]) -> None:
        found = table(doc)
        found[new] = found.pop(old)

    return go


def move(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["sources"]["ask_move"]["ask"]["q:move"]


def test_a_misspelt_key_is_refused_with_its_table() -> None:
    slip(rename(lambda d: d["rules"]["op"], "reads", "read"), "rules.op: unknown key 'read'")
    slip(
        rename(lambda d: d["sources"]["ask_move"], "only_when", "only_when_"),
        "sources.ask_move: unknown key 'only_when_'",
    )
    slip(rename(lambda d: d, "types", "tyeps"), "document: unknown key 'tyeps'")


SLIPS: list[tuple[str, Callable[[dict[str, Any]], object], str]] = [
    (
        "question kind twice",
        lambda d: move(d).update(choice="x"),
        "sources.ask_move.ask.q:move: a question names exactly one kind",
    ),
    (
        "only_when two refs",
        lambda d: d["sources"]["ask_move"]["only_when"].update({"d:x": ["y"]}),
        "sources.ask_move.only_when: one ref and its values, found ['d:kind', 'd:x']",
    ),
    (
        "unknown alias",
        lambda d: d["rules"]["op"].update(type="$opp"),
        "rules.op.type: unknown type alias $opp",
    ),
    (
        "alias cycle",
        lambda d: d["types"].update(op="$op"),
        "rules.op.type: type alias $op refers to itself (op -> op)",
    ),
    (
        "unknown labels entry",
        lambda d: move(d).update(labels="nope"),
        "sources.ask_move.ask.q:move.labels: no [labels] entry 'nope'",
    ),
    ("rule without fn", lambda d: d["rules"]["op"].pop("fn"), "rules.op: missing key 'fn'"),
    ("rule without type", lambda d: d["rules"]["op"].pop("type"), "rules.op: missing key 'type'"),
    (
        "parameter without type",
        lambda d: d["parameters"]["bars/move_at"].pop("type"),
        "parameters.bars/move_at: missing key 'type'",
    ),
    (
        "bad function",
        lambda d: d["rules"]["op"].update(fn="nope:x"),
        "rules.op.fn: cannot import 'nope:x'",
    ),
    (
        "function without a module",
        lambda d: (d.pop("module"), d["rules"]["op"].update(fn="folder_op")),
        "rules.kind.fn: 'folder_kind' has no module",
    ),
    (
        "target without a statement",
        lambda d: d["targets"]["tg:op"].pop("statement"),
        "targets.tg:op: missing key 'statement'",
    ),
]


@pytest.mark.parametrize(("label", "edit", "expect"), SLIPS, ids=[s[0] for s in SLIPS])
def test_a_slip_in_a_slice_document_is_refused_with_its_path(
    label: str, edit: Callable[[dict[str, Any]], object], expect: str
) -> None:
    slip(edit, expect)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("slip_probe")
    exec("async def a(r):\n    return 1\nconst = 'x'\n", module.__dict__)
    monkeypatch.setitem(sys.modules, "slip_probe", module)


@pytest.mark.usefixtures("probe")
def test_an_async_or_non_function_rule_is_refused() -> None:
    slip(
        lambda d: d["rules"]["op"].update(fn="slip_probe:a"),
        "rules.op.fn: 'slip_probe:a' is an async function",
    )
    slip(
        lambda d: d["rules"]["op"].update(fn="slip_probe:const"),
        "rules.op.fn: 'slip_probe:const' is not a function",
    )


def test_a_rule_function_and_a_relative_module_are_imported_by_name() -> None:
    refused(
        {"rules": {"r": {"fn": "..x:y", "type": "string"}}}, "rules.r.fn: cannot import '..x:y'"
    )
    refused({"rules": {"r": {"fn": ":", "type": "string"}}}, "rules.r.fn: cannot import ':'")


# ---- types ----------------------------------------------------------------------------------


def required(doc: dict[str, Any]) -> dict[str, Any]:
    doc["types"]["op"] = {"required": True}
    return doc


def test_a_required_type_takes_the_callers_string() -> None:
    built = slice_from(
        as_slice(required(slice_doc())), values=VALUES, types={"op": 'label{"a", "b"}'}
    )
    assert at(built.rules, "op").outputs == 'label{"a", "b"}'
    aliased = slice_from(as_slice(required(slice_doc())), values=VALUES, types={"op": "$unit"})
    assert at(aliased.rules, "op").outputs == UNIT


def test_a_required_type_not_supplied_is_refused() -> None:
    refused(
        required(slice_doc()),
        "types.op: the document leaves its type to the caller",
        values=VALUES,
    )


def test_a_required_type_used_only_inside_another_alias_is_still_named() -> None:
    doc = slice_doc()
    doc["types"]["inner"] = {"required": True}
    doc["types"]["outer"] = "list<$inner>"
    doc["rules"]["op"]["type"] = "$outer"
    refused(doc, "types.inner", values=VALUES)


@pytest.mark.parametrize(
    "marker", [{"required": False}, {"requried": True}, {"required": True, "type": "x"}]
)
def test_a_malformed_required_marker_is_refused(marker: dict[str, Any]) -> None:
    doc = slice_doc()
    doc["types"]["op"] = marker
    refused(doc, "types.op: a type string or", values=VALUES, types={"op": "x"})


def test_a_written_type_may_not_be_supplied_and_an_unknown_name_is_refused() -> None:
    refused(
        slice_doc(), "types.op: the document already gives a type", values=VALUES, types={"op": "x"}
    )
    refused(slice_doc(), "types.nope", values=VALUES, types={"nope": "x"})
    refused(
        required(slice_doc()),
        "rules.op.type: type alias $op refers to itself (op -> op)",
        values=VALUES,
        types={"op": "$op"},
    )


def test_design_from_takes_values_and_types_and_expands_names_in_state() -> None:
    doc = {
        "state": "record{request: $text, tools: $tools}",
        "module": "authoring_rules",
        "version": "1",
        "types": {"text": "string", "tools": {"required": True}},
        "parameters": {"bar": {"type": "number [0, 1]"}},
        "rules": {"act": {"fn": "act", "type": "$tools", "params": ["bar"]}},
    }
    built = design_from(doc, values={"bar": 0.7}, types={"tools": 'label{"a", "b"}'})
    assert built.parameters["bar"].value == 0.7
    assert at(built.rules, "act").outputs == 'label{"a", "b"}'
    assert built.state == 'record{request: string, tools: label{"a", "b"}}'


# ---- one owner for a value or a version -----------------------------------------------------


def test_a_value_has_one_owner() -> None:
    refused(slice_doc(), "parameters.bars/move_at: the document leaves its value to the caller")
    refused(
        slice_doc(),
        "values.bars/keep_at: the document already gives a value",
        values={**VALUES, "bars/keep_at": 0.1},
    )
    refused(
        slice_doc(),
        "values.bars/nope: the document declares no such parameter",
        values={**VALUES, "bars/nope": 1},
    )


def test_a_version_has_one_owner() -> None:
    refused(
        slice_doc(),
        "version: the document gives one and so does the caller; keep one",
        version="V",
        values=VALUES,
    )
    got = slice_from(as_slice(slice_doc()), values=VALUES)
    assert at(got.rules, "op").version == "1"


# ---- patterns in params ---------------------------------------------------------------------


def test_a_params_pattern_matches_this_documents_parameters_only() -> None:
    plan = at(slice_from(as_slice(slice_doc()), values=VALUES).rules, "plan")
    assert list(plan.params) == ["bars/move_at", "bars/keep_at"]
    slip(
        lambda d: d["rules"]["plan"].update(params=["nope/*"]),
        "rules.plan.params: pattern 'nope/*' matches no parameter",
    )
    slip(
        lambda d: d["rules"]["plan"].update(params=["bars/move_at", "bars/*"]),
        "rules.plan.params: 'bars/move_at' is named twice (pattern 'bars/*')",
    )
    other = {
        "parameters": {"other/bar": {"type": UNIT}},
        "rules": {"r": {"fn": "folder_op", "type": OP, "params": ["bars/*"]}},
        "module": "authoring_rules",
    }
    refused(
        other,
        "rules.r.params: pattern 'bars/*' matches no parameter of this document",
        version="V",
        values={"other/bar": 0.1},
    )


# ---------------------------------------------------------------------------------------------
# Tables


def table_slip(edit: Callable[[dict[str, Any]], object], expect: str) -> None:
    doc = table_doc()
    edit(doc)
    refused(doc, expect)


def grade(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["rules"]["grade"]


def again(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["rules"]["grade_again"]


def test_a_row_naming_an_unknown_input_or_without_then_is_refused() -> None:
    table_slip(
        lambda d: grade(d)["rows"].append({"rulez": ">= 1", "then": "none"}),
        "rules.grade.rows[4]: unknown input 'rulez'",
    )
    table_slip(
        lambda d: grade(d)["rows"].append({"n": ">= 1"}),
        "rules.grade.rows[4]: a row is a table with `then`",
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"thenn": "none"}),
        "rules.grade.rows[2]: a row is a table with `then`",
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"then": {"ref": "q:x", "x": 1}}),
        "rules.grade.rows[2].then: a value, or { ref",
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"then": {"ref": 3}}),
        "rules.grade.rows[2].then.ref: a string",
    )


@pytest.mark.parametrize("key", ["type", "hit", "inputs", "rows"])
def test_a_table_without_a_required_key_is_refused(key: str) -> None:
    table_slip(lambda d: grade(d).pop(key), f"rules.grade: missing key '{key}'")


def test_a_table_is_refused_for_what_it_may_not_hold() -> None:
    table_slip(lambda d: grade(d).update(hit="C#"), "rules.grade.hit")
    table_slip(lambda d: grade(d).update(order=["a"]), "rules.grade: unknown key 'order'")
    table_slip(
        lambda d: grade(d).update(fn="x:y"),
        "rules.grade: a rule is a function (`fn`) or a table (`rows`), not both",
    )
    table_slip(lambda d: grade(d).update(inputs={}), "rules.grade.inputs: one entry per input")
    table_slip(lambda d: grade(d).update(rows={}), "rules.grade.rows: a list of rows")


def test_an_input_named_then_is_refused() -> None:
    table_slip(
        lambda d: grade(d)["inputs"].update(then="q:a"),
        "rules.grade.inputs.then: `then` is a row's result, so no input may be named it",
    )


def test_a_table_input_is_a_ref_or_a_helper_call() -> None:
    table_slip(
        lambda d: grade(d)["inputs"].update(n=5), "rules.grade.inputs.n: a ref, or a helper call"
    )
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"size": 5}),
        "rules.grade.inputs.n: names 1 helpers (size) or reads no ref",
    )
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"mass": "q:x", "size": "q:y"}),
        "rules.grade.inputs.n: names 2 helpers (mass, size)",
    )
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"labl": "q:x", "labels": "p"}),
        "rules.grade.inputs.n: no key names a helper; the helpers are",
    )


def test_a_helper_call_finds_its_helper_by_name_in_any_key_position() -> None:
    first = table_doc()
    grade(first)["inputs"]["count"] = {"mass": "q:x", "labels": "p"}
    last = table_doc()
    grade(last)["inputs"]["count"] = {"labels": "p", "mass": "q:x"}
    want = Helper("mass", "q:x", labels="p")
    assert at(slice_from(as_slice(first)).rules, "grade").inputs[2] == want
    assert at(slice_from(as_slice(last)).rules, "grade").inputs[2] == want


def test_one_of_an_array_is_the_string_set_form() -> None:
    assert cell_form({"one_of": ["done", "empty"]}) == cell_form("{done, empty}")


def test_a_copy_keeps_its_bases_input_order_and_rows_and_changes_only_what_it_names() -> None:
    got = slice_from(as_slice(table_doc())).rules
    base, copy_ = at(got, "grade"), at(got, "grade_again")
    assert copy_.rows == base.rows
    assert copy_.inputs == ("t:state.m", "t:state.flag", Helper("size", "t:state.items"))
    assert copy_.provides == "d:grade_again"


def test_a_bad_like_is_refused() -> None:
    table_slip(
        lambda d: again(d).update(like="nope"),
        "rules.grade_again.like: 'nope' is not a table of this document",
    )
    table_slip(
        lambda d: again(d).update(like="folder_op"),
        "rules.grade_again.like: 'folder_op' is not a table of this document",
    )
    table_slip(
        lambda d: d["rules"].update(third={"like": "grade_again"}),
        "rules.third.like: 'grade_again' is itself a copy",
    )
    table_slip(
        lambda d: again(d).update(rows=[{"then": "x"}]), "rules.grade_again: unknown key 'rows'"
    )
    table_slip(
        lambda d: again(d)["inputs"].update(nope="x:y"),
        "rules.grade_again.inputs.nope: 'grade' has no such input",
    )
    table_slip(lambda d: again(d).update(hit="F"), "rules.grade_again: unknown key 'hit'")
    table_slip(
        lambda d: (again(d).pop("inputs"), again(d).pop("provides")),
        "rules.grade_again.like: changes nothing from 'grade'",
    )
    table_slip(
        lambda d: (
            again(d).pop("inputs"),
            again(d).pop("provides"),
            again(d).update(type="$grade"),
        ),
        "rules.grade_again.like: changes nothing from 'grade'",
    )
    refused(
        {
            "rules": {
                "c": {"like": "b", "type": "number"},
                "b": {"type": "string", "hit": "F", "rows": [{"then": "x"}]},
            }
        },
        "rules.c.like: 'b' has no `inputs` to copy",
    )
    refused(
        {"rules": {"r": {"like": "b", "type": "string"}, "b": 3}},
        "rules.r.like: 'b' is not a table of this document",
    )


# ---------------------------------------------------------------------------------------------
# Values of the wrong shape


@pytest.mark.parametrize(
    ("document", "expect"),
    [
        (
            {
                "parameters": {"a/x": {"type": "number", "value": 1}},
                "rules": {
                    "r": {"fn": "authoring_rules:folder_op", "type": "string", "params": "a/*"}
                },
            },
            "rules.r.params: a list of strings",
        ),
        (
            {"sources": {"s": {"only_when": {}}}},
            "sources.s.only_when: one ref and its values, found none",
        ),
        (
            {"sources": {"s": {"only_when": False}}},
            "sources.s.only_when: one ref and its values, found bool",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?"}}, "fetch": {"q:a": "string"}}}},
            "sources.s.fetch.q:a: the source already asks 'q:a'",
        ),
        ({"rules": []}, "rules: a table, found list"),
        ({"rules": {"r": {"fn": 3, "type": "string"}}}, "rules.r.fn: a string, found int"),
        (
            {"rules": {"r": {"fn": "authoring_rules:folder_op", "type": 3}}},
            "rules.r.type: a string, found int",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": []}}}}},
            "sources.s.ask.q:a.slot: an empty list names no slot",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "slot": "x"}}}}},
            "sources.s.ask.q:a.slot: names a slot but its source has no `grounded` rule",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "latent": ["a"]}}}}},
            "sources.s.ask.q:a.latent: a table",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": 3}}}}},
            "sources.s.ask.q:a.slot: a name",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": [["b"]]}}}}},
            "sources.s.ask.q:a.slot: a list of strings",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "latent": {"t": 3}}}}}},
            "sources.s.ask.q:a.latent.t: a role",
        ),
        ([], "document: a table"),
    ],
)
def test_a_value_of_the_wrong_shape_is_refused_with_its_path(document: Any, expect: str) -> None:
    refused(document, expect)


def test_only_when_takes_any_values_as_the_engine_does() -> None:
    got = slice_from({"sources": {"s": {"only_when": {"d:x": [True, 1]}}}})
    assert got.sources["s"].requirement == SourceRequirement("d:x", [True, 1])


def test_design_from_refuses_an_unknown_key_and_names_its_path() -> None:
    with pytest.raises(AuthoringError, match="document: unknown key 'rulez'"):
        design_from({"state": "record{}", "rulez": {}})


def test_a_sources_feedback_key_is_refused_as_an_unknown_key_with_its_path() -> None:
    with pytest.raises(AuthoringError, match=r"^sources\.s: unknown key 'feedback'; it may hold "):
        design_from({"state": "record{}", "sources": {"s": {"feedback": True}}})


def test_a_document_that_still_says_policy_is_refused_as_an_unknown_key() -> None:
    with pytest.raises(AuthoringError, match=r"^document: unknown key 'policy'; it may hold "):
        design_from({"state": "record{}", "policy": "d:a"})


# ---------------------------------------------------------------------------------------------
# What the reader leaves to load


def test_a_shape_load_refuses_is_not_refused_by_the_reader() -> None:
    # A bare string where a list belongs: the reader passes it on and load refuses it.
    cases: list[tuple[Callable[[dict[str, Any]], object], str]] = [
        (lambda d: d["rules"]["op"].update(reads="q:move"), "reads is not a list"),
        (lambda d: d["sources"]["ask_move"].update(needs="t:state"), "needs is not a list"),
        (
            lambda d: d["sources"]["ask_move"].update(only_when={"d:kind": "move"}),
            "requirement's values are not a list",
        ),
    ]
    for edit, says in cases:
        doc = slice_doc()
        edit(doc)
        design = design_from(doc, values=VALUES)
        with pytest.raises(InvalidDesign) as caught:
            load(design)
        assert says in str(caught.value)


def test_a_misspelt_then_is_refused_by_load_not_by_the_reader() -> None:
    doc = table_doc()
    grade(doc)["rows"][0]["then"] = "hihg"  # a value the table's type does not hold
    design = design_from(doc)
    with pytest.raises(InvalidDesign):
        load(design)
    # The same slip in the Python twin is refused the same way: it is the design's slip.
    twin = table_twin()
    bad = Table(
        "d:grade",
        GRADE,
        "F",
        at(twin.rules, "grade").inputs,
        [
            [">= bars/high", "-", "-", "hihg"],
            ["-", "-", "-", "none"],
        ],
    )
    with pytest.raises(InvalidDesign):
        load(Design(STATE, rules={"grade": bad}, parameters=dict(twin.parameters)))


def test_a_row_without_then_is_refused_even_when_its_other_key_is_an_input() -> None:
    # `then` misspelt as `thenn`: the row names no result, whatever else it holds.
    doc = table_doc()
    grade(doc)["rows"][0] = {"n": ">= bars/high", "thenn": "high"}
    with pytest.raises(AuthoringError, match=re.escape("rules.grade.rows[0]: a row is a table")):
        design_from(doc)


def test_a_row_key_that_is_not_an_input_is_refused_as_an_unknown_input() -> None:
    doc = table_doc()
    grade(doc)["rows"][0] = {"nn": ">= bars/high", "then": "high"}
    with pytest.raises(AuthoringError, match=re.escape("rules.grade.rows[0]: unknown input 'nn'")):
        design_from(doc)


def test_the_reader_reads_a_mapping_and_not_a_file() -> None:
    import ommat.authoring

    assert not hasattr(ommat.authoring, "read_toml")


@pytest.mark.parametrize("document", ["authoring_slice.toml", HERE / "authoring_slice.toml"])
def test_a_path_or_its_text_is_not_a_document(document: Any) -> None:
    for read in (slice_from, design_from):
        with pytest.raises(AuthoringError, match=r"^document: a table, found (str|\w+Path)"):
            read(document)


# ---------------------------------------------------------------------------------------------
# A grounded source, a candidates rule, slots and a text ref


GROUNDED_DOC: dict[str, Any] = tomllib.loads((HERE / "authoring_grounded.toml").read_text())


def grounded_twin() -> Design:
    street = [{"name": "street", "origin": {"candidate": "d:streets"}}]
    pair = [*street, {"name": "lamp", "origin": {"candidate": "d:streets"}}]
    return Design(
        "record{text: string}",
        sources={
            "lamps": Source(
                {
                    "q:lamps/out": Question("noul", Text("Is {street} lit?", street)),
                    "q:lamps/pair": Question("noul", Text("Is {street} lit by {lamp}?", pair)),
                    "q:lamps/rule": Question("noul", Text("Is {street} lit?", street)),
                },
                needs=["t:state"],
                oracle="classifier",
                cost=3,
                grounded="d:streets",
            ),
            "reader": Source(
                {
                    "q:reader/words": Question(
                        "extraction",
                        Text("Which words name a street?"),
                        value_type="text",
                        is_list=True,
                        text={"ref": "t:state.text"},
                    )
                },
                needs=["t:state"],
                oracle="classifier",
            ),
        },
        rules={
            "streets": Rule(
                "d:streets",
                "list<string>",
                rules.streets,
                version="1",
                reads=["t:state.text"],
                candidates=True,
            ),
            "pick": Rule("d:pick", "string", rules.pick, version="1", reads=["q:lamps/out"]),
        },
    )


def test_a_grounded_document_has_its_twins_hash() -> None:
    built = design_from(copy.deepcopy(GROUNDED_DOC))
    assert hash_of(built) == hash_of(grounded_twin())
    assert built.sources == grounded_twin().sources
    assert at(built.rules, "streets").candidates is True


def test_a_slot_name_a_list_and_a_rule_fill_the_wording_per_candidate() -> None:
    questions = design_from(copy.deepcopy(GROUNDED_DOC)).sources["lamps"].provides
    one = cast("Question", questions["q:lamps/out"]).wording
    two = cast("Question", questions["q:lamps/pair"]).wording
    named = cast("Question", questions["q:lamps/rule"]).wording
    assert isinstance(one, Text)
    assert isinstance(two, Text)
    assert isinstance(named, Text)
    assert [dict(s) for s in one.slots] == [
        {"name": "street", "origin": {"candidate": "d:streets"}}
    ]
    assert [s["name"] for s in two.slots] == ["street", "lamp"]
    assert all(s["origin"] == {"candidate": "d:streets"} for s in two.slots)
    assert one.slots == named.slots


def test_a_slot_naming_another_rule_needs_no_grounded_source() -> None:
    doc = {"sources": {"s": {"ask": {"q:a": {"noul": "{x}?", "slot": {"x": "d:other"}}}}}}
    wording = cast("Question", slice_from(doc).sources["s"].provides["q:a"]).wording
    assert isinstance(wording, Text)
    assert [dict(s) for s in wording.slots] == [{"name": "x", "origin": {"candidate": "d:other"}}]


def test_a_text_ref_is_wrapped_and_a_question_without_slots_has_none() -> None:
    reader = design_from(copy.deepcopy(GROUNDED_DOC)).sources["reader"].provides["q:reader/words"]
    assert cast("Question", reader).text == {"ref": "t:state.text"}
    assert cast("Question", reader).wording == Text("Which words name a street?")


def test_a_candidates_rule_defaults_to_false() -> None:
    assert at(slice_from(as_slice(slice_doc()), values=VALUES).rules, "op").candidates is False


def test_a_slot_that_is_not_a_name_list_or_table_is_refused() -> None:
    base: dict[str, Any] = {"grounded": "d:f", "ask": {"q:a": {"noul": "?"}}}
    for bad in [("a",), (), 1.5]:
        doc = {"sources": {"s": copy.deepcopy(base)}}
        doc["sources"]["s"]["ask"]["q:a"]["slot"] = bad
        refused(doc, "sources.s.ask.q:a.slot: a name, a list of names")


def test_a_tuple_of_latent_roles_is_refused() -> None:
    doc = {"sources": {"s": {"ask": {"q:a": {"noul": "?", "latent": {"tg:x": ("reflective",)}}}}}}
    refused(doc, "sources.s.ask.q:a.latent.tg:x: a role, or a list of roles")


# ---- latent targets -------------------------------------------------------------------------


def latent_of(question: dict[str, Any]) -> list[dict[str, str]]:
    doc = {"sources": {"s": {"ask": {"q:a": {"noul": "?", **question}}}}}
    got = cast("Question", slice_from(doc).sources["s"].provides["q:a"])
    return [{"target": m["target"], "role": str(m["role"])} for m in got.latent_targets]


def test_latent_targets_pass_through_and_roles_may_be_a_list() -> None:
    explicit = [{"target": "tg:a", "role": "reflective"}]
    assert latent_of({"latent_targets": explicit}) == explicit
    assert latent_of({"latent": {"tg:b": ["reflective", "formative"]}}) == [
        {"target": "tg:b", "role": "reflective"},
        {"target": "tg:b", "role": "formative"},
    ]


def test_explicit_latent_targets_come_before_the_latent_table() -> None:
    got = latent_of(
        {
            "latent_targets": [{"target": "tg:a", "role": "formative"}],
            "latent": {"tg:b": "reflective", "tg:c": ["formative"]},
        }
    )
    assert got == [
        {"target": "tg:a", "role": "formative"},
        {"target": "tg:b", "role": "reflective"},
        {"target": "tg:c", "role": "formative"},
    ]


def test_latent_targets_must_be_a_list_and_latent_a_table() -> None:
    base: dict[str, Any] = {"sources": {"s": {"ask": {"q:a": {"noul": "?"}}}}}
    for key, value, expect in [
        ("latent_targets", "tg:a", "sources.s.ask.q:a.latent_targets: a list"),
        ("latent", ["tg:a"], "sources.s.ask.q:a.latent: a table"),
    ]:
        doc = copy.deepcopy(base)
        doc["sources"]["s"]["ask"]["q:a"][key] = value
        refused(doc, expect)


# ---- tables: like, target, provides ---------------------------------------------------------

BASE_TYPE = 'label{"a", "b"}'


def small() -> dict[str, Any]:
    return {
        "state": "record{n: integer, m: integer}",
        "version": "1",
        "targets": {"tg:t": {"statement": "S"}},
        "rules": {
            "base": {
                "type": BASE_TYPE,
                "hit": "F",
                "inputs": {"n": "t:state.n"},
                "rows": [{"n": ">= 1", "then": "a"}, {"then": "b"}],
            }
        },
    }


def small_table(provides: str = "d:base", **kw: Any) -> Table:
    outputs = kw.pop("outputs", BASE_TYPE)
    return Table(provides, outputs, "F", ["t:state.n"], [[">= 1", "a"], ["-", "b"]], **kw)


def small_twin(**rules_: Table | Rule) -> Design:
    return Design(
        "record{n: integer, m: integer}",
        rules={"base": small_table(), **rules_},
        targets={"tg:t": Target("S")},
    )


def with_copy(**spec: Any) -> dict[str, Any]:
    doc = small()
    doc["rules"]["copy"] = {"like": "base", **spec}
    return doc


def test_a_copy_that_changes_the_type_only_keeps_the_rest_of_its_base() -> None:
    built = design_from(with_copy(type='label{"a", "b", "c"}'))
    twin = small_twin(copy=small_table("d:copy", outputs='label{"a", "b", "c"}'))
    assert hash_of(built) == hash_of(twin)
    assert at(built.rules, "copy").target is None


def test_a_copy_that_changes_the_target_only_keeps_the_rest_of_its_base() -> None:
    built = design_from(with_copy(target="tg:t"))
    assert at(built.rules, "copy").target == "tg:t"
    assert at(built.rules, "copy").outputs == BASE_TYPE
    assert at(built.rules, "base").target is None
    assert hash_of(built) == hash_of(small_twin(copy=small_table("d:copy", target="tg:t")))


def test_a_copy_that_changes_the_provides_only_keeps_the_rest_of_its_base() -> None:
    built = design_from(with_copy(provides="d:copy"))
    assert hash_of(built) == hash_of(small_twin(copy=small_table("d:copy")))
    # The reader passes `provides` on as written; load refuses a ref that is not the rule's own.
    other = design_from(with_copy(provides="d:other"))
    assert at(other.rules, "copy").provides == "d:other"
    with pytest.raises(InvalidDesign, match="a rule provides its own d: ref"):
        load(other)


def test_a_copy_takes_its_base_target_unless_it_names_its_own() -> None:
    doc = small()
    doc["rules"]["base"]["target"] = "tg:t"
    doc["rules"]["copy"] = {"like": "base", "provides": "d:copy"}
    assert at(design_from(doc).rules, "copy").target == "tg:t"
    doc["targets"]["tg:u"] = {"statement": "U"}
    doc["rules"]["copy"]["target"] = "tg:u"
    assert at(design_from(doc).rules, "copy").target == "tg:u"


def test_a_copy_of_a_base_without_rows_or_of_a_function_rule_is_refused() -> None:
    doc = with_copy(provides="d:copy")
    del doc["rules"]["base"]["rows"]
    refused(doc, "rules.base: missing key 'rows'")
    doc = with_copy(provides="d:copy")
    doc["rules"]["fn_rule"] = {"fn": "authoring_rules:pick", "type": "string"}
    doc["rules"]["copy"]["like"] = "fn_rule"
    refused(doc, "rules.copy.like: 'fn_rule' is not a table of this document")


def test_a_copy_that_repeats_its_base_is_refused() -> None:
    expect = "rules.copy.like: changes nothing from 'base'"
    refused(with_copy(type=BASE_TYPE), expect)
    refused(with_copy(inputs={"n": "t:state.n"}), expect)
    base = small()
    base["rules"]["base"]["target"] = "tg:t"
    base["rules"]["copy"] = {"like": "base", "target": "tg:t"}
    refused(base, expect)


def test_a_table_may_name_a_target_and_a_provides() -> None:
    doc = small()
    doc["rules"]["base"].update(target="tg:t", provides="d:base")
    built = design_from(doc)
    assert hash_of(built) == hash_of(
        Design(
            "record{n: integer, m: integer}",
            rules={"base": small_table(target="tg:t")},
            targets={"tg:t": Target("S")},
        )
    )
    assert at(built.rules, "base").target == "tg:t"
    doc["rules"]["base"]["provides"] = "d:custom"
    assert at(design_from(doc).rules, "base").provides == "d:custom"


def test_a_function_rule_may_name_its_own_provides_and_a_design_its_name() -> None:
    doc: dict[str, Any] = {
        "name": "mine",
        "state": "record{}",
        "rules": {"r": {"fn": "authoring_rules:pick", "type": "string", "provides": "d:r"}},
    }
    built = design_from(doc, version="V")
    assert built.name == "mine"
    twin = Design(
        "record{}",
        rules={"r": Rule("d:r", "string", rules.pick, version="V")},
        name="mine",
    )
    assert hash_of(built) == hash_of(twin)
    doc["rules"]["r"]["provides"] = "d:custom"
    assert at(design_from(doc, version="V").rules, "r").provides == "d:custom"


def test_a_function_named_by_a_dotted_qualname_is_found() -> None:
    kit = {"r": {"fn": "authoring_rules:Kit.pick", "type": "string"}}
    assert at(slice_from({"rules": kit}).rules, "r").function is rules.Kit.pick
    under_module = {
        "module": "authoring_rules",
        "rules": {"r": {"fn": "Kit.pick", "type": "string"}},
    }
    assert at(slice_from(under_module).rules, "r").function is rules.Kit.pick
    refused(
        {"rules": {"r": {"fn": "authoring_rules:Kit.nope", "type": "string"}}},
        "rules.r.fn: cannot import 'authoring_rules:Kit.nope'",
    )


# ---- patterns: `?`, `[...]` and case ---------------------------------------------------------


def pattern_doc(*params: str) -> dict[str, Any]:
    return {
        "module": "authoring_rules",
        "parameters": {
            "bars/a1": {"type": "integer", "value": 1},
            "bars/a2": {"type": "integer", "value": 2},
            "bars/b1": {"type": "integer", "value": 3},
        },
        "rules": {"r": {"fn": "pick", "type": "string", "params": list(params)}},
    }


def test_a_params_pattern_may_use_a_question_mark_or_a_bracket_set() -> None:
    def got(pattern: str) -> list[str]:
        return list(at(slice_from(pattern_doc(pattern), version="V").rules, "r").params)

    assert got("bars/a?") == ["bars/a1", "bars/a2"]
    assert got("bars/[ab]1") == ["bars/a1", "bars/b1"]
    assert got("bars/[!a]1") == ["bars/b1"]


def test_a_params_pattern_is_case_sensitive() -> None:
    refused(
        pattern_doc("BARS/*"),
        "rules.r.params: pattern 'BARS/*' matches no parameter of this document",
        version="V",
    )
    refused(
        pattern_doc("bars/A?"),
        "rules.r.params: pattern 'bars/A?' matches no parameter",
        version="V",
    )


# ---- more refusals --------------------------------------------------------------------------


def test_design_from_without_state_is_refused() -> None:
    with pytest.raises(AuthoringError, match=r"^document: missing key 'state'"):
        design_from({"name": "a"})


def test_an_unknown_key_in_a_question_a_parameter_or_a_target_is_refused() -> None:
    refused(
        {"sources": {"s": {"ask": {"q:a": {"noul": "?", "labelz": []}}}}},
        "sources.s.ask.q:a: unknown key 'labelz'",
    )
    refused(
        {"parameters": {"p": {"type": "integer", "valeu": 1}}},
        "parameters.p: unknown key 'valeu'",
    )
    refused(
        {
            "state": "record{}",
            "targets": {"tg:x": {"statement": "S", "valuez": 1}},
        },
        "targets.tg:x: unknown key 'valuez'",
    )


def test_a_module_that_is_not_a_string_is_refused() -> None:
    refused({"module": 3, "rules": {"r": {"fn": "pick", "type": "string"}}}, "module: a string")


def test_ask_that_is_not_a_table_is_refused() -> None:
    refused({"sources": {"s": {"ask": ["q:a"]}}}, "sources.s.ask: a table, found list")
    refused({"sources": {"s": {"fetch": "string"}}}, "sources.s.fetch: a table, found str")


def test_a_helper_call_with_no_key_is_refused() -> None:
    doc = table_doc()
    grade(doc)["inputs"]["n"] = {}
    refused(doc, "rules.grade.inputs.n: a ref, or a helper call such as")
