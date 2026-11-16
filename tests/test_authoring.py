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
                oracle="jev",
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
        "d:act",
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
        "d:grade",
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
    return {k: v for k, v in document.items() if k not in ("state", "policy", "targets")}


def refused(document: Any, expect: str, **kw: Any) -> None:
    # A whole document is read as a design, so its design keys are checked too.
    read = design_from if isinstance(document, dict) and "policy" in document else slice_from
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
    ("question kind twice", lambda d: move(d).update(choice="x"), "exactly one kind"),
    (
        "only_when two refs",
        lambda d: d["sources"]["ask_move"]["only_when"].update({"d:x": ["y"]}),
        "sources.ask_move.only_when",
    ),
    ("unknown alias", lambda d: d["rules"]["op"].update(type="$opp"), "unknown type alias $opp"),
    ("alias cycle", lambda d: d["types"].update(op="$op"), "refers to itself"),
    ("unknown labels entry", lambda d: move(d).update(labels="nope"), "no [labels] entry"),
    ("rule without fn", lambda d: d["rules"]["op"].pop("fn"), "missing key 'fn'"),
    ("rule without type", lambda d: d["rules"]["op"].pop("type"), "missing key 'type'"),
    (
        "parameter without type",
        lambda d: d["parameters"]["bars/move_at"].pop("type"),
        "missing key 'type'",
    ),
    (
        "bad function",
        lambda d: d["rules"]["op"].update(fn="nope:x"),
        "rules.op.fn: cannot import 'nope:x'",
    ),
    (
        "function without a module",
        lambda d: (d.pop("module"), d["rules"]["op"].update(fn="folder_op")),
        "has no module",
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
    slip(lambda d: d["rules"]["op"].update(fn="slip_probe:a"), "is an async function")
    slip(lambda d: d["rules"]["op"].update(fn="slip_probe:const"), "is not a function")


def test_a_rule_function_and_a_relative_module_are_imported_by_name() -> None:
    refused({"rules": {"r": {"fn": "..x:y", "type": "string"}}}, "cannot import")
    refused({"rules": {"r": {"fn": ":", "type": "string"}}}, "cannot import")


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
    refused(required(slice_doc()), "refers to itself", values=VALUES, types={"op": "$op"})


def test_design_from_takes_values_and_types_and_expands_names_in_state() -> None:
    doc = {
        "state": "record{request: $text, tools: $tools}",
        "policy": "d:act",
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
    refused(slice_doc(), "leaves its value to the caller")
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
    refused(slice_doc(), "keep one", version="V", values=VALUES)
    got = slice_from(as_slice(slice_doc()), values=VALUES)
    assert at(got.rules, "op").version == "1"


# ---- patterns in params ---------------------------------------------------------------------


def test_a_params_pattern_matches_this_documents_parameters_only() -> None:
    plan = at(slice_from(as_slice(slice_doc()), values=VALUES).rules, "plan")
    assert list(plan.params) == ["bars/move_at", "bars/keep_at"]
    slip(lambda d: d["rules"]["plan"].update(params=["nope/*"]), "matches no parameter")
    slip(
        lambda d: d["rules"]["plan"].update(params=["bars/move_at", "bars/*"]),
        "'bars/move_at' is named twice (pattern 'bars/*')",
    )
    other = {
        "parameters": {"other/bar": {"type": UNIT}},
        "rules": {"r": {"fn": "folder_op", "type": OP, "params": ["bars/*"]}},
        "module": "authoring_rules",
    }
    refused(other, "matches no parameter of this document", version="V", values={"other/bar": 0.1})


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
        lambda d: grade(d)["rows"].append({"n": ">= 1"}), "rows[4]: a row is a table with `then`"
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"thenn": "none"}),
        "rows[2]: a row is a table with `then`",
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"then": {"ref": "q:x", "x": 1}}),
        "rows[2].then: a value, or { ref",
    )
    table_slip(
        lambda d: grade(d)["rows"].__setitem__(2, {"then": {"ref": 3}}), "then.ref: a string"
    )


@pytest.mark.parametrize("key", ["type", "hit", "inputs", "rows"])
def test_a_table_without_a_required_key_is_refused(key: str) -> None:
    table_slip(lambda d: grade(d).pop(key), f"rules.grade: missing key '{key}'")


def test_a_table_is_refused_for_what_it_may_not_hold() -> None:
    table_slip(lambda d: grade(d).update(hit="C#"), "rules.grade.hit")
    table_slip(lambda d: grade(d).update(order=["a"]), "rules.grade: unknown key 'order'")
    table_slip(
        lambda d: grade(d).update(fn="x:y"), "a function (`fn`) or a table (`rows`), not both"
    )
    table_slip(lambda d: grade(d).update(inputs={}), "rules.grade.inputs: one entry per input")
    table_slip(lambda d: grade(d).update(rows={}), "rules.grade.rows: a list of rows")


def test_an_input_named_then_is_refused() -> None:
    table_slip(
        lambda d: grade(d)["inputs"].update(then="q:a"),
        "rules.grade.inputs.then: `then` is a row's result, so no input may be named it",
    )


def test_a_table_input_is_a_ref_or_a_helper_call() -> None:
    table_slip(lambda d: grade(d)["inputs"].update(n=5), "inputs.n: a ref, or a helper call")
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"size": 5}),
        "inputs.n: names 1 helpers (size) or reads no ref",
    )
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"mass": "q:x", "size": "q:y"}),
        "inputs.n: names 2 helpers (mass, size)",
    )
    table_slip(
        lambda d: grade(d)["inputs"].update(n={"labl": "q:x", "labels": "p"}),
        "inputs.n: no key names a helper; the helpers are",
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
    table_slip(lambda d: again(d).update(like="nope"), "grade_again.like: 'nope' is not a table")
    table_slip(lambda d: again(d).update(like="folder_op"), "is not a table")
    table_slip(lambda d: d["rules"].update(third={"like": "grade_again"}), "is itself a copy")
    table_slip(lambda d: again(d).update(rows=[{"then": "x"}]), "grade_again: unknown key 'rows'")
    table_slip(
        lambda d: again(d)["inputs"].update(nope="x:y"),
        "grade_again.inputs.nope: 'grade' has no such input",
    )
    table_slip(lambda d: again(d).update(hit="F"), "unknown key 'hit'")
    table_slip(
        lambda d: (again(d).pop("inputs"), again(d).pop("provides")),
        "grade_again.like: changes nothing from 'grade'",
    )
    table_slip(
        lambda d: (
            again(d).pop("inputs"),
            again(d).pop("provides"),
            again(d).update(type="$grade"),
        ),
        "changes nothing",
    )
    refused(
        {
            "rules": {
                "c": {"like": "b", "type": "number"},
                "b": {"type": "string", "hit": "F", "rows": [{"then": "x"}]},
            }
        },
        "has no `inputs` to copy",
    )
    refused(
        {"rules": {"r": {"like": "b", "type": "string"}, "b": 3}}, "is not a table of this document"
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
        ({"sources": {"s": {"only_when": {}}}}, "only_when: one ref and its values, found none"),
        ({"sources": {"s": {"only_when": False}}}, "found bool"),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?"}}, "fetch": {"q:a": "string"}}}},
            "a ref has one provider",
        ),
        ({"rules": []}, "rules: a table"),
        ({"rules": {"r": {"fn": 3, "type": "string"}}}, "rules.r.fn: a string"),
        (
            {"rules": {"r": {"fn": "authoring_rules:folder_op", "type": 3}}},
            "rules.r.type: a string",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": []}}}}},
            "an empty list names no slot",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "slot": "x"}}}}},
            "names a slot but its source has no `grounded` rule",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "latent": ["a"]}}}}},
            "latent: a table",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": 3}}}}},
            "slot: a name",
        ),
        (
            {"sources": {"s": {"grounded": "d:f", "ask": {"q:a": {"noul": "?", "slot": [["b"]]}}}}},
            "slot: a list of strings",
        ),
        (
            {"sources": {"s": {"ask": {"q:a": {"noul": "?", "latent": {"t": 3}}}}}},
            "latent.t: a role",
        ),
        ([], "document: a table"),
    ],
)
def test_a_value_of_the_wrong_shape_is_refused_with_its_path(document: Any, expect: str) -> None:
    refused(document, expect)


def test_only_when_takes_any_values_as_the_engine_does() -> None:
    got = slice_from({"sources": {"s": {"only_when": {"d:x": [True, 1]}}}})
    assert got.sources["s"].requirement == SourceRequirement("d:x", [True, 1])


def test_design_from_refuses_a_document_without_state_or_policy() -> None:
    refused_whole = {"state": "record{}"}
    with pytest.raises(AuthoringError, match="document: missing key 'policy'"):
        design_from(refused_whole)
    with pytest.raises(AuthoringError, match="document: unknown key 'rulez'"):
        design_from({"state": "record{}", "policy": "d:a", "rulez": {}})


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
        (lambda d: d.update(policy=3), "the policy is not a string"),
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
        load(Design(STATE, "d:grade", rules={"grade": bad}, parameters=dict(twin.parameters)))


def test_a_row_key_that_is_not_an_input_is_the_readers_to_refuse() -> None:
    # `then` misspelt as a key of a row is the reader's slip: a row names no result.
    doc = table_doc()
    grade(doc)["rows"][0] = {"n": ">= bars/high", "thenn": "high"}
    with pytest.raises(AuthoringError, match=re.escape("rows[0]: a row is a table with `then`")):
        design_from(doc)


def test_the_reader_reads_a_mapping_and_not_a_file() -> None:
    import ommat.authoring

    assert not hasattr(ommat.authoring, "read_toml")
    assert isinstance(SLICE_DOC, Mapping)
