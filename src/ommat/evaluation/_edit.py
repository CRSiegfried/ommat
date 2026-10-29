# pyright: reportPrivateUsage=false
"""Do-and-diff's first four forms and ``intervene``: edit a copy of the record, or of the
loaded design's document, and replay once.

Form 1 sets a ``q:`` or ``t:`` entry a returned call wrote to a value or to one of the six
markers a binding may return; form 2 sets a returned call to ``Failed`` with a structural code,
every entry it wrote ``Unavailable`` with that cause; form 3 sets a ``t:state`` field to a
value; form 4 replaces a rule's declaration by a one-row table of ``-`` cells over its reads,
outputting the set value or ``Unsure``, and loads the edited document. ``intervene`` takes forms
1 to 3 as a list and form 4 as ``rule``; each edit is checked against the record as read, so
the list's order changes nothing.

The record is read and refused ``unfinished``, ``nothing-to-replay`` and ``wrong-design``
first, as form 5 refuses it; ``intervene`` then refuses ``invalid-edit``, ``no-edit``,
``duplicate-target`` and ``conflicting-edits``; then each edit's own refusals in list order,
then the rule's. A record copy's log is edited, its ``hash`` set from format 3 to the edited
log's record hash, and read back through ``read_record`` (§record-hash-member). Replay reruns
the copy, its state entry, calls and cut, under the record's design or the modified one, so
the findings are what the edits changed downstream; the edited entry itself, answered as set,
is none. A plan is compared without its ``context``. Neither the record nor the loaded design
is changed, and nothing returned holds a ``Record``.
"""

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from ommat._answer import AnswerKind, ValueType
from ommat._canonical import decode, encode, record_hash
from ommat._decide import EntryRefusal
from ommat._design import InvalidDesign, LoadedDesign
from ommat._interpret import (
    _BINDING_MARKERS,
    _STRUCTURAL,
    _distinct,
    _marker_document,
    answer_written,
    no_provenance,
)
from ommat._ledger import _ITEM_DEPTH, _ORDERED, Record, _with_held_order
from ommat._load import load
from ommat._markers import Cause, CauseCode, Unavailable, Undetermined, Unsure
from ommat._refs import _RefError, candidate_values, code_point_sorted, declared_ref, parse_ref
from ommat._typed import decoded, is_a, typed
from ommat._values import HASH_FORMAT, KEY_ORDER_FORMAT, string_data, type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _replayed, _state, run_sync
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

_R = RefusalReason
_STATE = "t:state"
_Log = list[dict[str, Any]]
_Functions = Mapping[str, Callable[..., Any]] | None


@dataclass(frozen=True, slots=True)
class _SetEntry:
    ref: object
    value: object


@dataclass(frozen=True, slots=True)
class _FailCall:
    source: object
    code: object


@dataclass(frozen=True, slots=True)
class _SetField:
    path: object
    value: object


@dataclass(frozen=True, slots=True)
class _Constant:
    rule: object
    output: object


_Edit = _SetEntry | _FailCall | _SetField
_SHAPES: dict[frozenset[str], type[_Edit]] = {
    frozenset({"ref", "value"}): _SetEntry,
    frozenset({"source", "code"}): _FailCall,
    frozenset({"path", "value"}): _SetField,
}


# ---------------------------------------------------------------------------------------------
# The jobs


async def set_entry(
    record: object, loaded: LoadedDesign, ref: str, value: object
) -> dict[str, Any]:
    """Form 1: replay a copy of ``record`` with the ``q:`` or ``t:`` entry at ``ref`` set to
    ``value``; ``{"entry", "findings"}``, ``entry`` the edited log entry."""
    items, _, found = await _intervened(record, loaded, [_SetEntry(ref, value)], None, None)
    return {"entry": items[0], "findings": found}


async def fail_call(record: object, loaded: LoadedDesign, source: str, code: str) -> dict[str, Any]:
    """Form 2: replay a copy of ``record`` with the call listing ``source`` set to ``Failed``
    with ``code``; ``{"call", "findings"}``, ``call`` the edited call item."""
    items, _, found = await _intervened(record, loaded, [_FailCall(source, code)], None, None)
    return {"call": items[0], "findings": found}  # a call item precedes the entries it wrote


async def set_state_field(
    record: object, loaded: LoadedDesign, path: str, value: object
) -> dict[str, Any]:
    """Form 3: replay a copy of ``record`` with the ``t:state`` field at ``path``, a ref such as
    ``t:state.sender.address``, set to ``value``; ``{"entry", "findings"}``, ``entry`` the
    edited state entry."""
    items, _, found = await _intervened(record, loaded, [_SetField(path, value)], None, None)
    return {"entry": items[0], "findings": found}


async def constant_rule(
    record: object,
    loaded: LoadedDesign,
    rule: str,
    output: object,
    *,
    functions: _Functions = None,
) -> dict[str, Any]:
    """Form 4: replay ``record`` under ``loaded`` with the rule providing the ``d:`` ref
    ``rule`` replaced by a one-row table outputting ``output``, a value or ``Unsure``;
    ``{"document", "findings"}``, ``document`` the modified document, parsed."""
    _, document, found = await _intervened(record, loaded, [], _Constant(rule, output), functions)
    return {"document": document, "findings": found}


async def intervene(
    record: object,
    loaded: LoadedDesign,
    edits: object,
    *,
    rule: object = None,
    functions: _Functions = None,
) -> dict[str, Any]:
    """Forms 1 to 3 as ``edits``, each ``{"ref", "value"}``, ``{"source", "code"}`` or
    ``{"path", "value"}``, and form 4 as ``rule``, ``{"rule", "output"}`` or ``None``, on one
    copy, replayed once; ``{"items", "document", "findings"}``, ``items`` each log item the
    edits changed, once, in log order, ``document`` the modified document or ``None``."""
    items, document, found = await _intervened(record, loaded, edits, rule, functions, shaped=True)
    return {"items": items, "document": document, "findings": found}


def set_entry_sync(record: object, loaded: LoadedDesign, ref: str, value: object) -> dict[str, Any]:
    """``set_entry`` run to completion; ``RuntimeError`` inside a running event loop."""
    return run_sync("set_entry", lambda: set_entry(record, loaded, ref, value))


def fail_call_sync(record: object, loaded: LoadedDesign, source: str, code: str) -> dict[str, Any]:
    """``fail_call`` run to completion; ``RuntimeError`` inside a running event loop."""
    return run_sync("fail_call", lambda: fail_call(record, loaded, source, code))


def set_state_field_sync(
    record: object, loaded: LoadedDesign, path: str, value: object
) -> dict[str, Any]:
    """``set_state_field`` run to completion; ``RuntimeError`` inside a running event loop."""
    return run_sync("set_state_field", lambda: set_state_field(record, loaded, path, value))


def constant_rule_sync(
    record: object,
    loaded: LoadedDesign,
    rule: str,
    output: object,
    *,
    functions: _Functions = None,
) -> dict[str, Any]:
    """``constant_rule`` run to completion; ``RuntimeError`` inside a running event loop."""
    return run_sync(
        "constant_rule",
        lambda: constant_rule(record, loaded, rule, output, functions=functions),
    )


def intervene_sync(
    record: object,
    loaded: LoadedDesign,
    edits: object,
    *,
    rule: object = None,
    functions: _Functions = None,
) -> dict[str, Any]:
    """``intervene`` run to completion; ``RuntimeError`` inside a running event loop."""
    return run_sync(
        "intervene", lambda: intervene(record, loaded, edits, rule=rule, functions=functions)
    )


# ---------------------------------------------------------------------------------------------
# The one procedure


async def _intervened(
    record: object,
    loaded: LoadedDesign,
    edits: object,
    rule: object,
    functions: _Functions,
    *,
    shaped: bool = False,
) -> tuple[_Log, dict[str, Any] | None, list[dict[str, Any]]]:
    """The changed items in log order, the modified document or ``None``, and the findings.
    With ``shaped``, ``edits`` and ``rule`` are ``intervene``'s plain data, checked for shape;
    otherwise the form's own."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    read = _read(record, "record")
    log = read.log  # read once: every read decodes afresh
    _refuse_unfit(("record",), (read,), (log,), loaded)
    if shaped:
        edits, rule = _shaped(edits, rule)
    listed = cast("list[_Edit]", edits)
    constant = cast("_Constant | None", rule)
    _refuse_targets(listed, log)
    changed = _changes(listed, log, loaded)
    design, document = loaded, None
    if constant is not None:
        design, document = _constant(constant, loaded, functions)
    if not changed:
        return [], document, await _replayed(log, design, log)
    copy = _copy(log, read.unhashed, changed)
    edited = copy.log
    items = [edited[i] for i in sorted(changed)]
    return items, document, await _replayed(edited, design, edited)


def _shaped(edits: object, rule: object) -> tuple[list[_Edit], _Constant | None]:
    """``intervene``'s arguments as edits, refused ``invalid-edit`` then ``no-edit``."""
    if type(edits) is not list:
        raise EvaluationRefusal(
            f"edits is a {type_name(edits)}, not a list", reason=_R.invalid_edit
        )
    out: list[_Edit] = []
    for i, item in enumerate(cast("list[object]", edits)):
        form = _SHAPES.get(_keys(cast(object, item))) if type(item) is dict else None
        if form is None:
            raise EvaluationRefusal(
                f"edit {i} is not one of {{ref, value}}, {{source, code}} or {{path, value}}",
                reason=_R.invalid_edit,
            )
        out.append(form(**cast("dict[str, Any]", item)))
    constant: _Constant | None = None
    if rule is not None:
        if type(rule) is not dict or _keys(cast(object, rule)) != frozenset({"rule", "output"}):
            raise EvaluationRefusal("rule is not None or {rule, output}", reason=_R.invalid_edit)
        constant = _Constant(**cast("dict[str, Any]", rule))
    if not out and constant is None:
        raise EvaluationRefusal("edits is empty and rule is None", reason=_R.no_edit)
    return out, constant


def _keys(item: object) -> frozenset[str]:
    """An object's keys, by type alone; a key that is not exactly a ``str`` fits no shape."""
    keys = list(cast("dict[object, object]", item))
    return frozenset(cast("list[str]", keys)) if all(type(k) is str for k in keys) else frozenset()


def _refuse_targets(edits: list[_Edit], log: _Log) -> None:
    """Refuse ``duplicate-target``, two edits naming one ref, one call or one ``t:state`` path
    or a path within another, naming the later; then ``conflicting-edits``, a form 1 edit of an
    entry a form 2 edit's call wrote, naming both."""
    refs: set[str] = set()
    calls: dict[int, str] = {}
    paths: list[str] = []
    for edit in edits:
        if type(edit) is _SetEntry and type(edit.ref) is str:
            if edit.ref in refs:
                _duplicate(f"ref {edit.ref}", ref=edit.ref)
            refs.add(edit.ref)
        elif type(edit) is _FailCall and type(edit.source) is str:
            found = _call_of(edit.source, log)
            if found is not None:
                if found[0] in calls:
                    _duplicate(f"the call of source {edit.source}", source=edit.source)
                calls[found[0]] = edit.source
        elif type(edit) is _SetField and type(edit.path) is str:
            for path in paths:
                if _within(path, edit.path) or _within(edit.path, path):
                    _duplicate(f"path {edit.path}, beside {path}", path=edit.path)
            paths.append(edit.path)
    for edit in edits:
        if type(edit) is _SetEntry and type(edit.ref) is str:
            entry = _entry(edit.ref, log)
            source = None if entry is None else calls.get(cast(int, entry[1]["call"]))
            if source is not None:
                raise EvaluationRefusal(
                    f"ref {edit.ref} is an entry of the call source {source}'s edit fails",
                    reason=_R.conflicting_edits,
                    ref=edit.ref,
                    source=source,
                )


def _duplicate(
    what: str, *, ref: str | None = None, source: str | None = None, path: str | None = None
) -> NoReturn:
    raise EvaluationRefusal(
        f"two edits name {what}", reason=_R.duplicate_target, ref=ref, source=source, path=path
    )


def _within(outer: str, inner: str) -> bool:
    """Whether path ``inner`` is ``outer`` or a field beneath it."""
    return inner == outer or inner.startswith(outer + ".")


def _changes(edits: list[_Edit], log: _Log, loaded: LoadedDesign) -> dict[int, dict[str, Any]]:
    """Each edit checked against the log as read, in list order, and the items it changes by
    index; the ``t:state`` fields are set together on one state value."""
    changed: dict[int, dict[str, Any]] = {}
    fields: list[tuple[tuple[str, ...], object]] = []
    for edit in edits:
        if type(edit) is _SetEntry:
            changed.update(_set_entry(edit, log, loaded))
        elif type(edit) is _FailCall:
            changed.update(_fail_call(edit, log))
        else:
            fields.append(_set_field(cast(_SetField, edit), log, loaded))
    if fields:
        index, state = cast("tuple[int, dict[str, Any]]", _entry(_STATE, log))
        t = loaded._types[_STATE]
        value = decoded(state["value"], t)
        for names, given in fields:
            _put(value, names, given)
        forms = typed(value, t)
        assert forms is not None  # each field conformed alone, and none lies within another
        changed[index] = _rewritten(state, value=forms[0])
    return changed


# ---------------------------------------------------------------------------------------------
# Form 1: an entry


def _set_entry(edit: _SetEntry, log: _Log, loaded: LoadedDesign) -> dict[int, dict[str, Any]]:
    ref = edit.ref
    found = _entry(ref, log) if type(ref) is str else None
    if found is None:
        raise EvaluationRefusal(
            f"the record has no entry {_shown(ref)}", reason=_R.not_found, ref=_named(ref)
        )
    index, entry = found
    ref = cast(str, ref)
    if entry["call"] is None:
        raise EvaluationRefusal(f"entry {ref}'s call is null", reason=_R.null_call, ref=ref)
    call = log[cast(int, entry["call"])]
    if call["outcome"] != "returned":
        raise EvaluationRefusal(f"entry {ref}'s call is Failed", reason=_R.failed_call, ref=ref)
    value = edit.value
    if is_a(value, Undetermined):
        code = getattr(getattr(value, "cause", None), "code", None)
        if type(value) is not Unavailable or code not in _BINDING_MARKERS:
            raise EvaluationRefusal(
                f"entry {ref} may be set to a value or a marker a binding may return only",
                reason=_R.marker,
                ref=ref,
            )
        marker = _marker_document(Unavailable(Cause(cast(CauseCode, code), [ref])))
        return {index: _rewritten(entry, marker=marker)}
    written = (
        _q_value(ref, value, call, log) if ref.startswith("q:") else _t_value(ref, value, loaded)
    )
    if written is None:
        raise EvaluationRefusal(
            f"the value for entry {ref} does not conform", reason=_R.nonconforming, ref=ref
        )
    return {index: _rewritten(entry, value=written)}


def _q_value(ref: str, value: object, call: dict[str, Any], log: _Log) -> object | None:
    """An answer's document built from ``value`` in the binding's form, by the call item's
    question: its kind, its labels as asked, a candidate-valued set's rendered from the set's
    entry, or its value type and list flag; ``None`` when the answer's constructor refuses it.
    The record holds no text, so no span is bounded by one."""
    question = call["questions"][declared_ref(ref)]
    kind = AnswerKind(question["kind"])
    value_type = ValueType(question["value_type"]) if kind is AnswerKind.extraction else None
    written = answer_written(
        ref,
        value,
        kind,
        _labels(question, log),
        value_type,
        question.get("is_list", False),
        sys.maxsize,
    )
    return None if isinstance(written, Undetermined) else written[0]


def _labels(question: dict[str, Any], log: _Log) -> tuple[str, ...]:
    """A question's labels' names as asked: ``yes`` and ``no`` for an undescribed ``noul``,
    none for an extraction question, a candidate-valued set's fixed part with the set's
    candidate values at ``at``."""
    labels: object = question["labels"]
    if labels is None:
        return ("yes", "no") if question["kind"] == AnswerKind.noul else ()
    if type(labels) is dict:
        declared = cast("dict[str, Any]", labels)
        found = _entry(cast(str, declared["candidates"]), log)
        # A question is asked over a set holding a value only (§candidate-sets).
        assert found is not None
        sets = cast("dict[str, Any]", log[1].get("sets", {}))
        listed = candidate_values(found[1]["value"], sets.get(declared["candidates"]))
        at = cast(int, declared["at"])
        fixed = [_name(label) for label in cast("list[object]", declared["fixed"])]
        return (*fixed[:at], *listed, *fixed[at:])
    return tuple(_name(label) for label in cast("list[object]", labels))


def _name(label: object) -> str:
    return label if type(label) is str else cast("dict[str, str]", label)["name"]


def _t_value(ref: str, value: object, loaded: LoadedDesign) -> object | None:
    """``value``'s record form under the ref's declared type, a candidate set's members
    distinct; ``None`` when it does not conform."""
    forms = typed(value, loaded._types[declared_ref(ref)])
    if forms is None:
        return None
    if ref in loaded._candidate_sets and not _distinct(forms[0], loaded._candidate_sets[ref]):
        return None
    return forms[0]


# ---------------------------------------------------------------------------------------------
# Form 2: a call


def _fail_call(edit: _FailCall, log: _Log) -> dict[int, dict[str, Any]]:
    source = edit.source
    found = _call_of(source, log) if type(source) is str else None
    if found is None:
        raise EvaluationRefusal(
            f"the record has no call of source {_shown(source)}",
            reason=_R.not_found,
            source=_named(source),
        )
    source = cast(str, source)
    index, call = found
    if call["outcome"] != "returned":
        raise EvaluationRefusal(
            f"source {source}'s call is Failed already", reason=_R.failed_call, source=source
        )
    code = edit.code
    data = string_data(cast(str, code)) if issubclass(type(code), str) else None
    if data not in _STRUCTURAL:
        raise EvaluationRefusal(
            f"code {_shown(code)} is not raised, timed-out, invalid-return or unexpected-ref",
            reason=_R.fault_code,
            source=source,
        )
    wrote = [(i, e) for i, e in enumerate(log) if e["item"] == "entry" and e["call"] == index]
    failed = Unavailable(Cause(cast(str, data), code_point_sorted(e["ref"] for _, e in wrote)))
    # Each place gets its own copy: the read refuses a container at two places.
    outcome = {"failed": _marker_document(failed)["cause"]}
    # The hashed provenance members its format holds become null, claimed empty.
    provenance = {k: v for k, v in no_provenance().items() if k in call}
    changed = {index: {**call, "outcome": outcome, **provenance}}
    for i, entry in wrote:
        changed[i] = _rewritten(entry, marker=_marker_document(failed))
    return changed


# ---------------------------------------------------------------------------------------------
# Form 3: a state field


def _set_field(edit: _SetField, log: _Log, loaded: LoadedDesign) -> tuple[tuple[str, ...], object]:
    """The field path's names and the value, checked: the path names a field the recorded
    state holds, and the state with it set conforms."""
    path = edit.path
    names = _field_names(path)
    state = cast("dict[str, Any]", _state(log))
    t = loaded._types[_STATE]
    value = decoded(state["value"], t)
    if names is None or not _holds(value, names):
        raise EvaluationRefusal(
            f"the state holds no field {_shown(path)}", reason=_R.not_found, path=_named(path)
        )
    path = cast(str, path)
    if is_a(edit.value, Undetermined):
        raise EvaluationRefusal(
            f"state field {path} may be set to a value only", reason=_R.marker, path=path
        )
    _put(value, names, edit.value)
    if typed(value, t) is None:
        raise EvaluationRefusal(
            f"the state with {path} set does not conform", reason=_R.nonconforming, path=path
        )
    return names, edit.value


def _field_names(path: object) -> tuple[str, ...] | None:
    """The field names of a ``t:state`` ref with at least one, else ``None``."""
    if type(path) is not str:
        return None
    try:
        parsed = parse_ref(path)
    except _RefError:
        return None
    if f"{parsed.level}:{parsed.name}" != _STATE or parsed.candidate is not None:
        return None
    return parsed.fields or None


def _holds(value: object, names: tuple[str, ...]) -> bool:
    for name in names:
        if type(value) is not dict or name not in value:
            return False
        value = cast("dict[str, object]", value)[name]
    return True


def _put(value: object, names: tuple[str, ...], given: object) -> None:
    for name in names[:-1]:
        value = cast("dict[str, object]", value)[name]
    cast("dict[str, object]", value)[names[-1]] = given


# ---------------------------------------------------------------------------------------------
# Form 4: a constant rule


def _constant(
    constant: _Constant, loaded: LoadedDesign, functions: _Functions
) -> tuple[LoadedDesign, dict[str, Any]]:
    """The design loaded from ``loaded``'s document with the rule replaced, and that document,
    parsed."""
    ref = constant.rule
    document = cast("dict[str, Any]", decode(loaded.document))
    rules = cast("dict[str, dict[str, Any]]", document["rules"])
    name = next((n for n, r in rules.items() if type(ref) is str and r["provides"] == ref), None)
    if name is None:
        raise EvaluationRefusal(
            f"no rule provides {_shown(ref)}", reason=_R.not_found, ref=_named(ref)
        )
    ref = cast(str, ref)
    declared = rules[name]
    if declared.get("candidates", False) is not False or declared.get("grounded") is not None:
        raise EvaluationRefusal(
            f"rule {name} provides a candidate set or is grounded over one",
            reason=_R.candidate_rule,
            ref=ref,
        )
    output = constant.output
    if output is not Unsure and (is_a(output, Undetermined) or _marker_class(output)):
        raise EvaluationRefusal(
            f"rule {name} may output a value or Unsure only", reason=_R.marker, ref=ref
        )
    reads = (
        [i["read"] for i in declared["inputs"]]
        if declared["form"] == "table"
        else declared["reads"]
    )
    reads = list(dict.fromkeys(cast("list[str]", reads)))
    table: dict[str, Any] = {
        k: declared[k]
        for k in ("provides", "outputs", "candidates", "grounded", "target")
        if k in declared
    }
    table.update(
        form="table",
        hit="U",
        order=None,
        inputs=[{"read": r, "helper": None, "params": {}} for r in reads],
        rows=[
            {
                "cells": ["-"] * len(reads),
                "output": {"marker": "Unsure"} if output is Unsure else {"literal": output},
            }
        ],
    )
    rules[name] = table
    try:
        modified = load(document, functions=functions)
    except InvalidDesign as exc:
        raise EvaluationRefusal(
            f"the design with rule {name} constant does not load: {exc}",
            reason=_R.invalid_design,
            ref=ref,
            error=exc,
        ) from exc
    return modified, cast("dict[str, Any]", decode(modified.document))


def _marker_class(value: object) -> bool:
    return type(value) is type and issubclass(value, Undetermined)


# ---------------------------------------------------------------------------------------------
# The copy


def _copy(log: _Log, unhashed: dict[str, Any], changed: dict[int, dict[str, Any]]) -> Record:
    """The record copy: its log with the changed items, each ordered item's ``order`` derived
    afresh, its ``unhashed`` kept, ``hash`` from format 3 the edited log's record hash, read
    back through ``read_record``."""
    edited = [changed.get(i, item) for i, item in enumerate(log)]
    fmt = cast(int, log[0]["format"])
    held = [
        _with_held_order({k: v for k, v in item.items() if k != "order"})
        if fmt >= KEY_ORDER_FORMAT and item["item"] in _ORDERED
        else item
        for item in edited
    ]
    copy: dict[str, Any] = {"log": held, "unhashed": unhashed}
    if fmt >= HASH_FORMAT:
        copy["hash"] = record_hash([encode(item, _ITEM_DEPTH) for item in held])
    return _read(copy, "the copy")


def _rewritten(entry: dict[str, Any], **written: object) -> dict[str, Any]:
    """An entry with its ``value`` or ``marker`` replaced by ``written``'s one member."""
    kept = {k: v for k, v in entry.items() if k not in ("value", "marker")}
    return {**kept, **written}


# ---------------------------------------------------------------------------------------------
# Lookups


def _entry(ref: str, log: _Log) -> tuple[int, dict[str, Any]] | None:
    return next(
        ((i, e) for i, e in enumerate(log) if e["item"] == "entry" and e["ref"] == ref), None
    )


def _call_of(source: str, log: _Log) -> tuple[int, dict[str, Any]] | None:
    return next(
        ((i, e) for i, e in enumerate(log) if e["item"] == "call" and source in e["sources"]),
        None,
    )


def _named(value: object) -> str | None:
    """What a refusal names: the string given, else ``None``."""
    return value if type(value) is str else None


def _shown(value: object) -> str:
    return value if type(value) is str else f"of type {type_name(value)}"
