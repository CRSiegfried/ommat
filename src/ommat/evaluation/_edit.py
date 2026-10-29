# pyright: reportPrivateUsage=false
"""Do-and-diff's edits: ``set_entry``, ``fail_call``, ``set_state_field`` and
``constant_rule`` each change one input of a recorded turn, and ``intervene`` several; each
replays once.

``set_entry`` sets a ``q:`` or ``t:`` entry a returned call wrote to a value or to a marker a
binding may return and the record's format holds; ``fail_call`` sets a returned call to
``Failed`` with a structural code, every entry it wrote ``Unavailable`` with that cause;
``set_state_field`` sets a ``t:state`` field to a value; ``constant_rule`` replaces a rule's
declaration by a one-row table of ``-`` cells over its reads, outputting the set value or
``Unsure``, and loads the edited document. ``intervene`` takes the first three as a list of
edits and ``constant_rule``'s as ``rule``; their targets are disjoint, so the list's order
changes neither the copy nor the findings.

A ``loaded`` that is not exactly a ``LoadedDesign`` is an ``EntryRefusal`` before the record is
read; the record is then read and refused ``invalid-record``, ``unfinished``,
``nothing-to-replay`` and ``wrong-design``, as ``use_answers`` refuses it; ``intervene`` then
refuses ``invalid-edit``, ``no-edit``, ``duplicate-target`` and ``conflicting-edits``. Then
each edit's target is checked against the record as read, in list order, then the rule's; then
each value against its declared type alone, in list order; then every edit is applied to one
copy and each answer checked against it, its labels from a candidate set and its spans against
its text as edited; then the modified design is loaded. A record edit is made on a copy: its
log edited, its ``hash`` set from format 3 to the edited log's record hash, and read back
through ``read_record`` (§record-hash-member); ``constant_rule`` alone edits only the document
and makes no copy. Replay reruns the copy, its state entry, calls and cut, under the record's
design or the modified one, so the findings are what the edits changed downstream; the edited
entry itself, answered as set, is none. A plan is compared without its ``context``. Neither the
record nor the loaded design is changed, and nothing returned holds a ``Record``.
"""

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NoReturn, cast

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
from ommat._ledger import (
    _ITEM_DEPTH,
    _ORDERED,
    Record,
    _copy,
    _with_held_order,
    held_marker_code,
)
from ommat._load import load
from ommat._markers import Cause, Unavailable, Undetermined, Unsure
from ommat._read_view import _ViewError, walk
from ommat._refs import (
    _RefError,
    candidate_values,
    code_point_sorted,
    declared_ref,
    entry_ref,
    parse_ref,
)
from ommat._typed import decoded, is_a, typed
from ommat._values import HASH_FORMAT, KEY_ORDER_FORMAT, in_domain, type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _replayed, _run_sync
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


@dataclass(frozen=True, slots=True)
class _EntryAt:
    """A ``set_entry`` edit's entry, by index, and its call item, located."""

    edit: _SetEntry
    ref: str
    index: int
    entry: dict[str, Any]
    call: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CallAt:
    """A ``fail_call`` edit's call item, by index, and its code, located."""

    index: int
    call: dict[str, Any]
    code: str


@dataclass(frozen=True, slots=True)
class _FieldAt:
    """A ``set_state_field`` edit's path, its field names and its value."""

    path: str
    names: tuple[str, ...]
    value: object


# ---------------------------------------------------------------------------------------------
# The jobs


async def set_entry(
    record: object, loaded: LoadedDesign, ref: str, value: object
) -> dict[str, Any]:
    """Replay a copy of ``record`` with the ``q:`` or ``t:`` entry at ``ref`` set to ``value``;
    ``{"entry", "findings"}``, ``entry`` the edited log entry."""
    read, log = _checked(record, loaded)
    items, _, found = await _applied(read, log, loaded, [_SetEntry(ref, value)], None, None)
    return {"entry": items[0], "findings": found}


async def fail_call(record: object, loaded: LoadedDesign, source: str, code: str) -> dict[str, Any]:
    """Replay a copy of ``record`` with the call listing ``source`` set to ``Failed`` with
    ``code``; ``{"call", "findings"}``, ``call`` the edited call item."""
    read, log = _checked(record, loaded)
    items, _, found = await _applied(read, log, loaded, [_FailCall(source, code)], None, None)
    return {"call": items[0], "findings": found}  # a call item precedes the entries it wrote


async def set_state_field(
    record: object, loaded: LoadedDesign, path: str, value: object
) -> dict[str, Any]:
    """Replay a copy of ``record`` with the ``t:state`` field at ``path``, a ref such as
    ``t:state.sender.address``, set to ``value``; ``{"entry", "findings"}``, ``entry`` the
    edited state entry."""
    read, log = _checked(record, loaded)
    items, _, found = await _applied(read, log, loaded, [_SetField(path, value)], None, None)
    return {"entry": items[0], "findings": found}


async def constant_rule(
    record: object,
    loaded: LoadedDesign,
    rule: str,
    output: object,
    *,
    functions: _Functions = None,
) -> dict[str, Any]:
    """Replay ``record`` under ``loaded`` with the rule providing the ``d:`` ref ``rule``
    replaced by a one-row table outputting ``output``, a value or ``Unsure``;
    ``{"document", "findings"}``, ``document`` the modified document, parsed."""
    read, log = _checked(record, loaded)
    constant = _Constant(rule, output)
    _, document, found = await _applied(read, log, loaded, [], constant, functions)
    return {"document": document, "findings": found}


async def intervene(
    record: object,
    loaded: LoadedDesign,
    edits: object,
    *,
    rule: object = None,
    functions: _Functions = None,
) -> dict[str, Any]:
    """``set_entry``, ``fail_call`` and ``set_state_field`` edits as ``edits``, each
    ``{"ref", "value"}``, ``{"source", "code"}`` or ``{"path", "value"}``, and a
    ``constant_rule`` as ``rule``, ``{"rule", "output"}`` or ``None``, on one copy, replayed
    once; ``{"items", "document", "findings"}``, ``items`` each log item the edits changed,
    once, in log order, ``document`` the modified document or ``None``."""
    read, log = _checked(record, loaded)
    listed, constant = _shaped(edits, rule)
    items, document, found = await _applied(read, log, loaded, listed, constant, functions)
    return {"items": items, "document": document, "findings": found}


def set_entry_sync(record: object, loaded: LoadedDesign, ref: str, value: object) -> dict[str, Any]:
    """``set_entry`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync("set_entry", lambda: set_entry(record, loaded, ref, value))


def fail_call_sync(record: object, loaded: LoadedDesign, source: str, code: str) -> dict[str, Any]:
    """``fail_call`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync("fail_call", lambda: fail_call(record, loaded, source, code))


def set_state_field_sync(
    record: object, loaded: LoadedDesign, path: str, value: object
) -> dict[str, Any]:
    """``set_state_field`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync("set_state_field", lambda: set_state_field(record, loaded, path, value))


def constant_rule_sync(
    record: object,
    loaded: LoadedDesign,
    rule: str,
    output: object,
    *,
    functions: _Functions = None,
) -> dict[str, Any]:
    """``constant_rule`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync(
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
    return _run_sync(
        "intervene", lambda: intervene(record, loaded, edits, rule=rule, functions=functions)
    )


# ---------------------------------------------------------------------------------------------
# The one procedure


def _checked(record: object, loaded: LoadedDesign) -> tuple[Record, _Log]:
    """The record read, and its log, after the record checks."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    read = _read(record, "record")
    log = read.log  # read once: every read decodes afresh
    _refuse_unfit(("record",), (read,), (log,), loaded)
    return read, log


async def _applied(
    read: Record,
    log: _Log,
    loaded: LoadedDesign,
    edits: list[_Edit],
    constant: _Constant | None,
    functions: _Functions,
) -> tuple[_Log, dict[str, Any] | None, list[dict[str, Any]]]:
    """The changed items in log order, the modified document or ``None``, and the findings.

    Once duplicate and conflicting targets are refused, the edits' targets are disjoint, so no
    order among the edits changes the copy. The checks run in this order: each edit's target
    against the record as read, in list order, then the rule's; each value against its
    declared type alone, in list order; then, with every edit applied to one copy, each answer
    against that copy, its labels from a candidate set and its spans against its text as
    edited, in list order; then the load of the modified design. Nothing replays before every
    check passes."""
    _refuse_targets(edits, log)
    located = [_target(edit, log) for edit in edits]
    rule = None if constant is None else _rule_target(constant, log, loaded)
    edited = list(log)
    changed: set[int] = set()
    answers: list[_EntryAt] = []
    fields: list[_FieldAt] = []
    for at in located:
        if type(at) is _EntryAt:
            written = _alone(at, loaded)
            if written is None:
                answers.append(at)
                continue
            edited[at.index] = _rewritten(at.entry, **written)
            changed.add(at.index)
        elif type(at) is _CallAt:
            for index, item in _failed(at, log):
                edited[index] = item
                changed.add(index)
        elif type(at) is _FieldAt:
            _field_alone(at, log, loaded)
            fields.append(at)
    if fields:
        index, state = cast("tuple[int, dict[str, Any]]", _entry(_STATE, log))
        t = loaded._types[_STATE]
        value = decoded(state["value"], t)
        for at in fields:
            _put(value, at.names, at.value)
        forms = typed(value, t)
        assert forms is not None  # each field conformed alone, and none lies within another
        edited[index] = _rewritten(state, value=forms[0])
        changed.add(index)
    for at in answers:
        written = _q_value(at.ref, at.edit.value, at.call, edited, loaded)
        if written is None:
            _nonconforming(at.ref)
        edited[at.index] = _rewritten(edited[at.index], value=written)
        changed.add(at.index)
    design, document = loaded, None
    if rule is not None:
        design, document = _rule_loaded(*rule, functions)
    copied = _record_copy(edited, read.unhashed).log if changed else log
    items = [copied[i] for i in sorted(changed)]
    return items, document, await _replayed(copied, design, copied)


def _target(edit: _Edit, log: _Log) -> "_EntryAt | _CallAt | _FieldAt":
    """An edit's target located in the record as read, after its target refusals."""
    if type(edit) is _SetEntry:
        return _entry_target(edit, log)
    if type(edit) is _FailCall:
        return _call_target(edit, log)
    return _field_target(cast(_SetField, edit), log)


def _shaped(edits: object, rule: object) -> tuple[list[_Edit], _Constant | None]:
    """``intervene``'s arguments as edits, refused ``invalid-edit`` then ``no-edit``."""
    if type(edits) is not list:
        raise EvaluationRefusal(
            f"edits is a {type_name(edits)}, not a list", reason=_R.invalid_edit
        )
    out: list[_Edit] = []
    for i, item in enumerate(cast("list[object]", edits)):
        form = _SHAPES.get(_keys(item))
        if form is None:
            raise EvaluationRefusal(
                f"edit {i} is not one of {{ref, value}}, {{source, code}} or {{path, value}}",
                reason=_R.invalid_edit,
            )
        out.append(form(**cast("dict[str, Any]", item)))
    constant: _Constant | None = None
    if rule is not None:
        if _keys(rule) != frozenset({"rule", "output"}):
            raise EvaluationRefusal("rule is not None or {rule, output}", reason=_R.invalid_edit)
        constant = _Constant(**cast("dict[str, Any]", rule))
    if not out and constant is None:
        raise EvaluationRefusal("edits is empty and rule is None", reason=_R.no_edit)
    return out, constant


def _keys(item: object) -> frozenset[str]:
    """An exact ``dict``'s keys, read by type alone; none for anything else or a key that is
    not exactly a ``str``, which fit no shape."""
    if type(item) is not dict:
        return frozenset()
    keys = list(cast("dict[object, object]", item))
    return frozenset(cast("list[str]", keys)) if all(type(k) is str for k in keys) else frozenset()


def _refuse_targets(edits: list[_Edit], log: _Log) -> None:
    """Refuse ``duplicate-target``, two edits naming one ref, one call or one ``t:state`` path
    or a path within another, naming the later; then ``conflicting-edits``, a ``set_entry``
    edit of an entry a ``fail_call`` edit's call wrote, naming both."""
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


def _is_marker(value: object) -> bool:
    """A marker, or a marker class given in place of one."""
    return is_a(value, Undetermined) or (type(value) is type and issubclass(value, Undetermined))


def _not_found(what: str, given: object, member: Literal["ref", "source", "path"]) -> NoReturn:
    """``not-found`` for ``given``, named under ``member`` when it is a string."""
    named = given if type(given) is str else None
    shown = named if named is not None else f"of type {type_name(given)}"
    raise EvaluationRefusal(
        f"{what} {shown}",
        reason=_R.not_found,
        ref=named if member == "ref" else None,
        source=named if member == "source" else None,
        path=named if member == "path" else None,
    )


# ---------------------------------------------------------------------------------------------
# set_entry


def _entry_target(edit: _SetEntry, log: _Log) -> _EntryAt:
    """The entry and its call item; refused ``not-found``, ``null-call``, ``failed-call``,
    then ``marker`` for a marker no binding may return or the record's format does not hold
    under a returned call."""
    ref = edit.ref
    found = _entry(ref, log) if type(ref) is str else None
    if found is None:
        _not_found("the record has no entry", ref, "ref")
    ref = cast(str, ref)
    index, entry = found
    if entry["call"] is None:
        raise EvaluationRefusal(f"entry {ref}'s call is null", reason=_R.null_call, ref=ref)
    call = log[cast(int, entry["call"])]
    if call["outcome"] != "returned":
        raise EvaluationRefusal(f"entry {ref}'s call is Failed", reason=_R.failed_call, ref=ref)
    value = edit.value
    if _is_marker(value):
        code = _code(value)
        fmt = cast(int, log[0]["format"])
        if type(value) is not Unavailable or code not in _BINDING_MARKERS:
            _refuse_marker(f"entry {ref} may be set to a marker a binding may return only", ref)
        if not held_marker_code(cast(str, code), fmt):
            _refuse_marker(
                f"entry {ref} may not be set to {code}, which a format-{fmt} record does not "
                "hold under a returned call",
                ref,
            )
    return _EntryAt(edit, ref, index, entry, call)


def _code(marker: object) -> object:
    """A marker's cause code, or ``None`` for a marker class, read by attribute."""
    return getattr(getattr(marker, "cause", None), "code", None)


def _alone(at: _EntryAt, loaded: LoadedDesign) -> dict[str, object] | None:
    """The entry's new ``marker`` or ``value`` member, its value checked against its declared
    type alone, ``nonconforming`` otherwise; ``None`` for a ``q:`` value, whose answer is
    written on the copy."""
    ref, value = at.ref, at.edit.value
    if _is_marker(value):
        cause = Cause(cast(str, _code(value)), [ref])
        return {"marker": _marker_document(Unavailable(cause))}
    if ref.startswith("q:"):
        question = at.call["questions"][declared_ref(ref)]
        # Labels from a candidate set wait for the copy; an extraction answer is checked here
        # with no span bound, and against its text on the copy.
        if type(question["labels"]) is not dict and _q_value(ref, value, at.call, [], None) is None:
            _nonconforming(ref)
        return None
    written = _t_value(ref, value, loaded)
    if written is None:
        _nonconforming(ref)
    return {"value": written}


def _nonconforming(ref: str) -> NoReturn:
    raise EvaluationRefusal(
        f"the value for entry {ref} does not conform", reason=_R.nonconforming, ref=ref
    )


def _refuse_marker(message: str, ref: str) -> NoReturn:
    raise EvaluationRefusal(message, reason=_R.marker, ref=ref)


def _q_value(
    ref: str, value: object, call: dict[str, Any], log: _Log, loaded: LoadedDesign | None
) -> object | None:
    """An answer's document built from ``value`` in the binding's form, by the call item's
    question: its kind, its labels as asked, a candidate-valued set's rendered from the set's
    entry in ``log``, or its value type and list flag, an extraction answer's spans within its
    text as ``log`` holds it, unbounded with no ``loaded``; ``None`` when the answer's check
    refuses it."""
    question = call["questions"][declared_ref(ref)]
    kind = AnswerKind(question["kind"])
    labels = _labels(question, log)
    if labels is None:
        return None
    value_type = None
    bound = sys.maxsize
    if kind is AnswerKind.extraction:
        value_type = ValueType(question["value_type"])
        if loaded is not None:
            bound = _text_length(ref, log, loaded)
    is_list = question.get("is_list", False)
    written = answer_written(ref, value, kind, labels, value_type, is_list, bound)
    return None if isinstance(written, Undetermined) else written[0]


def _labels(question: dict[str, Any], log: _Log) -> tuple[str, ...] | None:
    """A question's labels' names as asked: ``yes`` and ``no`` for an undescribed ``noul``,
    none for an extraction question, a candidate-valued set's fixed part with the set's
    candidate values at ``at``; ``None`` when the set holds no value, so no answer fits."""
    labels: object = question["labels"]
    if labels is None:
        return ("yes", "no") if question["kind"] == AnswerKind.noul else ()
    if type(labels) is dict:
        declared = cast("dict[str, Any]", labels)
        found = _entry(cast(str, declared["candidates"]), log)
        if found is None or "value" not in found[1]:
            return None
        sets = cast("dict[str, Any]", log[1].get("sets", {}))
        listed = candidate_values(found[1]["value"], sets.get(declared["candidates"]))
        at = cast(int, declared["at"])
        fixed = [_name(label) for label in cast("list[object]", declared["fixed"])]
        return (*fixed[:at], *listed, *fixed[at:])
    return tuple(_name(label) for label in cast("list[object]", labels))


def _name(label: object) -> str:
    return label if type(label) is str else cast("dict[str, str]", label)["name"]


def _text_length(ref: str, log: _Log, loaded: LoadedDesign) -> int:
    """The length of an extraction question's text, read at its declared ref from the log's
    entry, as interpret bounds a span by; no bound when the log holds no string there."""
    declared = declared_ref(ref)
    question = loaded._design.sources[loaded._providers[declared]].provides[declared]
    text = cast("Mapping[str, str]", question.text)["ref"]
    found = _entry(entry_ref(text), log)
    if found is None or "value" not in found[1]:
        return sys.maxsize
    try:
        value = walk(found[1]["value"], parse_ref(text).fields, text)
    except _ViewError:
        return sys.maxsize
    return len(value) if type(value) is str else sys.maxsize


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
# fail_call


def _call_target(edit: _FailCall, log: _Log) -> _CallAt:
    """The call item listing the source; refused ``not-found``, ``failed-call``, then
    ``fault-code`` for a code that is not exactly a structural one's string."""
    source = edit.source
    found = _call_of(source, log) if type(source) is str else None
    if found is None:
        _not_found("the record has no call of source", source, "source")
    source = cast(str, source)
    if found[1]["outcome"] != "returned":
        raise EvaluationRefusal(
            f"source {source}'s call is Failed already", reason=_R.failed_call, source=source
        )
    code = edit.code
    if type(code) is not str or code not in _STRUCTURAL:
        shown = code if type(code) is str else f"of type {type_name(code)}"
        raise EvaluationRefusal(
            f"code {shown} is not raised, timed-out, invalid-return or unexpected-ref",
            reason=_R.fault_code,
            source=source,
        )
    return _CallAt(found[0], found[1], code)


def _failed(at: _CallAt, log: _Log) -> list[tuple[int, dict[str, Any]]]:
    """The call item set to ``Failed`` and each entry it wrote ``Unavailable``, by index."""
    index, call, code = at.index, at.call, at.code
    wrote = [(i, e) for i, e in enumerate(log) if e["item"] == "entry" and e["call"] == index]
    failed = Unavailable(Cause(code, code_point_sorted(e["ref"] for _, e in wrote)))
    # Each place gets its own copy: the read refuses a container at two places.
    outcome = {"failed": _marker_document(failed)["cause"]}
    # The hashed provenance members its format holds become null, claimed empty; before
    # format 5 they include snapshot.
    provenance = {k: v for k, v in {**no_provenance(), "snapshot": None}.items() if k in call}
    changed = [(index, {**call, "outcome": outcome, **provenance})]
    return changed + [(i, _rewritten(e, marker=_marker_document(failed))) for i, e in wrote]


# ---------------------------------------------------------------------------------------------
# set_state_field


def _field_target(edit: _SetField, log: _Log) -> _FieldAt:
    """The field path's names; refused ``not-found`` unless the path names a field the
    recorded state holds, then ``marker``."""
    path = edit.path
    names = _field_names(path)
    state = cast("tuple[int, dict[str, Any]]", _entry(_STATE, log))[1]
    if names is None or not _holds(state["value"], names):
        _not_found("the state holds no field", path, "path")
    path = cast(str, path)
    if _is_marker(edit.value):
        raise EvaluationRefusal(
            f"state field {path} may be set to a value only", reason=_R.marker, path=path
        )
    return _FieldAt(path, names, edit.value)


def _field_alone(at: _FieldAt, log: _Log, loaded: LoadedDesign) -> None:
    """Refuse ``nonconforming`` when the recorded state with this field set does not conform;
    fields within one another refused, each conforms alone as all do together."""
    state = cast("tuple[int, dict[str, Any]]", _entry(_STATE, log))[1]
    t = loaded._types[_STATE]
    value = decoded(state["value"], t)
    _put(value, at.names, at.value)
    if typed(value, t) is None:
        raise EvaluationRefusal(
            f"the state with {at.path} set does not conform",
            reason=_R.nonconforming,
            path=at.path,
        )


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
# constant_rule


def _rule_target(
    constant: _Constant, log: _Log, loaded: LoadedDesign
) -> tuple[dict[str, Any], str, str]:
    """The loaded design's document with the rule replaced, the rule's name and its ref;
    refused ``not-found``, ``candidate-rule``, then ``marker``. A rule reading a grounded
    declared ref, which the record's ``settings`` item names under ``grounded``, is a candidate
    rule: no table but one grounded over its set may read it."""
    ref = constant.rule
    document = cast("dict[str, Any]", decode(loaded.document))
    rules = cast("dict[str, dict[str, Any]]", document["rules"])
    name = next((n for n, r in rules.items() if type(ref) is str and r["provides"] == ref), None)
    if name is None:
        _not_found("no rule provides", ref, "ref")
    ref = cast(str, ref)
    declared = rules[name]
    reads = (
        [i["read"] for i in declared["inputs"]]
        if declared["form"] == "table"
        else declared["reads"]
    )
    reads = list(dict.fromkeys(cast("list[str]", reads)))
    grounded = cast("dict[str, str]", log[1].get("grounded", {}))
    if (
        declared.get("candidates", False) is not False
        or declared.get("grounded") is not None
        or any(_declared(r) in grounded for r in reads)
    ):
        raise EvaluationRefusal(
            f"rule {name} provides a candidate set, is grounded over one or reads a grounded ref",
            reason=_R.candidate_rule,
            ref=ref,
        )
    output = constant.output
    if output is not Unsure and _is_marker(output):
        raise EvaluationRefusal(
            f"rule {name} may output a value or Unsure only", reason=_R.marker, ref=ref
        )
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
                "output": {"marker": "Unsure"} if output is Unsure else _literal(output),
            }
        ],
    )
    rules[name] = table
    return document, name, ref


def _rule_loaded(
    document: dict[str, Any], name: str, ref: str, functions: _Functions
) -> tuple[LoadedDesign, dict[str, Any]]:
    """The modified design loaded, and its document parsed; ``invalid-design`` carrying the
    ``InvalidDesign`` load raised."""
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


def _literal(output: object) -> dict[str, object]:
    """``output``'s output entry: copied in exact types, so a container it holds at two places
    is two; one outside the value domain as given, for load to refuse."""
    return {"literal": _copy(output) if in_domain(output) else output}


def _declared(read: str) -> str:
    """A read's declared ref: its field path dropped; ``candidate`` reads as itself."""
    try:
        parsed = parse_ref(read)
    except _RefError:
        return read
    return f"{parsed.level}:{parsed.name}"


# ---------------------------------------------------------------------------------------------
# The copy


def _record_copy(log: _Log, unhashed: dict[str, Any]) -> Record:
    """The record copy of an edited log: each ordered item's ``order`` derived afresh, its
    ``unhashed`` kept, ``hash`` from format 3 the log's record hash, read back through
    ``read_record``."""
    fmt = cast(int, log[0]["format"])
    held = [
        _with_held_order({k: v for k, v in item.items() if k != "order"})
        if fmt >= KEY_ORDER_FORMAT and item["item"] in _ORDERED
        else item
        for item in log
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
