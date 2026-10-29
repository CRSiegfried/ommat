# pyright: reportPrivateUsage=false
"""The deciding inputs of an entry: each declared read of the rule providing it, classed by
whether some do-and-diff intervention on the read changes the entry (§do-and-diff).

Each intervention is one of forms 1 to 4 run through ``_edit``'s one procedure, so its checks,
its copy and its replay are those of ``set_entry``, ``fail_call``, ``set_state_field`` and
``constant_rule``. A ``q:`` or ``t:`` read is tried through its entry, a grounded declared
ref's through each of its entries: form 1 at each marker a binding may return and the record's
format holds, a ``q:`` answer at ``Unavailable`` with ``refused`` only, and a ``t:`` entry at
each value of a finite type; form 2 at each structural code, when the entry's call wrote it
alone. A ``t:state`` field is tried by form 3 at each value of a finite type, and a ``d:`` read
by form 4 at ``Unsure`` and each value of a finite type. A finite type is ``boolean``, a label,
or either nullable. A trial the form refuses for its target, a ``null-call``, ``failed-call``,
``not-found``, ``candidate-rule`` or ``invalid-design``, reaches nothing; so does one whose
replay asks outside the record, ``not-in-record``, ``ref-not-in-record`` or
``question-changed``. A trial changes the entry when replay gives an ``entry`` finding at its
ref.

A read is ``deciding`` when some trial changes the entry; else ``not-determined`` when it had
no trial or one reached nothing; else ``markers-only`` when some target is not of a finite type;
else ``not-deciding``. A ``candidate`` read, which no form reaches, is ``not-determined``.

The record is checked as the edits check it, then ``ref`` refused ``not-found`` unless it names
an entry of the record a rule provides, before any replay. The output is plain data.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from ommat._design import LoadedDesign
from ommat._interpret import _BINDING_MARKERS, _STRUCTURAL
from ommat._ledger import Record, held_marker_code
from ommat._markers import Cause, Unavailable, Unsure
from ommat._read_view import is_candidate_read
from ommat._refs import parse_ref
from ommat._replay import FindingKind
from ommat._types import Label, Nullable, RecordOf, Scalar, Type
from ommat.evaluation._diff import _run_sync
from ommat.evaluation._edit import (
    _STATE,
    _applied,
    _checked,
    _Constant,
    _declared,
    _Edit,
    _FailCall,
    _Log,
    _not_found,
    _SetEntry,
    _SetField,
)
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

_R = RefusalReason
_K = FindingKind
_Functions = Mapping[str, Callable[..., Any]] | None
_Trial = _Edit | _Constant

_UNREACHED = frozenset(
    {_R.null_call, _R.failed_call, _R.not_found, _R.candidate_rule, _R.invalid_design}
)
_OUTSIDE = frozenset({_K.not_in_record, _K.ref_not_in_record, _K.question_changed})


async def deciding_inputs(
    record: object, loaded: LoadedDesign, ref: str, *, functions: _Functions = None
) -> dict[str, str]:
    """Each declared read of the rule providing the ``d:`` entry at ``ref``, in declared order,
    to ``deciding``, ``not-deciding``, ``markers-only`` or ``not-determined``."""
    read, log = _checked(record, loaded)
    rule = _rule_of(ref, log, loaded)
    out: dict[str, str] = {}
    for r in dict.fromkeys(loaded._reads[rule]):
        trials, finite = _trials(r, log, loaded)
        out[r] = await _classed(ref, trials, finite, read, log, loaded, functions)
    return out


def deciding_inputs_sync(
    record: object, loaded: LoadedDesign, ref: str, *, functions: _Functions = None
) -> dict[str, str]:
    """``deciding_inputs`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync(
        "deciding_inputs", lambda: deciding_inputs(record, loaded, ref, functions=functions)
    )


def _rule_of(ref: object, log: _Log, loaded: LoadedDesign) -> str:
    """The id of the rule providing the entry at ``ref``; ``not-found`` unless the record holds
    that entry and a rule provides it."""
    if type(ref) is str and any(i["item"] == "entry" and i["ref"] == ref for i in log):
        provider = loaded._providers.get(_declared(ref))
        if provider is not None and provider in loaded._design.rules:
            return provider
    _not_found("no rule provides an entry of the record at", ref, "ref")


async def _classed(
    ref: str,
    trials: list[_Trial],
    finite: bool,
    read: Record,
    log: _Log,
    loaded: LoadedDesign,
    functions: _Functions,
) -> str:
    reached = bool(trials)
    for trial in trials:
        edits: list[_Edit] = [] if type(trial) is _Constant else [cast(_Edit, trial)]
        constant = trial if type(trial) is _Constant else None
        try:
            _, _, found = await _applied(read, log, loaded, edits, constant, functions)
        except EvaluationRefusal as exc:
            if exc.reason not in _UNREACHED:
                raise
            reached = False
            continue
        if any(f["kind"] in _OUTSIDE for f in found):
            reached = False
        elif any(f["kind"] is _K.entry and f["ref"] == ref for f in found):
            return "deciding"
    if not reached:
        return "not-determined"
    return "not-deciding" if finite else "markers-only"


def _trials(r: str, log: _Log, loaded: LoadedDesign) -> tuple[list[_Trial], bool]:
    """The interventions on read ``r``, and whether each target's type is finite, so its
    trials are exhaustive."""
    if is_candidate_read(r):
        return [], False
    parsed = parse_ref(r)
    declared = f"{parsed.level}:{parsed.name}"
    if declared == _STATE:
        values = _values(_field_type(loaded._types[_STATE], parsed.fields))
        return [_SetField(r, v) for v in values or ()], values is not None
    if parsed.level == "d":
        values = _values(loaded._types[declared])
        outputs = [Unsure, *(values or ())]
        return [_Constant(declared, o) for o in outputs], values is not None
    fmt = log[0]["format"]
    codes = ["refused"] if parsed.level == "q" else sorted(_BINDING_MARKERS)
    markers = [Unavailable(Cause(c)) for c in codes if held_marker_code(c, fmt)]
    values = None if parsed.level == "q" else _values(loaded._types[declared])
    trials: list[_Trial] = []
    for index, entry in enumerate(log):
        if entry["item"] != "entry" or _declared(entry["ref"]) != declared:
            continue
        at = entry["ref"]
        trials += [_SetEntry(at, v) for v in [*markers, *(values or ())]]
        if entry["call"] is not None and _alone(entry["call"], index, log):
            source = cast(str, log[entry["call"]]["sources"][0])
            trials += [_FailCall(source, c) for c in sorted(_STRUCTURAL)]
    return trials, values is not None


def _alone(call: int, index: int, log: _Log) -> bool:
    """Whether the call at ``call`` wrote the entry at ``index`` and no other."""
    return all(i == index for i, e in enumerate(log) if e["item"] == "entry" and e["call"] == call)


def _values(t: Type | None) -> list[object] | None:
    """Every value of a finite type, ``None`` for any other."""
    if isinstance(t, Nullable):
        inner = _values(t.inner)
        return None if inner is None else [*inner, None]
    if isinstance(t, Label):
        return list(t.values)
    if isinstance(t, Scalar) and t.name == "boolean":
        return [False, True]
    return None


def _field_type(t: Type, fields: tuple[str, ...]) -> Type | None:
    """The type of a state field through ``fields``, its definitions resolved; nullable when a
    record on the path is."""
    nullable = False
    for name in fields:
        if isinstance(t, Nullable):
            nullable, t = True, t.inner
        if not isinstance(t, RecordOf):
            return None
        t = dict(t.fields)[name]
    return Nullable(t) if nullable and not isinstance(t, Nullable) else t
