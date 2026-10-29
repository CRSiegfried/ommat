# pyright: reportPrivateUsage=false
"""The deciding inputs of an entry: each declared read of the rule providing it, classed by
whether some do-and-diff intervention on the read changes the entry (§do-and-diff).

Each intervention is one of forms 1 to 4 run through ``_edit``'s one procedure, so its checks,
its copy and its replay are those of ``set_entry``, ``fail_call``, ``set_state_field`` and
``constant_rule``. A ``q:`` or ``t:`` read is tried through its entry, a grounded declared
ref's through each of its entries: form 1 at each marker a binding may return and the record's
format holds, a ``q:`` answer at ``Unavailable`` with ``refused`` only, and a ``t:`` entry at
each value of a finite type; then form 2 at each structural code, once per call, when every
entry the call wrote is of the read's declared ref. A ``t:state`` field is tried by form 3 at each
value of a finite type, and a ``d:`` read by form 4 at ``Unsure`` and each value of a finite
type. A finite type is ``boolean``, a label, or either nullable. A trial the form refuses for
its target, ``null-call``, ``failed-call``, ``not-found`` or ``candidate-rule``, reaches
nothing, and so does one whose replay asks outside the record, ``not-in-record``,
``ref-not-in-record`` or ``question-changed``, whatever else it finds; any other refusal
propagates. A reached trial changes the entry when replay gives an ``entry`` finding at its
ref whose value, or marker kind and cause code, differ; ``rows``, ``derivation``, ``round`` or
a cause's refs alone are no change.

A read is ``deciding`` when a reached trial changes the entry; else ``not-determined`` when no
trial reached, as for a ``candidate`` read, which no form reaches; else ``markers-only`` when
some target is not of a finite type; else ``not-deciding``. Each trial is one intervention, so
``not-deciding`` says no single one changes the entry; joint effects are not explored.

The record is checked as the edits check it, then ``ref`` refused ``not-found`` unless it names
an entry of the record a rule provides, before any replay. The output is plain data.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from ommat._canonical import encode
from ommat._design import LoadedDesign
from ommat._interpret import _BINDING_MARKERS, _STRUCTURAL
from ommat._ledger import _ITEM_DEPTH, Record, held_marker_code
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
_Trial = tuple[list[_Edit], _Constant | None]

_UNREACHED = frozenset({_R.null_call, _R.failed_call, _R.not_found, _R.candidate_rule})
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
        trials, exhaustive = _trials(r, log, loaded)
        out[r] = await _classed(ref, trials, exhaustive, read, log, loaded, functions)
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
    if any(i["item"] == "entry" and i["ref"] == ref for i in log):
        provider = loaded._providers.get(_declared(cast(str, ref)))
        if provider is not None and provider in loaded._design.rules:
            return provider
    _not_found("the record has no entry a rule provides at ref", ref, "ref")


async def _classed(
    ref: str,
    trials: list[_Trial],
    exhaustive: bool,
    read: Record,
    log: _Log,
    loaded: LoadedDesign,
    functions: _Functions,
) -> str:
    """``deciding`` when a reached trial changes the entry at ``ref``; ``not-determined`` when
    no trial reached; else by whether the trials were exhaustive."""
    reached = False
    for edits, constant in trials:
        try:
            _, _, found = await _applied(read, log, loaded, edits, constant, functions)
        except EvaluationRefusal as exc:
            if exc.reason not in _UNREACHED:
                raise
            continue
        if any(f["kind"] in _OUTSIDE for f in found):
            continue  # answered outside the record: no finding of it counts
        reached = True
        if any(f["kind"] is _K.entry and f["ref"] == ref and _changed(f) for f in found):
            return "deciding"
    if not reached:
        return "not-determined"
    return "not-deciding" if exhaustive else "markers-only"


def _trials(r: str, log: _Log, loaded: LoadedDesign) -> tuple[list[_Trial], bool]:
    """The interventions on read ``r``, each as ``_applied``'s edits and rule, and whether they
    try every value of each target. A read with no trial is ``not-determined`` whatever the
    flag."""
    if is_candidate_read(r):
        return [], True
    parsed = parse_ref(r)
    declared = f"{parsed.level}:{parsed.name}"
    if declared == _STATE:
        values = _values(_field_type(loaded._types[_STATE], parsed.fields)) or []
        return [([_SetField(r, v)], None) for v in values], True
    if parsed.level == "d":
        values = _values(loaded._types[declared])
        outputs = [Unsure, *(values or ())]
        return [([], _Constant(declared, o)) for o in outputs], values is not None
    fmt = log[0]["format"]
    codes = ["refused"] if parsed.level == "q" else sorted(_BINDING_MARKERS)
    markers = [Unavailable(Cause(c)) for c in codes if held_marker_code(c, fmt)]
    values = None if parsed.level == "q" else _values(loaded._types[declared])
    trials: list[_Trial] = []
    calls: dict[int, None] = {}  # each sole writer once, in log order, after every entry's
    for entry in log:
        if entry["item"] != "entry" or _declared(entry["ref"]) != declared:
            continue
        trials.extend(([_SetEntry(entry["ref"], v)], None) for v in [*markers, *(values or ())])
        call = entry["call"]
        if call is not None and call not in calls and _sole_writer(call, declared, log):
            calls[call] = None
    for call in calls:
        source = cast(str, log[call]["sources"][0])
        trials.extend(([_FailCall(source, c.value)], None) for c in sorted(_STRUCTURAL))
    return trials, values is not None


def _changed(finding: dict[str, Any]) -> bool:
    """Whether an ``entry`` finding's two sides differ in value, or in marker kind and code: a
    change of ``rows``, ``derivation``, ``round`` or a cause's refs alone is none."""
    recorded, replayed = finding["recorded"], finding["replayed"]
    return recorded is None or replayed is None or _outcome(recorded) != _outcome(replayed)


def _outcome(entry: dict[str, Any]) -> bytes:
    """An entry's value, or its marker's kind and cause code, as canonical bytes."""
    if "marker" in entry:
        marker = entry["marker"]
        return encode(["marker", marker["kind"], marker["cause"]["code"]])
    return encode(["value", entry["value"]], _ITEM_DEPTH)


def _sole_writer(call: int, declared: str, log: _Log) -> bool:
    """Whether every entry the call at ``call`` wrote is of the declared ref ``declared``, as a
    grounded batch's are."""
    return all(
        _declared(e["ref"]) == declared for e in log if e["item"] == "entry" and e["call"] == call
    )


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


def _field_type(t: Type, fields: tuple[str, ...]) -> Type:
    """The declared type of a state field through ``fields``, its definitions resolved; a null
    record on the path holds no field, which ``set_state_field`` refuses ``not-found``."""
    for name in fields:
        if isinstance(t, Nullable):
            t = t.inner
        t = dict(cast(RecordOf, t).fields)[name]  # load checked the path
    return t
