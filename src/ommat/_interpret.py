# pyright: reportPrivateUsage=false
"""Interpret: from the board decide opened, ask the oracle bindings and fire the rules, round
by round over the turn's sorter, until the fixpoint or the round bound.

Round 0 is a rule pass. Each round from 1 is a source phase over the held
sources, each asked or given its own markers, then a rule phase. A held
source is ready when its needs are determined, a source requirement's ref counting as one,
its source requirement holds and its function ``when`` is true, tested in that order and in
source id order before the live frontier's calls start. The live frontier's calls run under one
``asyncio.TaskGroup``, each task holding its call's timeout and the binding's await alone; every
classification and append happens after the group closes, call items by first source id, then the
source entries by source id, then each rule wavefront sorted by id, then the ``round`` item. A batch
is the ready sources of one live frontier naming one oracle, called through that oracle's binding; a
source with no oracle is a batch of one through ``default``.

Three ends share one path: at the round bound after the rule phase, and at the budget or the
deadline in the source phase with no rule phase first, the pending sources are ``Unasked`` and
the sorter is drained. The budget is tested before a frontier is asked, then the
deadline, which is tested again when the frontier's task group closes; under replay the recorded cut
stands in for the clock.

Two boards: the live one the views and the needs read holds an ``Answer`` or an
``ExtractionAnswer`` for a ``q:`` ref and Python ``bytes`` for a ``bytes`` value; the ledger is
handed plain data.
"""

import asyncio
import contextvars
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, cast

from ommat import _form
from ommat._answer import (
    EXTRACTION_MEMBERS,
    Answer,
    AnswerKind,
    ExtractionAnswer,
    ValueType,
    answer_document,
    extraction_document,
    off_sum,
    span_bound,
)
from ommat._binding import CallResult, OracleBinding, RenderedQuestion, SourceCall
from ommat._canonical import decode, encode, hash_value, ordered, with_order
from ommat._component import (
    Called,
    at,
    base,
    call,
    exception_record,
    names_cause_ref,
    names_read,
    now,
)
from ommat._design import LoadedDesign, Question, Rule, Source, Table
from ommat._ledger import Ledger
from ommat._markers import Cause, CauseCode, Failed, Unasked, Unavailable, Undetermined, Unsure
from ommat._provenance import (
    DETAILS,
    HASHED_DETAILS,
    REASONS,
    precision_conforms,
    string_conforms,
    versions,
    versions_conform,
)
from ommat._read_view import (
    ComponentKind,
    ReadView,
    RequireStopped,
    Role,
    UndeclaredRead,
    _Stop,
)
from ommat._refs import code_point_sorted, declared_ref, grounded_ref
from ommat._sorter import TurnSorter
from ommat._table import evaluate, read_inputs
from ommat._typed import is_a, typed
from ommat._types import Type, b64url_encode
from ommat._values import (
    MAX_DOCUMENT_DEPTH,
    MAX_INTEGER,
    domain_error,
    guarded_repr,
    is_scalar_string,
    string_data,
)

_STATE = "t:state"
_DEFAULT = "default"

_C = CauseCode
_STRUCTURAL = frozenset({_C.raised, _C.timed_out, _C.invalid_return, _C.unexpected_ref})
_BINDING_MARKERS = frozenset(
    {
        _C.refused,
        _C.invalid_answer,
        _C.invalid_return,
        _C.unsupported,
        _C.timed_out,  # the binding's provider request ran out of the call's time
        _C.raised,  # the binding's provider request failed another way
    }
)
MAX_WARNINGS = 2**8
"""The most lines a call's ``warnings`` holds."""
MAX_WARNING_CODE_POINTS = 2**16
"""The most code points a call's ``warnings`` holds in all."""
_MARKER_KINDS: dict[type[Undetermined], str] = {
    Unsure: "Unsure",
    Unavailable: "Unavailable",
    Unasked: "Unasked",
}

Written = Undetermined | tuple[object, object]
"""What a ref gets: a marker, or a value in its record form and its live form."""


class _InterpretError(RuntimeError):
    """An engine defect interpret can see, which a caller cannot cause."""


@dataclass(slots=True)
class _WhenText:
    """What a function ``when`` leaves for ``unhashed``, under its source's lowest-ref entry: its
    raise, a raising truth test's included, and an undeclared read the design does not name."""

    error: Exception | None
    unnamed: str | None


async def interpret(
    ledger: Ledger,
    loaded: LoadedDesign,
    state: object,
    bindings: Mapping[str, OracleBinding],
    *,
    round_bound: int,
    timeout: float,  # noqa: ASYNC109 - the run setting; each call holds its own timeout
    budget: float | None,
    deadline: float | None,
    clock: float | None,
    cut: tuple[int, bool] | None,
    carried: frozenset[str],
) -> dict[str, object]:
    """Run the turn's rounds over the ledger decide opened, which holds the ``design`` and
    ``settings`` items and the state entry, and return the live board when interpret stops: every
    ref on it, as a value in its live form or a marker.

    ``state`` is the state entry's value as decide wrote it, in its live form. ``clock`` is
    the running loop's reading at decide's entry, from which the deadline runs, or ``None``
    under replay's internal entry, which reads no clock and passes no deadline. ``cut`` is
    then the recorded cut, the round the deadline cut and whether that round was asked, or
    ``None`` for a record with none. ``carried`` holds the ids of
    the carried providers, empty in this engine. Run settings, bindings and the
    carried set are decide's to check. A cancellation propagates: one
    delivered at an await with nothing of the round it interrupted appended, and a
    ``CancelledError`` a function ``when``, a rule or a method of a returned value raises with
    every item appended before it.
    """
    if carried:
        raise _InterpretError("carried providers are not supported in this engine")
    if deadline is not None and clock is None:
        raise _InterpretError("a deadline runs from decide's clock reading")
    if cut is not None and clock is not None:
        raise _InterpretError("the cut stands in for the clock, never beside it")
    end = None if deadline is None or clock is None else clock + deadline
    limits = _Limits(None if budget is None else _exact(budget), end, cut)
    return await _Turn(ledger, loaded, state, bindings, round_bound, timeout, carried, limits).run()


@dataclass(frozen=True, slots=True)
class _Limits:
    """What ends asking besides the round bound: the budget, exactly; the deadline's
    end on the running loop's clock; or under replay the recorded cut."""

    budget: Fraction | None
    end: float | None
    cut: tuple[int, bool] | None


# ---------------------------------------------------------------------------------------------
# One call, inside its task


@dataclass(slots=True)
class _Returned:
    """What a call's task saw: the wall-clock start, the latency, and the binding's return,
    its own raise, and whether the timeout expired."""

    start: str
    latency: float
    result: object
    error: Exception | None
    expired: bool
    escaped: BaseException | None
    """A ``BaseException`` the binding raised that is neither an ``Exception`` nor a
    cancellation, carried out of the task to propagate bare."""
    by_deadline: bool = False
    """Whether the call's timeout was the deadline's end, so an expiry is the deadline's."""


def _clock() -> float:
    """The running loop's clock, the one ``asyncio``'s timeouts read."""
    return asyncio.get_running_loop().time()


def _exact(tag: float) -> Fraction:
    """A cost tag or the budget as the decimal RFC 8785 writes it, exactly, as the choice helpers
    take a probability."""
    return Fraction(repr(tag))


async def _call(
    binding: OracleBinding, batch: tuple[SourceCall, ...], seconds: float, end: float | None
) -> _Returned:
    """One oracle binding call under its timeout, cut to the deadline's ``end`` when there is
    one, both taken at the call's start. The binding's
    await is the task's whole protected region; a cancellation passes
    through."""
    loop = asyncio.get_running_loop()
    start, began = now(), loop.time()
    result: object = None
    error: Exception | None = None
    escaped: BaseException | None = None
    expires = began + seconds if end is None else min(began + seconds, end)
    by_deadline = expires == end
    if end is not None and began >= end:
        # No time left: the call is not made, and is one the deadline cuts.
        return _Returned(start, 0.0, None, None, True, None, by_deadline)
    # The time the call has: the timeout itself, or the time left to the deadline when that is
    # smaller, positive since the deadline has not passed.
    left = float(end - began) if by_deadline and end is not None else float(seconds)
    timeout = asyncio.timeout_at(expires)
    try:
        async with timeout:
            try:
                result = await binding(batch, seconds=left)
            except Exception as exc:  # the binding's own raise, a late one included
                error = exc
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # out of the group bare, not in a group
                escaped = exc
    except TimeoutError:  # the timeout's own, raised when it cancelled the binding
        pass
    # Expiry, not the raise, decides timed-out: a binding may swallow the cancellation and
    # return, or raise a TimeoutError of its own. The timeout fired, or the clock reads at or
    # past its end as the call returns or raises, so one that never yields is judged alike.
    closed = loop.time()
    expired = timeout.expired() or closed >= expires
    return _Returned(start, closed - began, result, error, expired, escaped, by_deadline)


# ---------------------------------------------------------------------------------------------
# The batch and how its call is judged


@dataclass(frozen=True, slots=True)
class _Expanded:
    """A source's refs in a round: ``refs`` in code-point order, a per-candidate question's
    grounded; ``markers``, a per-candidate question's need marker by declared ref when its set
    holds one; ``candidates``, each grounded ref's candidate."""

    refs: tuple[str, ...]
    markers: dict[str, Undetermined]
    candidates: dict[str, str]


@dataclass(slots=True)
class _Batch:
    key: str
    """The binding's key: the sources' oracle name, or ``default``."""
    sources: tuple[str, ...]
    refs: tuple[str, ...]
    """Every ref the batch provides, in code-point order."""
    questions: dict[str, Question]
    """Each asked ``q:`` ref's declared question."""
    labels: dict[str, tuple[str, ...]]
    """The labels each ``q:`` ref was passed with."""
    texts: dict[str, int]
    """Each extraction ``q:`` ref's text length in code points, which its spans index."""
    sets: frozenset[str]
    """The batch's ``t:`` refs declared candidate sets, whose members must be distinct."""
    call: tuple[SourceCall, ...]
    timeout: float
    cost: Fraction


@dataclass(slots=True)
class _Judged:
    """A call's outcome, as the fault scope sorts it."""

    failed: Cause | None
    written: dict[str, Written]
    provenance: dict[str, object]
    """The call item's hashed details and ``claimed``; none on a failed call."""
    usage: dict[str, float] | None = None
    attempts: int | None = None
    unexpected: list[str] = field(default_factory=list[str])
    invalid: dict[str, str] = field(default_factory=dict[str, str])
    sums: dict[str, float] = field(default_factory=dict[str, float])
    """Each answer's sum, rounded once, where it lies off 1 by more than the flag allows."""
    truncated: list[str] = field(default_factory=list[str])
    """The ``q:`` refs the binding reported cut short, in code-point order."""
    warnings: list[str] = field(default_factory=list[str])
    """The binding's warning lines, in its order."""
    request_id: str | None = None
    calibrated: bool | None = None
    missing: dict[str, str] | None = None
    """Each null detail's reason; never on a failed call."""
    clients: dict[str, str] | None = None


def _judge(batch: _Batch, returned: _Returned, types: Mapping[str, Type]) -> _Judged:
    judged = _Judged(None, {}, no_provenance())
    codes: list[CauseCode] = []  # in fault-scope order; the first decides
    given: dict[str, object] = {}
    result = returned.result
    if returned.expired:
        codes.append(_C.timed_out)
    elif returned.error is not None:
        codes.append(_C.raised)
    elif type(result) is Failed:
        cause = getattr(result, "cause", None)
        code = getattr(cause, "code", None) if type(cause) is Cause else None
        if type(code) is CauseCode and code in _STRUCTURAL:
            codes.append(code)
        else:
            codes.append(_C.invalid_return)
            # A code the engine built is shown by its string; anything else by its repr's
            # guarded form, so no caller code runs.
            shown = code.value if type(code) is CauseCode else code
            judged.invalid["code"] = guarded_repr(shown if type(cause) is Cause else cause)
    elif type(result) is CallResult:
        values = cast(object, getattr(result, "values", None))
        if type(values) is not dict:
            codes.append(_C.invalid_return)
        else:
            provides = frozenset(batch.refs)
            items = list(cast("dict[object, object]", values).items())
            keys = [string_data(cast(str, k)) for k, _ in items if is_a(k, str)]
            if len(set(keys)) != len(keys):  # two keys of one string data
                codes.append(_C.invalid_return)
                items = []
            for key, value in items:
                ref = string_data(cast(str, key)) if is_a(key, str) else None
                if ref is not None and ref in provides and ref not in given:
                    given[ref] = value
                else:
                    judged.unexpected.append(ref if ref is not None else guarded_repr(key))
            if judged.unexpected:
                codes.append(_C.unexpected_ref)
        if not _provenance(result, judged):
            codes.append(_C.invalid_return)
        judged.usage = _usage(cast(object, getattr(result, "usage", None)), judged.invalid)
        judged.attempts = _attempts(cast(object, getattr(result, "attempts", None)), judged.invalid)
        judged.truncated = _truncated(
            cast(object, getattr(result, "truncated", ())), batch, judged.invalid
        )
        judged.warnings = _warnings(cast(object, getattr(result, "warnings", ())), judged.invalid)
        _unhashed_details(result, judged)
        if not _claimed(result, judged):
            codes.append(_C.invalid_return)
    else:
        codes.append(_C.invalid_return)
    if codes:
        judged.failed = Cause(codes[0], list(batch.refs))
        judged.provenance = no_provenance()
        judged.missing = None
        marker = Unavailable(judged.failed)
        judged.written = dict.fromkeys(batch.refs, marker)
        return judged
    for ref in batch.refs:
        if ref not in given:
            judged.written[ref] = Unavailable(Cause(_C.not_returned, [ref]))
        else:
            written = _returned_value(ref, given[ref], batch, types)
            judged.written[ref] = written
            if type(written) is tuple and type(written[1]) is Answer:
                # The flag changes no outcome: the answer is written as returned.
                total = off_sum(written[1])
                if total is not None:
                    judged.sums[ref] = total
    return judged


def _returned_value(ref: str, value: object, batch: _Batch, types: Mapping[str, Type]) -> Written:
    """A returned value or marker as written at its ref."""
    if is_a(value, Undetermined):
        return _binding_marker(ref, value)
    if ref in batch.questions:
        q = batch.questions[ref]
        value_type = None if q.value_type is None else ValueType(q.value_type)
        return answer_written(
            ref,
            value,
            AnswerKind(q.kind),
            batch.labels[ref],
            value_type,
            q.is_list,
            batch.texts.get(ref, 0),
        )
    forms = typed(value, types[ref])
    if forms is None or (ref in batch.sets and not _distinct(forms[0])):
        return Unavailable(Cause(_C.invalid_return, [ref]))
    return forms


def _distinct(members: object) -> bool:
    """Whether a candidate set's record-form members are distinct by canonical bytes."""
    listed = cast("list[object]", members)
    return len({encode(m) for m in listed}) == len(listed)


def answer_written(
    ref: str,
    value: object,
    kind: AnswerKind,
    labels: tuple[str, ...],
    value_type: ValueType | None,
    is_list: bool,
    text_length: int,
) -> Written:
    """A ``q:`` ref's returned value, not a marker, as interpret writes it: its answer's document
    and the answer, or ``invalid-answer`` when the value is not in its kind's form. An extraction
    answer's spans are bounded by ``text_length``, its text's length in code points."""
    if kind is AnswerKind.extraction:
        assert value_type is not None
        return _extraction(ref, value, value_type, is_list, text_length)
    try:
        built = Answer(kind, labels, cast("dict[str, float]", value))
    except ValueError:
        return Unavailable(Cause(_C.invalid_answer, [ref]))
    return answer_document(built), built


def _extraction(
    ref: str, value: object, value_type: ValueType, is_list: bool, text_length: int
) -> Written:
    """An extraction answer built from exactly the five members the binding returned, its span
    within its text and its document within the value domain; else ``invalid-answer``."""
    invalid = Unavailable(Cause(_C.invalid_answer, [ref]))
    if type(value) is not dict:
        return invalid
    members: dict[str, object] = {}
    for key, member in cast("dict[object, object]", value).items():
        name = string_data(cast(str, key)) if is_a(key, str) else None
        if name is None or name in members:
            return invalid
        members[name] = member
    if set(members) != set(EXTRACTION_MEMBERS):
        return invalid
    try:
        built = ExtractionAnswer(value_type, is_list, **cast(Any, members))
    except ValueError:
        return invalid
    if span_bound(built) > text_length:
        return invalid
    document = extraction_document(built)
    if domain_error(document) is not None:  # the size bounds: load bounds no extraction answer
        return invalid
    return document, built


def _truncated(value: object, batch: _Batch, invalid: dict[str, str]) -> list[str]:
    """A ``truncated`` of exactly a tuple of distinct exact ``str``, each exactly a question
    ref of the batch, in code-point order; else nothing, with the value kept in
    ``invalid``."""
    if type(value) is tuple:
        members = cast("tuple[object, ...]", value)
        refs = [m for m in members if type(m) is str and domain_error(m) is None]
        distinct = len(refs) == len(members) and len(set(refs)) == len(refs)
        if distinct and all(m in batch.questions for m in refs):
            return code_point_sorted(refs)
    invalid["truncated"] = guarded_repr(cast(object, value))
    return []


def warnings_conform(value: object) -> bool:
    """Whether ``value`` is a conforming ``warnings``: exactly a tuple of exact ``str``, each
    holding a character that is not whitespace and in the value domain, at most
    ``MAX_WARNINGS`` of them holding at most ``MAX_WARNING_CODE_POINTS`` code points in all,
    counted before any is read."""
    if type(value) is not tuple:
        return False
    members = cast("tuple[object, ...]", value)
    if len(members) > MAX_WARNINGS or not all(type(m) is str for m in members):
        return False
    lines = cast("tuple[str, ...]", members)
    return sum(len(m) for m in lines) <= MAX_WARNING_CODE_POINTS and all(
        m and not m.isspace() and domain_error(m) is None for m in lines
    )


def no_provenance() -> dict[str, object]:
    """The provenance of a call that sent no request: each hashed detail null, ``claimed``
    empty."""
    return {**dict.fromkeys(HASHED_DETAILS), "claimed": []}


def _provenance(result: object, judged: _Judged) -> bool:
    """Write each conforming hashed detail into ``judged``; ``False`` when any does not
    conform, its ``repr`` kept in ``invalid`` by the detail's name."""
    conforms = True
    for name in HASHED_DETAILS:
        value = cast(object, getattr(result, name, None))
        if name == "precision":
            ok = precision_conforms(value)
        elif name == "runtime":
            ok = versions_conform(value)
        else:  # a string's length is tested before its code points are read
            ok = string_conforms(value)
        if not ok:
            judged.invalid[name] = guarded_repr(value)
            conforms = False
        elif name == "runtime":
            judged.provenance[name] = versions(value)
        elif name == "precision" and type(value) is dict:
            judged.provenance[name] = dict(cast("dict[str, str]", value))
        elif name == "precision" or value is None:
            judged.provenance[name] = value
        else:
            judged.provenance[name] = string_data(cast(str, value))
    return conforms


def _held(name: str, result: object) -> bool:
    """Whether the detail ``name`` holds a value in what the binding returned, whatever its
    form: a ``usage`` figure is held when ``usage`` is a ``dict`` with its key."""
    if name.startswith("usage."):
        usage = cast(object, getattr(result, "usage", None))
        key = name.removeprefix("usage.")
        # Compared key by key, each an exact str: no hash or __eq__ of the binding's key runs.
        keys = cast("dict[object, object]", usage) if type(usage) is dict else {}
        return any(type(k) is str and k == key for k in keys)
    return cast(object, getattr(result, name, None)) is not None


def _unhashed_details(result: object, judged: _Judged) -> None:
    """Write ``request_id``, ``calibrated``, ``clients`` and ``missing`` when each conforms,
    else keep its ``repr`` in ``invalid``; never a fault. ``missing`` conforms when it is null
    or a ``dict`` from a detail's name to a reason, for details left null; an empty one is
    written nowhere, as null."""
    request_id = cast(object, getattr(result, "request_id", None))
    if string_conforms(request_id, exact=True):
        judged.request_id = cast("str | None", request_id)
    else:
        judged.invalid["request_id"] = guarded_repr(request_id)
    calibrated = cast(object, getattr(result, "calibrated", None))
    if calibrated is None or type(calibrated) is bool:
        judged.calibrated = calibrated
    else:
        judged.invalid["calibrated"] = guarded_repr(calibrated)
    clients = cast(object, getattr(result, "clients", None))
    if versions_conform(clients, exact=True):
        judged.clients = versions(clients)
    else:
        judged.invalid["clients"] = guarded_repr(clients)
    missing = cast(object, getattr(result, "missing", None))
    if missing is None:
        return
    if type(missing) is dict:
        pairs = list(cast("dict[object, object]", missing).items())
        if not pairs:
            return  # an empty missing is not written
        if all(
            type(k) is str and k in DETAILS and type(v) is str and v in REASONS for k, v in pairs
        ) and not any(_held(cast(str, k), result) for k, _ in pairs):
            judged.missing = {cast(str, k): cast(str, v) for k, v in pairs}
            return
    judged.invalid["missing"] = guarded_repr(cast(object, missing))


def _claimed(result: object, judged: _Judged) -> bool:
    """Write a conforming ``claimed`` sorted: exactly a tuple of distinct exact ``str``, each
    the name of a detail that holds a value; ``False`` when it does not conform."""
    value = cast(object, getattr(result, "claimed", ()))
    if type(value) is tuple:
        names = cast("tuple[object, ...]", value)
        distinct = all(type(n) is str for n in names) and len(set(names)) == len(names)
        if distinct and all(n in DETAILS and _held(cast(str, n), result) for n in names):
            judged.provenance["claimed"] = code_point_sorted(cast("tuple[str, ...]", names))
            return True
    judged.invalid["claimed"] = guarded_repr(cast(object, value))
    return False


def _warnings(value: object, invalid: dict[str, str]) -> list[str]:
    """A conforming ``warnings`` as a list in the binding's order; else nothing, with the value
    kept in ``invalid``. Never a fault."""
    if warnings_conform(value):
        return list(cast("tuple[str, ...]", value))
    invalid["warnings"] = guarded_repr(value)
    return []


def _binding_marker(ref: str, value: object) -> Unavailable:
    """One of the six markers a binding may return, naming ``ref`` alone; any other marker is
    ``invalid-return``. A ``timed-out`` or ``raised`` one is the binding's provider request,
    told from the engine's by its call's ``returned`` outcome."""
    if type(value) is Unavailable:
        cause = getattr(value, "cause", None)
        code = getattr(cause, "code", None) if type(cause) is Cause else None
        if type(code) is CauseCode and code in _BINDING_MARKERS:
            return Unavailable(Cause(code, [ref]))
    return Unavailable(Cause(_C.invalid_return, [ref]))


def _finite_number(value: object) -> bool:
    if type(value) is int:
        return abs(value) <= MAX_INTEGER
    return type(value) is float and math.isfinite(value)


def _usage(value: object, invalid: dict[str, str]) -> dict[str, float] | None:
    """A ``usage`` of the right type, else null with the value kept in ``invalid``."""
    if value is None:
        return None
    # In the domain, size bounds included, each key's length tested before its code points.
    if type(value) is dict and domain_error(cast(object, value)) is None:
        out: dict[str, float] = {}
        for key, number in cast("dict[object, object]", value).items():
            if type(key) is not str or not _finite_number(number):
                break
            out[key] = cast(float, number)
        else:
            return out
    invalid["usage"] = guarded_repr(cast(object, value))
    return None


def _attempts(value: object, invalid: dict[str, str]) -> int | None:
    if value is None:
        return None
    if type(value) is int and abs(value) <= MAX_INTEGER:
        return value
    invalid["attempts"] = guarded_repr(value)
    return None


def _rule_marker(
    value: object, loaded: LoadedDesign, board: Mapping[str, object]
) -> Undetermined | None:
    """A returned marker that is exactly one of the three, holding a well-formed ``Cause``
    whose refs each take a form a cause's refs take, a grounded ref only when its entry is on
    the board, rebuilt; else ``None``."""
    kind = cast("type[Undetermined]", type(value))
    if kind not in _MARKER_KINDS:
        return None
    cause = getattr(value, "cause", None)
    if type(cause) is not Cause:
        return None
    code = getattr(cause, "code", None)
    refs = cast(object, getattr(cause, "refs", None))
    if type(code) is not CauseCode or type(refs) is not tuple:
        return None
    members = cast("tuple[object, ...]", refs)
    if not all(type(r) is str and is_scalar_string(r) for r in members):
        return None
    # A ref of no declared form could carry the input's text; a grounded ref is a board ref
    # (§marker-causes) only when the turn wrote its entry, so no rule names a made-up candidate.
    if not all(names_cause_ref(loaded, cast(str, r)) or cast(str, r) in board for r in members):
        return None
    return kind(Cause(code, cast("tuple[str, ...]", members)))


def _marker_document(marker: Undetermined) -> dict[str, object]:
    cause = marker.cause
    return {
        "kind": _MARKER_KINDS[type(marker)],
        "cause": {"code": cause.code.value, "refs": list(cause.refs)},
    }


# ---------------------------------------------------------------------------------------------
# The board


def _fresh(value: object) -> object:
    """A copy of a board value no change reaches back from."""
    if type(value) is dict:
        return {k: _fresh(v) for k, v in cast("dict[str, object]", value).items()}
    if type(value) is list:
        return [_fresh(v) for v in cast("list[object]", value)]
    if type(value) is tuple:
        return tuple(_fresh(v) for v in cast("tuple[object, ...]", value))
    if type(value) is Answer:
        return Answer(value.kind, value.labels, dict(value.probabilities))
    return value


def _render_wording(template: object, values: Mapping[str, object]) -> Any:
    """A wording rendered: a template string by ``_render``; a structured one into fresh
    containers, each string value rendered and each key and other value as written."""
    if isinstance(template, str):
        return _render(template, values)
    if type(template) is dict:
        members = cast("dict[str, object]", template).items()
        return {k: _render_wording(v, values) for k, v in members}
    if type(template) is list:
        return [_render_wording(v, values) for v in cast("list[object]", template)]
    return template


def _render(template: str, values: Mapping[str, object]) -> str:
    """A wording rendered in one left-to-right pass: each escape as its brace and each
    placeholder as its slot's value rendered, never read for braces. Load
    check 9 has checked the template, so every placeholder names a slot."""
    out: list[str] = []
    i, n = 0, len(template)
    while i < n:
        pair = template[i : i + 2]
        if pair in ("{{", "}}"):
            out.append(pair[0])
            i += 2
        elif pair[0] == "{":
            close = template.index("}", i)
            out.append(_text(values[template[i + 1 : close]]))
            i = close + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


SORTED_TEXT: contextvars.ContextVar[bool] = contextvars.ContextVar("sorted_text", default=False)
"""Set by replay over a record before format 6, whose engine wrote slot text in canonical key
order, so its rerun renders as the turn did."""


def _text(value: object) -> str:
    """A slot value by the declared-text closed rule: a string as itself, a list as its
    members' renderings joined by ``, ``, any other value as its ordered encoding, each object in
    the order it holds, a ``bytes`` value as its record form's."""
    if type(value) is str:
        return value
    if type(value) in (list, tuple):
        return ", ".join(_text(member) for member in cast("Sequence[object]", value))
    if SORTED_TEXT.get():
        return encode(_record_form(value)).decode("utf-8")
    return ordered(_record_form(value))


def _record_form(value: object, *, bytes_form: Callable[[bytes], object] = b64url_encode) -> object:
    """A live board or leaf value in the record's form: each ``bytes`` as unpadded base64url,
    or as ``bytes_form`` gives it, and each answer as its document, at every depth, a tuple as
    a list. A slot's value is never an answer: load refuses a ``q:`` slot; a need's may be,
    which ``from_record`` compares and ``json_value`` writes with ``bytes_form`` refusing."""
    if type(value) is bytes:
        return bytes_form(value)
    if type(value) is Answer:
        return answer_document(value)
    if type(value) is ExtractionAnswer:
        return extraction_document(value)
    if type(value) is dict:
        return {
            k: _record_form(v, bytes_form=bytes_form)
            for k, v in cast("dict[str, object]", value).items()
        }
    if type(value) in (list, tuple):
        return [_record_form(v, bytes_form=bytes_form) for v in cast("Sequence[object]", value)]
    return value


# ---------------------------------------------------------------------------------------------
# The turn


class _Turn:
    def __init__(
        self,
        ledger: Ledger,
        loaded: LoadedDesign,
        state: object,
        bindings: Mapping[str, OracleBinding],
        round_bound: int,
        timeout: float,
        carried: frozenset[str],
        limits: _Limits,
    ) -> None:
        self.ledger = ledger
        self.loaded = loaded
        self.sources: Mapping[str, Source] = loaded._design.sources
        self.rules: Mapping[str, Rule | Table] = loaded._design.rules
        self.bindings = bindings
        self.round_bound = round_bound
        self.timeout = timeout
        self.carried = carried
        self.limits = limits
        self.board: dict[str, object] = {_STATE: state}
        self.sorter = TurnSorter(loaded._graph, carried)
        self.ran: set[str] = set()
        self.left_out: dict[str, Undetermined] = {}
        """Each candidate-valued question left out of its batch, with its own marker."""
        self.spent = Fraction(0)
        self.document = cast("dict[str, Any]", decode(loaded.document))["sources"]

    async def run(self) -> dict[str, object]:
        """Run the rounds and return the live board, which compile reads."""
        held = self.rule_phase(0)
        n = 0
        while held:  # the fixpoint: no source held after a rule phase
            n += 1
            cost, cut = await self.source_phase(held, n)
            if cut is not None:  # the budget or the deadline ended the turn's asking
                held = []
            else:
                held = self.rule_phase(n)
                if held and n >= self.round_bound:
                    self.end(Unasked(Cause(_C.round_bound)), held, n)
                    held = []
            # The exact sum, rounded once to the nearest float.
            rounded = float(cost)
            cut_by_deadline = cut is _C.deadline
            self.ledger.append(
                {"item": "round", "round": n, "cost": rounded, "deadline": cut_by_deadline}
            )
        if self.sorter.is_active():  # every node is done when interpret ends
            raise _InterpretError("interpret ended with a node not done")
        return self.board

    # -- the rule phase ----------------------------------------------------------------------

    def rule_phase(self, n: int) -> list[str]:
        """Fire wavefronts until one holds no rule; return the sources the wavefronts held."""
        held: list[str] = []
        while True:
            wave = self.sorter.get_ready()
            rules = code_point_sorted(node for node in wave if node in self.rules)
            held += [node for node in wave if node not in self.rules]
            if not rules:
                return held
            for rid in rules:
                self.fire(rid, n)
                self.sorter.done(rid)

    def fire(self, rid: str, n: int) -> None:
        """Call a rule on its read view and write its entry by the read view's order of outcomes:
        the first undeclared read, the first ``require`` stop, a raise, an invalid return,
        the returned value."""
        r = self.rules[rid]
        if isinstance(r, Table):
            self.fire_table(rid, r, n)
            return
        leaves = {path: self.loaded._leaf_value(path) for path in r.params}
        view = ReadView(
            Role(ComponentKind.rule, rid),
            r.reads,
            r.params,
            self.board,
            leaves,
            grounded=self.loaded._grounded,
        )
        called = call(self.loaded._callables[rid], view)
        stop, error, returned = called.stop, called.error, called.returned
        unnamed: str | None = None
        written: Written
        if isinstance(stop, UndeclaredRead):
            if names_read(self.loaded, stop):
                written = Unavailable(Cause(_C.undeclared_read, [rid, stop.read]))
            else:
                written = Unavailable(Cause(_C.undeclared_read, [rid]))
                unnamed = stop.read
        elif isinstance(stop, RequireStopped):
            written = type(stop.marker)(Cause(_C.required, [stop.ref]))
        elif error is not None:
            written = Unavailable(Cause(_C.raised, [rid]))
        elif called.awaitable:  # tested before any other test of the return
            written = Unavailable(Cause(_C.invalid_return, [rid]))
        else:
            # Never a truth test on what a rule returned: a marker's `__bool__` raises.
            if is_a(returned, Undetermined):
                value: Written | None = _rule_marker(returned, self.loaded, self.board)
            else:
                value = typed(returned, self.loaded._types[r.provides])
                if value is not None and r.candidates and not _distinct(value[0]):
                    value = None  # a candidate set's members are distinct, never merged
            written = Unavailable(Cause(_C.invalid_return, [rid])) if value is None else value
        index = self.write(r.provides, n, None, written, called.derivation)
        if called.text is not None:  # whatever the outcome, a stop's included
            self.ledger.write_unhashed("exceptions", exception_record(called.text), index)
        if unnamed is not None:
            self.ledger.write_unhashed("reads", unnamed, index)

    def fire_table(self, rid: str, r: Table, n: int) -> None:
        """Evaluate a rule table: its inputs read and its helpers called in input order, on a
        read view over its reads and leaves, then its rows; write its entry with ``rows``. A
        helper that raises, or a comparison of incomparable values, is ``Unavailable`` with
        ``raised``, no row tested."""
        table = self.loaded._tables[rid]
        reads, params = self.loaded._reads[rid], self.loaded._params[rid]
        leaves = {path: self.loaded._leaf_value(path) for path in params}
        view = ReadView(
            Role(ComponentKind.rule, rid),
            reads,
            params,
            self.board,
            leaves,
            grounded=self.loaded._grounded,
        )
        inputs = cast("list[dict[str, Any]]", table["inputs"])
        rows: list[int] = []
        raised: Exception | None = None
        written: Written
        try:
            values = read_inputs(inputs, view, leaves)
            outcome = evaluate(
                table["rows"], r.hit, table["order"], [i["read"] for i in inputs], values, leaves
            )
        except Exception as exc:  # a helper's raise, or an incomparable comparison
            raised = exc
            written = Unavailable(Cause(_C.raised, [rid]))
        else:
            rows = outcome.rows
            if isinstance(outcome.value, Undetermined):  # the engine's own marker
                written = outcome.value
            else:
                value = typed(outcome.value, self.loaded._types[r.provides])
                if value is not None and r.candidates and not _distinct(value[0]):
                    value = None  # a candidate set's members are distinct, never merged
                written = Unavailable(Cause(_C.invalid_return, [rid])) if value is None else value
        finally:
            view.close()
        index = self.write(r.provides, n, None, written, rows=rows)
        if raised is not None:
            self.ledger.write_unhashed("exceptions", exception_record(raised), index)

    # -- the source phase --------------------------------------------------------------------

    async def source_phase(self, held: list[str], n: int) -> tuple[Fraction, CauseCode | None]:
        """Ask or mark each held source and write the round's call items and source entries;
        then mark each held source done, or, when the budget or the deadline cuts the round,
        end it. Return the round's cost, exactly, and the cut's
        code, or ``None``."""
        order = code_point_sorted(held)
        marked: dict[str, Undetermined] = {}
        texts: dict[str, _WhenText] = {}
        ready: list[str] = []
        left: dict[str, dict[str, Undetermined]] = {}
        labels: dict[str, tuple[tuple[str, ...], tuple[str | None, ...]]] = {}
        # Each source's refs this round, a per-candidate question's grounded over its set.
        expanded = {sid: self.expanded(sid) for sid in order}
        # Readiness in source id order, before the TaskGroup, so a function `when` that
        # raises a cancellation falls at a fixed place. A source that
        # is not ready gets its own markers before the budget and the deadline are tested.
        for sid in order:
            if not expanded[sid].refs:
                continue  # nothing to ask over an empty set: not called, its `when` not either
            marker = self.need_marker(sid)
            if marker is None:
                marker = self.requirement_marker(sid)
            if marker is None:
                marker = self.when_marker(sid, texts)
            if marker is not None:
                marked[sid] = marker
                continue
            out = self.label_markers(sid, labels)
            out.update(expanded[sid].markers)
            if out:
                left[sid] = out
                self.left_out.update(out)
            # A source with no ref left to ask is not called, and is outside the frontier.
            if len(out) < len(expanded[sid].refs):
                ready.append(sid)
        batches = self.batches(ready, left, labels, expanded)
        cut = self.cut_before(batches, n)
        if cut is not None:
            batches = []  # the frontier is not asked, whole
        returned = await self.ask(batches)
        if cut is None and self.cut_after(batches, returned, n):
            cut = _C.deadline
        calls: dict[str, tuple[int, _Judged]] = {}
        cost = Fraction(0)
        for batch, outcome in zip(batches, returned, strict=True):
            judged = _judge(batch, outcome, self.loaded._types)
            index = self.ledger.append(self.call_item(batch, judged, n))
            unhashed: dict[str, object] = {
                "start": outcome.start,
                "latency": outcome.latency,
                "usage": judged.usage,
                "attempts": judged.attempts,
            }
            if judged.unexpected:
                unhashed["unexpected"] = judged.unexpected
            if judged.invalid:
                unhashed["invalid"] = judged.invalid
            if judged.sums:
                unhashed["sums"] = judged.sums
            if judged.truncated:
                unhashed["truncated"] = judged.truncated
            if judged.warnings:
                unhashed["warnings"] = judged.warnings
            for name in ("request_id", "calibrated", "missing", "clients"):
                if getattr(judged, name) is not None:
                    unhashed[name] = getattr(judged, name)
            self.ledger.write_unhashed("calls", unhashed, index)
            if outcome.error is not None:
                self.ledger.write_unhashed("exceptions", exception_record(outcome.error), index)
            for sid in batch.sources:
                calls[sid] = (index, judged)
            cost += batch.cost
        for sid in order:
            provided = expanded[sid].refs
            whole = len(left.get(sid, ())) == len(provided)
            if sid not in marked and sid not in calls and not whole:
                continue  # in a frontier the cut left unasked: the end marks it
            for i, ref in enumerate(provided):
                if sid in marked:
                    index = self.write(ref, n, None, marked[sid])
                    if i == 0 and sid in texts:  # keyed under the lowest ref
                        self.write_when_text(texts[sid], index)
                elif ref in left.get(sid, {}):
                    self.write(ref, n, None, left[sid][ref])
                else:
                    index, judged = calls[sid]
                    self.write(ref, n, index, judged.written[ref])
            self.ran.add(sid)
        self.spent += cost  # the budget's running total, kept exact
        if cut is None:
            for sid in order:
                self.sorter.done(sid)
        else:
            self.end(Unasked(Cause(cut)), order, n)
        return cost, cut

    def cut_before(self, batches: list[_Batch], n: int) -> CauseCode | None:
        """Whether the round is cut before its frontier is asked: by the budget when the cost
        spent plus the frontier's exceeds it, exactly; then by the deadline once the
        clock reads at least its end, or under replay at a cut round
        that was not asked."""
        budget, end, cut = self.limits.budget, self.limits.end, self.limits.cut
        if budget is not None:
            frontier = sum((b.cost for b in batches), Fraction(0))
            if self.spent + frontier > budget:
                return _C.over_budget
        if end is not None and _clock() >= end:
            return _C.deadline
        if cut == (n, False):
            return _C.deadline
        return None

    def cut_after(self, batches: list[_Batch], returned: list[_Returned], n: int) -> bool:
        """Whether the deadline has passed when the frontier's task group closes, or under
        replay this is the cut round and it was asked. A call the deadline's end expired counts as
        passed: asyncio fires a timeout up to its clock resolution early, so the clock may still
        read under the end."""
        end = self.limits.end
        expired = any(r.expired and r.by_deadline for r in returned)
        if end is not None and batches and (_clock() >= end or expired):
            return True
        return self.limits.cut == (n, True)

    def need_marker(self, sid: str) -> Undetermined | None:
        """A source's own marker when a need is undetermined, by the ref grammar's precedence,
        naming every need of the winning kind, a source requirement's ref counting as one; ``None``
        when every one is determined."""
        needs = set(self.loaded._needs[sid])
        requirement = self.sources[sid].requirement
        if requirement is not None:
            needs.add(requirement.ref)
        bases = {need: self.board[base(need)] for need in needs}
        unavailable = [need for need in needs if isinstance(bases[need], Unavailable)]
        if unavailable:
            # spec: Source requirements
            return Unavailable(Cause(_C.need_unavailable, code_point_sorted(unavailable)))
        undetermined = [need for need in needs if isinstance(bases[need], Undetermined)]
        if undetermined:
            # spec: Source requirements
            return Unasked(Cause(_C.need_undetermined, code_point_sorted(undetermined)))
        return None

    def requirement_marker(self, sid: str) -> Undetermined | None:
        """``Unasked`` with ``when-false`` naming the ref when the source requirement's
        determined entry equals none of its values by canonical bytes, never by ``==``; ``None``
        when it holds or there is none."""
        requirement = self.sources[sid].requirement
        if requirement is None:
            return None
        # Check 9 types the ref with no bytes at any depth, so its live form is a domain value
        # the encoding takes; the values are load's own bytes, never the caller's objects.
        if encode(self.board[requirement.ref]) in self.loaded._requirement_bytes[sid]:
            return None
        # spec: Source requirements
        return Unasked(Cause(_C.when_false, [requirement.ref]))

    def when_marker(self, sid: str, texts: dict[str, _WhenText]) -> Undetermined | None:
        """Call a function ``when`` on its read view and give the source's marker by
        the read view's order of outcomes: the first undeclared read, a raise, an awaitable
        returned, a truth test that raises, then false; ``None`` when it is true. Its text
        goes in ``texts``."""
        source = self.sources[sid]
        if source.when is None:
            return None
        state = self.loaded._types[_STATE]
        view = ReadView(
            Role(ComponentKind.when, sid), source.needs, (), self.board, {}, state=state
        )
        called: Called = call(self.loaded._callables[f"{sid}.when"], view)
        stop, error = called.stop, called.text
        unnamed: str | None = None
        marker: Undetermined | None
        if isinstance(stop, UndeclaredRead):
            if names_read(self.loaded, stop):
                marker = Unavailable(Cause(_C.undeclared_read, [sid, stop.read]))
            else:
                marker = Unavailable(Cause(_C.undeclared_read, [sid]))
                unnamed = stop.read
        elif isinstance(stop, RequireStopped):
            # Every need is determined and the state holds no marker.
            raise _InterpretError(f"require stopped the when of {sid!r} on a determined read")
        elif called.error is not None or called.awaitable:  # no truth test on an awaitable
            marker = Unavailable(Cause(_C.when_failed, [sid]))
        else:
            try:  # inside the protected region
                marker = None if bool(called.returned) else Unasked(Cause(_C.when_false, [sid]))
            except _Stop:  # a kept stop raised again: read from the record, no text
                marker = Unavailable(Cause(_C.when_failed, [sid]))
            except Exception as exc:
                error = exc
                marker = Unavailable(Cause(_C.when_failed, [sid]))
        if error is not None or unnamed is not None:
            texts[sid] = _WhenText(error, unnamed)
        return marker

    def write_when_text(self, text: _WhenText, index: int) -> None:
        if text.error is not None:
            self.ledger.write_unhashed("exceptions", exception_record(text.error), index)
        if text.unnamed is not None:
            self.ledger.write_unhashed("reads", text.unnamed, index)

    def batches(
        self,
        ready: list[str],
        left: Mapping[str, Mapping[str, Undetermined]],
        labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]],
        expanded: Mapping[str, "_Expanded"],
    ) -> list[_Batch]:
        """The live frontier's batches: the ready sources naming one oracle form one, each
        source with no oracle one of its own, in order of their first source id. ``ready`` is in
        source id order."""
        groups: list[list[str]] = []
        keyed: dict[str, list[str]] = {}
        for sid in ready:
            key = self.sources[sid].oracle
            if key is None:
                groups.append([sid])
            elif key in keyed:
                keyed[key].append(sid)
            else:
                keyed[key] = [sid]
                groups.append(keyed[key])
        return [self.batch(tuple(group), left, labels, expanded) for group in groups]

    def expanded(self, sid: str) -> "_Expanded":
        """A source's refs this round, in code-point order: each per-candidate question
        grounded over its set's candidates, or its declared ref alone when the set is off the
        board or holds a marker, with that marker naming the set; nothing over an empty set."""
        refs: list[str] = []
        markers: dict[str, Undetermined] = {}
        candidates: dict[str, str] = {}
        for ref in self.sources[sid].provides:
            set_ref = self.loaded._grounded.get(ref)
            if set_ref is None or set_ref not in self.board:
                refs.append(ref)
                continue
            value = self.board[set_ref]
            # Never a truth test on the set's value: a marker's `__bool__` raises.
            if is_a(value, Unavailable):
                markers[ref] = Unavailable(Cause(_C.need_unavailable, [set_ref]))
                refs.append(ref)
            elif is_a(value, Undetermined):
                markers[ref] = Unasked(Cause(_C.need_undetermined, [set_ref]))
                refs.append(ref)
            else:
                for candidate in cast("Sequence[str]", value):
                    grounded = grounded_ref(ref, candidate)
                    candidates[grounded] = candidate
                    refs.append(grounded)
        return _Expanded(tuple(code_point_sorted(refs)), markers, candidates)

    def label_markers(
        self, sid: str, labels: dict[str, tuple[tuple[str, ...], tuple[str | None, ...]]]
    ) -> dict[str, Undetermined]:
        """Each of a ready source's candidate-valued questions left out of its batch, with its
        marker, the first that applies: a marker on the set, an empty set, labels beyond
        check 10's bounds, a candidate equal to a fixed name; each names the set's ref. A
        question asked has its rendered labels put in ``labels``."""
        out: dict[str, Undetermined] = {}
        for ref in code_point_sorted(self.sources[sid].provides):
            set_ref = self.loaded._label_sets.get(ref)
            if set_ref is None:
                continue
            value = self.board[set_ref]
            # Never a truth test on the set's value: a marker's `__bool__` raises.
            if is_a(value, Unavailable):
                out[ref] = Unavailable(Cause(_C.need_unavailable, [set_ref]))
                continue
            if is_a(value, Undetermined):
                out[ref] = Unasked(Cause(_C.need_undetermined, [set_ref]))
                continue
            candidates = tuple(cast("Sequence[str]", value))
            names, descriptions = self.loaded._labels[ref]
            if not candidates:
                out[ref] = Unasked(Cause(_C.when_false, [set_ref]))
            elif len(names) + len(candidates) > _form.MAX_LABELS or (
                sum(map(len, names))
                + sum(len(d) for d in descriptions if d is not None)
                + sum(map(len, candidates))
                > _form.MAX_LABEL_CODE_POINTS
            ):
                out[ref] = Unavailable(Cause(_C.label_overflow, [set_ref]))
            elif not frozenset(names).isdisjoint(candidates):
                out[ref] = Unavailable(Cause(_C.label_collision, [set_ref]))
            else:
                # The candidates go after the first ``at`` fixed labels, each undescribed.
                k = self.loaded._label_at[ref]
                none = (None,) * len(candidates)
                labels[ref] = (
                    names[:k] + candidates + names[k:],
                    descriptions[:k] + none + descriptions[k:],
                )
        return out

    def batch(
        self,
        sids: tuple[str, ...],
        left: Mapping[str, Mapping[str, Undetermined]],
        labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]],
        expanded: Mapping[str, "_Expanded"],
    ) -> _Batch:
        refs: list[str] = []
        questions: dict[str, Question] = {}
        passed_labels: dict[str, tuple[str, ...]] = {}
        texts: dict[str, int] = {}
        sets: set[str] = set()
        calls: list[SourceCall] = []
        for sid in sids:
            source = self.sources[sid]
            # A question left out of the batch is neither passed nor among its refs.
            grounding = expanded[sid]
            provided = [r for r in grounding.refs if r not in left.get(sid, {})]
            sets.update(r for r in provided if r in self.loaded._candidate_sets)
            # A fresh copy per source, so no source's change reaches another's.
            needs = {
                need: _fresh(at(self.board, need))
                for need in code_point_sorted(self.loaded._needs[sid])
            }
            rendered: dict[str, RenderedQuestion] = {}
            for ref in provided:
                name = declared_ref(ref)  # a per-candidate entry's question is its declared ref's
                q = source.provides[name]
                if isinstance(q, Question):
                    kind = AnswerKind(q.kind)
                    # A structured wording renders from the document, each object in the order
                    # the document holds, so a design and its document ask alike.
                    written = self.document[sid]["provides"][name]["wording"]
                    declared = with_order(written["template"], written.get("order", {}))
                    values = self.slot_values(q, needs, grounding.candidates.get(ref))
                    wording = _render_wording(declared, values)
                    if kind is AnswerKind.extraction:
                        questions[ref], passed_labels[ref] = q, ()
                        rendered[ref] = self.extraction(ref, q, wording, needs, texts)
                        continue
                    # The names; each description beside its name, aligned; a candidate-valued
                    # set's rendered from the set's entry.
                    passed, descriptions = labels.get(name) or self.loaded._labels[name]
                    questions[ref], passed_labels[ref] = q, passed
                    rendered[ref] = RenderedQuestion(
                        kind, wording, passed, descriptions=descriptions
                    )
            t_refs = tuple(ref for ref in provided if ref not in rendered)
            calls.append(SourceCall(sid, self.loaded._payload(sid), needs, t_refs, rendered))
            refs += provided
        declared = [s.timeout for s in (self.sources[sid] for sid in sids) if s.timeout is not None]
        key = self.sources[sids[0]].oracle
        return _Batch(
            key=_DEFAULT if key is None else key,
            sources=sids,
            refs=tuple(code_point_sorted(refs)),
            questions=questions,
            labels=passed_labels,
            texts=texts,
            sets=frozenset(sets),
            call=tuple(calls),
            # The largest declared, else the run setting's.
            timeout=max(declared) if declared else self.timeout,
            cost=sum((_exact(self.sources[sid].cost) for sid in sids), Fraction(0)),
        )

    def extraction(
        self,
        ref: str,
        q: Question,
        wording: str,
        needs: Mapping[str, object],
        texts: dict[str, int],
    ) -> RenderedQuestion:
        """An extraction question as asked: the needs key of its text, whose length its spans
        are bounded by, and its reference date rendered; load added both refs to the needs and
        typed them exactly ``string``."""
        text = cast("Mapping[str, str]", q.text)["ref"]
        texts[ref] = len(cast(str, needs[text]))
        date = None if q.reference_date is None else _text(needs[q.reference_date["ref"]])
        return RenderedQuestion(
            AnswerKind.extraction,
            wording,
            (),
            value_type=ValueType(q.value_type),
            is_list=q.is_list,
            text=text,
            reference_date=date,
        )

    def slot_values(
        self, q: Question, needs: Mapping[str, object], candidate: str | None = None
    ) -> dict[str, object]:
        """Each slot's value by name: a board ref's from the source's needs, where load added
        it, a parameter's from the parameter block, and a candidate slot's the candidate the
        entry is asked about."""
        values: dict[str, object] = {}
        for slot in q.wording.slots:
            tag, origin = next(iter(slot["origin"].items()))
            if tag == "ref":
                values[slot["name"]] = needs[origin]
            elif tag == "param":
                values[slot["name"]] = self.loaded._leaf_value(origin)
            elif candidate is not None:  # a candidate slot of a per-candidate entry
                values[slot["name"]] = candidate
            else:  # pragma: no cover - load holds a candidate slot to a grounded question
                raise _InterpretError(f"a {tag} slot reached interpret with no candidate")
        return values

    def identity(self, ref: str) -> str:
        """The question identity: the hash of the question's design-document object, each
        parameter origin with its leaf's value in record form, no other slot's value. The document
        is left as it was."""
        declared = cast("dict[str, Any]", self.document[self.loaded._providers[ref]])
        question = cast("dict[str, Any]", declared["provides"][ref])
        wording = cast("dict[str, Any]", question["wording"])
        slots: list[dict[str, Any]] = []
        for slot in cast("list[dict[str, Any]]", wording["slots"]):
            origin = cast("dict[str, Any]", slot["origin"])
            if "param" in origin:
                path = cast(str, origin["param"])
                origin = {"param": path, "value": decode(self.loaded._leaf_bytes[path])}
            slots.append({**slot, "origin": origin})
        # The question's object is a holder of design members, held to no size bound; its
        # targets guide a reader and are not what was asked.
        hashed = {k: v for k, v in question.items() if k not in ("target", "latent_targets")}
        return hash_value({**hashed, "wording": {**wording, "slots": slots}}, MAX_DOCUMENT_DEPTH)

    async def ask(self, batches: list[_Batch]) -> list[_Returned]:
        """Run the live frontier's calls concurrently, one task per batch; a task that ended
        cancelled on its own is re-raised here, before the round's first append."""
        if not batches:
            return []
        end = self.limits.end
        # No timeout on the group: each task holds its own, cut to the deadline.
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(_call(self.bindings[b.key], b.call, b.timeout, end))
                for b in batches
            ]
        for task in tasks:  # in batch order, before the round's first append
            if task.cancelled():
                task.result()  # raises the task's CancelledError
            escaped = task.result().escaped
            if escaped is not None:
                raise escaped
        return [task.result() for task in tasks]

    def call_item(self, batch: _Batch, judged: _Judged, n: int) -> dict[str, object]:
        questions: dict[str, object] = {}
        # A per-candidate question's entries share one entry, under its declared ref.
        for ref in code_point_sorted({declared_ref(r) for r in batch.questions}):
            # The kind and labels, or value type and list flag, are those the identity hashes.
            declared = self.document[self.loaded._providers[ref]]["provides"][ref]
            asked: dict[str, object] = {
                "identity": self.identity(ref),
                "kind": declared["kind"],
                "labels": declared["labels"],
            }
            if declared["kind"] == AnswerKind.extraction:
                asked.update(value_type=declared["value_type"], is_list=declared["is_list"])
            questions[ref] = asked
        failed = judged.failed
        outcome: object = (
            "returned"
            if failed is None
            else {"failed": {"code": failed.code.value, "refs": list(failed.refs)}}
        )
        return {
            "item": "call",
            "round": n,
            "sources": list(batch.sources),
            "outcome": outcome,
            **judged.provenance,
            "questions": questions,
        }

    # -- the end and the drain ---------------------------------------------------------------

    def end(self, marker: Unasked, held: list[str], n: int) -> None:
        """End the turn's asking, as the round bound, the budget and the deadline each do:
        mark every pending source ``Unasked`` with ``marker`` by source id, mark the held
        sources done and drain the sorter."""
        asked = self.ran | self.carried
        pending = [s for s in code_point_sorted(self.sources) if s not in asked]
        # Every expansion before any write: a pending set's provider marked first must not
        # turn a grounded source's cut marker into the set's need marker.
        grounding = {sid: self.expanded(sid) for sid in pending}
        for sid in pending:
            for ref in grounding[sid].refs:
                # A question left out of its batch keeps its own marker under the cut.
                kept = self.left_out.get(ref, grounding[sid].markers.get(ref, marker))
                self.write(ref, n, None, kept)
        for sid in held:
            self.sorter.done(sid)
        self.drain(n)

    def drain(self, n: int) -> None:
        """``get_ready()`` until ``is_active()`` is false: a source is done with no write, a
        rule fires and is done. An empty wavefront while one is active is a returned node
        never done, which would spin."""
        while self.sorter.is_active():
            wave = self.sorter.get_ready()
            if not wave:
                raise _InterpretError("the drain found no node ready while a node is active")
            for node in code_point_sorted(wave):
                if node in self.rules:
                    self.fire(node, n)
                self.sorter.done(node)

    # -- the write ---------------------------------------------------------------------------

    def write(
        self,
        ref: str,
        n: int,
        call: int | None,
        written: Written,
        derivation: list[dict[str, list[str] | str]] | None = None,
        *,
        rows: list[int] | None = None,
    ) -> int:
        item: dict[str, object] = {"item": "entry", "ref": ref, "round": n, "call": call}
        if isinstance(written, Undetermined):  # the engine's own marker
            item["marker"] = _marker_document(written)
            live: object = written
        else:
            item["value"], live = written
        if derivation is not None:
            item["derivation"] = derivation
        if rows is not None:
            item["rows"] = rows
        index = self.ledger.append(item)
        self.board[ref] = live
        return index
