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
is the ready sources of one live frontier naming one oracle, called through that oracle's binding;
the sources that name none are one batch through ``default``.

Three ends share one path: at the round bound after the rule phase, and at the budget or the
deadline in the source phase with no rule phase first, the pending sources are ``Unasked`` and
the sorter is drained: the drain fires every unfired rule, the policy among them. The budget is
tested before a frontier is asked, then the deadline, which is tested again when the frontier's
task group closes; under replay the recorded cut stands in for the clock.

Two boards: the live one the views and the needs read holds an ``Answer`` or an
``ExtractionAnswer`` for a ``q:`` ref and Python ``bytes`` for a ``bytes`` value; the ledger is
handed plain data.
"""

import asyncio
import dataclasses
import enum
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
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
from ommat._binding import CallResult, OracleBinding, RenderedQuestion, Replaying, SourceCall
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
    usage_conforms,
    versions,
    versions_conform,
)
from ommat._read_view import (
    ComponentKind,
    Current,
    ReadView,
    RequireStopped,
    Role,
    UndeclaredRead,
    _Stop,
    is_candidate_read,
)
from ommat._refs import (
    candidate_values,
    code_point_sorted,
    declared_ref,
    grounded_ref,
    parse_ref,
)
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
    reuse: "Reuse | None" = None,
    replaying: bool = False,
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
    every item appended before it. ``reuse``, which only the evaluation jobs pass, is one
    record's reference run: the first run it is given keeps it, and a later one fires a rule
    from it where ``Reuse`` says the call would give the same. ``replaying``, which only the
    evaluation jobs pass, has a source's call kept and taken from ``reuse`` too; a call so
    taken writes the kept call's ``unhashed`` entry, so they pass it over a ledger that keeps
    none.
    """
    if carried:
        raise _InterpretError("carried providers are not supported in this engine")
    if deadline is not None and clock is None:
        raise _InterpretError("a deadline runs from decide's clock reading")
    if cut is not None and clock is not None:
        raise _InterpretError("the cut stands in for the clock, never beside it")
    end = None if deadline is None or clock is None else clock + deadline
    limits = _Limits(None if budget is None else _exact(budget), end, cut, clock)
    turn = _Turn(
        ledger, loaded, state, bindings, round_bound, timeout, carried, limits, reuse, replaying
    )
    # An answer's off sum is written to the call's ``unhashed`` entry alone, so a ledger that
    # keeps no ``unhashed`` write is spared it.
    turn.sums = not getattr(ledger, "drops_unhashed", False)
    return await turn.run()


@dataclass(frozen=True, slots=True)
class _Limits:
    """What ends asking besides the round bound: the budget, exactly; the deadline's
    end on the running loop's clock; or under replay the recorded cut."""

    budget: Fraction | None
    end: float | None
    cut: tuple[int, bool] | None
    clock: float | None = None
    """Decide's clock reading, or ``None`` under replay's internal entry."""


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
    sets: dict[str, str | None]
    """The batch's ``t:`` refs declared candidate sets, whose candidate values must be
    distinct, each with its candidate field."""
    call: tuple[SourceCall, ...]
    timeout: float
    cost: Fraction


@dataclass(frozen=True, slots=True)
class _Outcome:
    """One call of a rule or table: what it writes, its ``derivation`` or ``rows``, the
    exception or stop text for ``exceptions`` and an undeclared read for ``reads``."""

    written: "Written"
    derivation: list[dict[str, list[str] | str]] | None
    rows: list[int] | None
    exception: BaseException | None
    unnamed: str | None


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
    missing: dict[str, str] | None = None
    """Each null detail's reason; never on a failed call."""
    clients: dict[str, str] | None = None


def _judge(
    batch: _Batch, returned: _Returned, types: Mapping[str, Type], *, sums: bool = True
) -> _Judged:
    """A call's outcome, as the fault scope sorts it; each answer's off sum in ``sums`` only
    when ``sums``, as nothing but the call's ``unhashed`` entry reads it."""
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
            if sums and type(written) is tuple and type(written[1]) is Answer:
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
    forms = typed(value, types[declared_ref(ref)])  # a grounded t: ref's, its declared ref's
    if forms is None or (ref in batch.sets and not _distinct(forms[0], batch.sets[ref])):
        return Unavailable(Cause(_C.invalid_return, [ref]))
    return forms


def _distinct(members: object, field: str | None) -> bool:
    """Whether a candidate set's record-form members' candidate values are distinct by
    canonical bytes."""
    values = candidate_values(members, field)
    return len({encode(v) for v in values}) == len(values)


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
    """Write ``request_id``, ``clients`` and ``missing`` when each conforms,
    else keep its ``repr`` in ``invalid``; never a fault. ``missing`` conforms when it is null
    or a ``dict`` from a detail's name to a reason, for details left null; an empty one is
    written nowhere, as null."""
    request_id = cast(object, getattr(result, "request_id", None))
    if string_conforms(request_id, exact=True):
        judged.request_id = cast("str | None", request_id)
    else:
        judged.invalid["request_id"] = guarded_repr(request_id)
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


def _usage(value: object, invalid: dict[str, str]) -> dict[str, float] | None:
    """A ``usage`` copied when it conforms (``usage_conforms``), else null with the value kept
    in ``invalid``."""
    if value is None:
        return None
    if type(value) is dict and usage_conforms(cast("dict[object, object]", value)):
        return dict(cast("dict[str, float]", value))
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


def _on_board(board: Mapping[str, object], ref: str) -> bool:
    """Whether a read's entry, or its grounded declared ref's marker, is on the board."""
    p = parse_ref(ref)
    declared = f"{p.level}:{p.name}"
    return declared in board or grounded_ref(declared, cast(str, p.candidate)) in board


def _slot_origins(q: Question) -> list[tuple[str, str]]:
    """Each slot origin of a question's wording, as its tag and its value."""
    return [cast("tuple[str, str]", next(iter(slot["origin"].items()))) for slot in q.wording.slots]


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
        return _answer_copy(value)
    return value


def _answer_copy(answer: Answer) -> Answer:
    """``Answer(answer.kind, answer.labels, dict(answer.probabilities))``, the fields already
    checked and in their held forms, so none is checked again."""
    copy = object.__new__(Answer)
    object.__setattr__(copy, "kind", answer.kind)
    object.__setattr__(copy, "labels", answer.labels)
    object.__setattr__(copy, "probabilities", dict(answer.probabilities))
    return copy


def slot_value(loaded: LoadedDesign, ref: str, value: object) -> object:
    """A board-ref slot's value as declared text renders it: a candidate set of records as its
    candidate values, so a record candidate renders as its candidate field; any other as read."""
    field = loaded._candidate_sets.get(ref)
    if field is None or type(value) is not list:
        return value
    members: list[object] = cast("list[object]", value)
    try:
        return candidate_values(members, field)
    except (KeyError, TypeError):  # a member of another form: the set's own fault
        return members


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


def _text(value: object) -> str:
    """A slot value by the declared-text closed rule: a string as itself, a list as its
    members' renderings joined by ``, ``, any other value as its ordered encoding, each object in
    the order it holds, a ``bytes`` value as its record form's."""
    if type(value) is str:
        return value
    if type(value) in (list, tuple):
        return ", ".join(_text(member) for member in cast("Sequence[object]", value))
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
        reuse: "Reuse | None" = None,
        replaying: bool = False,
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
        self.sums = True
        """Whether an answer's off sum is worked out, for the call's ``unhashed`` entry;
        interpret clears it for a ledger that keeps no ``unhashed`` write."""
        self.document = loaded._unvalued.sources
        self.reuse = reuse
        self.keeping = reuse is not None and reuse.reference is None  # the reference run
        self.changed = None if reuse is None or self.keeping else reuse.changed(loaded)
        # A source's call is kept and taken from ``reuse`` only where the caller says so, and
        # never on a clock: a kept return carries its call's latency and expiry.
        self.replaying = (
            replaying and reuse is not None and limits.end is None and limits.clock is None
        )
        if self.keeping:
            cast(Reuse, reuse).calls = {}  # a reference run that raised left only part
            cast(Reuse, reuse).asked = {}

    async def run(self) -> dict[str, object]:
        """Run the rounds and return the live board, which decide reads."""
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
        if self.keeping:
            cast(Reuse, self.reuse).kept(self.loaded, self.board)
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
        """Run a rule, or a grounded rule once per candidate, and write its entries."""
        r = self.rules[rid]
        if r.grounded is not None:
            self.fire_grounded(rid, r, n)
            return
        self.write_outcome(r.provides, n, self.outcome(rid, r, None))

    def fire_grounded(self, rid: str, r: Rule | Table, n: int) -> None:
        """Run a grounded rule or table once per candidate, in its set's order, each on a read
        view whose current candidate is that candidate, then write every entry in ref
        code-point order. Over a set holding a marker, the set's need marker at the declared
        ref, with no ``derivation`` or ``rows``; over an empty set, nothing."""
        set_ref = cast(str, r.grounded)
        value = self.board[set_ref]
        # Never a truth test on the set's value: a marker's `__bool__` raises.
        if is_a(value, Unavailable):
            self.write(r.provides, n, None, Unavailable(Cause(_C.need_unavailable, [set_ref])))
            return
        if is_a(value, Undetermined):
            self.write(r.provides, n, None, Unasked(Cause(_C.need_undetermined, [set_ref])))
            return
        members = cast("Sequence[object]", value)
        values = candidate_values(members, self.loaded._candidate_sets[set_ref])
        outcomes: dict[str, _Outcome] = {}
        for member, candidate in zip(members, values, strict=True):
            current = Current(set_ref, member, candidate)
            outcomes[grounded_ref(r.provides, candidate)] = self.outcome(rid, r, current)
        for ref in code_point_sorted(outcomes):
            self.write_outcome(ref, n, outcomes[ref])

    def outcome(self, rid: str, r: Rule | Table, current: Current | None) -> "_Outcome":
        """A rule's outcome: from the reference run's call where ``Reuse`` allows, a table's
        whole and a function's judged afresh on this board; else from calling it, which a
        reference run keeps."""
        key = (rid, None if current is None else current.value)
        held = self.held(key, r, current)
        if isinstance(r, Table):
            if held is not None:
                return _copied_outcome(cast(_Outcome, held))
            outcome = self.table_outcome(rid, r, current)
            if self.keeping:
                cast(Reuse, self.reuse).calls[key] = outcome
            return outcome
        if held is not None:
            kept = cast(Called, held)
            called = dataclasses.replace(kept, derivation=cast(Any, _fresh(kept.derivation)))
        else:
            called = self.rule_call(rid, r, current)
            if self.keeping:
                cast(Reuse, self.reuse).calls[key] = called
        return self.rule_outcome(rid, r, called)

    def held(
        self, key: tuple[str, str | None], r: Rule | Table, current: Current | None
    ) -> "Called | _Outcome | None":
        """The reference run's call of rule ``key[0]`` for this candidate, when calling it
        again would give the same: it declares no leaf whose value changed, a function rule's
        callable is the reference's own, and every entry its read view reaches is the
        reference's."""
        reuse, changed = self.reuse, self.changed
        if reuse is None or changed is None:
            return None
        found = reuse.calls.get(key)
        if found is None:
            return None
        rid = key[0]
        if not changed.isdisjoint(self.loaded._params[rid]):
            return None
        reference = cast(LoadedDesign, reuse.reference)
        if not isinstance(r, Table) and (
            self.loaded._callables[rid] is not reference._callables[rid]
        ):
            return None
        for ref in self.reached(rid, r, current):
            if not _same(self.board.get(ref, _ABSENT), reuse.board.get(ref, _ABSENT)):
                return None
        return found

    def reached(self, rid: str, r: Rule | Table, current: Current | None) -> Iterator[str]:
        """Each board ref a rule's read view can reach, as ``ReadView.read`` reads it: each
        declared read's entry or the marker at its declared ref; for a grounded read not on
        the board, its set and each candidate's entry, or over the rule's own set the current
        candidate's alone; and the rule's own set, whose member ``candidate`` reads."""
        grounded = self.loaded._grounded
        bases: set[str] = set()
        for read in self.loaded._reads[rid]:
            if not is_candidate_read(read):
                p = parse_ref(read)
                bases.add(f"{p.level}:{p.name}")
        bases |= {grounded[b] for b in bases if b in grounded}
        if r.grounded is not None:
            bases.add(r.grounded)
        for b in code_point_sorted(bases):
            yield b
            if b not in grounded or b in self.board:
                continue
            set_ref = grounded[b]
            if current is not None and set_ref == current.set_ref:
                yield grounded_ref(b, current.value)
                continue
            members = self.board.get(set_ref)
            if type(members) is list:
                field_ = self.loaded._candidate_sets.get(set_ref)
                for candidate in candidate_values(cast("list[object]", members), field_):
                    yield grounded_ref(b, candidate)

    def write_outcome(self, ref: str, n: int, outcome: "_Outcome") -> None:
        index = self.write(ref, n, None, outcome.written, outcome.derivation, rows=outcome.rows)
        if outcome.exception is not None:  # whatever the outcome, a stop's included
            self.ledger.write_unhashed("exceptions", exception_record(outcome.exception), index)
        if outcome.unnamed is not None:
            self.ledger.write_unhashed("reads", outcome.unnamed, index)

    def rule_call(self, rid: str, r: Rule, current: Current | None) -> Called:
        """Call a function rule on its read view."""
        leaves = {path: self.loaded._leaf_value(path) for path in r.params}
        view = ReadView(
            Role(ComponentKind.rule, rid),
            r.reads,
            r.params,
            self.board,
            leaves,
            grounded=self.loaded._grounded,
            sets=self.loaded._candidate_sets,
            current=current,
        )
        return call(self.loaded._callables[rid], view)

    def rule_outcome(self, rid: str, r: Rule, called: Called) -> "_Outcome":
        """Judge a function rule's call by the read view's order of outcomes: the first
        undeclared read, the first ``require`` stop, a raise, an invalid return, the returned
        value."""
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
                value = self.typed_value(r.provides, returned)
            written = Unavailable(Cause(_C.invalid_return, [rid])) if value is None else value
        return _Outcome(written, called.derivation, None, called.text, unnamed)

    def typed_value(self, provides: str, returned: object) -> Written | None:
        """A component's returned value as written, or ``None`` when its type does not hold
        it or, for a candidate set, its candidate values repeat."""
        value = typed(returned, self.loaded._types[provides])
        if (
            value is not None
            and provides in self.loaded._candidate_sets
            and not _distinct(value[0], self.loaded._candidate_sets[provides])
        ):
            return None  # a candidate set's members are distinct, never merged
        return value

    def table_outcome(self, rid: str, r: Table, current: Current | None) -> "_Outcome":
        """Evaluate a rule table: its inputs read and its helpers called in input order, on a
        read view over its reads and leaves, then its rows, giving its ``rows``. A helper
        that raises, or a comparison of incomparable values, is ``Unavailable`` with
        ``raised``, no row tested."""
        table = self.loaded._table(rid)
        reads, params = self.loaded._reads[rid], self.loaded._params[rid]
        leaves = {path: self.loaded._leaf_value(path) for path in params}
        view = ReadView(
            Role(ComponentKind.rule, rid),
            reads,
            params,
            self.board,
            leaves,
            grounded=self.loaded._grounded,
            sets=self.loaded._candidate_sets,
            current=current,
        )
        inputs = cast("list[dict[str, Any]]", table["inputs"])
        rows: list[int] = []
        raised: Exception | None = None
        written: Written
        try:
            values = read_inputs(inputs, view, leaves)
            outcome = evaluate(
                table["rows"],
                r.hit,
                table["order"],
                [i["read"] for i in inputs],
                values,
                leaves,
                fields=self.loaded._candidate_sets,
            )
        except Exception as exc:  # a helper's raise, or an incomparable comparison
            raised = exc
            written = Unavailable(Cause(_C.raised, [rid]))
        else:
            rows = outcome.rows
            if isinstance(outcome.value, Undetermined):  # the engine's own marker
                marker = outcome.value
                # A grounded table's cause names a candidate read as its set's ref, once.
                refs = view.cause_refs(marker.cause.refs)
                written = type(marker)(Cause(marker.cause.code, list(refs)))
            else:
                value = self.typed_value(r.provides, outcome.value)
                written = Unavailable(Cause(_C.invalid_return, [rid])) if value is None else value
        finally:
            view.close()
        return _Outcome(written, None, rows, raised, None)

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
            for ref, slot_marker in self.slot_markers(sid, expanded[sid]).items():
                out.setdefault(ref, slot_marker)
            if out:
                left[sid] = out
                self.left_out.update(out)
            # A source with no ref left to ask is not called, and is outside the frontier.
            if len(out) < len(expanded[sid].refs):
                ready.append(sid)
        batches, reused = self.frontier(ready, left, labels, expanded, n)
        cut = self.cut_before(batches, n)
        if cut is not None:
            batches, reused = [], {}  # the frontier is not asked, whole
        returned = await self.ask(batches, reused)
        if cut is None and self.cut_after(batches, returned, n):
            cut = _C.deadline
        calls: dict[str, tuple[int, _Judged]] = {}
        cost = Fraction(0)
        for i, (batch, outcome) in enumerate(zip(batches, returned, strict=True)):
            asked = reused.get(i)
            if asked is None:
                judged = _judge(batch, outcome, self.loaded._types, sums=self.sums)
                item = self.call_item(batch, judged, n)
            else:  # the reference run's call, its live values fresh
                judged = dataclasses.replace(
                    asked.judged,
                    written={r: _copied_written(w) for r, w in asked.judged.written.items()},
                )
                item = asked.item
            index = self.ledger.append(item)
            if self.keeps_calls():
                self.keep_call(batch, outcome, judged, item, n)
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
            for name in ("request_id", "missing", "clients"):
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

    def frontier(
        self,
        ready: list[str],
        left: Mapping[str, Mapping[str, Undetermined]],
        labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]],
        expanded: Mapping[str, "_Expanded"],
        n: int,
    ) -> tuple[list[_Batch], dict[int, "_Asked"]]:
        """The live frontier's batches: the ready sources naming one oracle form one, and those
        naming none one more under ``default``, in order of their first source id. ``ready`` is
        in source id order. A batch the reference run asked alike (``reusable``) is its batch,
        kept by index in the second member, and is not built."""
        groups: list[list[str]] = []
        keyed: dict[str, list[str]] = {}
        for sid in ready:
            oracle = self.sources[sid].oracle
            key = _DEFAULT if oracle is None else oracle  # load refuses ``default`` as a name
            if key in keyed:
                keyed[key].append(sid)
            else:
                keyed[key] = [sid]
                groups.append(keyed[key])
        batches: list[_Batch] = []
        reused: dict[int, _Asked] = {}
        for i, group in enumerate(groups):
            asked = self.reusable(tuple(group), n, left, labels, expanded)
            if asked is not None:
                reused[i] = asked
                batches.append(asked.batch)
            else:
                batches.append(self.batch(tuple(group), left, labels, expanded))
        return batches, reused

    # -- a source's call from the reference run ---------------------------------------------

    def keeps_calls(self) -> bool:
        """Whether this run is the reference run and keeps its sources' calls."""
        return self.keeping and self.replaying

    def reusable(
        self,
        sids: tuple[str, ...],
        n: int,
        left: Mapping[str, Mapping[str, Undetermined]],
        labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]],
        expanded: Mapping[str, "_Expanded"],
    ) -> "_Asked | None":
        """The reference run's call of exactly these sources in round ``n``, when asking them
        again would give the same: the run is ``replaying``, so has no clock, their binding
        answers by the batch and its construction (``Replaying``) and was built as the
        reference's was, no question of theirs declares a leaf whose value changed, and the
        batch ``batch`` would build passes what the kept one passed: each source's refs asked,
        each need's value as ``at`` reads it, by ``_same``, and each question's labels and their
        descriptions. A question's wording is rendered from these, its leaves and its
        candidate, which its ref names. The sources' readiness and their left-out refs' markers
        are judged afresh before this."""
        reuse, changed = self.reuse, self.changed
        if reuse is None or changed is None or not self.replaying:
            return None
        asked = reuse.asked.get(sids)
        if asked is None or asked.round != n:
            return None
        binding = self.bindings.get(asked.batch.key)
        if not isinstance(binding, Replaying) or binding.construction() != asked.construction:
            return None
        for sid, kept in zip(sids, asked.batch.call, strict=True):
            if not changed.isdisjoint(reuse.source_params(sid, self.sources[sid])):
                return None
            grounding = expanded[sid]
            out = left.get(sid, {})
            provided = [r for r in grounding.refs if r not in out]
            # A source provides t: refs or q: refs, never both, each in code-point order.
            if provided != [*kept.refs, *kept.questions]:
                return None
            named = set(self.loaded._needs[sid])
            for ref in provided:
                named.update(self.grounded_slots(sid, ref, grounding.candidates.get(ref)))
            if len(named) != len(kept.needs):
                return None
            for need in named:
                if need not in kept.needs or not _same(at(self.board, need), kept.needs[need]):
                    return None
            for ref, q in kept.questions.items():
                if q.kind is not AnswerKind.extraction:
                    name = declared_ref(ref)
                    found = labels.get(name) or self.loaded._labels[name]
                    if found != (q.labels, q.descriptions):
                        return None
        return asked

    def keep_call(
        self, batch: _Batch, returned: _Returned, judged: _Judged, item: dict[str, object], n: int
    ) -> None:
        """Keep the reference run's call of ``batch``, with what its binding noted of it and how
        the binding was built, when it answers by the batch and its construction."""
        binding = self.bindings[batch.key]
        if isinstance(binding, Replaying):
            effects = binding.effects(frozenset(batch.sources), frozenset(batch.refs))
            asked = _Asked(n, batch, returned, judged, item, binding.construction(), effects)
            cast(Reuse, self.reuse).asked[batch.sources] = asked

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
                for candidate in candidate_values(value, self.loaded._candidate_sets[set_ref]):
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
            if set_ref is None or set_ref not in self.board:  # off the board: the cut's marker
                continue
            value = self.board[set_ref]
            # Never a truth test on the set's value: a marker's `__bool__` raises.
            if is_a(value, Unavailable):
                out[ref] = Unavailable(Cause(_C.need_unavailable, [set_ref]))
                continue
            if is_a(value, Undetermined):
                out[ref] = Unasked(Cause(_C.need_undetermined, [set_ref]))
                continue
            candidates = tuple(candidate_values(value, self.loaded._candidate_sets[set_ref]))
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
        sets: dict[str, str | None] = {}
        calls: list[SourceCall] = []
        for sid in sids:
            source = self.sources[sid]
            # A question left out of the batch is neither passed nor among its refs.
            grounding = expanded[sid]
            provided = [r for r in grounding.refs if r not in left.get(sid, {})]
            sets.update(
                (r, self.loaded._candidate_sets[r])
                for r in provided
                if r in self.loaded._candidate_sets
            )
            # A fresh copy per source, so no source's change reaches another's; a slot on a
            # grounded declared ref is each asked candidate's need, under its grounded ref.
            named = set(self.loaded._needs[sid])
            for ref in provided:
                named.update(self.grounded_slots(sid, ref, grounding.candidates.get(ref)))
            needs = {need: _fresh(at(self.board, need)) for need in code_point_sorted(named)}
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
            sets=sets,
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
                value = needs[self.slot_key(origin, candidate)]
                values[slot["name"]] = slot_value(self.loaded, origin, value)
            elif tag == "param":
                values[slot["name"]] = self.loaded._leaf_value(origin)
            elif candidate is not None:  # a candidate slot of a per-candidate entry
                values[slot["name"]] = candidate
            else:  # pragma: no cover - load holds a candidate slot to a grounded question
                raise _InterpretError(f"a {tag} slot reached interpret with no candidate")
        return values

    def identity(self, ref: str) -> str:
        """The question identity of ``ref`` under the turn's design (``held_identity``)."""
        return held_identity(self.loaded, ref)

    def slot_key(self, origin: str, candidate: str | None) -> str:
        """A slot ref's needs key: a ref on a grounded declared ref, in a per-candidate
        question, as the candidate's grounded ref, field path kept; any other as spelt."""
        p = parse_ref(origin)
        declared = f"{p.level}:{p.name}"
        if candidate is None or declared not in self.loaded._grounded:
            return origin
        return grounded_ref(declared, candidate) + "".join(f".{f}" for f in p.fields)

    def grounded_slots(self, sid: str, ref: str, candidate: str | None) -> list[str]:
        """The needs keys of a per-candidate entry's slots on grounded declared refs."""
        if candidate is None:
            return []
        q = self.sources[sid].provides[declared_ref(ref)]
        if not isinstance(q, Question):
            return []
        keys = (self.slot_key(o, candidate) for tag, o in _slot_origins(q) if tag == "ref")
        return [k for k in keys if parse_ref(k).candidate is not None]

    def slot_markers(self, sid: str, grounding: "_Expanded") -> dict[str, Undetermined]:
        """Each per-candidate entry left out of its batch for a marker on a grounded slot:
        ``Unavailable`` with ``need-unavailable`` when one holds ``Unavailable``, else
        ``Unasked`` with ``need-undetermined``, naming each needs key of that kind once, in
        code-point order."""
        out: dict[str, Undetermined] = {}
        for ref in grounding.refs:
            keys = self.grounded_slots(sid, ref, grounding.candidates.get(ref))
            # A slot's entry off the board, its provider cut, leaves the cut's marker.
            found = {k: at(self.board, k) for k in keys if _on_board(self.board, k)}
            unavailable = [k for k, v in found.items() if is_a(v, Unavailable)]
            undetermined = [k for k, v in found.items() if is_a(v, Undetermined)]
            if unavailable:
                keyed = code_point_sorted(set(unavailable))
                out[ref] = Unavailable(Cause(_C.need_unavailable, keyed))
            elif undetermined:
                keyed = code_point_sorted(set(undetermined))
                out[ref] = Unasked(Cause(_C.need_undetermined, keyed))
        return out

    async def ask(
        self, batches: list[_Batch], reused: Mapping[int, "_Asked"] | None = None
    ) -> list[_Returned]:
        """Run the live frontier's calls concurrently, one task per batch; a task that ended
        cancelled on its own is re-raised here, before the round's first append. A batch in
        ``reused`` is not called: its task gives the binding back what it noted of the
        reference's call, in the batch's place among the calls, and returns that call's
        return."""
        if not batches:
            return []
        reused = reused or {}
        if len(reused) == len(batches):  # nothing to call
            for i, b in enumerate(batches):
                cast(Replaying, self.bindings[b.key]).restore(reused[i].effects)
            return [reused[i].returned for i in range(len(batches))]
        end = self.limits.end
        # No timeout on the group: each task holds its own, cut to the deadline.
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(
                    _restored(cast(Replaying, self.bindings[b.key]), reused[i])
                    if i in reused
                    else _call(self.bindings[b.key], b.call, b.timeout, end)
                )
                for i, b in enumerate(batches)
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
        # A question the source's asking would leave out keeps its own marker under the cut,
        # judged on the board as the cut finds it, before any cut marker is written.
        own: dict[str, Undetermined] = {}
        for sid in pending:
            own.update(grounding[sid].markers)
            if self.gated_at_cut(sid):
                continue  # the source's own markers come first, and the cut stands for them
            for ref, left in self.label_markers(sid, {}).items():
                own.setdefault(ref, left)
            for ref, slot_marker in self.slot_markers(sid, grounding[sid]).items():
                own.setdefault(ref, slot_marker)
        for sid in pending:
            for ref in grounding[sid].refs:
                kept = self.left_out.get(ref, own.get(ref, marker))
                self.write(ref, n, None, kept)
        for sid in held:
            self.sorter.done(sid)
        self.drain(n)

    def gated_at_cut(self, sid: str) -> bool:
        """Whether a source the cut finds unasked has a marker of its own on the board as the
        cut finds it: a need holding a marker, or a determined source requirement that does
        not hold. Its ``when`` is not run."""
        needs = set(self.loaded._needs[sid])
        requirement = self.sources[sid].requirement
        if requirement is not None:
            needs.add(requirement.ref)
        if any(is_a(self.board.get(base(n)), Undetermined) for n in needs):
            return True
        if requirement is None or requirement.ref not in self.board:
            return False
        return self.requirement_marker(sid) is not None

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


# ---------------------------------------------------------------------------------------------
# A reference run's calls, which an evaluation job's later runs of the same record reuse


class Reuse:
    """One record's reference run, kept by an evaluation job that reruns the record under
    several designs: each rule's call by rule id and candidate, a function rule's ``Called``
    and a table's outcome; each source call by its batch's sources, with its round, batch,
    return, judged outcome, call item, its binding's construction key and what the binding
    noted of it; and the live board the run ended with. The first run handed it keeps it. A
    later run under a design whose document is the reference's but for its leaves' values fires
    a rule from the reference's call, not calling it, when the rule declares no leaf whose value
    differs, a function rule's callable is the reference's own, and every entry its read view
    reaches is the reference's, by ``_same``: a rule's value being a function of what it reads,
    its leaves and its candidate (§rule-contract, §read-view), the run writes what calling it
    would. A function's call is judged afresh on the run's board.

    A source's readiness is judged afresh, its function ``when`` called as ever, and so are the
    markers of the refs left out of its batch. Where interpret is told it is ``replaying``, the
    run has no clock and the binding answers by the batch and its construction (``Replaying``,
    the fixture), a later run's batch of the very sources the reference called together in the
    same round, under a binding built alike, takes that call's outcome, not calling the
    binding: its entries, live values fresh, its call item, its cost charged again to the
    budget, and the stale refs and pinned answers the binding noted, when no question of the
    batch declares a leaf whose value differs and the batch passes what the kept one passed
    (``_Turn.reusable``). The round is the graph's, so the budget and the cut reach a source
    through it alone; a source now asked or left unasked, asked with other batchmates, or passed
    another need's value, label or ref is asked as now. A run under any other design keeps and
    reuses nothing."""

    def __init__(self) -> None:
        self.reference: LoadedDesign | None = None
        self.board: dict[str, object] = {}
        self.calls: dict[tuple[str, str | None], Called | _Outcome] = {}
        self.asked: dict[tuple[str, ...], _Asked] = {}
        self.params: dict[str, frozenset[str]] = {}

    def source_params(self, sid: str, source: Source) -> frozenset[str]:
        """The leaves a source declares, its questions' wording slots', kept per source id:
        a design differing from the reference's in leaf values alone declares the same."""
        found = self.params.get(sid)
        if found is None:
            found = self.params[sid] = frozenset(
                origin
                for q in source.provides.values()
                if isinstance(q, Question)
                for tag, origin in _slot_origins(q)
                if tag == "param"
            )
        return found

    def kept(self, loaded: LoadedDesign, board: dict[str, object]) -> None:
        """Keep a finished reference run under ``loaded``."""
        self.reference, self.board = loaded, board

    def changed(self, loaded: LoadedDesign) -> frozenset[str] | None:
        """The leaves whose values differ between ``loaded`` and the reference design, or
        ``None`` when the documents differ elsewhere, a leaf's type included."""
        reference = cast(LoadedDesign, self.reference)
        if (
            not _shares_all_but_leaves(loaded, reference)
            and loaded._unvalued.shape != reference._unvalued.shape
        ):
            return None
        mine, theirs = loaded._leaf_bytes, reference._leaf_bytes
        return frozenset(p for p in mine if mine[p] != theirs[p])


@dataclass(frozen=True, slots=True)
class _Asked:
    """A reference run's source call: its round, batch, return, judged outcome and call item,
    its binding's construction key, and what the binding noted of it."""

    round: int
    batch: _Batch
    returned: _Returned
    judged: _Judged
    item: dict[str, object]
    construction: object
    effects: object


async def _restored(binding: Replaying, asked: _Asked) -> _Returned:
    """A kept call in its batch's place among the round's calls: the binding notes what it
    noted of it again, and its return is given."""
    binding.restore(asked.effects)
    return asked.returned


def _copied_written(written: Written) -> Written:
    """A kept source entry's write with its live value fresh, so no run changes the
    reference's board; the record form is copied by the ledger."""
    if isinstance(written, Undetermined):
        return written
    return written[0], _fresh(written[1])


_LEAF_PARTS = frozenset({"_design", "_document", "_hash", "_grid", "_leaf_bytes"})
"""The parts of a loaded design ``with_parameters`` builds afresh when it loads nothing."""


def _held_alike(u: object, v: object) -> bool:
    """The same object, or a ``Design``'s snapshot of the same mapping: its every key holding
    the same object."""
    if u is v:
        return True
    if type(u) is not MappingProxyType or type(v) is not MappingProxyType:
        return False
    x, y = cast("Mapping[object, object]", u), cast("Mapping[object, object]", v)
    return x.keys() == y.keys() and all(x[k] is y[k] for k in x)


def _shares_all_but_leaves(a: LoadedDesign, b: LoadedDesign) -> bool:
    """Whether two loaded designs hold every part but their leaves' values as the same
    objects, as ``with_parameters`` builds one from the other when it loads nothing."""
    slots = (n for n in LoadedDesign.__slots__ if n not in _LEAF_PARTS)
    if any(getattr(a, n) is not getattr(b, n) for n in slots):
        return False
    x, y = a._design, b._design
    fields = (f.name for f in dataclasses.fields(x) if f.name != "parameters")
    if not all(_held_alike(getattr(x, n), getattr(y, n)) for n in fields):
        return False
    return x.parameters.keys() == y.parameters.keys() and all(
        x.parameters[p].type == y.parameters[p].type for p in x.parameters
    )


_ABSENT = object()
"""A ref not on a board."""
_SAME_FIELDS = (Answer, ExtractionAnswer, Cause, Unsure, Unavailable, Unasked)


def _same(a: object, b: object) -> bool:
    """Whether two live board values are alike in everything a rule can tell apart: exact
    types at every depth, a mapping's keys in order, a float's bits (every NaN alike), an
    answer's and a marker's every field. A value of any other type is the same only as
    itself."""
    if a is b:
        return True
    kind = type(a)
    if kind is not type(b):
        return False
    if kind is float:
        return cast(float, a).hex() == cast(float, b).hex()
    if kind in (str, int, bool, bytes) or isinstance(a, enum.Enum):
        return a == b
    if kind is dict:
        x, y = cast("dict[object, object]", a), cast("dict[object, object]", b)
        return (
            len(x) == len(y)
            and all(_same(i, j) for i, j in zip(x, y, strict=True))
            and all(_same(x[k], y[k]) for k in x)
        )
    if kind is list or kind is tuple:
        x2, y2 = cast("Sequence[object]", a), cast("Sequence[object]", b)
        return len(x2) == len(y2) and all(_same(i, j) for i, j in zip(x2, y2, strict=True))
    if kind in _SAME_FIELDS:
        return all(
            _same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(cast(Any, a))
        )
    return False


def _copied_outcome(outcome: "_Outcome") -> "_Outcome":
    """A kept table outcome with fresh containers, so no run changes the reference's."""
    written = outcome.written
    if not isinstance(written, Undetermined):
        written = (_fresh(written[0]), _fresh(written[1]))
    rows = None if outcome.rows is None else list(outcome.rows)
    return _Outcome(written, None, rows, outcome.exception, outcome.unnamed)


# The question identities a turn or an evaluation job reads and never changes, kept by design
# hash and ref, bounded, the oldest entry dropped first. A read is a plain ``dict.get``; a write
# takes the lock, so two threads evicting at once never race for the same oldest key.
_IDENTITIES: dict[tuple[str, str], str] = {}
_IDENTITIES_HELD = 1 << 16
_CACHE_LOCK = threading.Lock()


def _remember(cache: dict[Any, Any], key: Any, value: Any, held: int) -> None:
    """Keep ``value`` under ``key``, dropping the oldest entry past ``held``."""
    with _CACHE_LOCK:
        if key not in cache and len(cache) >= held:
            cache.pop(next(iter(cache)), None)
        cache[key] = value


def held_identity(loaded: LoadedDesign, ref: str) -> str:
    """``question_identity`` of declared ``q:`` ref ``ref``, computed once per design hash."""
    key = (loaded.design_hash, ref)
    found = _IDENTITIES.get(key)
    if found is None:
        found = question_identity(loaded, loaded._unvalued.sources, ref)
        _remember(_IDENTITIES, key, found, _IDENTITIES_HELD)
    return found


def question_identity(loaded: LoadedDesign, sources: Mapping[str, Any], ref: str) -> str:
    """The question identity of declared ``q:`` ref ``ref``: the hash of the question's
    design-document object, each parameter origin with its leaf's value in record form, no
    other slot's value, its targets left out. ``sources`` is the design document's ``sources``
    object, which is left as it was."""
    declared = cast("dict[str, Any]", sources[loaded._providers[ref]])
    question = cast("dict[str, Any]", declared["provides"][ref])
    wording = cast("dict[str, Any]", question["wording"])
    slots: list[dict[str, Any]] = []
    for slot in cast("list[dict[str, Any]]", wording["slots"]):
        origin = cast("dict[str, Any]", slot["origin"])
        if "param" in origin:
            path = cast(str, origin["param"])
            origin = {"param": path, "value": decode(loaded._leaf_bytes[path])}
        slots.append({**slot, "origin": origin})
    # The question's object is a holder of design members, held to no size bound; its targets
    # guide a reader and are not what was asked.
    hashed = {k: v for k, v in question.items() if k not in ("target", "latent_targets")}
    return hash_value({**hashed, "wording": {**wording, "slots": slots}}, MAX_DOCUMENT_DEPTH)
