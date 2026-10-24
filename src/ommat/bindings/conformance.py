"""The binding conformance suite: ``check`` holds an oracle binding to ommat's binding protocol.

``check(binding, *, batches=None, deterministic=False)`` calls the binding with each batch, as
interpret does, ``binding(batch, seconds=s)``, or with the suite's own when ``batches`` is
``None``: a source of each answer kind, one of them needing an image, and a ``t:`` source, each
as its own batch, then one question asked over three label sets as a candidate-valued question
is rendered, then all of the sources in one, built through the checked constructors and shaped as
interpret builds them. It returns the failures as a list of strings, each
``<check>: batch <n>: <what>`` but for ``seconds``, which names no batch, and an empty list
passes. It needs no test runner.

The checks, by name:

- ``seconds``: the binding takes ``seconds`` as a keyword, read from its signature before any
  call; a binding that does not is this one failure, and no batch is sent;
- ``returns``: the binding returns exactly a ``CallResult``, or exactly a ``Failed`` whose
  cause is a code a binding may fail a call with; it never raises;
- ``values``: ``values`` is a ``dict`` holding every ref the batch provides, its ``t:`` refs and
  its question keys, and no other key;
- ``markers``: each marker in ``values`` is ``Unavailable`` with a code a binding may return;
- ``answers``: each ``q:`` value that is not a marker is in its kind's form, as interpret's own
  test judges it, an extraction's spans bounded by its text in ``needs``;
- ``per-ref-cause``: a result with a ``timed-out`` or ``raised`` ref holds a ``warnings`` line
  when the binding sent a request, its ``attempts`` above 0; a result that sent none, as a
  replay's, owes none;
- ``warnings`` and ``precision``: each conforms as interpret tests it;
- ``provenance``: each detail of the provenance profile, ``request_id``, ``calibrated``,
  ``usage``, ``attempts`` and ``clients`` in its form, ``claimed`` naming only details holding
  a value,
  ``missing`` only details and reasons of the profile and never a detail holding a value; a
  result that sent a request gives each detail a value or a reason, and one that sent none has
  no provenance;
- ``unsupported-provenance``: a result whose every ref is ``unsupported`` has no provenance,
  ``usage``, ``attempts``, ``request_id``, ``calibrated``, ``missing`` or ``clients``, and
  ``truncated`` empty;
- ``deterministic``, when asked: two calls with one batch return equal ``CallResult``s but
  for ``request_id``, which a provider gives each request afresh.

A binding that passes may still misread its provider: the suite holds it to the protocol only.
A binding that returns a ``Failed`` for every batch passes every check but ``deterministic``.
"""

import asyncio
import inspect
from collections.abc import Callable, Iterable
from dataclasses import replace
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType
from ommat._binding import CallResult, OracleBinding, RenderedQuestion, SourceCall
from ommat._decide import RunSettings
from ommat._interpret import (
    _BINDING_MARKERS,  # pyright: ignore[reportPrivateUsage]
    _STRUCTURAL,  # pyright: ignore[reportPrivateUsage]
    answer_written,
    warnings_conform,
)
from ommat._markers import Cause, CauseCode, Failed, Unavailable, Undetermined
from ommat._provenance import (
    DETAILS,
    HASHED_DETAILS,
    REASONS,
    STRING_DETAILS,
    precision_conforms,
    string_conforms,
    usage_conforms,
    versions_conform,
)
from ommat._typed import is_a
from ommat._values import MAX_INTEGER, described, string_data

__all__ = ["check"]

Batch = tuple[SourceCall, ...]

_UNSENT = (
    *HASHED_DETAILS,
    "usage",
    "attempts",
    "request_id",
    "calibrated",
    "missing",
    "clients",
)
"""The members a result that sent no request holds null."""
_PER_REF = frozenset({CauseCode.timed_out, CauseCode.raised})
_SECONDS = float(RunSettings().timeout)
"""The seconds each call has: the run settings' default timeout."""
_PHOTO = {"media_type": "image/png", "data": b"\x89PNG\r\n\x1a\n"}
"""An image need: the eight bytes that open a PNG file, which no binding is asked to decode."""

_TEXT = "Filed on 10 March: the lamp on Elm Row and the one on Quay Street failed on 3 March."
_NEEDS = {"t:state.filed_on": "2026-03-10", "t:state.text": _TEXT}


def _sources() -> tuple[SourceCall, ...]:
    """One source of each answer kind, a ``t:`` source, and one asking a described noul and a
    structured wording, in source id order."""
    return (
        SourceCall(
            "choice",
            None,
            {"t:state.text": _TEXT},
            (),
            {
                "q:choice/asset": RenderedQuestion(
                    AnswerKind.choice,
                    "Which asset is the report about?",
                    ("lighting", "other"),
                    descriptions=("A street light or a lit sign", None),
                )
            },
        ),
        SourceCall(
            "extraction",
            None,
            dict(_NEEDS),
            (),
            {
                "q:extraction/noticed": RenderedQuestion(
                    AnswerKind.extraction,
                    "The date the problem was first noticed",
                    (),
                    value_type=ValueType.date,
                    text="t:state.text",
                    reference_date="2026-03-10",
                ),
                "q:extraction/places": RenderedQuestion(
                    AnswerKind.extraction,
                    "Each street the report names",
                    (),
                    value_type=ValueType.text,
                    is_list=True,
                    text="t:state.text",
                ),
            },
        ),
        SourceCall(
            "noul",
            None,
            {"t:state.photo": dict(_PHOTO), "t:state.text": _TEXT},
            (),
            {
                "q:noul/hazard": RenderedQuestion(
                    AnswerKind.noul, "Does the report describe a danger to people?", ("yes", "no")
                )
            },
        ),
        SourceCall(
            "score",
            None,
            {"t:state.text": _TEXT},
            (),
            {
                "q:score/severity": RenderedQuestion(
                    AnswerKind.score,
                    "How serious is the problem?",
                    ("minor", "moderate", "serious"),
                )
            },
        ),
        SourceCall("tool", None, {"t:state.text": _TEXT}, ("t:tool/open_orders",), {}),
        SourceCall(
            "wording",
            None,
            {"t:state.text": _TEXT},
            (),
            {
                "q:wording/hazard": RenderedQuestion(
                    AnswerKind.noul,
                    "Does the report describe a danger to people?",
                    ("yes", "no"),
                    descriptions=("Someone could be hurt", None),
                ),
                "q:wording/kind": RenderedQuestion(
                    AnswerKind.choice,
                    {"task": "What kind of report is it?", "examples": ["a lamp out", None]},
                    ("lighting", "other"),
                ),
            },
        ),
    )


_LABEL_SETS = (
    ("none", "several", "ana@example.com", '"Ana" <a@x.org>'),
    ("none", "several", "]", "Zo\u00eb \u00c5ngstr\u00f6m", "\u674e"),
    ("ana@example.com", "bob@example.com", "none", "several"),
)
"""Three renderings of one candidate-valued question: its fixed part, then two sets' members,
and a set's members before its fixed part."""


def _label_sets() -> list[Batch]:
    """One ``q:`` ref asked over three label sets, as a candidate-valued question is rendered
    from two turns' sets: a binding that keeps labels by ref fails ``answers``."""
    return [
        (
            SourceCall(
                "pick",
                None,
                {"t:state.text": _TEXT},
                (),
                {
                    "q:pick/sender": RenderedQuestion(
                        AnswerKind.choice,
                        "Who sent the report?",
                        labels,
                        descriptions=(None,) * len(labels),
                    )
                },
            ),
        )
        for labels in _LABEL_SETS
    ]


def _default_batches() -> list[Batch]:
    """The suite's own batches: each source alone, then one question over three label sets,
    then all of the sources in one batch."""
    sources = _sources()
    return [*((s,) for s in sources), *_label_sets(), sources]


def check(
    binding: OracleBinding,
    *,
    batches: Iterable[Batch] | None = None,
    deterministic: bool = False,
) -> list[str]:
    """Call ``binding`` with each batch and return the protocol's failures; empty passes.

    It runs its own event loop, so inside a running one it raises ``RuntimeError``."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("check was called inside a running event loop")
    chosen = _default_batches() if batches is None else list(batches)
    if not _takes_seconds(binding):
        return ["seconds: the binding does not take seconds as a keyword"]
    return asyncio.run(_check(binding, chosen, deterministic))


def _takes_seconds(binding: object) -> bool:
    """Whether ``binding(batch, seconds=...)`` binds to its signature; true when the signature
    cannot be read, so the call itself is then the test."""
    try:
        signature = inspect.signature(cast("Callable[..., object]", binding))
    except (TypeError, ValueError):
        return True
    try:
        signature.bind((), seconds=_SECONDS)
    except TypeError:
        return False
    return True


async def _check(binding: OracleBinding, batches: list[Batch], deterministic: bool) -> list[str]:
    failures: list[str] = []
    for n, batch in enumerate(batches):

        def fail(name: str, what: str, n: int = n) -> None:
            failures.append(f"{name}: batch {n}: {what}")

        first = await _call(binding, batch, fail)
        if first is _RAISED:
            continue
        _judge(batch, first, fail)
        if deterministic:
            again = await _call(binding, batch, fail)
            if again is _RAISED:
                continue
            if type(first) is not CallResult or type(again) is not CallResult:
                fail("deterministic", "a call did not return a CallResult")
            elif replace(first, request_id=None) != replace(again, request_id=None):
                fail("deterministic", "two calls with one batch returned different results")
    return failures


_RAISED = object()


async def _call(binding: OracleBinding, batch: Batch, fail: Callable[[str, str], None]) -> object:
    try:
        return await binding(batch, seconds=_SECONDS)
    except Exception as error:  # the suite reports a raise, never propagates it
        fail("returns", f"raised {type(error).__name__}")
        return _RAISED


def _judge(batch: Batch, result: object, fail: Callable[[str, str], None]) -> None:
    if type(result) is Failed:
        cause = cast(object, getattr(result, "cause", None))
        code = getattr(cause, "code", None) if type(cause) is Cause else None
        if not (type(code) is CauseCode and code in _STRUCTURAL):
            fail("returns", "a Failed whose cause is not one a binding may fail a call with")
        return
    if type(result) is not CallResult:
        fail("returns", f"returned {described(result)}, not a CallResult or a Failed")
        return
    provided = [ref for call in batch for ref in (*call.refs, *call.questions)]
    values = cast(object, result.values)
    if type(values) is not dict:
        fail("values", f"values is {described(values)}, not a dict")
        return
    held = _held(cast("dict[object, object]", values), fail)
    missing = [ref for ref in provided if ref not in held]
    extra = sorted(ref for ref in held if ref not in provided)
    if missing:
        fail("values", f"no value for {', '.join(missing)}")
    if extra:
        fail("values", f"a key the batch does not provide: {', '.join(extra)}")
    questions = {ref: (call, q) for call in batch for ref, q in call.questions.items()}
    per_ref = False
    for ref, value in held.items():
        if isinstance(value, Undetermined):
            if not _binding_marker(value):
                fail("markers", f"{ref} holds a marker a binding may not return")
            elif cast(Unavailable, value).cause.code in _PER_REF:
                per_ref = True
        elif ref in questions and not _in_form(ref, value, *questions[ref]):
            fail("answers", f"{ref} is not in its kind's form")
    called = _sent(result)
    if per_ref and called and not result.warnings:
        fail("per-ref-cause", "a timed-out or raised ref with no warnings line")
    if not warnings_conform(cast(object, result.warnings)):
        fail("warnings", "warnings does not conform")
    if not precision_conforms(cast(object, result.precision)):
        fail("precision", f"precision {described(result.precision)} does not conform")
    for problem in _provenance_problems(result):
        fail("provenance", problem)
    refused = [held.get(ref) for ref in provided]
    if provided and all(_unsupported(v) for v in refused):
        set_ = [name for name in _UNSENT if getattr(result, name) not in (None, 0, {})]
        set_ += [name for name in ("claimed", "truncated") if getattr(result, name) != ()]
        if set_:
            fail("unsupported-provenance", f"every ref is unsupported, yet {', '.join(set_)} set")


def _sent(result: CallResult) -> bool:
    """Whether the binding says it sent a request: ``attempts`` above 0, an integer of the value
    domain as interpret keeps one."""
    attempts = cast(object, result.attempts)
    return type(attempts) is int and 0 < attempts <= MAX_INTEGER


def _holds(result: CallResult, detail: str) -> bool:
    if detail.startswith("usage."):
        usage = cast(object, result.usage)
        return type(usage) is dict and detail.removeprefix("usage.") in usage
    return cast(object, getattr(result, detail)) is not None


def _provenance_problems(result: CallResult) -> list[str]:
    """What breaks the provenance profile: a detail out of its form, a ``claimed`` or a
    ``missing`` out of theirs, and, on a call that sent a request, a detail with neither a
    value nor a reason, or with both."""
    problems = [
        f"{name} {described(getattr(result, name))} is not a string or null"
        for name in STRING_DETAILS
        if not string_conforms(cast(object, getattr(result, name)))
    ]
    if not versions_conform(cast(object, result.runtime)):
        problems.append("runtime is not null or a non-empty dict of strings")
    if not versions_conform(cast(object, result.clients), exact=True):
        problems.append("clients is not null or a non-empty dict of exact strings")
    if not string_conforms(cast(object, result.request_id), exact=True):
        problems.append("request_id is not null or an exact string")
    calibrated = cast(object, result.calibrated)
    if not (calibrated is None or type(calibrated) is bool):
        problems.append("calibrated is not null or a bool")
    usage = cast(object, result.usage)
    if usage is not None and not (
        type(usage) is dict and usage_conforms(cast("dict[object, object]", usage))
    ):
        problems.append("usage is not null or a dict of strings to numbers")
    claimed = cast(object, result.claimed)
    names = cast("tuple[object, ...]", claimed) if type(claimed) is tuple else None
    strings = names is not None and all(type(n) is str for n in names)
    if names is None or not strings or len(set(names)) != len(names):
        problems.append("claimed is not a tuple of distinct detail names")
        names = None
    else:
        for name in names:
            if not (name in DETAILS and _holds(result, cast(str, name))):
                problems.append(f"claimed names {described(name)}, no detail holding a value")
    attempts = cast(object, result.attempts)
    if attempts is not None and not (type(attempts) is int and 0 <= attempts <= MAX_INTEGER):
        problems.append("attempts is not null or an integer of at least 0")
    missing = cast(object, result.missing)
    reasons: dict[object, object] = {}
    if missing is not None:
        if type(missing) is not dict:
            problems.append("missing is not null or a dict")
        else:
            reasons = cast("dict[object, object]", missing)
            for name, reason in reasons.items():
                if not (type(name) is str and name in DETAILS):
                    problems.append(f"missing names {described(name)}, no detail")
                elif not (type(reason) is str and reason in REASONS):
                    problems.append(f"missing gives {name} {described(reason)}, no reason")
    sent = _sent(result)
    for name in sorted(DETAILS):
        held, given = _holds(result, name), name in reasons
        if held and given:
            problems.append(f"{name} holds a value and a reason")
        elif sent and not held and not given:
            problems.append(f"{name} holds no value and no reason")
    if not sent:
        set_ = [n for n in HASHED_DETAILS if cast(object, getattr(result, n)) is not None]
        if type(cast(object, result.claimed)) is tuple and result.claimed != ():
            set_.append("claimed")
        if set_:
            problems.append(f"no request was sent, yet {', '.join(set_)} set")
    return problems


def _in_form(ref: str, value: object, call: SourceCall, question: RenderedQuestion) -> bool:
    """Whether interpret would write ``value`` at ``ref`` as an answer, not ``invalid-answer``."""
    text = call.needs.get(question.text) if question.text is not None else None
    length = len(text) if type(text) is str else 0
    written = answer_written(
        ref,
        value,
        question.kind,
        question.labels,
        question.value_type,
        question.is_list,
        length,
    )
    return not isinstance(written, Undetermined)


def _held(values: dict[object, object], fail: Callable[[str, str], None]) -> dict[str, Any]:
    """The values by each key's string data; a key that is not a string, or two keys of one
    string data, are failures."""
    held: dict[str, Any] = {}
    for key, value in values.items():
        if not is_a(key, str):
            fail("values", f"a key {described(key)} is not a string")
            continue
        ref = string_data(cast(str, key))
        if ref in held:
            fail("values", f"two keys of one string data, {ref}")
        held[ref] = value
    return held


def _binding_marker(value: object) -> bool:
    """Whether a marker is one a binding may return: exactly ``Unavailable`` with exactly a
    ``Cause`` whose code is in the binding's set."""
    if type(value) is not Unavailable:
        return False
    cause = cast(object, getattr(value, "cause", None))
    code = getattr(cause, "code", None) if type(cause) is Cause else None
    return type(code) is CauseCode and code in _BINDING_MARKERS


def _unsupported(value: object) -> bool:
    return _binding_marker(value) and cast(Unavailable, value).cause.code is CauseCode.unsupported
