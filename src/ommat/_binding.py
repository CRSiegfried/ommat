"""What an oracle binding call passes and returns.

A binding is an async function the caller supplies, one per oracle name and one under
``default``. It takes the batch, a tuple of ``SourceCall``s in source id order, and returns a
``CallResult`` or a ``Failed`` outcome, or raises. Interpret is the one engine producer of a
``SourceCall`` and the one reader of a ``CallResult``. The three types are frozen.
``RenderedQuestion`` and ``SourceCall`` check their fields when built, raising ``TypeError`` for a
field of the wrong type and ``ValueError`` for a wrong value, so a hand-built
one is refused at the line that built it and a binding reads each field as its type says.
``CallResult`` checks nothing: interpret tests it at the write, so a bad value fails its ref or
the call by the fault scope rather than raising inside the binding.
"""

import math
from collections.abc import Awaitable
from dataclasses import KW_ONLY, dataclass
from typing import Any, Protocol, TypeVar, cast

from ommat._answer import AnswerKind, ValueType
from ommat._values import described, string_data

_E = TypeVar("_E", AnswerKind, ValueType)


def _member(enum: type[_E], value: object, field: str) -> _E:
    """``value`` as a member of ``enum``: the member, or a string naming one."""
    if not issubclass(type(value), str):
        name = enum.__name__
        raise TypeError(f"{field} {described(value)} is not a member of {name} or its string")
    try:
        return enum(string_data(cast(str, value)))
    except ValueError:
        raise ValueError(f"{field} {described(value)} names no {enum.__name__}") from None


def _string(value: object, field: str) -> str:
    if not issubclass(type(value), str):
        raise TypeError(f"{field} {described(value)} is not a string")
    return string_data(cast(str, value))


def _optional_string(value: object, field: str) -> str | None:
    return None if value is None else _string(value, field)


def _keyed(value: object, field: str) -> dict[str, Any]:
    """An exact dict with string keys, each key stored as its string data, the values as given."""
    if type(value) is not dict:
        raise TypeError(f"{field} {described(value)} is not a dict")
    items = cast("dict[object, object]", value).items()
    if not all(issubclass(type(k), str) for k, _ in items):
        raise TypeError(f"{field} holds a key that is not a string")
    if all(type(k) is str for k, _ in items):
        return cast("dict[str, Any]", value)
    return {string_data(cast(str, k)): v for k, v in items}


def _strings(value: object, field: str) -> tuple[str, ...]:
    """An exact tuple of strings, each stored as its string data."""
    if type(value) is not tuple:
        raise TypeError(f"{field} {described(value)} is not a tuple")
    return tuple(_string(m, f"a member of {field}") for m in cast("tuple[object, ...]", value))


def _wording(value: object) -> "str | dict[str, Any] | list[Any]":
    """A rendered wording: a string, or a structured one of exact JSON types, rebuilt with
    each key and string as its string data."""
    if isinstance(value, str):
        return _string(value, "wording")
    if type(value) not in (dict, list):
        raise TypeError(f"wording {described(value)} is not a string, an object or an array")
    return cast("dict[str, Any] | list[Any]", _json(value))


def _json(value: object) -> object:
    t = type(value)
    if t is dict:
        out: dict[str, object] = {}
        for k, v in cast("dict[object, object]", value).items():
            if not isinstance(k, str):
                raise TypeError(f"wording holds a key {described(k)} that is not a string")
            out[_string(k, "a wording key")] = _json(v)
        return out
    if t is list:
        return [_json(v) for v in cast("list[object]", value)]
    if isinstance(value, str):
        return _string(value, "a wording value")
    if value is None or t in (bool, int) or (t is float and math.isfinite(cast(float, value))):
        return value
    raise TypeError(f"wording holds {described(value)}, which is no JSON value")


@dataclass(frozen=True, slots=True)
class RenderedQuestion:
    """A ``q:`` ref's question as asked: its kind, the filled wording and its labels' names in
    order, ``yes`` and ``no`` for ``noul`` and none for ``extraction``. ``descriptions``,
    keyword-only, is aligned with ``labels``: each label's description, or ``None`` where it
    declares none, so ``(None, None)`` for ``noul`` and ``()`` for ``extraction``; built with
    none, it is ``None`` for every label.
    An extraction question also gives, keyword-only, its ``value_type``, ``is_list``, ``text``,
    the ``needs`` key holding the text its spans index, and ``reference_date``, the rendered
    date the model resolves a date written without a year against, or ``None`` when the
    question declares none; a probability kind's question leaves them at ``None``, ``False``,
    ``None`` and ``None``.
    A ``choice`` over a candidate-valued label set is rendered at each call: its labels are the
    fixed part's first ``at`` names, then the set's members in order, then the rest of the
    fixed part, each member's description ``None``,
    so one ``q:`` ref's labels may differ from call to call. The set is in the source's
    ``needs`` only when the source declares it a need."""

    kind: AnswerKind
    wording: "str | dict[str, Any] | list[Any]"
    labels: tuple[str, ...]
    _: KW_ONLY
    descriptions: tuple[str | None, ...] = ()
    value_type: ValueType | None = None
    is_list: bool = False
    text: str | None = None
    reference_date: str | None = None

    def __post_init__(self) -> None:
        kind = _member(AnswerKind, self.kind, "kind")
        wording = _wording(self.wording)
        labels = _strings(self.labels, "labels")
        if kind is AnswerKind.noul and labels != ("yes", "no"):
            raise ValueError("a noul question's labels are not yes and no")
        if kind is AnswerKind.extraction and labels:
            raise ValueError("an extraction question holds labels")
        given = cast(object, self.descriptions)
        if type(given) is not tuple:
            raise TypeError(f"descriptions {described(given)} is not a tuple")
        descriptions = cast("tuple[object, ...]", given) or (None,) * len(labels)
        if len(descriptions) != len(labels):
            raise ValueError("descriptions is not as long as labels")
        aligned = tuple(_optional_string(d, "a description") for d in descriptions)
        value_type = (
            None if self.value_type is None else _member(ValueType, self.value_type, "value_type")
        )
        if type(cast(object, self.is_list)) is not bool:
            raise TypeError(f"is_list {described(self.is_list)} is not a bool")
        text = _optional_string(self.text, "text")
        reference_date = _optional_string(self.reference_date, "reference_date")
        if kind is AnswerKind.extraction:
            if value_type is None or text is None:
                raise ValueError("an extraction question lacks its value_type or text")
        elif (value_type, self.is_list, text, reference_date) != (None, False, None, None):
            raise ValueError(f"a {kind} question sets value_type, is_list, text or reference_date")
        for name, value in (
            ("kind", kind),
            ("wording", wording),
            ("labels", labels),
            ("descriptions", aligned),
            ("value_type", value_type),
            ("text", text),
            ("reference_date", reference_date),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SourceCall:
    """One source of a batch: its id, its payload, its needs' values in the form a read
    returns, keyed by each need as declared, the ``t:`` refs it provides and the ``q:`` refs
    it is asked, each in code-point order. Interpret passes ``needs`` as a fresh
    copy: changing it reaches nothing else. A source provides ``t:`` or ``q:`` refs, never
    both, so ``refs`` and ``questions`` are not both non-empty. ``payload`` is the design's
    and is not checked."""

    id: str
    payload: Any
    needs: dict[str, Any]
    refs: tuple[str, ...]
    questions: dict[str, RenderedQuestion]

    def __post_init__(self) -> None:
        sid = _string(self.id, "id")
        needs = _keyed(self.needs, "needs")
        refs = _strings(self.refs, "refs")
        questions = _keyed(self.questions, "questions")
        for question in questions.values():
            if type(question) is not RenderedQuestion:
                raise TypeError(f"questions holds {described(question)}, not a RenderedQuestion")
        if refs and questions:
            raise ValueError("a source call holds both t: refs and questions")
        object.__setattr__(self, "id", sid)
        object.__setattr__(self, "needs", needs)
        object.__setattr__(self, "refs", refs)
        object.__setattr__(self, "questions", questions)


@dataclass(frozen=True, slots=True)
class CallResult:
    """What a binding returns for its batch: ``values``, a ``dict`` from each ref the batch
    provides to its value, a ``q:`` ref's a ``dict`` from label name to probability or, for an
    extraction question, exactly the keys ``value``, ``guess``, ``p_null``, ``confidence`` and
    ``span``; then the call's provenance, hashed on the call item: ``provider``, the provider's
    name, ``model_requested`` and ``model_reported``, ``revision``, the exact version that
    answered, ``runtime``, the code that ran the model as a ``dict`` from package to version,
    ``elicitation``, how it was asked, ``precision``, how the provider rounds probabilities
    (decimal places, ``"unrounded"``, ``{"step": s}`` or ``None`` when unknown), ``prompt``,
    and ``claimed``, the details the source asserts and cannot verify; and kept outside every
    hash: ``usage``, ``attempts``, ``request_id``, ``calibrated``, the provider's claim that its
    probabilities are calibrated, ``missing``, from each detail a call that sent a request
    leaves null to its reason, ``clients``, the versions of the libraries that made the call,
    ``truncated``, the ``q:`` refs whose answer came from a response the provider cut short, and
    ``warnings``, the binding's own lines about the call. A binding that sent no request returns
    no provenance. It checks nothing when built; interpret tests it at the write."""

    values: dict[str, Any]
    _: KW_ONLY
    provider: str | None = None
    model_requested: str | None = None
    model_reported: str | None = None
    revision: str | None = None
    runtime: dict[str, str] | None = None
    elicitation: str | None = None
    precision: "int | str | dict[str, str] | None" = None
    prompt: str | None = None
    claimed: tuple[str, ...] = ()
    usage: dict[str, float] | None = None
    attempts: int | None = None
    request_id: str | None = None
    calibrated: bool | None = None
    missing: dict[str, str] | None = None
    clients: dict[str, str] | None = None
    truncated: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class OracleBinding(Protocol):
    """The type of an oracle binding: ``async def binding(batch, *, seconds)``. ``seconds`` is
    the time the call has, from its start to its timeout's end; what it returns is tested at
    the write, so it is typed as anything."""

    def __call__(
        self, batch: tuple[SourceCall, ...], /, *, seconds: float
    ) -> Awaitable[object]: ...
