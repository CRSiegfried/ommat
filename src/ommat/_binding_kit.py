"""The binding kit: what a binding author calls. ``extraction_value`` writes an extraction
answer in the form a binding returns, its temporal values in ommat's grammar;
``bounded_warnings`` makes any lines a conforming ``warnings``; ``json_value`` writes a need's
value as JSON in its record form; ``image_need`` reads an image need's media type and bytes;
``prompt_text`` writes a prompt as text; ``candidate_of`` reads a grounded ref's candidate;
and ``call_result`` folds each request's ``RequestProvenance`` into the call's ``CallResult``.

Their public home is ``ommat.bindings``. They live here, in the core, so that ``from_record``
calls the same functions without ``import ommat`` importing ``ommat.bindings``. Each is public
under the same deprecation rule as ``ommat.__all__``, and none changes the helper library's
version.
"""

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast

from ommat._binding import CallResult
from ommat._interpret import (
    MAX_WARNING_CODE_POINTS,
    MAX_WARNINGS,
    _record_form,  # pyright: ignore[reportPrivateUsage]
)
from ommat._provenance import DETAILS
from ommat._refs import parse_ref
from ommat._values import described, escape_surrogates

_TAIL = 64
"""Code points kept free for the last line, which counts the lines left out."""


def extraction_value(
    value: object, guess: object, p_null: object, confidence: object, span: object
) -> dict[str, object]:
    """The five-member ``dict`` an extraction question's ref takes in ``CallResult.values``.

    In ``value`` and ``guess``, or in each member of a list or tuple of either, a
    ``datetime.datetime``, ``datetime.date``, ``datetime.time`` or ``datetime.timedelta`` is
    written in ommat's grammar, chosen by its exact type:

    - a date as ``YYYY-MM-DD``;
    - a time as ``HH:MM:SS``, then ``.ffffff`` when it has microseconds, then ``Z`` for a zero
      offset or ``+HH:MM`` or ``-HH:MM`` for another, when it is aware;
    - a datetime as its date, ``T`` and its time;
    - a duration as ``P``, ``nD`` for its days, then ``T`` and ``nH``, ``nM``, ``nS`` for its
      non-zero hours, minutes and seconds; ``PT0S`` for zero.

    A tuple becomes a list. ``span``, a ``(start, end)`` pair or a sequence of pairs, becomes
    lists. ``ValueError``, which the binding returns as ``invalid-answer`` on that ref, refuses
    a negative duration, a duration with a fraction of a second, an offset that is not a whole
    number of minutes, and an aware value whose zone gives no offset. Nothing else is checked:
    interpret validates the answer."""
    return {
        "value": _rendered(value),
        "guess": _rendered(guess),
        "p_null": p_null,
        "confidence": confidence,
        "span": _span(span),
    }


def _rendered(value: object) -> object:
    if type(value) is list or type(value) is tuple:
        return [_temporal(v) for v in cast("list[object] | tuple[object, ...]", value)]
    return _temporal(value)


def _temporal(value: object) -> object:
    t = type(value)
    if t is datetime.datetime:
        moment = cast(datetime.datetime, value)
        return f"{moment.date().isoformat()}T{_clock(moment, moment.tzinfo, moment.utcoffset())}"
    if t is datetime.date:
        return cast(datetime.date, value).isoformat()
    if t is datetime.time:
        clock = cast(datetime.time, value)
        return _clock(clock, clock.tzinfo, clock.utcoffset())
    if t is datetime.timedelta:
        return _duration(cast(datetime.timedelta, value))
    return value


def _clock(
    value: datetime.time | datetime.datetime,
    zone: datetime.tzinfo | None,
    offset: datetime.timedelta | None,
) -> str:
    """``HH:MM:SS``, six fraction digits when there are microseconds, and the offset."""
    clock = f"{value.hour:02d}:{value.minute:02d}:{value.second:02d}"
    if value.microsecond:
        clock += f".{value.microsecond:06d}"
    if zone is None:
        return clock
    if offset is None:
        raise ValueError(f"the zone of {described(value)} gives no offset")
    if offset.microseconds or offset.seconds % 60:
        raise ValueError(f"the offset {offset} is not a whole number of minutes")
    minutes = (offset.days * 86400 + offset.seconds) // 60
    if minutes == 0:
        return f"{clock}Z"
    sign = "-" if minutes < 0 else "+"
    hours, minutes = divmod(abs(minutes), 60)
    return f"{clock}{sign}{hours:02d}:{minutes:02d}"


def _duration(value: datetime.timedelta) -> str:
    if value < datetime.timedelta(0):
        raise ValueError(f"the duration {value} is negative")
    if value.microseconds:
        raise ValueError(f"the duration {value} has a fraction of a second")
    hours, rest = divmod(value.seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    clock = "".join(f"{n}{unit}" for n, unit in ((hours, "H"), (minutes, "M"), (seconds, "S")) if n)
    if not value.days and not clock:
        return "PT0S"
    days = f"{value.days}D" if value.days else ""
    return f"P{days}T{clock}" if clock else f"P{days}"


def _span(span: object) -> object:
    """A pair as a list; a sequence of pairs as a list of lists; anything else as given."""
    if type(span) is not list and type(span) is not tuple:
        return span
    members = cast("list[object] | tuple[object, ...]", span)
    if all(type(m) is list or type(m) is tuple for m in members) and members:
        return [list(cast("list[object] | tuple[object, ...]", m)) for m in members]
    return list(members)


def bounded_warnings(lines: Iterable[str]) -> tuple[str, ...]:
    """A ``warnings`` that conforms, from any iterable of ``str``, in its order.

    Each surrogate code point is escaped as the ledger escapes one, and a line that is empty or
    whitespace alone is left out. Past 2^8 lines or 2^16 code points in all, counted after the
    escapes, the lines stop at the first that does not fit, and a last line says how many were
    left out, ``1 more line left out`` or ``<n> more lines left out``; the lines left out as
    whitespace are not counted. A member that is not a ``str`` is a ``TypeError``."""
    kept_lines: list[str] = []
    for line in lines:
        if not issubclass(type(line), str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"a warnings line {described(line)} is not a string")
        escaped = escape_surrogates(line)
        if escaped and not escaped.isspace():
            kept_lines.append(escaped)
    kept: list[str] = []
    total = 0
    for i, line in enumerate(kept_lines):
        rest = len(kept_lines) - i
        last = rest == 1
        room = len(kept) < MAX_WARNINGS - 1 or last
        limit = MAX_WARNING_CODE_POINTS if last else MAX_WARNING_CODE_POINTS - _TAIL
        if not room or total + len(line) > limit:
            kept.append(f"{rest} more {'line' if last else 'lines'} left out")
            break
        kept.append(line)
        total += len(line)
    return tuple(kept)


def json_value(value: object) -> object:
    """A need's value in record form, as JSON data: each ``Answer`` and ``ExtractionAnswer`` as
    its document, each ``dict`` and each list or tuple rebuilt, a tuple as a list, each type
    matched exactly, and any other value as given. ``bytes`` at any depth, which a JSON part
    cannot carry, is a ``TypeError``."""
    return _record_form(value, bytes_form=_no_bytes)


def _no_bytes(value: bytes) -> object:
    raise TypeError("a need holds bytes, which a JSON part cannot carry")


_IMAGE = frozenset({"media_type", "data"})
"""The members of a need that is an image."""


def image_need(value: object) -> tuple[str, bytes] | None:
    """The ``(media_type, data)`` pair of a need that is an image: a ``dict`` of exactly
    ``media_type``, a ``str`` naming an image media type, starting ``image/``, and ``data``,
    the image's ``bytes``, each type matched exactly; ``None`` for any other value."""
    if type(value) is not dict or frozenset(cast("dict[str, object]", value)) != _IMAGE:
        return None
    image = cast("dict[str, object]", value)
    media_type, data = image["media_type"], image["data"]
    if type(media_type) is not str or not media_type.startswith("image/"):
        return None
    if type(data) is not bytes:
        return None
    return media_type, data


def prompt_text(prompt: object) -> str | None:
    """A prompt as the record's text: a ``str`` or ``None`` as given; a mapping as
    ``json.dumps(prompt)`` with its defaults, each mapping's keys in the order given at every
    depth and sequences in order, so the text keeps the order the model read. A key that is not
    a ``str``, which JSON would write as one and so could repeat, and any other value, within a
    mapping or in place of one, is a ``TypeError``."""
    if prompt is None or isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, Mapping):
        raise TypeError(f"a prompt {described(prompt)} is not a str, a mapping or None")
    _string_keys(cast(object, prompt))
    return json.dumps(prompt, default=_object)


def candidate_of(ref: object) -> str | None:
    """A grounded ref's candidate, the bracketed JSON string read as the grounded ref form
    says; ``None`` for a ref with no bracket, a malformed one, or anything but a string."""
    if not isinstance(ref, str):
        return None
    try:
        return parse_ref(ref).candidate
    except ValueError:  # outside the ref grammar
        return None


def _string_keys(value: object) -> None:
    """``TypeError`` for a mapping key, at any depth, that is not a ``str``."""
    if isinstance(value, Mapping):
        for key, member in cast("Mapping[object, object]", value).items():
            if not isinstance(key, str):
                raise TypeError(f"a prompt holds the key {described(key)}, not a str")
            _string_keys(member)
    elif isinstance(value, (list, tuple)):
        for member in cast("Sequence[object]", value):
            _string_keys(member)


def _object(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(cast("Mapping[object, object]", value))
    raise TypeError(f"a prompt holds {described(value)}, which JSON cannot carry")


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestProvenance:
    """One provider request's provenance, which ``call_result`` folds into the call's: the
    profile's details, ``prompt`` the text ``prompt_text`` gives, ``runtime`` and ``clients``
    mappings from package to version, ``claimed`` the details the provider asserts and cannot
    verify, ``usage`` a mapping from string to number, ``attempts`` 0 for a request never
    sent, ``missing`` from each detail the request left null to its reason, and ``truncated``
    the ``q:`` refs the provider cut short. It checks nothing when built."""

    provider: str | None = None
    model_requested: str | None = None
    model_reported: str | None = None
    revision: str | None = None
    runtime: Mapping[str, str] | None = None
    elicitation: str | None = None
    prompt: str | None = None
    claimed: tuple[str, ...] = ()
    usage: Mapping[str, float] | None = None
    attempts: int = 0
    request_id: str | None = None
    calibrated: bool | None = None
    missing: Mapping[str, str] | None = None
    clients: Mapping[str, str] | None = None
    truncated: tuple[str, ...] = ()


def call_result(
    values: dict[str, object],
    requests: Iterable[RequestProvenance],
    *,
    precision: "int | str | dict[str, str] | None",
    warnings: tuple[str, ...] = (),
) -> CallResult:
    """The ``CallResult`` of a call sent as ``requests``, read only from those whose
    ``attempts`` is above 0: each distinct ``provider``, ``model_requested``,
    ``model_reported``, ``revision``, ``elicitation``, ``prompt`` and ``request_id`` one per
    line in the order first given, ``None`` when none has one; ``runtime`` and ``clients`` each
    merged; ``claimed`` the union, sorted; ``calibrated`` false when any request that holds one
    says false, true when every one that holds one says true, ``None`` when none holds one;
    ``missing`` a detail's reason when every request gives one for it and the result holds no
    value for it, the first code other than ``not-sent``, else ``not-sent``;
    ``usage`` summed per key over the requests that report it, ``None`` when none does;
    ``attempts`` summed; ``truncated`` sorted and distinct; ``warnings`` as given. When no
    request made an attempt, the result has no provenance and ``attempts``, ``precision`` and
    ``missing`` are ``None``, as a binding that called no model returns them. A package at two
    versions in ``runtime`` or ``clients`` is a ``ValueError``."""
    given = list(requests)
    sent = [r for r in given if type(r.attempts) is int and r.attempts > 0]
    usage: dict[str, float] = {}
    for request in sent:
        for key, figure in (request.usage or {}).items():
            usage[key] = usage[key] + figure if key in usage else figure

    def joined(name: str) -> str | None:
        return _joined(cast("str | None", getattr(r, name)) for r in sent)

    says = [r.calibrated for r in sent if r.calibrated is not None]
    # Read by identity, so a value off its form folds to None and never to a bool.
    if any(x is False for x in says):
        calibrated: bool | None = False
    elif says and all(x is True for x in says):
        calibrated = True
    else:
        calibrated = None
    result = CallResult(
        values,
        provider=joined("provider"),
        model_requested=joined("model_requested"),
        model_reported=joined("model_reported"),
        revision=joined("revision"),
        elicitation=joined("elicitation"),
        prompt=joined("prompt"),
        request_id=joined("request_id"),
        runtime=_merged("runtime", (r.runtime for r in sent)),
        precision=precision if sent else None,
        claimed=tuple(sorted({name for r in sent for name in r.claimed})),
        usage=usage or None,
        attempts=sum(r.attempts for r in sent) if sent else None,
        calibrated=calibrated,
        clients=_merged("clients", (r.clients for r in sent)),
        truncated=tuple(sorted({ref for r in sent for ref in r.truncated})),
        warnings=warnings,
    )
    return replace(result, missing=_missing(result, sent))


def _merged(name: str, mappings: Iterable[Mapping[str, str] | None]) -> dict[str, str] | None:
    """The mappings merged; ``None`` when none holds a package. A package at two versions is a
    ``ValueError``."""
    merged: dict[str, str] = {}
    for mapping in mappings:
        for package, version in (mapping or {}).items():
            if merged.setdefault(package, version) != version:
                raise ValueError(
                    f"{name} names {package} at two versions, {merged[package]} and {version}"
                )
    return merged or None


def _missing(result: CallResult, sent: list[RequestProvenance]) -> dict[str, str] | None:
    """Each detail every sent request gives a reason for and the result holds no value for,
    with the first reason other than ``not-sent``, else ``not-sent``."""
    if not sent:
        return None
    reasons: dict[str, str] = {}
    for detail in DETAILS:
        given = [(r.missing or {}).get(detail) for r in sent]
        if any(type(g) is not str for g in given) or _holds(result, detail):
            continue  # a request with no reason, or one not a str, gives the call none
        codes = cast("list[str]", given)
        reasons[detail] = next((c for c in codes if c != "not-sent"), "not-sent")
    return dict(sorted(reasons.items())) or None


def _holds(result: CallResult, detail: str) -> bool:
    if detail.startswith("usage."):
        return result.usage is not None and detail.removeprefix("usage.") in result.usage
    return getattr(result, detail) is not None


def _joined(values: Iterable[str | None]) -> str | None:
    """The distinct values in the order first given, one per line, a newline within a value
    kept as given; ``None`` when there are none."""
    distinct = list(dict.fromkeys(v for v in values if v is not None))
    return "\n".join(distinct) if distinct else None
