"""The provenance profile: the details a call states about who answered and how, the reasons a
detail may lack a value for, and the forms of ``precision``.

Each call item from format 5 holds the hashed details and ``claimed``; its ``calls`` entry in
``unhashed`` holds the rest, with ``missing``, the reason for each detail a call that sent a
request leaves null, and ``clients``, the versions of the libraries that made the call, which
is no detail.
"""

import math
import re
from typing import cast

from ommat._typed import is_a
from ommat._values import MAX_INTEGER, domain_error, string_data

STRING_DETAILS = (
    "provider",
    "model_requested",
    "model_reported",
    "revision",
    "elicitation",
    "prompt",
)
"""The hashed details that are each a string or null."""

HASHED_DETAILS = (*STRING_DETAILS, "runtime", "precision")
"""The details the call item holds, inside the hash."""

USAGE_DETAILS = ("usage.input_tokens", "usage.output_tokens", "usage.passes")
"""The ``usage`` figures that are each a detail of their own."""

USAGE_KEYS = tuple(detail.removeprefix("usage.") for detail in USAGE_DETAILS)
"""The keys a ``usage`` may hold."""

DETAILS: frozenset[str] = frozenset((*HASHED_DETAILS, *USAGE_DETAILS, "request_id"))
"""Every detail's name: the closed set a ``missing`` key and a ``claimed`` name come from."""

RECORDED_DETAILS: frozenset[str] = DETAILS | {"calibrated"}
"""Every detail's name a record may hold: ``calibrated`` only before record format 7."""

REASONS: frozenset[str] = frozenset(
    {"not-reported", "not-pinned", "not-sent", "hidden", "not-applicable", "not-exposed"}
)
"""Why a detail holds no value: the closed set a ``missing`` reason comes from."""

UNROUNDED = "unrounded"
"""The ``precision`` of a provider that does not round."""

MAX_PRECISION = 17
"""The most decimal places a call's ``precision`` names: past it a double holds no more."""

_STEP = re.compile(r"0\.[0-9]{0,16}[1-9]")
_POWER_OF_TEN = re.compile(r"0\.0*1")


def precision_conforms(value: object, *, profile: bool = True) -> bool:
    """Whether ``value`` is a conforming ``precision``: ``None``, an exact ``int`` from 0 to
    ``MAX_PRECISION``, or with ``profile`` the string ``unrounded`` or exactly
    ``{"step": s}``, ``s`` an exact ``str`` of ``0.`` then at most 17 digits ending in one
    from 1 to 9 that is no power of ten, which a count of places gives."""
    if value is None or (type(value) is int and 0 <= value <= MAX_PRECISION):
        return True
    if not profile:
        return False
    if type(value) is str:
        return value == UNROUNDED
    if type(value) is not dict:
        return False
    members = cast("dict[object, object]", value)
    if len(members) != 1:
        return False
    ((key, step),) = members.items()  # no code of the binding's key runs
    return (
        type(key) is str
        and key == "step"
        and type(step) is str
        and _STEP.fullmatch(step) is not None
        and _POWER_OF_TEN.fullmatch(step) is None
    )


def string_conforms(value: object, *, exact: bool = False) -> bool:
    """Whether ``value`` is null or a string of the value domain, a ``str`` subclass read by
    its string data; with ``exact``, only an exact ``str``."""
    if value is None:
        return True
    kind = type(value) is str if exact else is_a(value, str)
    return kind and domain_error(value) is None


def versions_conform(value: object, *, exact: bool = False) -> bool:
    """Whether ``value`` is null or exactly a non-empty ``dict`` from string to string within
    the value domain, a ``str`` subclass read by its string data: a ``runtime``; with
    ``exact``, from exact ``str`` to exact ``str``: a ``clients``."""
    if value is None:
        return True
    if type(value) is not dict:
        return False
    members = cast("dict[object, object]", value)

    def string(x: object) -> bool:
        return type(x) is str if exact else is_a(x, str)

    if not members or not all(string(k) and string(v) for k, v in members.items()):
        return False
    return domain_error(members) is None


def versions(value: object) -> dict[str, str] | None:
    """A conforming ``runtime`` or ``clients`` copied, each string by its string data."""
    if value is None:
        return None
    members = cast("dict[str, str]", value)
    return {string_data(k): string_data(v) for k, v in members.items()}


def usage_conforms(value: dict[object, object]) -> bool:
    """Whether a ``usage`` dict conforms as interpret keeps one: in the value domain, each key
    an exact ``str`` of ``USAGE_KEYS`` and each figure exactly an ``int`` of magnitude at most
    2^53 - 1 or a finite ``float``."""
    if domain_error(value) is not None:
        return False
    for key, figure in value.items():
        if type(key) is not str or key not in USAGE_KEYS:
            return False
        if type(figure) is int:
            if abs(figure) > MAX_INTEGER:
                return False
        elif not (type(figure) is float and math.isfinite(figure)):
            return False
    return True
