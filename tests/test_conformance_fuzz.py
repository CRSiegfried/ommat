"""The conformance suite runs none of a binding's code: every member of a ``CallResult``, and
every place inside one, given a value whose dunders raise, or a ``str``, ``int``, ``dict`` or
``tuple`` subclass whose dunders raise, makes ``check`` report a finding or pass, never raise,
with ``deterministic`` on and off. A ``RequestProvenance`` checks nothing, so its members reach
the suite only through ``call_result``, which a binding author calls; that path is covered by
building results directly here."""

# pyright: reportPrivateUsage=false

from typing import Any

import pytest

from ommat import AnswerKind, CallResult, RenderedQuestion, SourceCall
from ommat.bindings import RequestProvenance, call_result
from ommat.bindings.conformance import check


class Boom(Exception):
    """What a hostile value raises."""


def _boom(*_: object, **__: object) -> Any:
    raise Boom


_DUNDERS = (
    "__eq__",
    "__ne__",
    "__hash__",
    "__bool__",
    "__len__",
    "__iter__",
    "__str__",
    "__repr__",
    "__format__",
    "__lt__",
    "__contains__",
    "__getitem__",
)


class Hostile:
    """Every dunder a reader might call raises, and so do its class and its attributes."""

    @property
    def __class__(self) -> type:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise Boom

    def __getattr__(self, name: str) -> Any:
        raise Boom


for _name in _DUNDERS:
    setattr(Hostile, _name, _boom)


def _hostile_subclass(base: type) -> type:
    namespace: dict[str, Any] = dict.fromkeys(_DUNDERS, _boom)
    namespace["__getattr__"] = _boom
    if base is not str and base is not int:
        namespace["__slots__"] = ()
    return type(f"Hostile{base.__name__.title()}", (base,), namespace)


HStr = _hostile_subclass(str)
HInt = _hostile_subclass(int)
HDict = _hostile_subclass(dict)
HTuple = _hostile_subclass(tuple)


def _hashable(base: type, hash_of: object) -> type:
    """A hostile subclass that can stand as a key: its hash works, every other dunder raises."""
    namespace: dict[str, Any] = dict.fromkeys(_DUNDERS, _boom)
    namespace["__getattr__"] = _boom

    def hashed(self: object) -> int:
        return hash(hash_of)

    namespace["__hash__"] = hashed
    if base is not str and base is not int and base is not object:
        namespace["__slots__"] = ()
    return type(f"HostileKey{base.__name__.title()}", (base,), namespace)


def _keys() -> list[object]:
    """Hostile keys, each hashing as a name the suite looks up, so a lookup reaches __eq__."""
    return [
        _hashable(object, "input_tokens")(),
        _hashable(str, "input_tokens")("x"),
        _hashable(str, "provider")("provider"),
        _hashable(int, 1)(1),
    ]


def _hostiles() -> list[object]:
    return [
        Hostile(),
        HStr("provider"),
        HStr("input_tokens"),
        HInt(1),
        HDict({"a": "1"}),
        HTuple(("revision",)),
    ]


REF = "t:a/n"
Q = "q:b/yes"
ONE: list[tuple[SourceCall, ...]] = [
    (
        SourceCall("a", None, {}, (REF,), {}),
        SourceCall(
            "b",
            None,
            {},
            (),
            {Q: RenderedQuestion(AnswerKind.noul, "Is it so?", ("yes", "no"))},
        ),
    )
]
YES = {"yes": 0.8, "no": 0.2}


def _places(bad: object, key: object) -> list[tuple[str, dict[str, Any]]]:
    """Each place a value may stand in a result, holding ``bad``, or ``key`` where a key."""
    out: list[tuple[str, dict[str, Any]]] = [
        ("values", {"values": bad}),
        ("value", {"values": {REF: bad, Q: YES}}),
        ("answer", {"values": {REF: 1, Q: bad}}),
        ("probability", {"values": {REF: 1, Q: {"yes": bad, "no": 0.2}}}),
        ("label key", {"values": {REF: 1, Q: {key: 0.8, "no": 0.2}}}),
        ("key", {"values": {key: 1}}),
        ("precision", {"precision": bad}),
        ("step", {"precision": {"step": bad}}),
        ("step key", {"precision": {key: "0.05"}}),
        ("claimed", {"claimed": bad}),
        ("claimed item", {"claimed": (bad,)}),
        ("usage", {"usage": bad}),
        ("usage key", {"usage": {key: 1}}),
        ("usage value", {"usage": {"input_tokens": bad}}),
        ("missing", {"missing": bad}),
        ("missing key", {"missing": {key: "hidden"}}),
        ("missing value", {"missing": {"provider": bad}}),
        ("clients", {"clients": bad}),
        ("clients key", {"clients": {key: "1"}}),
        ("clients value", {"clients": {"idia": bad}}),
        ("runtime", {"runtime": bad}),
        ("runtime key", {"runtime": {key: "1"}}),
        ("runtime value", {"runtime": {"torch": bad}}),
        ("warnings", {"warnings": bad}),
        ("warnings item", {"warnings": (bad,)}),
        ("truncated", {"truncated": bad}),
        ("truncated item", {"truncated": (bad,)}),
        ("attempts", {"attempts": bad}),
        ("calibrated", {"calibrated": bad}),
        ("request_id", {"request_id": bad}),
    ]
    for name in ("provider", "model_requested", "model_reported", "revision", "elicitation"):
        out.append((name, {name: bad}))
    out.append(("prompt", {"prompt": bad}))
    return out


def _binding(members: dict[str, Any]) -> Any:
    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        values = members.get("values", {REF: 1, Q: YES})
        rest = {k: v for k, v in members.items() if k != "values"}
        return CallResult(values, **rest)

    return binding


def _cases() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    keys = _keys()
    for i, bad in enumerate(_hostiles()):
        key = keys[i % len(keys)]
        for place, members in _places(bad, key):
            cases.append((f"hostile{i}:{place}", members))
            cases.append((f"hostile{i}:{place}:sent", {**members, "attempts": 1}))
    return cases


CASES = _cases()


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(("place", "members"), CASES, ids=[c[0] for c in CASES])
def test_check_never_raises_on_a_hostile_value(
    place: str, members: dict[str, Any], deterministic: bool
) -> None:
    failures = check(_binding(members), batches=ONE, deterministic=deterministic)
    assert type(failures) is list
    assert all(type(f) is str for f in failures)


_REQUEST_MEMBERS = (
    "provider",
    "model_requested",
    "model_reported",
    "revision",
    "runtime",
    "elicitation",
    "prompt",
    "claimed",
    "usage",
    "request_id",
    "calibrated",
    "missing",
    "clients",
    "truncated",
)


def _request_cases() -> list[tuple[str, dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any]]] = []
    for i, bad in enumerate(_hostiles()):
        for name in _REQUEST_MEMBERS:
            cases.append((f"hostile{i}:{name}", {name: bad}))
    return cases


REQUEST_CASES = _request_cases()


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize(("place", "members"), REQUEST_CASES, ids=[c[0] for c in REQUEST_CASES])
def test_check_never_raises_on_a_hostile_request_folded_by_call_result(
    place: str, members: dict[str, Any], deterministic: bool
) -> None:
    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        request = RequestProvenance(attempts=1, **members)
        return call_result({REF: 1, Q: YES}, [request, RequestProvenance(attempts=1)], precision=2)

    failures = check(binding, batches=ONE, deterministic=deterministic)
    assert type(failures) is list
    assert all(type(f) is str for f in failures)
