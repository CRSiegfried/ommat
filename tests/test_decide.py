# pyright: reportPrivateUsage=false
"""Decide: the entry refusals, the run settings, the state entry and the codec, the decision and
the policy's entry on every path, the broken entry, the observer's exception and the
cancellation at every point they can land, ``decide_sync``, ``unhashed``, and golden records
with the record hash and the design hash checked in, against a scripted fixture binding."""

import asyncio
import dataclasses
import enum
import functools
import gc
import inspect
import os
import random
import subprocess
import sys
import time
import types
import warnings
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import ommat
from counting import SmallBounds
from designs import minimal
from ommat import (
    Cause,
    CauseCode,
    Decision,
    DefectWarning,
    EntryRefusal,
    LoadedDesign,
    Question,
    Record,
    Rule,
    RunSettings,
    Source,
    Table,
    Text,
    Unasked,
    Unavailable,
    Unsure,
    When,
    _decide,
    decide,
    decide_sync,
    escalation,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._decide import run_settings
from ommat._ledger import decision_entry, finished
from ommat._values import UNREPRESENTABLE
from ommat.helpers import pick_with_margin
from scripted import (
    Batch,
    Claims,
    NotACoroutine,
    Scripted,
    apart,
    bound_all,
    clashing,
    generator_coroutine,
    returns,
    started_raising_on_close,
    stop,
)

pytestmark = pytest.mark.filterwarnings("ignore::ommat.DefectWarning")  # defects are the point here

# ---------------------------------------------------------------------------------------------
# The slice design. The golden hashes below cover these functions' source: change one and the
# design hash changes.


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


def echo(r: Any) -> Any:
    return r.read("t:state.subject")


def archive_policy(r: Any) -> str:
    return "archive" if r.read("d:route") == "yes" else "keep"


def ok_policy(r: Any) -> str:
    return "keep"


def undecided_policy(r: Any) -> str:
    return "archive" if r.require("d:route") == "yes" else "keep"


def cancel_policy(r: Any) -> str:
    raise asyncio.CancelledError


ACT = t.label("archive", "keep")


def act_rule(function: Callable[[Any], object] = archive_policy, *reads: str) -> Rule:
    """The policy's rule, ``d:act``, reading ``d:route`` unless ``reads`` are given."""
    return Rule("d:act", ACT, function, version="1", reads=list(reads or ("d:route",)))


STATE_T = t.record(
    subject=t.string,
    to=t.string,
    blob=t.bytes_,
    weight=t.number(),
    sender=t.record(address=t.string),
)
ASK = Source(
    {"q:spam": Question("noul", Text("Is it spam?"))},
    needs=["t:state.subject"],
    oracle="mail",
    cost=1,
)
BODY = Source({"t:body": t.string}, needs=["t:state.subject"], oracle="mail")
"""Asked with ``ask`` in one call: the two share the oracle ``mail``, and ``ask`` is the
batch's first source, so its scripted reply answers both."""
LOOK = Source({"t:size": t.integer()}, needs=["t:state"])
RULES = {
    "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:spam"]),
    "echo": Rule("d:echo", t.string, echo, version="1", reads=["t:state.subject", "t:body"]),
}


def slice_design(*, act: Rule | None = None, **changes: Any) -> LoadedDesign:
    """The slice design, its policy ``d:act``: ``act`` or ``act_rule()``, beside ``RULES`` or
    the ``rules`` given."""
    rules = changes.pop("rules", RULES)
    fields: dict[str, Any] = {
        "state": STATE_T,
        "sources": {"ask": ASK, "body": BODY, "look": LOOK},
        "rules": {**rules, "act": act_rule() if act is None else act},
        "policy": "d:act",
    }
    fields.update(changes)
    return load(minimal(**fields))


LOADED = slice_design()
STATE: dict[str, Any] = {
    "subject": "hello",
    "to": "dana@example.com",
    "blob": b"\x00\x01",
    "weight": 3,
    "sender": {"address": "a@example.com"},
}
SPAM = {"q:spam": {"yes": 0.9, "no": 0.1}, "t:body": "text of the mail"}
UNCLEAR = {"q:spam": {"yes": 0.52, "no": 0.48}, "t:body": "text of the mail"}
"""Within the margin, so ``route`` abstains."""
REPLIES: dict[str, object] = {"ask": returns(SPAM), "look": returns({"t:size": 4})}

GOLDEN_DESIGN = "sha256:f9f63dd387fed20583784895b4212501bb750a3717c35ebaa5c766c767cc8bb8"
GOLDEN_RECORD = "sha256:eac4b883e564ef1528a223825340abb3b0516bf067d68f251afe4dfae25f1f2f"
GOLDEN_BROKEN = "sha256:87116dbff18451aa04ad3a2fb85ee750696f103caba257c603ad5e0b42340822"
GOLDEN_CANCELLED = (
    "sha256:9657cecd82a522226d0eb004d6743c31371a251d2834f3d278ab7dbc65c21574",
    "sha256:4be24d9a01741c473aef8e050a9adc4fd88387283cc70cb61535bdd395ae84e7",
)
GOLDEN_UNDECIDED = (
    "sha256:fd616aeadf632cf3760774b847f1c1437e361d0dc70054e09c4dbaa1b1531a89",
    "sha256:95bbe1bdbd52064b72916ddb25f85a367477b5b935dd68ec34ffe5d054c56a08",
)
"""Each a pair: the design hash, then the record hash."""


def scripted(**changes: object) -> Scripted:
    return Scripted({**REPLIES, **changes})


def bound(binding: object) -> dict[str, Any]:
    """One binding under every key the slice design needs: ``default`` and ``mail``."""
    return {"default": binding, "mail": binding}


def bindings(**changes: object) -> dict[str, Any]:
    return bound(scripted(**changes))


def go(
    loaded: LoadedDesign = LOADED,
    state: object = STATE,
    binds: Mapping[str, Any] | None = None,
    settings: Any = None,
    **keywords: Any,
) -> Decision:
    return decide_sync(loaded, state, bindings() if binds is None else binds, settings, **keywords)


T = Any


def caught(coroutine: Coroutine[Any, Any, T]) -> BaseException:
    """Run a coroutine to the exception it raises, caught inside the loop's task, so the
    instance decide raised is the one returned."""

    async def main() -> BaseException:
        try:
            await coroutine
        except BaseException as exc:  # the test reads whatever decide raised
            return exc
        raise AssertionError("decide returned")

    return asyncio.run(main())


def deciding(
    loaded: LoadedDesign = LOADED,
    state: object = STATE,
    binds: Mapping[str, Any] | None = None,
    settings: Any = None,
    **keywords: Any,
) -> Coroutine[Any, Any, Decision]:
    return decide(loaded, state, bindings() if binds is None else binds, settings, **keywords)


def record_of(exc: BaseException) -> Record:
    record = getattr(exc, "record", None)
    assert type(record) is Record
    return record


def kinds(record: Record) -> list[str]:
    return [item["item"] for item in record.log]


def policy_entries(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "entry" and i["ref"] == "d:act"]


BROKEN_KINDS = ["design", "settings", "entry", "entry"]
"""A broken entry's log: ``t:state`` and the policy's ref ``Unavailable``, and nothing after."""


class Seen:
    """An observer that keeps each index and item, raising ``raises`` at index ``at``."""

    def __init__(self, at: int | None = None, raises: BaseException | None = None) -> None:
        self.items: list[tuple[int, dict[str, Any]]] = []
        self.at = at
        self.raises = raises or RuntimeError("the observer failed")

    def __call__(self, index: int, item: dict[str, Any]) -> None:
        self.items.append((index, item))
        if index == self.at:
            raise self.raises


class Counting:
    """A codec that counts its calls and returns ``value``, or the state when none."""

    def __init__(self, value: object = None, raises: BaseException | None = None) -> None:
        self.calls: list[object] = []
        self.value = value
        self.raises = raises

    def __call__(self, state: object) -> object:
        self.calls.append(state)
        if self.raises is not None:
            raise self.raises
        return state if self.value is None else self.value


def roundtrips(record: Record) -> None:
    """A record decide wrote reads back with the same bytes, and both views take it."""
    assert read_record(record.document).document == record.document
    redact(record, b"\x07" * 32)
    escalation(record)


# ---------------------------------------------------------------------------------------------
# A returned turn and the golden records


def test_a_turn_returns_a_decision_whose_record_holds_every_stage() -> None:
    decision = go()
    record = decision.record
    assert kinds(record) == [
        "design",
        "settings",
        "entry",
        "call",
        "call",
        "entry",
        "entry",
        "entry",
        "entry",
        "entry",
        "entry",
        "round",
    ]
    assert decision.estimate == "archive"
    assert policy_entries(record)[0]["value"] == "archive"
    assert decision.design_hash == record.design_hash == LOADED.design_hash
    assert record.cause is None
    roundtrips(record)


def test_the_golden_record() -> None:
    record = go().record
    assert LOADED.design_hash == GOLDEN_DESIGN
    assert record.record_hash == GOLDEN_RECORD


def test_the_golden_broken_entry() -> None:
    record = go(state={"subject": "no other field"}).record
    assert record.log[2:] == broken_items(INVALID)
    assert record.record_hash == GOLDEN_BROKEN


def golden_cancelled() -> Record:
    """A policy raising a cancellation: every round kept, then the ``cause`` item. No timing reaches
    it."""
    loaded = slice_design(act=act_rule(cancel_policy))
    exc = caught(deciding(loaded))
    assert type(exc) is asyncio.CancelledError
    return record_of(exc)


def golden_undecided() -> Record:
    """A policy requiring ``d:route``, which abstains: the policy's entry is ``Unsure``."""
    loaded = slice_design(act=act_rule(undecided_policy))
    return go(loaded, binds=bindings(ask=returns(UNCLEAR))).record


GOLDENS: dict[str, tuple[Callable[[], Record], tuple[str, str]]] = {
    "cancelled": (golden_cancelled, GOLDEN_CANCELLED),
    "undecided": (golden_undecided, GOLDEN_UNDECIDED),
}


def test_the_golden_cancelled_record() -> None:
    record = golden_cancelled()
    assert record.log[-1] == {"item": "cause", "cause": {"code": "cancelled", "refs": []}}
    assert policy_entries(record) == []
    assert record.cause == {"code": "cancelled", "refs": []}
    assert (record.design_hash, record.record_hash) == GOLDEN_CANCELLED


def test_the_golden_undecided_record() -> None:
    record = golden_undecided()
    (entry,) = policy_entries(record)
    assert entry["marker"] == {"kind": "Unsure", "cause": {"code": "required", "refs": ["d:route"]}}
    route = next(i for i in record.log if i["item"] == "entry" and i["ref"] == "d:route")
    view = escalation(record)
    assert view == {
        "kind": "Unsure",
        "cause": {"code": "required", "refs": ["d:route"]},
        "root_cause": route["marker"]["cause"],
        "defect": False,  # an oracle abstained
        "record_hash": record.record_hash,
    }
    assert route["marker"]["cause"]["code"] != "required"
    assert record.log[-1]["item"] == "round"
    assert record.cause is None
    assert (record.design_hash, record.record_hash) == GOLDEN_UNDECIDED


_GOLDEN_PROBE = (
    "from test_decide import GOLDENS, LOADED, go\n"
    "print(LOADED.design_hash, go().record.record_hash,"
    " go(state={'subject': 'no other field'}).record.record_hash)\n"
    "for name, (make, _) in GOLDENS.items():\n"
    "    record = make()\n"
    "    print(name, record.design_hash, record.record_hash)\n"
)


@pytest.mark.parametrize("seed", ["0", "12345"])
def test_the_golden_hashes_hold_under_every_hash_seed(seed: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _GOLDEN_PROBE],
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=Path(__file__).parent,
        env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.stdout.splitlines() == [
        f"{GOLDEN_DESIGN} {GOLDEN_RECORD} {GOLDEN_BROKEN}",
        *(f"{name} {pair[0]} {pair[1]}" for name, (_, pair) in GOLDENS.items()),
    ]


def test_the_design_and_settings_items_bytes() -> None:
    log = go().record.log
    assert log[0] == {"item": "design", "hash": LOADED.design_hash, "format": 8, "helpers": 4}
    assert log[1] == {
        "item": "settings",
        "settings": {
            "round_bound": 8,
            "timeout": 60,
            "budget": None,
            "deadline": None,
            "certificate_cap": None,
        },
        "providers": dict(LOADED._providers),
        "classes": LOADED._classes,
        "grounded": {},
        "sets": {},
        "policy": "d:act",
    }
    assert "t:state" not in log[1]["providers"]
    assert "t:state" in log[1]["classes"]


def test_no_settings_writes_the_bytes_of_the_default_settings() -> None:
    assert go().record.record_hash == go(settings=RunSettings()).record.record_hash


def test_settings_in_force_are_recorded() -> None:
    record = go(settings=RunSettings(round_bound=1, timeout=0.5)).record
    assert record.log[1]["settings"]["round_bound"] == 1
    assert record.log[1]["settings"]["timeout"] == 0.5


def test_shuffled_latencies_give_identical_records() -> None:
    sources = apart({f"s{i}": Source({f"t:v{i}": t.string}) for i in range(5)})
    loaded = load(minimal(state=t.record(), sources=sources))
    hashes: set[str] = set()
    orders: set[tuple[str, ...]] = set()
    for seed in range(6):
        rng = random.Random(seed)
        finished: list[str] = []

        async def reply(
            batch: Batch,
            rng: random.Random = rng,
            finished: list[str] = finished,
            *,
            seconds: float,
        ) -> object:
            await asyncio.sleep(rng.random() / 100)
            finished.append(batch[0].id)
            return returns(dict.fromkeys(batch[0].refs, "v"))

        binding = Scripted(dict.fromkeys(sources, reply))
        decision = decide_sync(loaded, {}, bound_all(binding, sources))
        hashes.add(decision.record.record_hash)
        orders.add(tuple(finished))
    assert len(orders) > 1
    assert len(hashes) == 1


# ---------------------------------------------------------------------------------------------
# The decision


def test_a_decision_is_built_only_by_decide() -> None:
    decision = go()
    with pytest.raises(TypeError, match="only by decide"):
        Decision(object(), decision.record)


def test_a_decision_is_frozen() -> None:
    decision = go()
    with pytest.raises(AttributeError):
        decision.estimate = "keep"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del decision.record  # type: ignore[misc]
    with pytest.raises(AttributeError):
        decision.extra = 1  # type: ignore[attr-defined]


def test_a_decision_has_only_its_three_public_attributes() -> None:
    public = {n for n in dir(Decision) if not n.startswith("_")}
    assert public == {"estimate", "record", "design_hash"}
    assert Decision.__slots__ == ("_record",)


def test_the_design_hash_and_the_estimate_are_derived_from_the_record() -> None:
    decision = go()
    assert decision.design_hash == decision.record.log[0]["hash"]
    assert decision.estimate == policy_entries(decision.record)[0]["value"]
    assert repr(decision) == f"Decision(estimate='archive', record={decision.record!r})"


def test_the_estimate_of_a_value_is_the_value_as_recorded() -> None:
    keep = go(binds=bindings(ask=returns({**SPAM, "q:spam": {"yes": 0.1, "no": 0.9}})))
    assert keep.estimate == "keep"
    blob = go(slice_design(act=Rule("d:act", t.bytes_, lambda r: b"\x00", version="1")))
    assert blob.estimate == "AA"  # a bytes value as the record holds it


def test_the_estimate_of_a_marker_is_the_marker_rebuilt_with_its_cause() -> None:
    estimate = golden_undecided_decision().estimate
    assert type(estimate) is Unsure
    assert estimate.cause == Cause(CauseCode.required, ["d:route"])
    broken = go(state={"subject": "no other field"}).estimate
    assert type(broken) is Unavailable
    assert broken.cause == Cause(CauseCode.invalid_return, ["t:state"])


def unasked_policy(r: Any) -> str:
    return "archive" if r.require("q:spam") == "yes" else "keep"


def test_an_unasked_estimate_is_rebuilt_with_the_cause_its_policy_wrote() -> None:
    """Under a budget cut ``q:spam`` is ``Unasked``; a policy requiring it carries the kind."""
    loaded = slice_design(act=act_rule(unasked_policy, "q:spam"))
    estimate = go(loaded, settings=RunSettings(budget=0.5)).estimate
    assert type(estimate) is Unasked
    assert estimate.cause == Cause(CauseCode.required, ["q:spam"])


def test_an_estimate_read_from_a_record_with_no_policy_entry_is_an_engine_defect() -> None:
    cancelled = record_of(cancelled_mid_turn())
    decision = Decision(_decide._BUILT_BY_DECIDE, cancelled)
    with pytest.raises(_decide._EngineDefect, match="no entry at the policy's ref"):
        decision.estimate  # noqa: B018 - the read raises


def test_interpret_ending_with_no_policy_entry_is_an_engine_defect_with_no_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_policy(*args: Any, **keywords: Any) -> dict[str, object]:
        return {"t:state": STATE}

    monkeypatch.setattr(_decide, "interpret", no_policy)
    seen = Seen()
    exc = caught(deciding(observer=seen))
    assert type(exc) is _decide._EngineDefect
    assert isinstance(exc, RuntimeError)
    assert "d:act" in str(exc)
    assert not hasattr(exc, "record")
    assert [item["item"] for _, item in seen.items] == ["design", "settings", "entry"]


def golden_undecided_decision() -> Decision:
    return go(slice_design(act=act_rule(undecided_policy)), binds=bindings(ask=returns(UNCLEAR)))


# ---------------------------------------------------------------------------------------------
# The entry refusals


class Weird(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Any:
        raise RuntimeError("cannot iterate")

    def __len__(self) -> int:
        return 1


class Classless:
    @property
    def __class__(self) -> type:  # type: ignore[override]
        raise RuntimeError("no class")


class Sub(RunSettings):
    pass


class Three(enum.IntEnum):
    three = 3


class Called:
    """Callable through its type's ``__call__``; nothing else of it is read."""

    def __call__(self, *args: object) -> object:
        return args[-1]


class Untouchable:
    """Not callable, with a raising ``__class__`` and attribute read: callable() reads its
    type's call slot alone."""

    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the argument was read: {name}")


def instance_call() -> object:
    """An object with ``__call__`` set on the instance, which ``callable`` does not count."""

    class Bare:
        pass

    bare = Bare()
    bare.__call__ = lambda *args: None  # type: ignore[attr-defined]
    return bare


class Odd(str):
    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return other is self


REFUSED: list[tuple[str, dict[str, Any]]] = [
    ("a Design", {"loaded": minimal()}),
    ("None loaded", {"loaded": None}),
    ("a dict of settings", {"settings": {"round_bound": 8}}),
    ("a RunSettings subclass", {"settings": Sub()}),
    ("a list of bindings", {"binds": [("default", None)]}),
    ("a mapping whose copy raises", {"binds": Weird()}),
    ("a raising __class__", {"binds": Classless()}),
    ("no default binding", {"binds": {}}),
    ("an undeclared oracle name", {"binds": {**bindings(), "post": scripted()}}),
    ("a key that is not a string", {"binds": {**bindings(), 1: scripted()}}),
    ("two keys of one string data", {"binds": {"default": scripted(), Odd("default"): 1}}),
    ("a non-null certificate cap", {"settings": RunSettings(certificate_cap=1)}),
    ("a codec that is an int", {"codec": 5}),
    ("a codec with an instance __call__", {"codec": instance_call()}),
    ("a codec that cannot be read", {"codec": Untouchable()}),
    ("an observer that is a str", {"observer": "print"}),
    ("an observer with an instance __call__", {"observer": instance_call()}),
    ("an observer that cannot be read", {"observer": Untouchable()}),
]


@pytest.mark.parametrize(("case", "arguments"), REFUSED, ids=[c for c, _ in REFUSED])
def test_an_entry_refusal_writes_no_record_and_calls_no_observer_or_codec(
    case: str, arguments: dict[str, Any]
) -> None:
    seen, codec = Seen(), Counting()
    keywords: dict[str, Any] = {"observer": seen, "codec": codec, **arguments}
    exc = caught(deciding(**keywords))
    assert type(exc) is EntryRefusal
    assert isinstance(exc, ValueError)
    assert not hasattr(exc, "record")
    assert seen.items == []
    assert codec.calls == []


def test_a_codec_and_an_observer_callable_by_their_type_are_taken() -> None:
    decision = go(state=object(), codec=Called(), observer=Called())
    assert kinds(decision.record) == BROKEN_KINDS  # Called returns the state object
    assert decision.record.unhashed["codec"] == "Called"


@pytest.mark.parametrize(
    "settings",
    [
        RunSettings(round_bound=True),
        RunSettings(round_bound=1.0),  # type: ignore[arg-type]
        RunSettings(round_bound=Three.three),
        RunSettings(round_bound=0),
        RunSettings(round_bound=2**53),
        RunSettings(timeout=float("nan")),
        RunSettings(timeout=float("inf")),
        RunSettings(timeout=0),
        RunSettings(timeout=-1.5),
        RunSettings(timeout=True),
        RunSettings(timeout="60"),  # type: ignore[arg-type]
        RunSettings(timeout=2**53),
        RunSettings(budget=-1),
        RunSettings(budget=float("nan")),
        RunSettings(budget=True),
        RunSettings(deadline=0),
        RunSettings(deadline=-0.5),
        RunSettings(deadline=float("inf")),
        RunSettings(certificate_cap=True),  # type: ignore[arg-type]
        RunSettings(certificate_cap=0),
    ],
    ids=repr,
)
def test_a_setting_outside_its_bounds_is_refused(settings: RunSettings) -> None:
    with pytest.raises(EntryRefusal):
        go(settings=settings)


def test_settings_at_their_bounds_are_accepted() -> None:
    record = go(settings=RunSettings(round_bound=2**53 - 1, timeout=2**53 - 1)).record
    assert record.log[1]["settings"]["round_bound"] == 2**53 - 1
    run = run_settings(1, 1e-9, 0, 1e300, None)
    assert run.item() == {
        "round_bound": 1,
        "timeout": 1e-9,
        "budget": 0,
        "deadline": 1e300,
        "certificate_cap": None,
    }


def test_run_settings_validates_nothing() -> None:
    assert RunSettings(round_bound="x").round_bound == "x"  # type: ignore[arg-type]
    assert RunSettings() == RunSettings(8, 60, None, None, None)


def test_a_settings_object_of_another_type_is_refused_before_any_value_is_read() -> None:
    reads: list[str] = []

    class Counted:
        def __getattribute__(self, name: str) -> object:
            reads.append(name)
            return object.__getattribute__(RunSettings(), name)

    exc = caught(deciding(settings=Counted()))
    assert type(exc) is EntryRefusal
    assert reads == []


def test_a_design_with_no_source_needs_no_binding_and_takes_default() -> None:
    loaded = load(minimal())
    assert go(loaded, {}, {}).estimate is True  # the minimal policy reads nothing
    assert go(loaded, {}, {"default": object()}).record.log[2]["ref"] == "t:state"


def test_an_oracle_needs_its_binding_and_default_only_for_a_source_with_none() -> None:
    keyed = slice_design(
        sources={"ask": ASK, "body": BODY, "look": Source({"t:size": t.integer()}, oracle="m")}
    )
    with pytest.raises(EntryRefusal) as caught_refusal:
        go(keyed, binds={"default": scripted(), "mail": scripted()})
    assert str(caught_refusal.value) == 'no binding is given for the oracle "m"'
    only = slice_design(
        sources={"look": Source({"t:size": t.integer()}, oracle="m")},
        rules={},
        act=act_rule(ok_policy, "t:state"),
    )
    # Every binding is given, so entry passes and the key's binding is called.
    binding = Scripted({"look": returns({"t:size": 4})})
    log = go(only, binds={"m": binding}).record.log
    assert [i["sources"] for i in log if i["item"] == "call"] == [["look"]]
    assert [b[0].id for b in binding.batches] == ["look"]


def test_a_str_subclass_key_is_its_string_data() -> None:
    class Key(str):
        pass

    binding = scripted()
    decision = go(binds={Key("default"): binding, Key("mail"): binding})
    assert decision.estimate == "archive"


def test_two_keys_of_one_string_data_are_refused() -> None:
    with pytest.raises(EntryRefusal) as refused:
        go(binds={"default": scripted(), Odd("default"): scripted()})
    assert str(refused.value) == 'two bindings keys name "default"'


@pytest.mark.parametrize(
    ("binds", "message"),
    [
        ({"mail": None}, 'no binding is given under "default" for the sources with no oracle'),
        ({"default": None}, 'no binding is given for the oracle "mail"'),
        (
            {"default": None, "mail": None, "post": None},
            'the bindings key "post" is not an oracle name the design declares',
        ),
        (
            {"default": None, "mail": None, 1: None},
            "the bindings key 1 is not an oracle name the design declares",
        ),
    ],
)
def test_each_binding_refusal_names_the_oracle_or_default(
    binds: dict[object, object], message: str
) -> None:
    with pytest.raises(EntryRefusal) as refused:
        go(binds=cast("dict[str, Any]", binds))
    assert str(refused.value) == message


def test_the_bindings_are_copied_once_and_only_the_copy_is_read() -> None:
    binds: dict[str, Any] = bindings()
    first = binds["default"]

    async def swap(batch: Batch, *, seconds: float) -> object:
        binds["default"] = Scripted({})
        return returns(SPAM)

    first.replies["ask"] = swap
    decision = go(binds=binds)
    assert decision.estimate == "archive"
    assert len(first.batches) == 2


# ---------------------------------------------------------------------------------------------
# The state entry


def state_value(record: Record) -> Any:
    item = record.log[2]
    assert item["ref"] == "t:state"
    return item["value"]


def test_the_state_entry_drops_undeclared_fields_at_every_depth() -> None:
    state = {**STATE, "extra": 1, "sender": {"address": "a@example.com", "name": "A"}}
    value = state_value(go(state=state).record)
    assert value == {
        "subject": "hello",
        "to": "dana@example.com",
        "blob": "AAE",
        "weight": 3,
        "sender": {"address": "a@example.com"},
    }


def test_bytes_are_base64url_in_the_record_and_bytes_and_floats_live() -> None:
    seen = Seen()
    binding = scripted()
    record = go(binds=bound(binding), observer=seen).record
    assert state_value(record)["blob"] == "AAE"
    (look,) = [b[0] for b in binding.batches if b[0].id == "look"]
    live = cast("dict[str, Any]", look.needs["t:state"])
    assert live["blob"] == b"\x00\x01"
    assert type(live["weight"]) is float
    assert type(seen.items[2][1]["value"]["weight"]) is float  # the stored copy
    assert record.document.count(b'"weight":3') == 1


def test_mutating_the_state_after_entry_changes_nothing() -> None:
    state = {**STATE, "sender": {"address": "a@example.com"}}

    async def mutate(batch: Batch, *, seconds: float) -> object:
        state["subject"] = "changed"
        cast("dict[str, Any]", state["sender"])["address"] = "x"
        return returns(SPAM)

    record = go(state=state, binds=bindings(ask=mutate)).record
    assert state_value(record)["subject"] == "hello"
    (echoed,) = [i for i in record.log if i["item"] == "entry" and i["ref"] == "d:echo"]
    assert echoed["value"] == "hello"


def test_the_codecs_value_takes_an_int_as_a_float_and_drops_an_undeclared_field() -> None:
    value = {**STATE, "weight": 3, "sender": {"address": "a@example.com", "name": "A"}}
    seen = Seen()
    record = go(state=object(), codec=Counting(value=value), observer=seen).record
    assert state_value(record)["sender"] == {"address": "a@example.com"}
    stored = seen.items[2][1]["value"]
    assert type(stored["weight"]) is float
    assert stored["sender"] == {"address": "a@example.com"}
    assert record.document.count(b'"weight":3') == 1


def test_the_codec_returning_the_callers_object_is_copied_too() -> None:
    state = dict(STATE)
    codec = Counting(value=state)
    seen = Seen()
    go(state=object(), codec=codec, observer=seen)
    state["subject"] = "after"
    assert seen.items[2][1]["value"]["subject"] == "hello"


class Big(enum.IntEnum):
    one = 1


class Plain(enum.StrEnum):
    hello = "hello"


class DictSub(dict[str, Any]):
    pass


NONCONFORMING: list[tuple[str, object]] = [
    ("a missing field", {k: v for k, v in STATE.items() if k != "to"}),
    ("not a record", ["hello"]),
    ("a bool under number", {**STATE, "weight": True}),
    ("an IntEnum under number", {**STATE, "weight": Big.one}),
    ("NaN under number", {**STATE, "weight": float("nan")}),
    ("an int past 2^53 - 1", {**STATE, "weight": 2**53}),
    ("a bytearray under bytes", {**STATE, "blob": bytearray(b"\x00")}),
    ("a base64 string under bytes", {**STATE, "blob": "AAE"}),
    ("a dict subclass", DictSub(STATE)),
    ("a nested dict subclass", {**STATE, "sender": DictSub(address="a")}),
    ("a lone surrogate", {**STATE, "subject": "\ud800"}),
]


@pytest.mark.parametrize(("case", "state"), NONCONFORMING, ids=[c for c, _ in NONCONFORMING])
def test_a_nonconforming_state_is_a_broken_entry_naming_t_state(case: str, state: object) -> None:
    for codec in (None, Counting(value=state)):
        decision = go(state=state if codec is None else object(), codec=codec)
        assert decision.record.cause == INVALID
        assert kinds(decision.record) == BROKEN_KINDS
        assert "exceptions" not in decision.record.unhashed


def test_conforming_edge_values_are_accepted() -> None:
    state = {**STATE, "subject": Plain.hello, "weight": 0.5}
    assert state_value(go(state=state).record)["subject"] == "hello"
    tupled = load(minimal(state=t.record(tags=t.list_(t.string))))
    assert state_value(go(tupled, {"tags": ("a", "b")}, {}).record) == {"tags": ["a", "b"]}


def test_an_absent_nullable_field_does_not_conform_and_a_present_null_does() -> None:
    loaded = load(minimal(state=t.record(note=t.nullable(t.string))))
    assert go(loaded, {}, {}).record.cause == INVALID
    assert state_value(go(loaded, {"note": None}, {}).record) == {"note": None}


def test_an_empty_record_state_holds_an_empty_object() -> None:
    loaded = load(minimal())
    assert state_value(go(loaded, {"x": 1}, {}).record) == {}


# ---------------------------------------------------------------------------------------------
# The codec and the broken entry

INVALID = {"code": "invalid-return", "refs": ["t:state"]}
RAISED = {"code": "raised", "refs": ["codec"]}


def broken_items(cause: dict[str, Any], policy: str = "d:act") -> list[dict[str, Any]]:
    """A broken entry's items after the settings item: ``t:state`` and the policy's ref
    ``Unavailable`` with the entry cause in round 0, and nothing after."""
    marker = {"kind": "Unavailable", "cause": cause}
    # The marker's members as the engine writes every marker: kind, then cause.
    order = {"1": [1, 0]}
    entry = {"item": "entry", "round": 0, "call": None, "marker": marker, "order": order}
    return [
        {**entry, "ref": "t:state"},
        {**entry, "ref": policy},
    ]


def test_the_codec_is_called_once_with_the_state_object() -> None:
    state = object()
    codec = Counting(value=STATE)
    decision = go(state=state, codec=codec)
    assert codec.calls == [state]
    assert decision.estimate == "archive"


def test_an_async_codecs_coroutine_is_a_broken_entry_and_no_warning_leaks() -> None:
    async def codec(state: object) -> object:
        return STATE

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        decision = go(state=object(), codec=cast(Any, codec))
        gc.collect()
    assert decision.record.log[2:] == broken_items(INVALID)
    others = [str(w.message) for w in caught_warnings if w.category is not DefectWarning]
    assert others == []
    assert [w.category for w in caught_warnings].count(DefectWarning) == 1


@pytest.fixture
def untyped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decide's type test of the codec's return fails if reached: an awaitable is refused
    before it."""

    def never(value: object, type_: object) -> object:
        raise AssertionError("the return reached the type test")

    monkeypatch.setattr(_decide, "typed", never)


@pytest.mark.usefixtures("untyped")
def test_a_sync_codec_returning_a_coroutine_is_a_broken_entry_and_no_warning_leaks() -> None:
    async def inner(state: object) -> object:
        return STATE

    def codec(state: object) -> object:
        return inner(state)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        decision = go(state=object(), codec=codec)
        gc.collect()
    assert decision.record.cause == INVALID
    assert "exceptions" not in decision.record.unhashed
    others = [str(w.message) for w in caught_warnings if w.category is not DefectWarning]
    assert others == []
    assert [w.category for w in caught_warnings].count(DefectWarning) == 1


@pytest.mark.usefixtures("untyped")
def test_a_codec_returning_an_awaitable_is_a_broken_entry_and_it_is_not_closed() -> None:
    awaitable = NotACoroutine()

    def codec(state: object) -> object:
        return awaitable

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == INVALID
    assert not awaitable.closed


@pytest.mark.usefixtures("untyped")
def test_a_codecs_generator_coroutine_is_a_broken_entry_closed_at_the_close() -> None:
    ran: list[str] = []
    generator = generator_coroutine(ran, started=True)

    def codec(state: object) -> object:
        return generator

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == INVALID
    assert inspect.getgeneratorstate(generator) == inspect.GEN_CLOSED
    assert ran == ["finally"]


@pytest.mark.usefixtures("untyped")
def test_a_codecs_coroutine_re_raising_a_kept_stop_on_close_is_a_broken_entry() -> None:
    def codec(state: object) -> object:
        return started_raising_on_close(stop())

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == INVALID
    assert "exceptions" not in decision.record.unhashed


@pytest.mark.parametrize("make_", [Claims, lambda: clashing(stop())], ids=["claims", "kept-stop"])
def test_a_codecs_return_that_is_no_awaitable_by_its_type_meets_the_type_test(
    make_: Callable[[], object], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    typed = _decide.typed

    def spy(value: object, type_: Any) -> Any:
        seen.append(value)
        return typed(value, type_)

    monkeypatch.setattr(_decide, "typed", spy)
    returned = make_()

    def codec(state: object) -> object:
        return returned

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == INVALID
    assert seen == [returned]
    assert "exceptions" not in decision.record.unhashed


def test_a_codecs_coroutine_raising_on_close_keeps_the_broken_entry_and_its_text() -> None:
    def codec(state: object) -> object:
        return started_raising_on_close("on close")

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == INVALID
    text = decision.record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "on close")
    roundtrips(decision.record)


def test_a_codec_return_whose_awaitable_test_raises_is_the_codecs_raise() -> None:
    def codec(state: object) -> object:
        return clashing()

    decision = go(state=object(), codec=codec)
    assert decision.record.cause == RAISED
    text = decision.record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "the type test raised")


def test_a_raising_codec_is_a_broken_entry_with_its_text_under_the_state_index() -> None:
    codec = Counting(raises=ValueError("bad state"))
    binding = scripted()
    calls: list[object] = []

    def counted(r: Any) -> str:
        calls.append(r)
        return archive_policy(r)

    loaded = slice_design(act=act_rule(counted))
    decision = go(loaded, codec=codec, binds=bound(binding))
    record = decision.record
    assert record.log[2:] == broken_items(RAISED)
    assert decision.estimate == Unavailable(Cause(CauseCode.raised, ["codec"]))
    assert "cause" not in kinds(record)
    assert record.cause == RAISED
    text = record.unhashed["exceptions"]["2"]
    assert (text["type"], text["message"]) == ("ValueError", "bad state")
    assert "ValueError: bad state" in text["traceback"]
    assert calls == []
    assert binding.batches == []
    roundtrips(record)


def test_a_broken_entry_writes_the_policy_of_the_design_and_runs_no_rule() -> None:
    decision = go(load(minimal()), {}, {}, codec=Counting(raises=ValueError("x")))
    assert decision.record.log[2:] == broken_items(RAISED, "d:decision")
    assert escalation(decision.record) == {
        "kind": "Unavailable",
        "cause": RAISED,
        "root_cause": RAISED,
        "defect": True,
        "record_hash": decision.record.record_hash,
    }


def test_a_codec_exception_whose_text_raises_is_unrepresentable() -> None:
    class Mute(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no text")

    record = go(codec=Counting(raises=Mute())).record
    assert record.unhashed["exceptions"]["2"]["message"] == UNREPRESENTABLE


def test_a_codecs_keyboard_interrupt_propagates_with_no_record() -> None:
    seen = Seen()
    exc = caught(deciding(codec=Counting(raises=KeyboardInterrupt()), observer=seen))
    assert type(exc) is KeyboardInterrupt
    assert not hasattr(exc, "record")
    assert [item["item"] for _, item in seen.items] == ["design", "settings"]


def test_an_identity_codec_and_no_codec_write_one_log() -> None:
    plain = go().record

    def identity(state: object) -> object:
        return state

    coded = go(codec=identity).record
    assert coded.record_hash == plain.record_hash
    assert coded.log == plain.log
    assert "codec" not in plain.unhashed
    assert (
        coded.unhashed["codec"]
        == "test_an_identity_codec_and_no_codec_write_one_log.<locals>.identity"
    )


def named(name: object) -> Any:
    class Codec:
        def __call__(self, state: object) -> object:
            return state

    codec = Codec()
    object.__setattr__(codec, "__qualname__", name)
    return codec


class Raising:
    def __getattr__(self, name: str) -> object:
        if name == "__qualname__":
            raise RuntimeError("no name")
        raise AttributeError(name)

    def __call__(self, state: object) -> object:
        return state


def encode_state(state: object) -> object:
    return state


@pytest.mark.parametrize(
    ("codec", "name"),
    [
        (encode_state, "encode_state"),
        (named("custom"), "custom"),
        (named(5), "named.<locals>.Codec"),
        (Counting(), "Counting"),
        (Raising(), UNREPRESENTABLE),
    ],
    ids=["function", "set", "not a str", "instance", "raising"],
)
def test_the_codecs_name_in_unhashed(codec: Any, name: str) -> None:
    assert go(codec=codec).record.unhashed["codec"] == name


def test_a_cancelled_error_from_the_codec_is_a_delivered_cancellation() -> None:
    exc = caught(deciding(codec=Counting(raises=asyncio.CancelledError())))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[2:] == [{"item": "cause", "cause": {"code": "cancelled", "refs": []}}]
    assert record.cause == {"code": "cancelled", "refs": []}
    assert set(record.unhashed["turn"]) == {"start", "end"}
    roundtrips(record)


# ---------------------------------------------------------------------------------------------
# The observer's exception


def test_the_observer_sees_every_item_and_the_record_is_the_same_without_one() -> None:
    seen = Seen()
    record = go(observer=seen).record
    assert [index for index, _ in seen.items] == list(range(len(record.log)))
    assert [item for _, item in seen.items] == record.log
    assert go().record.record_hash == record.record_hash


NORMAL_LOG = go().record.log
with warnings.catch_warnings(action="ignore", category=DefectWarning):
    BROKEN_LOG = go(codec=Counting(raises=ValueError("bad"))).record.log


@pytest.mark.parametrize("at", range(len(NORMAL_LOG)))
def test_the_observer_raising_at_any_item_ends_the_turn_with_the_record_so_far(at: int) -> None:
    raised = RuntimeError("the observer failed")
    exc = caught(deciding(observer=Seen(at, raised)))
    assert exc is raised
    record = record_of(exc)
    assert record.log == NORMAL_LOG[: at + 1]
    assert "cause" not in kinds(record)
    assert record.cause is None
    assert set(record.unhashed["turn"]) == {"start", "end"}
    assert str(at) not in record.unhashed.get("calls", {})  # none of that item's members
    roundtrips(record)


def test_a_record_an_observer_ended_is_finished_once_it_holds_the_policy_s_entry() -> None:
    ended = [
        record_of(caught(deciding(observer=Seen(at, RuntimeError("x"))))).log
        for at in range(len(NORMAL_LOG))
    ]
    held = [decision_entry(log) is not None for log in ended]
    assert [finished(log) for log in ended] == held
    assert True in held
    assert False in held


@pytest.mark.parametrize("at", range(len(BROKEN_LOG)))
def test_the_observer_raising_at_a_broken_entry_keeps_the_record_so_far(at: int) -> None:
    exc = caught(deciding(observer=Seen(at), codec=Counting(raises=ValueError("bad"))))
    record = record_of(exc)
    assert record.log == BROKEN_LOG[: at + 1]
    exceptions = record.unhashed.get("exceptions", {})
    assert ("2" in exceptions) == (at > 2)  # none of the state entry's when it raised there
    roundtrips(record)


def test_the_observer_raising_at_a_broken_entry_is_not_chained_to_the_codecs_exception() -> None:
    raised = RuntimeError("the observer failed")
    exc = caught(deciding(observer=Seen(2, raised), codec=Counting(raises=ValueError("bad"))))
    assert exc is raised
    assert exc.__context__ is None
    assert record_of(exc).log[-1] == broken_items(RAISED)[0]


def test_the_unhashed_members_precede_the_design_item() -> None:
    exc = caught(deciding(observer=Seen(0), codec=encode_state))
    unhashed = record_of(exc).unhashed
    assert set(unhashed) == {"turn", "engine", "codec"}
    assert record_of(exc).log == NORMAL_LOG[:1]


@pytest.mark.parametrize("raises", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
def test_any_base_exception_the_observer_raises_is_its_own_with_no_cause_item(
    raises: BaseException,
) -> None:
    exc = caught(deciding(observer=Seen(1, raises)))
    assert exc is raises
    record = record_of(exc)
    assert kinds(record) == ["design", "settings"]
    assert record.cause is None


# ---------------------------------------------------------------------------------------------
# Cancellation

CAUSE_ITEM: dict[str, Any] = {"item": "cause", "cause": {"code": "cancelled", "refs": []}}


def cancelled_mid_turn(observer: Any = None) -> BaseException:
    async def main() -> BaseException:
        started = asyncio.Event()

        async def ask(batch: Any, *, seconds: float) -> object:
            started.set()
            await asyncio.sleep(600)
            return returns(SPAM)

        task = asyncio.create_task(deciding(binds=bindings(ask=ask), observer=observer))
        await started.wait()  # the call is under way
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    return asyncio.run(main())


def test_a_cancellation_in_interpret_carries_the_record_ending_in_the_cause_item() -> None:
    record = record_of(cancelled_mid_turn())
    assert record.log == [*NORMAL_LOG[:3], CAUSE_ITEM]
    assert record.cause == {"code": "cancelled", "refs": []}
    assert set(record.unhashed) == {"turn", "engine"}
    roundtrips(record)
    assert escalation(record) is None


def cancelling(r: Any) -> Any:
    raise asyncio.CancelledError


def test_a_rule_raising_a_cancellation_in_round_1_keeps_the_rounds_items_so_far() -> None:
    rules = {
        **RULES,
        "route": Rule("d:route", t.label("no", "yes"), cancelling, version="1", reads=["q:spam"]),
    }
    exc = caught(deciding(slice_design(rules=rules)))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    log = record.log
    assert log[-1] == CAUSE_ITEM
    round_1 = log[3:-1]
    assert round_1
    assert all(item.get("round") == 1 for item in round_1)
    assert {item["item"] for item in round_1} == {"call", "entry"}  # no round item
    assert "round" not in kinds(record)
    assert record.cause == {"code": "cancelled", "refs": []}
    roundtrips(record)


def test_a_policy_raising_a_cancellation_writes_no_policy_entry() -> None:
    exc = caught(deciding(slice_design(act=act_rule(cancelling))))
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[1:-1] == NORMAL_LOG[1 : len(record.log) - 1]  # another design hash
    assert kinds(record)[-1] == "cause"
    assert policy_entries(record) == []
    roundtrips(record)


class CancellingRepr:
    """A returned key whose ``repr`` would cancel; the engine names its type instead."""

    def __repr__(self) -> str:
        raise asyncio.CancelledError


def test_a_returned_keys_own_repr_is_never_run() -> None:
    reply = returns({**SPAM, CancellingRepr(): 1})  # type: ignore[dict-item]
    record = go(binds=bindings(ask=reply)).record
    unexpected = [c.get("unexpected") for c in record.unhashed["calls"].values()]
    assert ["<test_decide.CancellingRepr>"] in unexpected
    roundtrips(record)


def test_a_cancellation_before_the_body_runs_carries_no_record() -> None:
    seen, codec = Seen(), Counting()

    async def main() -> BaseException:
        task = asyncio.create_task(deciding(observer=seen, codec=codec))
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    exc = asyncio.run(main())
    assert not hasattr(exc, "record")
    assert seen.items == []
    assert codec.calls == []


def test_an_existing_record_is_overwritten() -> None:
    raised = asyncio.CancelledError()
    raised.record = "an outer turn's"  # type: ignore[attr-defined]
    exc = caught(deciding(codec=Counting(raises=raised)))
    assert exc is raised
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_a_cancellation_that_refuses_the_attribute_is_re_raised_unchanged() -> None:
    class Stubborn(asyncio.CancelledError):
        def __setattr__(self, name: str, value: object) -> None:
            raise AttributeError(name)

    raised = Stubborn()
    exc = caught(deciding(codec=Counting(raises=raised)))
    assert exc is raised
    assert not hasattr(exc, "record")


@pytest.mark.parametrize("raises", [RuntimeError("obs"), KeyboardInterrupt()])
def test_the_observer_raising_at_the_cause_item_loses_to_the_cancellation(
    raises: BaseException,
) -> None:
    cancel = asyncio.CancelledError()
    exc = caught(deciding(codec=Counting(raises=cancel), observer=Seen(2, raises)))
    assert exc is cancel
    assert exc.__context__ is raises
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_the_observer_at_the_cause_item_of_a_cancellation_in_interpret() -> None:
    exc = cancelled_mid_turn(observer=Seen(3))
    assert type(exc.__context__) is RuntimeError
    assert record_of(exc).log[-1] == CAUSE_ITEM


def test_a_base_exception_from_a_binding_propagates_bare() -> None:
    async def interrupt(batch: Batch, *, seconds: float) -> object:
        raise SystemExit(3)

    exc = caught(deciding(binds=bindings(ask=interrupt)))
    assert type(exc) is SystemExit
    assert not hasattr(exc, "record")


# ---------------------------------------------------------------------------------------------
# decide_sync


def test_decide_sync_inside_a_running_loop_raises_before_doing_anything() -> None:
    seen, codec = Seen(), Counting()

    async def main() -> None:
        with pytest.raises(RuntimeError, match="running event loop"):
            decide_sync(cast(Any, None), STATE, {}, codec=codec, observer=seen)

    asyncio.run(main())
    assert seen.items == []
    assert codec.calls == []


def test_decide_sync_returns_a_decision_whose_record_ends_with_interpret_s_last_item() -> None:
    decision = decide_sync(LOADED, STATE, bindings())
    assert type(decision) is Decision
    assert kinds(decision.record)[-1] == "round"


def test_decide_is_awaitable_and_equals_decide_sync() -> None:
    decision = asyncio.run(decide(LOADED, STATE, bindings()))
    assert decision.record.record_hash == go().record.record_hash


# ---------------------------------------------------------------------------------------------
# unhashed


def test_unhashed_holds_the_turn_the_engine_and_the_calls() -> None:
    unhashed = go().record.unhashed
    assert set(unhashed) == {"turn", "engine", "calls"}
    assert unhashed["engine"] == ommat.__version__
    turn = unhashed["turn"]
    assert set(turn) == {"start", "end"}
    assert turn["start"].endswith("Z")
    assert turn["end"] >= turn["start"]
    assert set(unhashed["calls"]) == {"3", "4"}


def test_a_codec_adds_only_its_name() -> None:
    assert set(go(codec=encode_state).record.unhashed) == {"turn", "engine", "calls", "codec"}


# ---------------------------------------------------------------------------------------------
# Records decide writes, read back and viewed

SCENARIOS: dict[str, Callable[[], Record]] = {
    "returned": lambda: go().record,
    "undecided": golden_undecided,
    "broken": lambda: go(codec=Counting(raises=ValueError("x"))).record,
    "invalid": lambda: go(state={}).record,
    "cancelled": lambda: record_of(cancelled_mid_turn()),
    "observer-ended": lambda: record_of(caught(deciding(observer=Seen(5)))),
    "design only": lambda: record_of(caught(deciding(observer=Seen(0)))),
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_record_decide_writes_reads_back_and_takes_both_views(scenario: str) -> None:
    roundtrips(SCENARIOS[scenario]())


def test_a_when_raising_a_cancellation_in_round_2_keeps_round_1_whole() -> None:
    def cancels(r: Any) -> Any:
        raise asyncio.CancelledError

    later = Source({"t:later": t.string}, needs=["d:route"], when=When(cancels, version="1"))
    exc = caught(
        deciding(slice_design(sources={"ask": ASK, "body": BODY, "look": LOOK, "later": later}))
    )
    assert type(exc) is asyncio.CancelledError
    record = record_of(exc)
    assert record.log[-1] == CAUSE_ITEM
    round_1 = record.log[3:-1]
    assert {item["round"] for item in round_1} == {1}
    assert round_1[-1]["item"] == "round"
    roundtrips(record)


# ---------------------------------------------------------------------------------------------
# The budget and the deadline through decide


class Slow:
    """A codec that takes ``seconds`` of the turn's time."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self, state: object) -> object:
        time.sleep(self.seconds)
        return state


def rounds(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "round"]


def test_the_codecs_time_counts_against_the_deadline() -> None:
    binding = scripted()
    record = go(binds=bound(binding), settings=RunSettings(deadline=0.01), codec=Slow(0.05))
    assert binding.batches == []
    assert rounds(record.record) == [{"item": "round", "round": 1, "cost": 0, "deadline": True}]
    log = record.record.log
    marker: dict[str, Any] = {"kind": "Unasked", "cause": {"code": "deadline", "refs": []}}
    assert [
        i["marker"] for i in log if i["item"] == "entry" and i["ref"] in ("q:spam", "t:size")
    ] == [
        marker,
        marker,
    ]
    roundtrips(record.record)


def test_a_budget_within_its_bounds_runs_and_cuts_the_frontier_over_it() -> None:
    binding = scripted()
    record = go(binds=bound(binding), settings=RunSettings(budget=0.5)).record
    assert binding.batches == []
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 0, "deadline": False}]
    assert record.log[1]["settings"]["budget"] == 0.5
    record = go(settings=RunSettings(budget=1, deadline=600)).record
    assert rounds(record) == [{"item": "round", "round": 1, "cost": 1, "deadline": False}]


@pytest.mark.parametrize(
    "settings",
    [RunSettings(round_bound=1), RunSettings(budget=0.5), RunSettings(deadline=0.01)],
    ids=["round bound", "budget", "deadline"],
)
def test_the_policy_fires_once_at_every_cut_and_interpret_ends_the_turn(
    settings: RunSettings,
) -> None:
    """A cut drains the sorter, so the policy's rule fires there as any rule does."""
    codec = Slow(0.05) if settings.deadline is not None else None
    record = go(settings=settings, codec=codec).record
    (entry,) = policy_entries(record)
    assert kinds(record)[-1] == "round"
    assert ("value" in entry) != ("marker" in entry)
    roundtrips(record)


def test_a_cancellation_during_a_deadline_bounded_call_records_cancelled_and_no_call() -> None:
    async def main() -> BaseException:
        started = asyncio.Event()

        async def ask(batch: Any, *, seconds: float) -> object:
            started.set()
            await asyncio.sleep(600)
            return returns(SPAM)

        binding = scripted(ask=ask)
        task = asyncio.ensure_future(
            deciding(binds=bound(binding), settings=RunSettings(deadline=600))
        )
        await started.wait()  # the call is under way, its deadline far off
        task.cancel()
        try:
            await task
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("decide returned")

    record = record_of(asyncio.run(main()))
    assert record.cause == {"code": "cancelled", "refs": []}
    assert "call" not in kinds(record)
    assert "round" not in kinds(record)


class _Opaque:
    def __repr__(self) -> str:
        raise RuntimeError("repr was called")


class _Raising(type):
    def __getattribute__(cls, name: str) -> Any:
        raise RuntimeError(f"the metaclass was read for {name}")


class _Hidden(metaclass=_Raising):
    pass


@pytest.mark.parametrize("value", [10**5000, -(10**5000), _Opaque()], ids=["huge", "neg", "repr"])
@pytest.mark.parametrize(
    "field", ["round_bound", "timeout", "budget", "deadline", "certificate_cap"]
)
def test_a_setting_that_cannot_be_written_is_an_entry_refusal(field: str, value: object) -> None:
    exc = caught(deciding(settings=RunSettings(**cast(dict[str, Any], {field: value}))))
    assert type(exc) is EntryRefusal


def test_a_huge_setting_is_named_by_its_bit_length() -> None:
    with pytest.raises(EntryRefusal, match=r"^the round bound integer of 16610 bits is not"):
        run_settings(10**5000, 60, None, None, None)


@pytest.mark.parametrize("key", [10**5000, _Opaque(), _Hidden()], ids=["huge", "repr", "meta"])
def test_a_bindings_key_that_cannot_be_written_is_an_entry_refusal(key: object) -> None:
    exc = caught(deciding(binds={key: scripted()}))  # type: ignore[dict-item]
    assert type(exc) is EntryRefusal


@pytest.mark.parametrize(
    "keywords",
    [
        {"loaded": _Hidden()},
        {"settings": _Hidden()},
        {"observer": _Hidden()},
        {"codec": _Hidden()},
        {"binds": _Hidden()},
    ],
    ids=["loaded", "settings", "observer", "codec", "binds"],
)
def test_an_argument_of_a_hostile_metaclass_is_an_entry_refusal(keywords: dict[str, Any]) -> None:
    exc = caught(deciding(**keywords))
    assert type(exc) is EntryRefusal


def test_a_state_over_the_value_domains_size_bounds_is_a_broken_entry(
    small_bounds: SmallBounds,
) -> None:
    big = {**STATE, "subject": "x" * 257}
    decision = go(state=big)
    assert decision.record.cause == INVALID
    assert kinds(decision.record) == BROKEN_KINDS


def test_a_long_wording_loads_and_decides(small_bounds: SmallBounds) -> None:
    wording = "Is it spam? " + "Consider every word of it. " * 20  # past the 2^8 code points set
    ask = dataclasses.replace(
        ASK, provides={**ASK.provides, "q:spam": Question("noul", Text(wording))}
    )
    decision = go(slice_design(sources={"ask": ask, "body": BODY, "look": LOOK}))
    assert decision.estimate == "archive"
    assert read_record(decision.record.document).document == decision.record.document


def a_list(r: Any) -> list[str]:
    return ["a", "b"]


def test_a_candidate_set_policys_decision_is_the_set() -> None:
    policy = Rule("d:decision", t.list_(t.string), a_list, version="1", candidates=True)
    decision = decide_sync(load(minimal(rules={"decision": policy})), {}, {})
    assert decision.estimate == ["a", "b"]


def reads_the_policy(r: Any) -> bool:
    return r.read("d:decision") is not None


def negated(r: Any) -> bool:
    return not r.read("d:decision")


def test_a_rule_a_table_and_a_source_reading_the_policys_ref_load_and_run_after_it() -> None:
    reader = Rule("d:r", t.boolean, negated, version="1", reads=["d:decision"])
    table = Table("d:tab", t.boolean, "F", ["d:decision"], [["-", True]])
    asks = Source({"t:x": t.string}, needs=["d:decision"])
    design = minimal(rules={"r": reader, "tab": table}, sources={"s": asks})
    binding = Scripted({"s": returns({"t:x": "a"})})
    decision = decide_sync(load(design), {}, {"default": binding})
    assert decision.estimate is True
    entries = {i["ref"]: i for i in decision.record.log if i["item"] == "entry"}
    assert entries["d:r"]["value"] is False
    assert entries["d:tab"]["value"] is True
    assert entries["t:x"]["value"] == "a"
    (call,) = binding.batches[0]
    assert call.needs == {"d:decision": True}
    index = {ref: decision.record.log.index(item) for ref, item in entries.items()}
    assert index["d:decision"] < min(index["d:r"], index["d:tab"], index["t:x"])


def test_a_function_when_reading_the_policys_ref_runs_when_its_source_needs_it() -> None:
    gated = Source(
        {"t:x": t.string}, needs=["d:decision"], when=When(reads_the_policy, version="1")
    )
    loaded = load(dataclasses.replace(minimal(), sources={"s": gated}))
    record = decide_sync(loaded, {}, {"default": Scripted({"s": returns({"t:x": "a"})})}).record
    (found,) = [item for item in record.log if item.get("ref") == "t:x"]
    assert found["value"] == "a"


def test_a_function_when_reading_the_policys_ref_undeclared_fails_at_decide() -> None:
    # A function when declares no reads, so load cannot see this read of the policy's ref.
    gated = Source({"t:x": t.string}, when=When(reads_the_policy, version="1"))
    loaded = load(dataclasses.replace(minimal(), sources={"s": gated}))
    record = decide_sync(loaded, {}, {"default": Scripted({"t:x": "a"})}).record
    (found,) = [item for item in record.log if item.get("ref") == "t:x"]
    assert found["marker"] == {
        "kind": "Unavailable",
        "cause": {"code": "undeclared-read", "refs": ["s", "d:decision"]},
    }


# ---------------------------------------------------------------------------------------------
# An async observer is refused at entry


async def async_observer(index: int, item: Any) -> None:
    return None


async def async_gen_observer(index: int, item: Any) -> Any:
    yield index


class AsyncCall:
    async def __call__(self, index: int, item: Any) -> None:
        return None

    async def method(self, index: int, item: Any) -> None:
        return None


class SyncCall:
    def __call__(self, index: int, item: Any) -> None:
        return None


def sync_observer(index: int, item: Any) -> None:
    return None


class PartialCall:
    """A sync callable whose ``__call__`` is no plain function: a partial, never bound."""

    __call__ = functools.partial(sync_observer)


def _deep(depth: int) -> Any:
    """A sync observer behind ``depth`` bound methods; nested partials flatten, so they cannot
    make a chain."""

    def ignore(*args: object) -> None:
        return None

    observer: Any = ignore
    for _ in range(depth):
        observer = types.MethodType(observer, object())
    return observer


@pytest.mark.parametrize(
    "observer",
    [
        async_observer,
        async_gen_observer,
        functools.partial(async_observer),
        functools.partial(functools.partial(async_observer)),
        AsyncCall(),
        AsyncCall().method,
        _deep(33),  # a chain past the bound counts as async
    ],
)
def test_an_async_observer_is_refused_by_decide_and_replay(observer: Any) -> None:
    with pytest.raises(EntryRefusal, match="async"):
        go(observer=observer)
    record = go().record
    with pytest.raises(EntryRefusal, match="async"):
        replay_sync(record, LOADED, observer=observer)


@pytest.mark.parametrize("observer", [sync_observer, SyncCall(), PartialCall(), _deep(31)])
def test_a_sync_observer_is_accepted(observer: Any) -> None:
    assert go(observer=observer).estimate == "archive"
