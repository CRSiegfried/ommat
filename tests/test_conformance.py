# pyright: reportPrivateUsage=false
"""The binding conformance suite, ``ommat.bindings.conformance.check``: the bindings ommat ships
pass it, and a binding that breaks each rule fails the check that names it."""

import asyncio
import dataclasses
from typing import Any

import pytest

import helpdesk_design
import helpdesk_run
import street_reports_design as works
import test_street_reports as street
from ommat import (
    CallResult,
    Cause,
    Failed,
    LoadedDesign,
    PlanKind,
    SourceCall,
    Unavailable,
    Unsure,
    decide_sync,
    from_record,
    load,
)
from ommat._binding import OracleBinding
from ommat._fixture import FixtureBinding, RecordedCalls
from ommat._interpret import MAX_WARNINGS
from ommat.bindings import image_need
from ommat.bindings.conformance import _default_batches, check

Batch = tuple[SourceCall, ...]
BindingFn = OracleBinding


def capturing(inner: BindingFn, seen: list[Batch]) -> BindingFn:
    async def binding(batch: Batch, *, seconds: float) -> object:
        seen.append(batch)
        return await inner(batch, seconds=seconds)

    return binding


def street_turn(name: str) -> tuple[LoadedDesign, Any, list[Batch]]:
    """A street-reports turn decided live, its record and every batch its bindings received."""
    loaded = load(works.DESIGN)
    report, given, open_orders = street.TURNS[name]
    seen: list[Batch] = []
    inner = street.fixed(given(), open_orders, [])
    binding = capturing(inner, seen)
    state = {"report": report, "filed_on": street.FILED_ON}
    decision = decide_sync(loaded, state, dict.fromkeys(street.ORACLES, binding))
    return loaded, decision, seen


# --- the bindings ommat ships ---------------------------------------------------------------


@pytest.mark.parametrize("name", list(street.TURNS))
def test_from_record_passes_on_the_batches_its_record_answers(name: str) -> None:
    loaded, decision, seen = street_turn(name)
    assert check(from_record(decision.record, loaded), batches=seen, deterministic=True) == []


@pytest.mark.parametrize("name", list(street.TURNS))
def test_the_fixture_binding_passes_on_the_batches_its_record_answers(name: str) -> None:
    loaded, decision, seen = street_turn(name)
    fixture = FixtureBinding(RecordedCalls(decision.record.log), loaded)
    assert check(fixture, batches=seen, deterministic=True) == []


def test_from_record_passes_on_the_suites_own_batches() -> None:
    # The record holds none of the suite's sources, so every ref is unsupported, with a reason.
    loaded, decision, _ = street_turn("execute")
    assert check(from_record(decision.record, loaded), deterministic=True) == []


def test_the_tutorials_binding_passes_on_its_designs_batches_and_the_suites() -> None:
    loaded = load(helpdesk_design.DESIGN)
    seen: list[Batch] = []
    binding = capturing(helpdesk_run.fixed_answers, seen)
    for message in helpdesk_run.MESSAGES:
        bindings = {"classifier": binding, "reader": binding}
        decide_sync(loaded, {"message": message}, bindings)
    kinds = {q.kind for batch in seen for call in batch for q in call.questions.values()}
    assert kinds == {"noul", "choice", "extraction"}
    assert check(helpdesk_run.fixed_answers, batches=seen, deterministic=True) == []
    assert check(helpdesk_run.fixed_answers, deterministic=True) == []


def test_every_batch_interpret_builds_over_the_example_designs_passes() -> None:
    # The producer side: a turn over each answer kind, described and undescribed labels, a
    # reference date, a list and a t: source raises nothing, and its batches are ones a
    # conforming binding answers.
    seen: list[Batch] = []
    for name in street.TURNS:
        loaded, decision, batches = street_turn(name)
        assert decision.plan.kind is not PlanKind.broken
        seen += batches
        assert check(from_record(decision.record, loaded), batches=batches) == []
    kinds = {q.kind for batch in seen for call in batch for q in call.questions.values()}
    assert kinds == {"noul", "choice", "score", "extraction"}
    assert any(call.refs for batch in seen for call in batch)


# --- the suite's own batches --------------------------------------------------------------


def test_the_suites_batches_hold_each_kind_alone_and_then_together() -> None:
    batches = _default_batches()
    assert [len(b) for b in batches] == [1, 1, 1, 1, 1, 5]
    together = batches[-1]
    assert [s.id for s in together] == sorted(s.id for s in together)
    kinds = [q.kind for s in together for q in s.questions.values()]
    assert sorted(kinds) == ["choice", "extraction", "extraction", "noul", "score"]
    for source in together:
        assert list(source.questions) == sorted(source.questions)
        for question in source.questions.values():
            if question.text is not None:
                assert question.text in source.needs
    (tool,) = (s for s in together if s.refs)
    assert tool.questions == {}


async def answering(batch: Batch, *, seconds: float) -> object:
    """A conforming binding: each question answered by its kind, each t: ref a value."""
    values: dict[str, Any] = {}
    for call in batch:
        for ref in call.refs:
            values[ref] = []
        for ref, question in call.questions.items():
            if question.kind == "extraction":
                values[ref] = {
                    "value": None,
                    "guess": None,
                    "p_null": 0.9,
                    "confidence": None,
                    "span": None,
                }
            else:
                share = 1 / len(question.labels)
                values[ref] = dict.fromkeys(question.labels, share)
    return CallResult(values, warnings=("answered by a test",), precision=2)


def test_a_conforming_binding_passes() -> None:
    assert check(answering, deterministic=True) == []


def test_a_binding_that_refuses_everything_passes_with_null_provenance() -> None:
    async def refusing(batch: Batch, *, seconds: float) -> object:
        refs = [r for call in batch for r in (*call.refs, *call.questions)]
        marker = Unavailable(Cause("unsupported"))
        return CallResult(dict.fromkeys(refs, marker), warnings=("no provider here",))

    assert check(refusing) == []


def test_a_failed_call_passes_but_is_never_deterministic() -> None:
    async def failing(_: Batch, *, seconds: float) -> object:
        return Failed(Cause("raised"))

    assert check(failing) == []
    assert check(failing, batches=_default_batches()[:1], deterministic=True) == [
        "deterministic: batch 0: a call did not return a CallResult"
    ]


# --- each check fails a binding that breaks it --------------------------------------------


def changed(**members: Any) -> BindingFn:
    """``answering``'s result with ``members`` replaced."""

    async def binding(batch: Batch, *, seconds: float) -> object:
        result = await answering(batch, seconds=seconds)
        assert isinstance(result, CallResult)
        return dataclasses.replace(result, **members)

    return binding


ONE = _default_batches()[2:3]  # the noul source alone
NOUL = "q:noul/hazard"


async def raising(_: Batch, *, seconds: float) -> object:
    raise RuntimeError("down")


async def plain(_: Batch, *, seconds: float) -> object:
    return {NOUL: {"yes": 0.5, "no": 0.5}}


async def failed_unsupported(_: Batch, *, seconds: float) -> object:
    return Failed(Cause("unsupported"))


async def missing(_: Batch, *, seconds: float) -> object:
    return CallResult({})


async def extra(_: Batch, *, seconds: float) -> object:
    return CallResult({NOUL: {"yes": 0.5, "no": 0.5}, "q:other": {}})


async def not_a_dict(_: Batch, *, seconds: float) -> object:
    return CallResult([NOUL])  # pyright: ignore[reportArgumentType]


async def bad_key(_: Batch, *, seconds: float) -> object:
    return CallResult({NOUL: {"yes": 1.0, "no": 0.0}, 3: 1})  # pyright: ignore[reportArgumentType]


async def unsure(_: Batch, *, seconds: float) -> object:
    return CallResult({NOUL: Unsure(Cause("abstained"))})


async def not_returned(_: Batch, *, seconds: float) -> object:
    return CallResult({NOUL: Unavailable(Cause("not-returned"))})


async def unsupported_with_a_snapshot(_: Batch, *, seconds: float) -> object:
    marker = Unavailable(Cause("unsupported"))
    return CallResult({NOUL: marker}, snapshot="model-1", truncated=(NOUL,))


class Counter:
    def __init__(self) -> None:
        self.n = 0

    async def __call__(self, _: Batch, *, seconds: float) -> object:
        self.n += 1
        return CallResult({NOUL: {"yes": self.n / 10, "no": 1 - self.n / 10}})


@pytest.mark.parametrize(
    ("binding", "failures"),
    [
        (raising, ["returns: batch 0: raised RuntimeError"]),
        (plain, ["returns: batch 0: returned <dict>, not a CallResult or a Failed"]),
        (
            failed_unsupported,
            ["returns: batch 0: a Failed whose cause is not one a binding may fail a call with"],
        ),
        (missing, ["values: batch 0: no value for q:noul/hazard"]),
        (extra, ["values: batch 0: a key the batch does not provide: q:other"]),
        (not_a_dict, ["values: batch 0: values is <list>, not a dict"]),
        (bad_key, ["values: batch 0: a key 3 is not a string"]),
        (unsure, ["markers: batch 0: q:noul/hazard holds a marker a binding may not return"]),
        (
            not_returned,
            ["markers: batch 0: q:noul/hazard holds a marker a binding may not return"],
        ),
        (changed(warnings=["a list"]), ["warnings: batch 0: warnings does not conform"]),
        (changed(warnings=(" ",)), ["warnings: batch 0: warnings does not conform"]),
        (
            changed(warnings=("x",) * (MAX_WARNINGS + 1)),
            ["warnings: batch 0: warnings does not conform"],
        ),
        (changed(precision=18), ["precision: batch 0: precision 18 does not conform"]),
        (changed(precision=True), ["precision: batch 0: precision True does not conform"]),
        (
            unsupported_with_a_snapshot,
            [
                "unsupported-provenance: batch 0: every ref is unsupported, yet snapshot, "
                "truncated set"
            ],
        ),
    ],
    ids=lambda v: getattr(v, "__name__", None) if callable(v) else None,
)
def test_a_binding_that_breaks_a_rule_fails_its_check(
    binding: BindingFn, failures: list[str]
) -> None:
    assert check(binding, batches=ONE) == failures


class _Ref(str):
    __slots__ = ()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


def test_two_keys_of_one_string_data_fail_values() -> None:
    async def twice(_: Batch, *, seconds: float) -> object:
        answer = {"yes": 0.5, "no": 0.5}
        return CallResult({NOUL: answer, _Ref(NOUL): answer})

    assert check(twice, batches=ONE) == [
        "values: batch 0: two keys of one string data, q:noul/hazard"
    ]


def test_a_binding_that_answers_differently_each_call_is_not_deterministic() -> None:
    assert check(Counter(), batches=ONE) == []
    assert check(Counter(), batches=ONE, deterministic=True) == [
        "deterministic: batch 0: two calls with one batch returned different results"
    ]


def test_a_raise_on_the_second_call_is_reported_once() -> None:
    calls: list[int] = []

    async def once(batch: Batch, *, seconds: float) -> object:
        calls.append(1)
        if len(calls) > 1:
            raise ValueError("again")
        return await answering(batch, seconds=seconds)

    assert check(once, batches=ONE, deterministic=True) == ["returns: batch 0: raised ValueError"]


def test_check_refuses_to_run_inside_a_running_event_loop() -> None:
    async def inside() -> None:
        with pytest.raises(RuntimeError, match="inside a running event loop"):
            check(answering)

    asyncio.run(inside())


# --- format 4: seconds, the per-ref causes and the answers' form ----------------------------


def test_a_binding_without_the_seconds_keyword_fails_before_any_call() -> None:
    calls: list[Batch] = []

    async def old(batch: Batch) -> object:
        calls.append(batch)
        return CallResult({})

    assert check(old) == ["seconds: the binding does not take seconds as a keyword"]  # pyright: ignore[reportArgumentType]
    assert calls == []


class _Unreadable:
    """A binding whose signature cannot be read, so the call itself is the test."""

    __signature__ = 3

    async def __call__(self, batch: Batch, *, seconds: float) -> object:
        return await answering(batch, seconds=seconds)


def test_a_binding_whose_signature_cannot_be_read_is_called() -> None:
    assert check(_Unreadable(), batches=ONE) == []


def test_each_call_receives_the_suites_seconds() -> None:
    got: list[float] = []

    async def noting(batch: Batch, *, seconds: float) -> object:
        got.append(seconds)
        return await answering(batch, seconds=seconds)

    assert check(noting, batches=ONE, deterministic=True) == []
    assert got == [60.0, 60.0]


def per_ref(code: str, *lines: str, attempts: int | None = 2) -> BindingFn:
    async def binding(batch: Batch, *, seconds: float) -> object:
        marker = Unavailable(Cause(code))
        return CallResult({NOUL: marker}, warnings=lines, attempts=attempts)

    return binding


@pytest.mark.parametrize("code", ["timed-out", "raised"])
def test_a_per_ref_timed_out_or_raised_needs_a_warnings_line(code: str) -> None:
    assert check(per_ref(code, "the request failed: HTTP 504"), batches=ONE) == []
    assert check(per_ref(code), batches=ONE) == [
        "per-ref-cause: batch 0: a timed-out or raised ref with no warnings line"
    ]
    # A binding that called no provider, as a replay's fixture, owes no line.
    assert check(per_ref(code, attempts=None), batches=ONE) == []


async def misshapen(batch: Batch, *, seconds: float) -> object:
    values: dict[str, Any] = {}
    for call in batch:
        for ref in call.refs:
            values[ref] = []
        for ref, question in call.questions.items():
            if question.kind == "extraction":
                # A span past the end of the text.
                values[ref] = {
                    "value": None,
                    "guess": "x",
                    "p_null": 0.5,
                    "confidence": None,
                    "span": [0, 10_000],
                }
            else:
                values[ref] = {"not a label": 1.0}
    return CallResult(values)


def test_a_value_not_in_its_kinds_form_fails_answers() -> None:
    failures = check(misshapen, batches=_default_batches()[-1:])
    assert failures == [
        f"answers: batch 0: {ref} is not in its kind's form"
        for ref in [
            "q:choice/asset",
            "q:extraction/noticed",
            "q:extraction/places",
            "q:noul/hazard",
            "q:score/severity",
        ]
    ]


def test_the_suites_batches_hold_an_image_need() -> None:
    (noul,) = (s for s in _default_batches()[-1] if s.id == "noul")
    photo = noul.needs["t:state.photo"]
    assert set(photo) == {"media_type", "data"}
    assert photo["media_type"] == "image/png"
    assert type(photo["data"]) is bytes
    assert image_need(photo) is not None
