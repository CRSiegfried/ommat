"""Record format 10: a call's ``extras``, one open object per request it sent, kept outside
every hash; ``usage`` of ``input_tokens`` and ``output_tokens`` alone, ``usage.passes`` no
detail; and one provider and one requested model per call, each a name with no line boundary.
What interpret writes, what the ledger reads format by format, how ``call_result`` folds the
requests, and the conformance suite's checks."""

import asyncio
import dataclasses
import json
import sys
from typing import Any, cast

import pytest

from counting import SmallBounds
from ledger_logs import NORMAL_8, NORMAL_9, NORMAL_10, rehashed
from ommat import CallResult, InvalidRecord, decide_sync, read_record
from ommat._binding import OracleBinding
from ommat._provenance import (
    DETAILS,
    LINE_BOUNDARIES,
    MAX_EXTRAS,
    MAX_EXTRAS_CODE_POINTS,
    RECORDED_DETAILS,
    USAGE_KEYS,
    extras_conform,
    name_conforms,
)
from ommat.bindings import RequestProvenance, call_result
from ommat.bindings.conformance import (
    _default_batches,  # pyright: ignore[reportPrivateUsage]
    check,
)
from scripted import sleeping
from test_conformance import answering
from test_provenance import LOADED, YES, Batch, Colliding, call_item, calls_entry, turn

# ---------------------------------------------------------------------------------------------
# The provenance profile's lists


def test_usage_holds_two_keys_and_passes_is_only_a_recorded_detail() -> None:
    assert USAGE_KEYS == ("input_tokens", "output_tokens")
    assert "usage.passes" not in DETAILS
    assert "usage.passes" in RECORDED_DETAILS
    assert "calibrated" in RECORDED_DETAILS


def test_the_line_boundaries_are_those_splitlines_splits_at() -> None:
    splits = {chr(c) for c in range(sys.maxunicode + 1) if len(f"a{chr(c)}b".splitlines()) == 2}
    assert splits == set(LINE_BOUNDARIES)


@pytest.mark.parametrize("name", [None, "", "openai", "gpt-5.1-2025-11-13", "a b\tc"])
def test_a_name_without_a_line_boundary_conforms(name: str | None) -> None:
    assert name_conforms(name)


@pytest.mark.parametrize("boundary", list(LINE_BOUNDARIES))
def test_a_name_holding_a_line_boundary_does_not_conform(boundary: str) -> None:
    assert not name_conforms(f"openai{boundary}anthropic")
    assert not name_conforms(boundary)


def test_a_name_off_the_string_form_does_not_conform() -> None:
    assert not name_conforms(3)
    assert not name_conforms("\ud800")


# ---------------------------------------------------------------------------------------------
# extras' form


@pytest.mark.parametrize(
    ("extras", "attempts"),
    cast(
        "list[Any]",
        [
            (None, None),
            ((), None),
            ((), 0),
            ([], 1),
            (({"passes": 2},), 1),
            (({"passes": 2}, {}), 2),
            (({"a": [1, "x", None, True, {"b": 1.5}]},), 3),
            (({"a": 1},), None),
        ],
    ),
)
def test_an_extras_in_form_conforms(extras: Any, attempts: Any) -> None:
    assert extras_conform(extras, attempts)


class Sub(dict[str, Any]):
    pass


@pytest.mark.parametrize(
    ("extras", "attempts"),
    cast(
        "list[Any]",
        [
            ("a string", None),
            ({"passes": 2}, 1),
            (3, None),
            (({"passes": 2},), 0),
            (({"a": 1}, {"b": 2}), 1),
            (({1: "x"},), 1),
            (({Colliding("x"): 1},), 1),
            ((Sub(a=1),), 1),
            (([1],), 1),
            (("x",), 1),
            ((None,), 1),
            (({"a": b"x"},), 1),
            (({"a": float("nan")},), 1),
            (({"a": float("inf")},), 1),
            (({"a": {1, 2}},), 1),
            (({"a": 2**53},), 1),
            (({"a": "\ud800"},), 1),
            (({},) * (MAX_EXTRAS + 1), MAX_EXTRAS + 1),
            (({"a": "x" * (MAX_EXTRAS_CODE_POINTS - 9)},), 1),
        ],
    ),
    ids=[
        "str",
        "dict",
        "int",
        "past attempts 0",
        "past attempts",
        "int key",
        "str subclass key",
        "dict subclass",
        "list element",
        "str element",
        "null element",
        "bytes",
        "nan",
        "inf",
        "set",
        "past the integers",
        "surrogate",
        "too many",
        "too long",
    ],
)
def test_an_extras_off_its_form_does_not_conform(extras: Any, attempts: Any) -> None:
    assert not extras_conform(extras, attempts)


def test_the_extras_bounds_are_inclusive() -> None:
    assert extras_conform(({"a": 1},) * MAX_EXTRAS, MAX_EXTRAS)
    # [{"a":"…"}] is the text plus 10 code points.
    assert extras_conform(({"a": "x" * (MAX_EXTRAS_CODE_POINTS - 10)},), 1)


# ---------------------------------------------------------------------------------------------
# What interpret writes


def test_a_calls_extras_is_written_one_object_per_request_in_order() -> None:
    extras: tuple[dict[str, Any], ...] = ({"passes": 2, "note": "x"}, {})
    record = turn(attempts=2, extras=extras)
    assert call_item(record)["outcome"] == "returned"
    assert calls_entry(record)["extras"] == [{"note": "x", "passes": 2}, {}]
    assert record.log[0]["format"] == 11
    back = read_record(record.document)
    assert back.unhashed["calls"] == record.unhashed["calls"]


def test_a_list_extras_is_written_as_a_tuple_is() -> None:
    assert calls_entry(turn(attempts=1, extras=[{"a": 1}]))["extras"] == [{"a": 1}]


def test_extras_change_no_hash() -> None:
    one = turn(attempts=2, extras=({"passes": 1}, {}))
    other = turn(attempts=2, extras=({"passes": 9}, {"cached": True}))
    none = turn(attempts=2)
    assert one.record_hash == other.record_hash == none.record_hash
    assert one.log == none.log


@pytest.mark.parametrize("extras", [(), [], None, ({},), ({}, {})])
def test_an_empty_extras_is_not_written(extras: Any) -> None:
    entry = calls_entry(turn(attempts=2, extras=extras))
    assert "extras" not in entry
    assert "invalid" not in entry


@pytest.mark.parametrize(
    "extras",
    ["a string", ({"a": b"x"},), ({1: "x"},), ({"a": 1}, {"b": 2}, {"c": 3})],
    ids=["str", "bytes", "int key", "past attempts"],
)
def test_an_extras_off_its_form_is_kept_in_invalid_and_is_no_fault(extras: Any) -> None:
    record = turn(attempts=2, extras=extras)
    assert call_item(record)["outcome"] == "returned"
    entry = calls_entry(record)
    assert "extras" not in entry
    assert entry["invalid"]["extras"]
    assert entry["attempts"] == 2


def test_an_extras_key_of_the_binding_runs_no_code() -> None:
    record = turn(attempts=1, extras=({Colliding("x"): 1},))
    assert call_item(record)["outcome"] == "returned"
    assert "extras" in calls_entry(record)["invalid"]


def test_extras_beside_a_structural_fault_are_still_written() -> None:
    async def faulted(batch: Batch, *, seconds: float) -> object:
        return CallResult({"q:a/yes": YES, "t:nope": 1}, attempts=1, extras=({"passes": 3},))

    record = decide_sync(LOADED, {"text": "a"}, {"o": faulted}).record
    assert call_item(record)["outcome"]["failed"]["code"] == "unexpected-ref"
    assert calls_entry(record)["extras"] == [{"passes": 3}]


def test_a_late_result_keeps_no_extras() -> None:
    late = sleeping(1, CallResult({"q:a/yes": YES}, attempts=1, extras=({"passes": 3},)))
    from ommat import RunSettings

    record = decide_sync(
        LOADED, {"text": "a"}, {"o": late}, settings=RunSettings(timeout=0.01)
    ).record
    assert call_item(record)["outcome"]["failed"]["code"] == "timed-out"
    assert "extras" not in calls_entry(record)


def test_the_extras_written_are_a_copy_of_what_the_binding_returned() -> None:
    held: dict[str, Any] = {"nested": {"n": 1}}

    async def keeping(batch: Batch, *, seconds: float) -> object:
        return CallResult({"q:a/yes": YES}, attempts=1, extras=(held,))

    record = decide_sync(LOADED, {"text": "a"}, {"o": keeping}).record
    held["nested"]["n"] = 2
    assert calls_entry(record)["extras"] == [{"nested": {"n": 1}}]


def test_a_reused_call_takes_an_unaliased_copy_of_its_extras() -> None:
    from ommat._interpret import _Judged, _reused  # pyright: ignore[reportPrivateUsage]

    kept = _Judged(None, {}, {}, extras=[{"nested": {"n": 1}}])
    again = _reused(kept)
    assert again.extras == kept.extras
    assert again.extras is not kept.extras
    assert again.extras[0]["nested"] is not kept.extras[0]["nested"]


@pytest.mark.parametrize("boundary", ["\n", "\r", "\x85", "\u2028"])
@pytest.mark.parametrize("member", ["provider", "model_requested"])
def test_a_provider_or_requested_model_holding_a_line_boundary_is_invalid_return(
    member: str, boundary: str
) -> None:
    record = turn(attempts=1, revision="r1", **{member: f"one{boundary}two"})
    call = call_item(record)
    assert call["outcome"] == {"failed": {"code": "invalid-return", "refs": ["q:a/yes"]}}
    assert (call[member], call["revision"]) == (None, None)
    assert member in calls_entry(record)["invalid"]


def test_a_reported_model_and_an_elicitation_may_still_hold_lines() -> None:
    record = turn(model_reported="one\ntwo", elicitation="a\nb")
    call = call_item(record)
    assert call["outcome"] == "returned"
    assert (call["model_reported"], call["elicitation"]) == ("one\ntwo", "a\nb")


@pytest.mark.parametrize(
    "members",
    [
        {"usage": {"input_tokens": 1, "passes": 2}},
        {"missing": {"usage.passes": "not-reported"}},
    ],
    ids=["usage", "missing"],
)
def test_a_passes_a_binding_still_returns_is_nulled_into_invalid_with_no_error(
    members: dict[str, Any],
) -> None:
    record = turn(attempts=1, **members)
    assert call_item(record)["outcome"] == "returned"
    entry = calls_entry(record)
    (name,) = members
    assert name in entry["invalid"]
    assert entry.get(name) is None


def test_a_claim_on_passes_is_invalid_return() -> None:
    record = turn(attempts=1, usage={"passes": 2}, claimed=("usage.passes",))
    assert call_item(record)["outcome"]["failed"]["code"] == "invalid-return"


# ---------------------------------------------------------------------------------------------
# The ledger, format by format

CALL = {"start": "2026-10-20T00:00:00Z", "latency": 0.25, "usage": None, "attempts": 2}
LOGS = {8: NORMAL_8, 9: NORMAL_9, 10: NORMAL_10}


def doc_of(
    fmt: int, entry: dict[str, Any] | None = None, call: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A format-``fmt`` document whose ``ask_spam`` call, item 7, has ``entry`` as its
    ``calls`` entry and ``call``'s members."""
    log = json.loads(json.dumps(LOGS[fmt]))
    log[7].update(call or {})
    unhashed = {"calls": {"7": {**CALL, **(entry or {})}}}
    return rehashed({"log": log, "unhashed": unhashed})


PASSES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "usage": ({"usage": {"input_tokens": 3, "passes": 2}}, {}),
    "missing": ({"missing": {"usage.passes": "not-reported"}}, {}),
    "claimed": ({}, {"claimed": ["usage.passes"]}),
}


@pytest.mark.parametrize("place", list(PASSES))
@pytest.mark.parametrize("fmt", [8, 9])
def test_a_format_8_or_9_record_holding_passes_reads(fmt: int, place: str) -> None:
    entry, call = PASSES[place]
    record = read_record(doc_of(fmt, entry, call))
    assert record.log[0]["format"] == fmt


@pytest.mark.parametrize("place", list(PASSES))
def test_a_format_10_record_holding_passes_is_refused(place: str) -> None:
    entry, call = PASSES[place]
    with pytest.raises(InvalidRecord, match="passes"):
        read_record(doc_of(10, entry, call))


@pytest.mark.parametrize("fmt", [8, 9])
def test_extras_before_format_10_is_refused(fmt: int) -> None:
    with pytest.raises(InvalidRecord, match=f"holds extras, which a format-{fmt} record"):
        read_record(doc_of(fmt, {"extras": [{"passes": 1}]}))
    with pytest.raises(InvalidRecord, match=f"holds extras, which a format-{fmt} record"):
        read_record(doc_of(fmt, {"invalid": {"extras": "'x'"}}))


def test_a_format_10_extras_and_its_invalid_key_read() -> None:
    record = read_record(doc_of(10, {"extras": [{"passes": 1, "z": [1, {"b": None}]}, {}]}))
    assert record.unhashed["calls"]["7"]["extras"] == [{"passes": 1, "z": [1, {"b": None}]}, {}]
    read_record(doc_of(10, {"invalid": {"extras": "'x'"}}))


def test_a_format_10_extras_changes_no_record_hash() -> None:
    one = read_record(doc_of(10, {"extras": [{"passes": 1}]}))
    other = read_record(doc_of(10, {"extras": [{"passes": 2}, {"q": "text"}]}))
    none = read_record(doc_of(10))
    assert one.record_hash == other.record_hash == none.record_hash


def test_the_extras_canonical_order_sorts_keys_and_keeps_the_list() -> None:
    doc = doc_of(10, {"extras": [{"b": 1, "a": 2}, {"c": {"z": 0, "y": 1}}]})
    record = read_record(json.dumps(doc).encode())
    assert b'"extras":[{"a":2,"b":1},{"c":{"y":1,"z":0}}]' in record.document
    assert list(record.unhashed["calls"]["7"]["extras"][0]) == ["a", "b"]


@pytest.mark.parametrize(
    ("extras", "says"),
    cast(
        "list[Any]",
        [
            ([], "is empty"),
            ([{}], "is empty"),
            ([{}, {}], "is empty"),
            ({"passes": 1}, "is not a list"),
            ([1], "is not an object"),
            (["x"], "is not an object"),
            ([{"a": 1}, {"b": 2}, {"c": 3}], "more objects than its 2 attempts"),
        ],
    ),
)
def test_an_extras_off_its_shape_is_refused_at_the_read(extras: Any, says: str) -> None:
    with pytest.raises(InvalidRecord, match=says):
        read_record(doc_of(10, {"extras": extras}))


def test_an_extras_beside_a_null_attempts_reads() -> None:
    read_record(doc_of(10, {"attempts": None, "extras": [{"a": 1}, {"b": 2}, {"c": 3}]}))


def test_the_extras_are_size_tested_as_one_value_at_the_read(small_bounds: SmallBounds) -> None:
    doc = doc_of(10, {"extras": [{"a": "x" * 300}]})
    small_bounds()
    with pytest.raises(InvalidRecord, match="entry 0's 'extras' is outside the value domain"):
        read_record(doc)


@pytest.mark.parametrize("member", ["provider", "model_requested"])
def test_a_format_10_provider_or_requested_model_holding_a_line_is_refused(member: str) -> None:
    with pytest.raises(InvalidRecord, match=f"{member} holds a line boundary"):
        read_record(doc_of(10, call={member: "one\ntwo"}))
    read_record(doc_of(9, call={member: "one\ntwo"}))
    read_record(doc_of(10, call={"model_reported": "one\ntwo"}))


# ---------------------------------------------------------------------------------------------
# call_result's fold


def test_the_fold_writes_one_extras_object_per_sent_request_in_the_order_given() -> None:
    requests = [
        RequestProvenance(attempts=1, extras={"passes": 2}),
        RequestProvenance(attempts=0, extras={"passes": 9}),  # never sent
        RequestProvenance(attempts=2),
        RequestProvenance(attempts=1, extras={"passes": 1, "note": "x"}),
    ]
    result = call_result({}, requests, precision=None)
    assert result.extras == ({"passes": 2}, {}, {"passes": 1, "note": "x"})
    assert result.attempts == 4


def test_the_fold_writes_no_extras_when_every_request_reports_none() -> None:
    requests = [RequestProvenance(attempts=1), RequestProvenance(attempts=1, extras={})]
    assert call_result({}, requests, precision=None).extras == ()
    assert call_result({}, [], precision=None).extras == ()
    unsent = [RequestProvenance(attempts=0, extras={"a": 1})]
    assert call_result({}, unsent, precision=None).extras == ()


def test_the_fold_copies_each_extras_mapping_into_a_dict() -> None:
    held = {"passes": 2}
    result = call_result({}, [RequestProvenance(attempts=1, extras=held)], precision=None)
    assert type(result.extras[0]) is dict
    assert result.extras[0] is not held


@pytest.mark.parametrize(
    ("member", "says"), [("provider", "providers"), ("model_requested", "requested models")]
)
def test_the_fold_refuses_two_providers_or_requested_models(member: str, says: str) -> None:
    requests = [
        dataclasses.replace(RequestProvenance(attempts=1), **{member: "a"}),
        dataclasses.replace(RequestProvenance(attempts=1), **{member: "b"}),
    ]
    with pytest.raises(
        ValueError, match=f"call_result: a call's requests name two {says}, a and b"
    ):
        call_result({}, requests, precision=None)


@pytest.mark.parametrize("member", ["provider", "model_requested"])
def test_a_null_beside_a_provider_or_model_and_an_unsent_other_are_one(member: str) -> None:
    requests = [
        dataclasses.replace(RequestProvenance(attempts=1), **{member: "a"}),
        RequestProvenance(attempts=1),
        dataclasses.replace(RequestProvenance(attempts=1), **{member: "a"}),
        dataclasses.replace(RequestProvenance(attempts=0), **{member: "b"}),  # never sent
    ]
    assert getattr(call_result({}, requests, precision=None), member) == "a"


def test_the_fold_still_joins_reported_models_and_elicitations() -> None:
    requests = [
        RequestProvenance(attempts=1, model_reported="m1", elicitation="e1"),
        RequestProvenance(attempts=1, model_reported="m2", elicitation="e1"),
    ]
    result = call_result({}, requests, precision=None)
    assert (result.model_reported, result.elicitation) == ("m1\nm2", "e1")


# ---------------------------------------------------------------------------------------------
# The conformance suite


def changed(**members: Any) -> OracleBinding:
    async def binding(batch: Batch, *, seconds: float) -> object:
        result = await answering(batch, seconds=seconds)
        assert isinstance(result, CallResult)
        usage = cast("dict[str, Any]", members.get("usage") or {})
        given = {f"usage.{key}" for key in usage}
        missing = {d: r for d, r in (result.missing or {}).items() if d not in given}
        return dataclasses.replace(result, missing=missing or None, **members)

    return binding


def failures(binding: OracleBinding, **kwargs: Any) -> list[str]:
    return check(binding, batches=_default_batches()[:1], **kwargs)


def test_an_extras_in_form_passes() -> None:
    assert failures(changed(extras=({"passes": 1, "cached_tokens": 4},))) == []


def test_a_usage_holding_passes_fails_the_provenance_check() -> None:
    found = failures(changed(usage={"input_tokens": 1, "passes": 2}))
    assert any("usage is not null or a dict of input_tokens or output_tokens" in f for f in found)


@pytest.mark.parametrize(
    "extras",
    [({"a": b"x"},), ({"a": 1}, {"b": 2}), "x", ({1: 2},)],
    ids=["bytes", "past attempts", "str", "int key"],
)
def test_an_extras_off_its_form_fails_the_provenance_check(extras: Any) -> None:
    found = failures(changed(extras=extras))
    assert any(f.startswith("provenance:") and "extras" in f for f in found)


@pytest.mark.parametrize("member", ["provider", "model_requested"])
def test_a_provider_or_requested_model_holding_a_line_fails_the_provenance_check(
    member: str,
) -> None:
    found = failures(changed(**{member: "a\nb"}))
    assert any(f.startswith("provenance:") and f"{member} " in f for f in found)


def test_an_all_unsupported_result_with_extras_is_reported() -> None:
    from ommat import Cause, Unavailable

    async def refusing(batch: Batch, *, seconds: float) -> object:
        refs = [r for call in batch for r in (*call.refs, *call.questions)]
        marker = Unavailable(Cause("unsupported"))
        return CallResult(dict.fromkeys(refs, marker), extras=({"a": 1},), warnings=("no",))

    found = failures(refusing)
    assert any(f.startswith("unsupported-provenance:") and "extras" in f for f in found)


def test_the_deterministic_check_leaves_extras_out() -> None:
    calls = 0

    async def counting(batch: Batch, *, seconds: float) -> object:
        nonlocal calls
        calls += 1
        result = await answering(batch, seconds=seconds)
        assert isinstance(result, CallResult)
        return dataclasses.replace(result, extras=({"latency_ms": calls},))

    assert failures(counting, deterministic=True) == []


def test_answering_returns_a_call_result() -> None:  # the suite's own stand-in
    assert isinstance(asyncio.run(answering(_default_batches()[0], seconds=1.0)), CallResult)
