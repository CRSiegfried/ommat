# pyright: reportPrivateUsage=false
"""The binding kit in ``ommat.bindings``: ``extraction_value`` writes temporal values in
ommat's grammar, ``bounded_warnings`` makes any lines a conforming ``warnings``, ``json_value``
writes a need in record form, ``image_need`` reads an image need, ``prompt_text`` writes a
prompt's text and ``call_result`` folds each request's provenance into the call's."""

import datetime as dt
import json
from types import MappingProxyType
from typing import Any

import pytest

import ommat._binding_kit as core
import ommat.bindings
from ommat import Answer, AnswerKind, CallResult, ExtractionAnswer, ValueType
from ommat._interpret import MAX_WARNING_CODE_POINTS, MAX_WARNINGS, _record_form, warnings_conform
from ommat.bindings import (
    RequestProvenance,
    bounded_warnings,
    call_result,
    extraction_value,
    image_need,
    json_value,
    prompt_text,
)

UTC = dt.UTC
EAST = dt.timezone(dt.timedelta(hours=5, minutes=45))
WEST = dt.timezone(dt.timedelta(hours=-5, minutes=-30))


def test_the_package_re_exports_the_core_functions() -> None:
    assert ommat.bindings.__all__ == [
        "OracleBinding",
        "RequestProvenance",
        "bounded_warnings",
        "call_result",
        "candidate_of",
        "extraction_value",
        "image_need",
        "json_value",
        "prompt_text",
    ]
    assert ommat.bindings.__all__ == sorted(ommat.bindings.__all__)
    from ommat import _binding

    for name in ommat.bindings.__all__:
        home = _binding if name == "OracleBinding" else core
        assert getattr(ommat.bindings, name) is getattr(home, name)


def test_extraction_value_returns_the_five_members_in_order() -> None:
    got = extraction_value("A-1", None, 0.1, 0.9, (3, 6))
    assert got == {"value": "A-1", "guess": None, "p_null": 0.1, "confidence": 0.9, "span": [3, 6]}
    assert list(got) == ["value", "guess", "p_null", "confidence", "span"]


RENDERED: list[tuple[object, str, ValueType]] = [
    (dt.date(2026, 3, 3), "2026-03-03", ValueType.date),
    (dt.date(5, 1, 2), "0005-01-02", ValueType.date),
    (dt.time(9, 5, 7), "09:05:07", ValueType.time),
    (dt.time(9, 5, 7, 120), "09:05:07.000120", ValueType.time),
    (dt.time(9, 5, 7, 0, UTC), "09:05:07Z", ValueType.time),
    (dt.time(9, 5, 7, 0, EAST), "09:05:07+05:45", ValueType.time),
    (dt.time(9, 5, 7, 0, WEST), "09:05:07-05:30", ValueType.time),
    (dt.datetime(2026, 3, 3, 21, 0), "2026-03-03T21:00:00", ValueType.datetime),
    (
        dt.datetime(2026, 3, 3, 21, 0, 0, 500000, UTC),
        "2026-03-03T21:00:00.500000Z",
        ValueType.datetime,
    ),
    (dt.datetime(2026, 3, 3, 21, 0, 0, 0, WEST), "2026-03-03T21:00:00-05:30", ValueType.datetime),
    (dt.timedelta(0), "PT0S", ValueType.duration),
    (dt.timedelta(days=1), "P1D", ValueType.duration),
    (dt.timedelta(days=2, hours=3), "P2DT3H", ValueType.duration),
    (dt.timedelta(seconds=90), "PT1M30S", ValueType.duration),
    (dt.timedelta(hours=25), "P1DT1H", ValueType.duration),
    (dt.timedelta(hours=1, seconds=5), "PT1H5S", ValueType.duration),
]


@pytest.mark.parametrize(("value", "text", "value_type"), RENDERED)
def test_a_temporal_value_is_written_in_ommats_grammar(
    value: object, text: str, value_type: ValueType
) -> None:
    got = extraction_value(value, value, None, None, None)
    assert got["value"] == got["guess"] == text
    # The answer's constructor holds the grammar, so it is the oracle.
    answer = ExtractionAnswer(value_type, False, got["value"], None, None, None, None)
    assert answer.value == text


def test_a_list_or_tuple_is_written_member_by_member_as_a_list() -> None:
    days = (dt.date(2026, 3, 3), dt.date(2026, 3, 4))
    got = extraction_value(days, list(days), None, None, ((0, 7), [9, 16]))
    assert got["value"] == got["guess"] == ["2026-03-03", "2026-03-04"]
    assert got["span"] == [[0, 7], [9, 16]]
    assert extraction_value([], None, None, None, [])["span"] == []


@pytest.mark.parametrize(
    "value", ["text", 3.5, 7, None, True, dt.datetime, b"x", {"a": 1}], ids=repr
)
def test_any_other_value_passes_through_unchecked(value: object) -> None:
    assert extraction_value(value, None, None, None, None)["value"] == value


class _Day(dt.date):
    pass


def test_a_subclass_of_a_temporal_type_is_not_written() -> None:
    # The exact type chooses; interpret then judges the value.
    day = _Day(2026, 3, 3)
    assert extraction_value(day, None, None, None, None)["value"] is day


class _NoOffset(dt.tzinfo):
    def utcoffset(self, _: object) -> None:
        return None


REFUSED: list[tuple[object, str]] = [
    (dt.timedelta(seconds=-1), "is negative"),
    (dt.timedelta(seconds=1, microseconds=5), "fraction of a second"),
    (dt.time(9, 0, 0, 0, dt.timezone(dt.timedelta(seconds=30))), "whole number of minutes"),
    (
        dt.datetime(2026, 3, 3, 9, 0, 0, 0, dt.timezone(dt.timedelta(minutes=1, microseconds=1))),
        "whole number of minutes",
    ),
    (dt.time(9, 0, 0, 0, _NoOffset()), "gives no offset"),
    (dt.datetime(2026, 3, 3, 9, 0, 0, 0, _NoOffset()), "gives no offset"),
]


@pytest.mark.parametrize(("value", "message"), REFUSED)
def test_a_temporal_value_the_grammar_cannot_hold_is_refused(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extraction_value(value, None, None, None, None)
    with pytest.raises(ValueError, match=message):
        extraction_value(None, [value], None, None, None)


# --- bounded_warnings ---------------------------------------------------------------------


def test_lines_within_the_bounds_are_kept_in_order() -> None:
    lines = ["b", "a", "b"]
    assert bounded_warnings(iter(lines)) == ("b", "a", "b")
    assert bounded_warnings([]) == ()


def test_the_lines_stop_at_either_bound_with_a_count_of_those_left_out() -> None:
    many = [f"line {n}" for n in range(MAX_WARNINGS + 10)]
    kept = bounded_warnings(many)
    assert len(kept) == MAX_WARNINGS
    assert kept[-1] == "11 more lines left out"
    exact = [f"line {n}" for n in range(MAX_WARNINGS)]
    assert bounded_warnings(exact) == tuple(exact)
    long = ["x" * (MAX_WARNING_CODE_POINTS // 2)] * 3
    assert bounded_warnings(long) == (long[0], "2 more lines left out")
    near = ["x" * (MAX_WARNING_CODE_POINTS - 100), "y" * 80]
    assert bounded_warnings(near) == tuple(near)
    over = ["x" * (MAX_WARNING_CODE_POINTS - 100), "y" * 150]
    assert bounded_warnings(over) == (over[0], "1 more line left out")


def test_a_surrogate_is_escaped_as_the_ledger_escapes_it() -> None:
    assert bounded_warnings(["bad \udc80 byte"]) == ("bad \\udc80 byte",)


def test_the_bounds_are_counted_after_the_escapes() -> None:
    # Each surrogate is six code points once escaped.
    lines = ["\ud800" * 10_000, "\ud800" * 1_000]
    assert sum(map(len, lines)) <= MAX_WARNING_CODE_POINTS
    kept = bounded_warnings(lines)
    assert kept == ("\\ud800" * 10_000, "1 more line left out")
    assert warnings_conform(kept)


def test_a_line_of_whitespace_alone_is_left_out_and_not_counted() -> None:
    assert bounded_warnings(["", " \t\n", "kept", chr(0x2003)]) == ("kept",)
    many = [" "] * 5 + [f"line {n}" for n in range(MAX_WARNINGS + 1)]
    assert bounded_warnings(many)[-1] == "2 more lines left out"


class _Line(str):
    __slots__ = ()


def test_a_string_subclass_is_kept_as_its_string_data() -> None:
    (line,) = bounded_warnings([_Line("note")])
    assert type(line) is str


def test_a_member_that_is_not_a_string_is_refused() -> None:
    bad: Any = ["ok", 3]
    with pytest.raises(TypeError, match="warnings line 3 is not a string"):
        bounded_warnings(bad)


@pytest.mark.parametrize(
    "lines",
    [
        [],
        ["a"] * (MAX_WARNINGS * 2),
        ["x" * MAX_WARNING_CODE_POINTS] * 2,
        ["\udfff" * MAX_WARNING_CODE_POINTS],
        [" ", "y"],
    ],
    ids=["none", "many", "long", "escaped", "spaces"],
)
def test_what_it_returns_conforms(lines: list[str]) -> None:
    assert warnings_conform(bounded_warnings(lines))


# --- json_value ---------------------------------------------------------------------------


def test_a_need_is_written_in_record_form() -> None:
    topic = Answer(AnswerKind.choice, ("road", "other"), {"road": 0.75, "other": 0.25})
    found = ExtractionAnswer(ValueType.text, True, ["Elm Row"], None, 0.1, 0.9, [[12, 19]])
    need = {"t:a": "text", "t:b": (1, 2.5, None, True), "q:x": topic, "q:y": found}
    assert json_value(need) == {
        "t:a": "text",
        "t:b": [1, 2.5, None, True],
        "q:x": {
            "kind": "choice",
            "labels": ["road", "other"],
            "probabilities": {"road": 0.75, "other": 0.25},
        },
        "q:y": {
            "kind": "extraction",
            "value_type": "text",
            "is_list": True,
            "value": ["Elm Row"],
            "guess": None,
            "p_null": 0.1,
            "confidence": 0.9,
            "span": [[12, 19]],
        },
    }


def test_a_need_without_bytes_is_its_record_form() -> None:
    need = {"t:a": [{"b": (1, "x")}], "t:c": "y"}
    assert json_value(need) == _record_form(need)


@pytest.mark.parametrize(
    "need",
    [b"x", {"a": b"x"}, {"a": [(1, b"x")]}, [{"a": {"b": b""}}]],
    ids=["top", "member", "in-a-tuple-in-a-list", "deep"],
)
def test_bytes_at_any_depth_are_refused(need: object) -> None:
    with pytest.raises(TypeError, match="a need holds bytes, which a JSON part cannot carry"):
        json_value(need)


class _Text(str):
    __slots__ = ()


def test_any_other_value_is_as_given() -> None:
    proxy = MappingProxyType({"a": 1})
    text = _Text("x")
    written: Any = json_value({"p": proxy, "t": text, "s": frozenset[int]()})
    assert written["p"] is proxy
    assert written["t"] is text
    assert written["s"] == frozenset()


def test_the_record_form_still_writes_bytes_as_base64url() -> None:
    assert _record_form({"a": b"\x00\x01"}) == {"a": "AAE"}


# --- image_need ---------------------------------------------------------------------------

PNG = b"\x89PNG\r\n\x1a\n"


def test_an_image_need_is_its_media_type_and_bytes() -> None:
    assert image_need({"media_type": "image/png", "data": PNG}) == ("image/png", PNG)
    assert image_need({"data": b"", "media_type": "image/svg+xml"}) == ("image/svg+xml", b"")


class _Dict(dict[str, object]):
    pass


class _Bytes(bytes):
    pass


@pytest.mark.parametrize(
    "value",
    [
        None,
        PNG,
        "image/png",
        {"media_type": "image/png"},
        {"media_type": "image/png", "data": PNG, "name": "a.png"},
        {"media_type": "text/plain", "data": PNG},
        {"media_type": "IMAGE/PNG", "data": PNG},
        {"media_type": _Text("image/png"), "data": PNG},
        {"media_type": "image/png", "data": "iVBORw"},
        {"media_type": "image/png", "data": bytearray(PNG)},
        {"media_type": "image/png", "data": _Bytes(PNG)},
        _Dict(media_type="image/png", data=PNG),
        MappingProxyType({"media_type": "image/png", "data": PNG}),
    ],
    ids=[
        "none",
        "bytes",
        "str",
        "no-data",
        "extra-member",
        "not-an-image-type",
        "upper-case",
        "str-subclass",
        "data-a-str",
        "bytearray",
        "bytes-subclass",
        "dict-subclass",
        "mapping",
    ],
)
def test_any_other_value_is_not_an_image_need(value: object) -> None:
    assert image_need(value) is None


# --- prompt_text --------------------------------------------------------------------------


def test_a_str_or_none_is_as_given() -> None:
    assert prompt_text(None) is None
    assert prompt_text("") == ""
    text = _Text("asked")
    assert prompt_text(text) is text


def test_a_mapping_keeps_its_key_order_byte_for_byte() -> None:
    prompt = MappingProxyType(
        {
            "model": "m",
            "input": ({"role": "user", "content": MappingProxyType({"z": 1, "a": "é"})},),
            "answers": [2.5, None, True],
        }
    )
    assert prompt_text(prompt) == (
        '{"model": "m", "input": [{"role": "user", "content": {"z": 1, "a": "\\u00e9"}}], '
        '"answers": [2.5, null, true]}'
    )


def test_a_mapping_is_what_json_dumps_gives_in_the_order_given() -> None:
    prompt = {"b": [{"d": 1, "c": [1, 2]}], "a": "x"}
    assert prompt_text(prompt) == json.dumps(prompt)
    assert prompt_text({"a": "x", "b": [{"c": [1, 2], "d": 1}]}) != prompt_text(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        3,
        ["a"],
        {"a": b"x"},
        {"a": {1, 2}},
        {"a": [object()]},
        {1: "a"},
        {"a": [{None: 1}]},
        {"a": ({2: 1},)},
    ],
    ids=["number", "list", "bytes", "set", "object", "int key", "nested null key", "tuple key"],
)
def test_any_other_value_is_refused(prompt: object) -> None:
    with pytest.raises(TypeError):
        prompt_text(prompt)


# --- call_result --------------------------------------------------------------------------


def test_the_requests_provenance_is_joined_summed_and_sorted() -> None:
    first = RequestProvenance(
        revision="s1",
        elicitation="passthrough",
        prompt="p1",
        usage={"input_tokens": 10, "output_tokens": 2},
        attempts=1,
        truncated=("q:b", "q:a"),
    )
    second = RequestProvenance(
        revision="s1",
        elicitation="passthrough",
        prompt="p2",
        usage={"input_tokens": 5, "passes": 2},
        attempts=2,
        truncated=("q:a",),
    )
    values: dict[str, object] = {"q:a": {"yes": 1.0, "no": 0.0}}
    result = call_result(values, iter([first, second]), precision=2, warnings=("w",))
    assert result == CallResult(
        values,
        revision="s1",
        elicitation="passthrough",
        prompt="p1\np2",
        usage={"input_tokens": 15, "output_tokens": 2, "passes": 2},
        attempts=3,
        truncated=("q:a", "q:b"),
        warnings=("w",),
        precision=2,
    )
    assert result.values is values


def test_a_request_with_none_holds_no_line() -> None:
    result = call_result(
        {},
        [RequestProvenance(attempts=1), RequestProvenance(revision="s", attempts=1)],
        precision=None,
    )
    assert (result.revision, result.elicitation, result.prompt, result.usage) == (
        "s",
        None,
        None,
        None,
    )
    assert result.attempts == 2


@pytest.mark.parametrize(
    "requests",
    [[], [RequestProvenance()], [RequestProvenance(usage={}), RequestProvenance()]],
    ids=["none", "never-sent", "no-figures"],
)
def test_a_call_that_made_no_attempt_holds_null_attempts_and_precision(
    requests: list[RequestProvenance],
) -> None:
    result = call_result({}, requests, precision=2)
    assert (result.attempts, result.precision, result.usage) == (None, None, None)


def test_a_provenance_checks_nothing_when_built() -> None:
    odd: Any = 3
    assert RequestProvenance(revision=odd, attempts=odd).attempts == 3
