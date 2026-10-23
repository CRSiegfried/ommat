"""The oracle binding's types: what a call passes, ``SourceCall``
and ``RenderedQuestion``, and what it returns, ``CallResult``."""

import dataclasses
from typing import Any, cast

import pytest

import ommat
from ommat import AnswerKind, CallResult, RenderedQuestion, SourceCall, ValueType


def test_the_three_binding_types_are_public() -> None:
    assert {"CallResult", "SourceCall", "RenderedQuestion"} <= set(ommat.__all__)


def test_a_call_result_defaults_every_provenance_member_to_null() -> None:
    result = CallResult({"t:x": 1})
    assert result.values == {"t:x": 1}
    assert (result.snapshot, result.elicitation, result.prompt) == (None, None, None)
    assert (result.usage, result.attempts) == (None, None)
    assert result.truncated == ()
    assert (result.warnings, result.precision) == ((), None)


def test_a_call_result_takes_its_provenance_by_keyword_only() -> None:
    with pytest.raises(TypeError):
        CallResult({}, "snap")  # pyright: ignore[reportCallIssue]
    result = CallResult(
        {}, snapshot="m-1", elicitation="e", prompt="p", usage={"in": 3}, attempts=2
    )
    assert (result.snapshot, result.elicitation, result.prompt) == ("m-1", "e", "p")
    assert (result.usage, result.attempts) == ({"in": 3}, 2)


def test_a_call_result_checks_nothing_when_built() -> None:
    # Interpret tests it at the write, so a bad value fails its ref or the call by the fault scope,
    # not as a raise inside the binding.
    bad: Any = cast(Any, object())
    result = CallResult(bad, snapshot=bad, elicitation=bad, prompt=bad, usage=bad, attempts=True)
    assert result.values is bad
    assert result.attempts is True


@pytest.mark.parametrize(
    "value",
    [
        CallResult({}),
        SourceCall("s", None, {}, (), {}),
        RenderedQuestion(AnswerKind.noul, "Is it?", ("yes", "no")),
    ],
)
def test_the_binding_types_are_frozen(value: object) -> None:
    field = dataclasses.fields(cast(Any, value))[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(value, field, None)


def test_a_source_call_holds_what_the_protocol_names() -> None:
    question = RenderedQuestion(AnswerKind.choice, "Which?", ("a", "b"))
    call = SourceCall("ask", {"model": "m"}, {"t:state.subject": "hi"}, (), {"q:x": question})
    assert [f.name for f in dataclasses.fields(call)] == [
        "id",
        "payload",
        "needs",
        "refs",
        "questions",
    ]
    assert [f.name for f in dataclasses.fields(question)] == [
        "kind",
        "wording",
        "labels",
        "descriptions",
        "value_type",
        "is_list",
        "text",
        "reference_date",
    ]
    # The descriptions and the extraction members are keyword-only.
    assert [f.name for f in dataclasses.fields(question) if f.kw_only] == [
        "descriptions",
        "value_type",
        "is_list",
        "text",
        "reference_date",
    ]
    assert call.questions["q:x"].labels == ("a", "b")
    # A probability kind's question leaves the extraction members at their defaults.
    assert (question.value_type, question.is_list, question.text, question.reference_date) == (
        None,
        False,
        None,
        None,
    )


def test_a_rendered_question_built_without_descriptions_is_aligned_with_its_labels() -> None:
    question = RenderedQuestion(AnswerKind.choice, "Which?", ("a", "b"))
    assert question.descriptions == (None, None)
    noul = RenderedQuestion(AnswerKind.noul, "Is it?", ("yes", "no"))
    assert noul.descriptions == (None, None)
    extraction = RenderedQuestion(
        AnswerKind.extraction, "What?", (), value_type=ValueType.text, text="t:state.body"
    )
    assert extraction.descriptions == ()
    given = RenderedQuestion(AnswerKind.choice, "Which?", ("a", "b"), descriptions=("x", None))
    assert given.descriptions == ("x", None)


def test_a_rendered_question_takes_its_kind_and_value_type_as_a_member_or_its_string() -> None:
    question = RenderedQuestion(
        "extraction",  # pyright: ignore[reportArgumentType]
        "When?",
        (),
        value_type="date",  # pyright: ignore[reportArgumentType]
        text="t:state.body",
        reference_date="2026-03-10",
    )
    assert question.kind is AnswerKind.extraction
    assert question.value_type is ValueType.date


class _Text(str):
    __slots__ = ()


def test_a_rendered_question_stores_a_string_subclass_as_its_string_data() -> None:
    question = RenderedQuestion(AnswerKind.choice, _Text("Which?"), (_Text("a"),))
    assert type(question.wording) is str
    assert type(question.labels[0]) is str


def test_a_source_call_stores_a_string_subclass_key_as_its_string_data() -> None:
    call = SourceCall(_Text("s"), None, {_Text("t:state"): 1}, (_Text("t:x"),), {})
    assert [type(k) for k in (call.id, *call.needs, *call.refs)] == [str, str, str]
    asked = SourceCall("s", None, {}, (), {_Text("q:x"): Q})
    assert [type(k) for k in asked.questions] == [str]


A: Any = AnswerKind.choice
X: dict[str, Any] = {"value_type": ValueType.text, "text": "t:state.body"}


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "message"),
    [
        ((3, "Which?", ("a",)), {}, TypeError, "kind 3 is not a member of AnswerKind"),
        (("pick", "Which?", ("a",)), {}, ValueError, "names no AnswerKind"),
        ((A, b"Which?", ("a",)), {}, TypeError, "wording <bytes> is not a string"),
        ((A, "Which?", ["a"]), {}, TypeError, "labels <list> is not a tuple"),
        ((A, "Which?", ("a", 1)), {}, TypeError, "a member of labels 1 is not a string"),
        ((AnswerKind.noul, "Is it?", ("no", "yes")), {}, ValueError, "labels are not yes and no"),
        ((AnswerKind.extraction, "What?", ("a",)), X, ValueError, "holds labels"),
        ((A, "Which?", ("a",)), {"descriptions": ["x"]}, TypeError, "descriptions <list>"),
        ((A, "Which?", ("a", "b")), {"descriptions": ("x",)}, ValueError, "as long as labels"),
        ((A, "Which?", ("a",)), {"descriptions": (1,)}, TypeError, "a description 1 is not"),
        ((A, "Which?", ("a",)), {"value_type": ValueType.text}, ValueError, "sets value_type"),
        ((A, "Which?", ("a",)), {"is_list": True}, ValueError, "sets value_type"),
        ((A, "Which?", ("a",)), {"is_list": 0}, TypeError, "is_list 0 is not a bool"),
        ((A, "Which?", ("a",)), {"text": "t:state.body"}, ValueError, "sets value_type"),
        ((A, "Which?", ("a",)), {"reference_date": "2026"}, ValueError, "sets value_type"),
        ((A, "Which?", ("a",)), {"text": 3}, TypeError, "text 3 is not a string"),
        ((AnswerKind.extraction, "What?", ()), {"text": "t:x"}, ValueError, "lacks its value_type"),
        ((AnswerKind.extraction, "What?", ()), {**X, "value_type": "day"}, ValueError, "names no"),
        ((AnswerKind.extraction, "What?", ()), {**X, "is_list": 1}, TypeError, "is_list 1"),
        ((AnswerKind.extraction, "What?", ()), {**X, "text": None}, ValueError, "or text"),
        ((AnswerKind.extraction, "What?", ()), {**X, "reference_date": 5}, TypeError, "date 5"),
    ],
)
def test_a_rendered_question_refuses_a_field_out_of_shape(
    args: tuple[Any, ...], kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        RenderedQuestion(*args, **kwargs)


Q = RenderedQuestion(AnswerKind.noul, "Is it?", ("yes", "no"))
E: dict[Any, Any] = {}
SOURCE_CALLS: list[tuple[tuple[Any, ...], type[Exception], str]] = [
    ((1, None, E, (), E), TypeError, "id 1 is not a string"),
    (("s", None, [], (), E), TypeError, "needs <list> is not a dict"),
    (("s", None, {1: "x"}, (), E), TypeError, "needs holds a key"),
    (("s", None, E, ["t:x"], E), TypeError, "refs <list> is not a tuple"),
    (("s", None, E, (1,), E), TypeError, "a member of refs 1"),
    (("s", None, E, (), [Q]), TypeError, "questions <list> is not a dict"),
    (("s", None, E, (), {1: Q}), TypeError, "questions holds a key"),
    (("s", None, E, (), {"q:x": "Is it?"}), TypeError, "not a RenderedQuestion"),
    (("s", None, E, ("t:x",), {"q:x": Q}), ValueError, "both t: refs and questions"),
]


@pytest.mark.parametrize(("args", "error", "message"), SOURCE_CALLS)
def test_a_source_call_refuses_a_field_out_of_shape(
    args: tuple[Any, ...], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SourceCall(*args)


def test_a_source_call_leaves_its_payload_unchecked() -> None:
    payload = object()
    assert SourceCall("s", payload, {}, (), {}).payload is payload
