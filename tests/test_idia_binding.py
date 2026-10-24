"""``ommat.bindings.idia``: each source as one idia request, each answer mapped to the
value ommat takes, each refusal and marker as the binding's decisions say, over idia's
scripted leaf and ``Dispatcher``."""

import asyncio
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

idia = pytest.importorskip("idia")

from idia import (  # noqa: E402
    UNBOUNDED,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Fault,
    ImagePart,
    JsonPart,
    LabelAnswer,
    Notice,
    NoulAnswer,
    ProviderError,
    Refusal,
    RetryPolicy,
    Rounding,
    Router,
    Rule,
    RuleFailure,
    Span,
    TextPart,
    Usage,
    WireRequest,
)
from idia.providers.scripted import Reply, Scripted  # noqa: E402

from ommat import (  # noqa: E402
    Answer,
    AnswerKind,
    CallResult,
    Cause,
    Failed,
    RenderedQuestion,
    SourceCall,
    Unavailable,
    ValueType,
)
from ommat import ExtractionAnswer as OmmatExtraction  # noqa: E402
from ommat.bindings.conformance import check  # noqa: E402
from ommat.bindings.idia import oracle_binding  # noqa: E402

MODULE = Path(__file__).parent.parent / "src" / "ommat" / "bindings" / "idia.py"
Batch = tuple[SourceCall, ...]

NOUL = RenderedQuestion(AnswerKind.noul, "Is anyone in danger?", ("yes", "no"))
ASSET = RenderedQuestion(
    AnswerKind.choice,
    "Which asset?",
    ("road", "other"),
    descriptions=("The road surface", None),
)
LEVEL = RenderedQuestion(AnswerKind.score, "How serious?", ("minor", "serious"))
TEXT = "The lamp on Elm Row has been out since 3 March; seen at 9 pm on Quay Street too."


def extraction(
    value_type: ValueType = ValueType.text,
    *,
    is_list: bool = False,
    text: str = "t:state.report",
    reference_date: str | None = None,
) -> RenderedQuestion:
    return RenderedQuestion(
        AnswerKind.extraction,
        "What does the report name?",
        (),
        value_type=value_type,
        is_list=is_list,
        text=text,
        reference_date=reference_date,
    )


def caps(**overrides: Any) -> Capabilities:
    """A full leaf record taking every kind ommat asks, text, JSON and images."""
    fields: dict[str, Any] = {
        "kinds": frozenset({"noul", "choice", "score", "extraction"}),
        "modalities": frozenset({"text", "json", "image"}),
        "image_carriers": frozenset({"array"}),
        "max_input_tokens": 65_536,
        "max_question_tokens": 65_536,
        "max_body_bytes": UNBOUNDED,
        "max_questions": UNBOUNDED,
        "max_images": 4,
        "max_images_per_question": 4,
        "max_image_bytes": UNBOUNDED,
        "max_total_image_bytes": UNBOUNDED,
        "max_labels": 20,
        "max_levels": 20,
        "max_total_options": UNBOUNDED,
        "min_levels": 1,
        "min_image_side": 0,
        "image_media_types": None,
        "key_patterns": (),
        "recounts_state_per_question": False,
        "returns_refusal": True,
        "calibrated": False,
        "snapshot_pinnable": True,
        "truncates_silently": False,
        "confidence": {},
        "rounding": Rounding(probability_decimals=2, score_decimals=2),
        "opt_in_features": frozenset(),
        "enabled_features": frozenset(),
        "feature_kinds": {},
        "feature_modalities": {},
        "carries_noul_descriptions": True,
        "carries_structured_instructions": True,
    }
    fields.update(overrides)
    return Capabilities(**fields)


def answer_all(wire_request: WireRequest, deadline: Deadline) -> dict[str, Any]:
    """An answer to each question by its kind."""
    out: dict[str, Any] = {}
    for q in wire_request.questions:
        if q.kind == "noul":
            out[q.key] = NoulAnswer(probability=0.7)
        elif q.kind in ("choice", "score"):
            share = 1 / len(q.labels)
            out[q.key] = LabelAnswer(probabilities=tuple((label.name, share) for label in q.labels))
        else:
            out[q.key] = ExtractionAnswer(value=None, p_null=0.9)
    return out


def replying(answers: Any) -> Any:
    """A script answering every call with ``answers``."""

    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return answers

    return script


def built(script: Any = answer_all, **overrides: Any) -> tuple[Any, Scripted]:
    leaf = Scripted(capabilities=caps(**overrides), script=script)
    return oracle_binding(Dispatcher(leaf, retry=RetryPolicy.without_delay())), leaf


def source(
    sid: str = "triage",
    questions: dict[str, RenderedQuestion] | None = None,
    needs: dict[str, Any] | None = None,
    refs: tuple[str, ...] = (),
) -> SourceCall:
    asked = {"q:triage/hazard": NOUL} if questions is None and not refs else questions or {}
    return SourceCall(sid, None, {"t:state.report": TEXT} if needs is None else needs, refs, asked)


def call(binding: Any, *sources: SourceCall, seconds: float = 60.0) -> CallResult:
    return asyncio.run(binding(sources, seconds=seconds))


def code(result: CallResult, ref: str) -> str:
    marker = result.values[ref]
    assert isinstance(marker, Unavailable)
    return marker.cause.code.value


# ---------------------------------------------------------------------------------------------
# Built


def test_a_provider_that_is_not_a_dispatcher_is_refused() -> None:
    leaf = Scripted(capabilities=caps(), script=answer_all)
    with pytest.raises(ValueError, match="takes a Dispatcher over one leaf, not a Scripted"):
        oracle_binding(leaf)
    router = Router({"noul": Dispatcher(leaf)})
    with pytest.raises(ValueError, match="not a Router"):
        oracle_binding(router)


def test_a_dispatcher_that_re_asks_is_refused() -> None:
    leaf = Scripted(capabilities=caps(), script=answer_all)
    with pytest.raises(ValueError, match="re-ask off"):
        oracle_binding(Dispatcher(leaf, reask=True))


class Trim:
    name = "trim"

    def apply(self, state: Any, limit: int, estimator: Any) -> Any:
        raise AssertionError("never applied")


def test_a_dispatcher_whose_overflow_policy_trims_is_refused() -> None:
    leaf = Scripted(capabilities=caps(), script=answer_all)
    with pytest.raises(ValueError, match="overflow policy is Fail"):
        oracle_binding(Dispatcher(leaf, overflow=Trim()))


@pytest.mark.parametrize("decimals", [18, -1])
def test_a_leaf_declaring_decimals_no_precision_holds_is_refused(decimals: int) -> None:
    rounding = Rounding(probability_decimals=decimals, score_decimals=None)
    leaf = Scripted(capabilities=caps(rounding=rounding), script=answer_all)
    with pytest.raises(ValueError, match=f"probability decimals are 0 to 17, not {decimals}"):
        oracle_binding(Dispatcher(leaf))


@pytest.mark.parametrize("decimals", [0, 17, None])
def test_a_leaf_declaring_decimals_precision_holds_is_taken(decimals: int | None) -> None:
    rounding = Rounding(probability_decimals=decimals, score_decimals=None)
    binding, _ = built(rounding=rounding)
    expected = "unrounded" if decimals is None else decimals  # idia's None is unrounded
    assert call(binding, source()).precision == expected


# ---------------------------------------------------------------------------------------------
# Mapping in


def test_a_label_source_sends_its_needs_as_one_json_part_keyed_by_ommats_refs() -> None:
    binding, leaf = built()
    needs = {"t:state.report": TEXT, "t:state.filed_on": "2026-03-10"}
    questions = {"q:triage/asset": ASSET, "q:triage/hazard": NOUL, "q:triage/severity": LEVEL}
    call(binding, source(questions=questions, needs=needs))
    (sent,) = leaf.calls
    (part,) = sent.wire_request.state.parts
    assert isinstance(part, JsonPart)
    assert dict(part.value) == needs  # type: ignore[arg-type]
    assert [q.key for q in sent.wire_request.questions] == list(questions)
    asset = sent.wire_request.questions[0]
    assert [(label.name, label.description) for label in asset.labels] == [
        ("road", "The road surface"),
        ("other", None),
    ]
    assert sent.wire_request.questions[1].labels == ()


def test_a_q_need_and_a_list_need_are_sent_as_json() -> None:
    binding, leaf = built()
    topic = Answer(AnswerKind.choice, ("road", "other"), {"road": 0.75, "other": 0.25})
    found = OmmatExtraction(ValueType.text, True, ["Elm Row"], None, 0.1, 0.9, [[12, 19]])
    needs = {"q:triage/asset": topic, "q:details/places": found, "t:state.tags": ["a", "b"]}
    call(binding, source(needs=needs))
    (part,) = leaf.calls[0].wire_request.state.parts
    assert isinstance(part, JsonPart)
    assert part.value == {
        "q:triage/asset": {
            "kind": "choice",
            "labels": ("road", "other"),
            "probabilities": {"road": 0.75, "other": 0.25},
        },
        "q:details/places": {
            "kind": "extraction",
            "value_type": "text",
            "is_list": True,
            "value": ("Elm Row",),
            "guess": None,
            "p_null": 0.1,
            "confidence": 0.9,
            "span": ((12, 19),),
        },
        "t:state.tags": ("a", "b"),
    }


def test_an_extraction_source_sends_its_text_first_and_its_date_as_an_option() -> None:
    binding, leaf = built()
    needs = {"t:state.report": TEXT, "t:state.filed_on": "2026-03-10"}
    noticed = extraction(ValueType.date, reference_date="2026-03-10")
    call(binding, source("details", {"q:details/noticed": noticed}, needs))
    (sent,) = leaf.calls
    assert sent.wire_request.state.parts == (TextPart(text=TEXT),)
    assert dict(sent.wire_request.options) == {"reference_date": "2026-03-10"}
    (question,) = sent.wire_request.questions
    assert (question.kind, question.value_type, question.is_list) == ("extraction", "date", False)


def test_an_image_need_is_an_image_part_after_the_first() -> None:
    binding, leaf = built()
    png = b"\x89PNG\r\n\x1a\n"
    photo = {"media_type": "image/png", "data": png}
    record = {"media_type": "text/plain", "body": "not an image"}
    needs = {"t:state.report": TEXT, "t:state.photo": photo, "t:state.note": record}
    call(binding, source(needs=needs))
    call(binding, source("details", {"q:details/places": extraction(is_list=True)}, needs))
    label, reading = (c.wire_request.state.parts for c in leaf.calls)
    assert label[1:] == reading[1:] == (ImagePart(data=png, media_type="image/png"),)
    assert isinstance(label[0], JsonPart)
    assert dict(label[0].value) == {"t:state.report": TEXT, "t:state.note": record}  # type: ignore[arg-type]
    assert reading[0] == TextPart(text=TEXT)


def test_each_request_takes_the_deadline_of_the_calls_seconds() -> None:
    binding, leaf = built()
    before = Deadline.after(30.0).at
    call(binding, source(), source("other", {"q:other/x": NOUL}), seconds=30.0)
    after = Deadline.after(30.0).at
    deadlines = {c.deadline.at for c in leaf.calls}
    assert len(deadlines) == 1
    assert before <= deadlines.pop() <= after


# ---------------------------------------------------------------------------------------------
# Refused before any call: unsupported, the reason a warnings line


def test_seconds_idia_cannot_hold_send_nothing() -> None:
    binding, leaf = built()
    tool = source("registry", refs=("t:registry/open",))
    for seconds in (31 * 86_400.0, 1e12):
        result = call(binding, source(), tool, seconds=seconds)
        assert {code(result, r) for r in result.values} == {"unsupported"}
        assert set(result.values) == {"q:triage/hazard", "t:registry/open"}
        (line,) = result.warnings
        assert "Deadline.after" in line
    assert leaf.calls == []


def test_a_t_ref_is_unsupported_and_its_sibling_answers() -> None:
    binding, _ = built()
    result = call(binding, source(), source("registry", refs=("t:registry/open",)))
    assert code(result, "t:registry/open") == "unsupported"
    assert result.values["q:triage/hazard"] == {"yes": 0.7, "no": idia.noul_complement(0.7)}
    assert result.warnings == (
        "registry: unsupported: the idia binding asks q: refs only, not t:registry/open",
    )


def unsendable() -> list[tuple[SourceCall, str]]:
    two_texts = {
        "q:r/a": extraction(text="t:state.report"),
        "q:r/b": extraction(text="t:state.note"),
    }
    two_dates = {
        "q:r/a": extraction(ValueType.date, reference_date="2026-03-10"),
        "q:r/b": extraction(ValueType.date, reference_date="2026-03-11"),
    }
    needs = {"t:state.report": TEXT, "t:state.note": "x"}
    return [
        (source("r", two_texts, needs), "read different texts, t:state.note, t:state.report"),
        (source("r", two_dates), "different reference dates, 2026-03-10, 2026-03-11"),
        (source(needs={"t:state.blob": b"\x00"}), "a need holds bytes"),
    ]


@pytest.mark.parametrize(("bad", "reason"), unsendable())
def test_a_source_one_request_cannot_hold_is_unsupported_with_its_reason(
    bad: SourceCall, reason: str
) -> None:
    binding, leaf = built()
    good = source("good", {"q:good/x": NOUL})
    result = call(binding, bad, good)
    assert {code(result, ref) for ref in bad.questions} == {"unsupported"}
    assert result.values["q:good/x"] == {"yes": 0.7, "no": idia.noul_complement(0.7)}
    (line,) = result.warnings
    assert line.startswith(f"{bad.id}: unsupported: ")
    assert reason in line
    assert [c.wire_request.questions[0].key for c in leaf.calls] == ["q:good/x"]


ONE = RenderedQuestion(AnswerKind.choice, "Who?", ("ana@x.org",))


def test_a_choice_of_fewer_than_two_labels_is_unsupported_alone_and_its_siblings_are_asked() -> (
    None
):
    binding, leaf = built()
    result = call(binding, source("triage", {"q:triage/one": ONE, "q:triage/x": NOUL}))
    assert code(result, "q:triage/one") == "unsupported"
    assert result.values["q:triage/x"] == {"yes": 0.7, "no": idia.noul_complement(0.7)}
    (line,) = result.warnings
    assert line.startswith("triage: unsupported q:triage/one: ")
    assert "two labels" in line
    assert [q.key for c in leaf.calls for q in c.wire_request.questions] == ["q:triage/x"]


def test_a_source_whose_every_question_idia_refuses_sends_no_request() -> None:
    binding, leaf = built()
    good = source("good", {"q:good/x": NOUL})
    result = call(binding, source("triage", {"q:triage/one": ONE}), good)
    assert code(result, "q:triage/one") == "unsupported"
    assert result.values["q:good/x"] == {"yes": 0.7, "no": idia.noul_complement(0.7)}
    assert [q.key for c in leaf.calls for q in c.wire_request.questions] == ["q:good/x"]


def test_candidate_shaped_labels_are_sent_and_answered_by_name() -> None:
    labels = ("none", "several", '"Ana" <a@x.org>', "]", "Zoë Ångström", "李")
    q = RenderedQuestion(AnswerKind.choice, "Who?", labels, descriptions=(None,) * len(labels))
    binding, leaf = built()
    result = call(binding, source("triage", {"q:triage/who": q}))
    (sent,) = [q for c in leaf.calls for q in c.wire_request.questions]
    assert tuple(label.name for label in sent.labels) == labels
    assert set(cast("dict[str, float]", result.values["q:triage/who"])) == set(labels)


def test_a_source_fit_refuses_is_unsupported_with_fits_message() -> None:
    binding, leaf = built(kinds=frozenset({"noul"}))
    result = call(binding, source(questions={"q:triage/asset": ASSET}))
    assert code(result, "q:triage/asset") == "unsupported"
    (line,) = result.warnings
    assert line.startswith("triage: unsupported: ")
    assert "choice" in line
    assert leaf.calls == []
    assert result.precision is None  # no source was sent


# ---------------------------------------------------------------------------------------------
# Mapping out


def test_a_noul_is_yes_as_given_and_no_its_exact_complement() -> None:
    binding, _ = built(replying({"q:triage/hazard": NoulAnswer(probability=0.58)}))
    result = call(binding, source())
    assert result.values["q:triage/hazard"] == {"yes": 0.58, "no": 0.42}


def test_a_label_answer_is_its_probabilities_by_name() -> None:
    probabilities = (("road", 0.25), ("other", 0.75))
    binding, _ = built(replying({"q:triage/asset": LabelAnswer(probabilities=probabilities)}))
    result = call(binding, source(questions={"q:triage/asset": ASSET}))
    assert result.values["q:triage/asset"] == {"road": 0.25, "other": 0.75}


def test_a_refusal_is_refused_and_a_fault_invalid_answer_naming_its_rules() -> None:
    fault = Fault(
        key="q:triage/asset",
        failures=(
            RuleFailure(rule=Rule.sum_tolerance, observed=1.4, bound=1.0),
            RuleFailure(rule=Rule.sum_tolerance, observed=1.4, bound=1.0),
        ),
        mapped=None,
    )
    binding, _ = built(replying({"q:triage/hazard": Refusal(), "q:triage/asset": fault}))
    result = call(binding, source(questions={"q:triage/asset": ASSET, "q:triage/hazard": NOUL}))
    assert code(result, "q:triage/hazard") == "refused"
    assert code(result, "q:triage/asset") == "invalid-answer"
    assert "triage: invalid-answer q:triage/asset: sum_tolerance" in result.warnings


def reading(answer: ExtractionAnswer, question: RenderedQuestion) -> Any:
    binding, _ = built(replying({"q:r/x": answer}))
    return call(binding, source("r", {"q:r/x": question}))


def test_an_extraction_is_the_five_members_with_its_span_as_a_pair() -> None:
    answer = ExtractionAnswer(
        value="Elm Row", confidence=0.9, p_null=0.01, span=Span(part=0, start=12, end=19)
    )
    result = reading(answer, extraction())
    assert result.values["q:r/x"] == {
        "value": "Elm Row",
        "guess": None,
        "p_null": 0.01,
        "confidence": 0.9,
        "span": [12, 19],
    }


def test_a_list_extraction_has_one_pair_per_item() -> None:
    spans = (Span(part=0, start=12, end=19), Span(part=0, start=59, end=70))
    answer = ExtractionAnswer(value=("Elm Row", "Quay Street"), span=spans)
    result = reading(answer, extraction(is_list=True))
    assert result.values["q:r/x"]["value"] == ["Elm Row", "Quay Street"]
    assert result.values["q:r/x"]["span"] == [[12, 19], [59, 70]]


def test_a_temporal_extraction_is_written_in_ommats_grammar() -> None:
    answer = ExtractionAnswer(value=datetime.time(21, 0), guess=None, p_null=0.04)
    result = reading(answer, extraction(ValueType.time))
    assert result.values["q:r/x"]["value"] == "21:00:00"


@pytest.mark.parametrize(
    ("answer", "value_type", "message"),
    [
        (ExtractionAnswer(value=datetime.timedelta(seconds=-1)), ValueType.duration, "is negative"),
        (
            ExtractionAnswer(value="Elm Row", span=Span(part=1, start=0, end=3)),
            ValueType.text,
            "a span into part 1, not the text",
        ),
    ],
)
def test_an_extraction_ommat_cannot_hold_is_invalid_answer_with_a_line(
    answer: ExtractionAnswer, value_type: ValueType, message: str
) -> None:
    result = reading(answer, extraction(value_type))
    assert code(result, "q:r/x") == "invalid-answer"
    (line,) = result.warnings
    assert line.startswith("r: invalid-answer q:r/x: ")
    assert message in line


# ---------------------------------------------------------------------------------------------
# A wire request that did not answer: its refs alone


def failing(error: ProviderError) -> Any:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        if wire_request.questions[0].key == "q:fails/x":
            return error
        return answer_all(wire_request, deadline)

    return script


@pytest.mark.parametrize("retryable", [False, True])
def test_a_source_that_fails_inside_the_deadline_is_raised_and_its_sibling_stands(
    retryable: bool,
) -> None:
    error = ProviderError("HTTP 500: busy", status_code=500, is_retryable=retryable)
    binding, _ = built(failing(error), max_questions=UNBOUNDED)
    result = call(binding, source(), source("fails", {"q:fails/x": NOUL}))
    assert code(result, "q:fails/x") == "raised"
    assert isinstance(result.values["q:triage/hazard"], dict)
    assert "fails: no answer for q:fails/x: ProviderError status 500: HTTP 500: busy" in (
        result.warnings
    )


def test_a_wire_request_the_deadline_cannot_hold_is_timed_out_and_its_sibling_stands() -> None:
    # Packed one question per wire request, the first stalls past the deadline, so idia
    # never sends the second: its error is DeadlineExceeded.
    async def stall(wire_request: WireRequest, deadline: Deadline) -> Any:
        if wire_request.questions[0].key == "q:triage/asset":
            await asyncio.sleep(5)
        return answer_all(wire_request, deadline)

    binding, leaf = built(stall, max_questions=1)
    questions = {"q:triage/asset": ASSET, "q:triage/hazard": NOUL}
    result = call(
        binding, source(questions=questions), source("other", {"q:other/x": NOUL}), seconds=0.3
    )
    assert code(result, "q:triage/hazard") == "timed-out"
    # The stall's retry cannot fit the deadline either, so it ends in DeadlineExceeded too.
    assert code(result, "q:triage/asset") == "timed-out"
    assert isinstance(result.values["q:other/x"], dict)
    assert [
        c.wire_request.questions[0].key
        for c in leaf.calls
        if c.wire_request.questions[0].key != "q:other/x"
    ] == ["q:triage/asset"]
    assert any(
        line.startswith("triage: no answer for q:triage/hazard: DeadlineExceeded")
        for line in result.warnings
    )


def test_a_retry_the_deadline_cannot_fit_is_timed_out_with_the_attempts_error() -> None:
    busy = ProviderError("HTTP 503: busy", status_code=503, is_retryable=True, retry_after=30.0)
    leaf = Scripted(capabilities=caps(), script=replying(busy))
    binding = oracle_binding(Dispatcher(leaf, retry=RetryPolicy(attempts=3)))
    result = call(binding, source(), seconds=1.0)
    assert code(result, "q:triage/hazard") == "timed-out"
    assert len(leaf.calls) == 1
    assert result.attempts == 1
    assert "triage: failed-attempt: ProviderError status 503: HTTP 503: busy" in result.warnings
    assert any(
        line.startswith("triage: no answer for q:triage/hazard: DeadlineExceeded")
        for line in result.warnings
    )


def test_a_credential_error_is_raised() -> None:
    early = ProviderError("HTTP 401: no", status_code=401)
    binding, _ = built(replying(early))
    assert code(call(binding, source()), "q:triage/hazard") == "raised"


def test_a_source_providing_nothing_is_asked_nothing() -> None:
    binding, leaf = built()
    result = call(binding, SourceCall("empty", None, {}, (), {}))
    assert result.values == {}
    assert leaf.calls == []
    assert (result.attempts, result.precision) == (None, None)


def test_a_call_that_reaches_no_model_holds_null_attempts_and_precision() -> None:
    binding, leaf = built()
    result = call(binding, source(), seconds=1e-9)
    assert code(result, "q:triage/hazard") == "timed-out"
    assert leaf.calls == []
    assert (result.attempts, result.precision, result.revision) == (None, None, None)


def test_a_packed_source_keeps_the_answers_of_the_wire_requests_that_answered() -> None:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        if wire_request.questions[0].key == "q:triage/severity":
            return ProviderError("HTTP 503: down", status_code=503)
        return answer_all(wire_request, deadline)

    binding, leaf = built(script, max_questions=1)
    questions = {"q:triage/asset": ASSET, "q:triage/hazard": NOUL, "q:triage/severity": LEVEL}
    result = call(binding, source(questions=questions))
    assert len(leaf.calls) == 3
    assert code(result, "q:triage/severity") == "raised"
    assert isinstance(result.values["q:triage/asset"], dict)
    assert isinstance(result.values["q:triage/hazard"], dict)
    assert result.attempts == 3  # one each, the failed request's not retried


# ---------------------------------------------------------------------------------------------
# Provenance


def test_the_sources_run_concurrently_and_their_provenance_is_joined_and_summed() -> None:
    started = 0
    both = asyncio.Event()

    async def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        nonlocal started
        started += 1
        if started == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)
        usage = Usage(input_tokens=10, output_tokens=2, passes=1)
        return Reply(answers=answer_all(wire_request, deadline), usage=usage)

    binding, _ = built(script)
    result = call(binding, source(), source("other", {"q:other/x": NOUL}))
    assert result.usage == {"input_tokens": 20, "output_tokens": 4, "passes": 2}
    assert result.attempts == 2
    assert result.precision == 2
    assert result.revision is not None
    prompts = (result.prompt or "").split("\n")
    assert len(prompts) == 2
    assert all(json.loads(p) for p in prompts)


def test_a_truncated_outcomes_refs_are_truncated() -> None:
    warning = Notice(code=idia.WARNING_PROVIDER_TRUNCATED, message="cut short")

    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return Reply(answers=answer_all(wire_request, deadline), warnings=(warning,))

    binding, _ = built(script)
    result = call(binding, source(questions={"q:triage/asset": ASSET, "q:triage/hazard": NOUL}))
    assert result.truncated == ("q:triage/asset", "q:triage/hazard")
    assert f"triage: {idia.WARNING_PROVIDER_TRUNCATED}: cut short" in result.warnings


def test_each_failed_attempt_is_a_line() -> None:
    attempts = iter([ProviderError("HTTP 429: slow down", status_code=429, is_retryable=True)])

    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return next(attempts, None) or answer_all(wire_request, deadline)

    binding, _ = built(script)
    result = call(binding, source())
    assert result.attempts == 2
    assert result.warnings[0] == (
        "triage: failed-attempt: ProviderError status 429: HTTP 429: slow down"
    )


def test_a_batch_with_nothing_sent_holds_null_provenance() -> None:
    binding, _ = built()
    result = call(binding, source("registry", refs=("t:registry/open",)))
    assert (result.revision, result.prompt, result.usage, result.attempts) == (
        None,
        None,
        None,
        None,
    )


def test_a_leafs_own_exception_propagates() -> None:
    def broken(wire_request: WireRequest, deadline: Deadline) -> Any:
        raise RuntimeError("the leaf's bug")

    binding, _ = built(broken)
    with pytest.raises(BaseExceptionGroup) as group:
        call(binding, source())
    assert group.group_contains(RuntimeError, match="the leaf's bug")


# ---------------------------------------------------------------------------------------------
# The suite, the import guard and the module's sites


def test_the_binding_passes_the_conformance_suite_over_a_scripted_leaf() -> None:
    binding, _ = built()
    assert check(binding, deterministic=True) == []


def test_import_ommat_imports_no_idia_module() -> None:
    probe = "import sys, ommat, ommat.bindings; print('idia' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_the_import_guard_names_the_extra() -> None:
    probe = (
        "import sys; sys.modules['idia'] = None\n"
        "try:\n    import ommat.bindings.idia\n"
        "except ModuleNotFoundError as e:\n    print(e)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert 'needs the "idia" extra: pip install "ommat[idia]"' in result.stdout


SITE = re.compile(r"\bUnavailable\(|\bFailed\(|\bround\(|\braise\b")


SPEC = Path(__file__).resolve().parent.parent / "docs" / "spec" / "README.md"
CITES = re.compile(r"# spec: ([^:]+?)(?::|$)")


def test_every_marker_refusal_and_raise_site_names_its_spec_section() -> None:
    headings = set(re.findall(r"^## (.+?)\s*$", SPEC.read_text(encoding="utf-8"), re.MULTILINE))
    lines = MODULE.read_text(encoding="utf-8").splitlines()
    sites = [n for n, line in enumerate(lines) if SITE.search(line.split("#")[0])]
    assert sites
    for n in sites:
        cited = CITES.findall(lines[n]) + CITES.findall(lines[n - 1].strip())
        assert cited, f"line {n + 1}: {lines[n].strip()}"
        assert set(cited) <= headings, f"line {n + 1}: {cited}"


def test_the_module_ignores_no_type_check() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "pyright: ignore" not in text
    assert "type: ignore" not in text


# ---------------------------------------------------------------------------------------------
# A noul's described yes and no, and a structured instruction


DESCRIBED_NOUL = RenderedQuestion(
    AnswerKind.noul,
    "Is anyone in danger?",
    ("yes", "no"),
    descriptions=("Someone could be hurt", None),
)
STRUCTURED = RenderedQuestion(
    AnswerKind.noul,
    {"task": "Is anyone in danger?", "criteria": ["a person nearby", "traffic"]},
    ("yes", "no"),
)


def sent(leaf: Scripted) -> list[Any]:
    return [q for c in leaf.calls for q in c.wire_request.questions]


def test_a_described_noul_is_sent_with_idias_yes_and_no_labels() -> None:
    binding, leaf = built()
    result = call(binding, source(questions={"q:triage/hazard": DESCRIBED_NOUL}))
    assert set(cast("dict[str, float]", result.values["q:triage/hazard"])) == {"yes", "no"}
    (q,) = sent(leaf)
    assert [(label.name, label.description) for label in q.labels] == [
        ("yes", "Someone could be hurt"),
        ("no", None),
    ]


def test_an_undescribed_noul_is_sent_with_no_labels() -> None:
    binding, leaf = built()
    call(binding, source())
    (q,) = sent(leaf)
    assert q.labels == ()


def test_a_structured_instruction_is_sent_as_idias_json() -> None:
    binding, leaf = built()
    call(binding, source(questions={"q:triage/hazard": STRUCTURED}))
    (q,) = sent(leaf)
    assert dict(q.instruction) == {
        "task": "Is anyone in danger?",
        "criteria": ("a person nearby", "traffic"),
    }


@pytest.mark.parametrize(
    ("question", "flag", "says"),
    [
        (DESCRIBED_NOUL, "carries_noul_descriptions", "a noul's yes and no descriptions"),
        (STRUCTURED, "carries_structured_instructions", "a structured instruction"),
    ],
)
def test_a_form_the_provider_does_not_carry_is_unsupported_alone(
    question: RenderedQuestion, flag: str, says: str
) -> None:
    binding, leaf = built(**{flag: False})
    result = call(binding, source(questions={"q:triage/a": question, "q:triage/b": NOUL}))
    assert code(result, "q:triage/a") == "unsupported"
    assert result.values["q:triage/b"] == {"yes": 0.7, "no": idia.noul_complement(0.7)}
    (line,) = result.warnings
    assert line == f"triage: unsupported q:triage/a: the provider does not carry {says}"
    assert [q.key for q in sent(leaf)] == ["q:triage/b"]


# ---------------------------------------------------------------------------------------------
# The provenance profile, as idia's leaves fill it


def test_a_local_leafs_runtime_reasons_and_claims_reach_the_call_under_ommats_names() -> None:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return Reply(
            answers=answer_all(wire_request, deadline),
            usage=Usage(input_tokens=12, output_tokens=None),
            versions={"python": "3.11.9", "gliner2": "1.2.0", "torch": "2.5.1"},
            client_versions={"huggingface_hub": "0.26.0"},
            unreported={"usage.output_tokens": "not-applicable", "usage.passes": "not-reported"},
            claimed=frozenset({"model_reported", "snapshot"}),
        )

    binding, _ = built(script)
    result = call(binding, source())
    assert result.runtime == {"python": "3.11.9", "gliner2": "1.2.0", "torch": "2.5.1"}
    assert result.claimed == ("model_reported", "revision")  # idia's snapshot is the revision
    assert result.revision == "scripted-1"
    assert result.missing == {
        "usage.output_tokens": "not-applicable",
        "usage.passes": "not-reported",
    }
    assert result.clients is not None
    assert "huggingface_hub" in result.clients
    assert result.clients["idia"] == idia.__version__
    assert (result.provider, result.model_requested) == ("scripted", "scripted-1")


def test_a_wire_alias_is_an_unpinned_revision_and_a_runtime_that_does_not_apply() -> None:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return Reply(answers=answer_all(wire_request, deadline), pinned=False)

    binding, _ = built(script)
    result = call(binding, source())
    assert (result.revision, result.runtime) == (None, None)
    assert result.missing is not None
    assert result.missing["revision"] == "not-pinned"
    assert result.missing["runtime"] == "not-applicable"
    assert any("not a pinned version" in line for line in result.warnings)


def test_a_failed_request_gives_not_sent_to_each_detail_it_holds_no_value_for() -> None:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        raise ProviderError("HTTP 500: busy", status_code=500, is_retryable=False)

    binding, _ = built(script)
    result = call(binding, source())
    assert code(result, "q:triage/hazard") == "raised"
    assert result.claimed == ()
    assert result.missing is not None
    assert set(result.missing.values()) == {"not-sent"}
    assert {"revision", "model_reported", "prompt"} <= set(result.missing)
    assert result.precision == 2  # the leaf's rounding, a value whatever the request did


def test_a_call_whose_requests_name_one_package_at_two_versions_is_invalid_return() -> None:
    seen = iter(("2.5", "2.4"))

    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        return Reply(answers=answer_all(wire_request, deadline), versions={"torch": next(seen)})

    binding, _ = built(script)
    other = source("second", {"q:second/hazard": NOUL})
    result = asyncio.run(binding((source(), other), seconds=60.0))
    assert result == Failed(Cause("invalid-return"))


def test_a_call_of_an_answered_and_a_failed_request_keeps_the_answered_ones_details() -> None:
    def script(wire_request: WireRequest, deadline: Deadline) -> Any:
        if any(q.key.startswith("q:second/") for q in wire_request.questions):
            raise ProviderError("HTTP 500: busy", status_code=500, is_retryable=False)
        return Reply(answers=answer_all(wire_request, deadline), claimed=frozenset({"snapshot"}))

    binding, _ = built(script)
    other = source("second", {"q:second/hazard": NOUL})
    result = asyncio.run(binding((source(), other), seconds=60.0))
    assert code(result, "q:second/hazard") == "raised"
    assert result.revision == "scripted-1"  # the answered request's
    assert result.claimed == ("revision",)
    assert result.missing is None or "revision" not in result.missing


def test_idias_details_and_reasons_are_ommats_under_the_binding_names() -> None:
    from ommat._provenance import DETAILS, REASONS
    from ommat.bindings.idia import _detail  # pyright: ignore[reportPrivateUsage]

    mapped = {_detail(d) for d in idia.DETAILS} | {"provider", "model_requested", "precision"}
    assert mapped == DETAILS
    assert set(idia.REASONS) == REASONS
