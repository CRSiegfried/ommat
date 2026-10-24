"""An oracle binding over a local span model run in-process.

The model loads once, before decide, and each call runs its inference in a worker thread, one at
a time. The binding answers extraction questions of value type ``text`` and marks every other
ref ``unsupported``, with its reason as a line of ``warnings``.
"""

import asyncio
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol

from ommat import CallResult, Cause, RenderedQuestion, SourceCall, Unavailable
from ommat.bindings import (
    RequestProvenance,
    bounded_warnings,
    call_result,
    extraction_value,
    image_need,
    prompt_text,
)

THRESHOLD = 0.5

NOT_GIVEN = {
    "request_id": "not-reported",
    "calibrated": "not-reported",
    "usage.input_tokens": "not-reported",
    "usage.output_tokens": "not-applicable",  # an encoder writes no tokens
    "usage.passes": "not-reported",
}
"""Why each detail the model cannot give is null."""


class SpanModel(Protocol):
    """The shape the binding asks of a model: the library that runs it, names for its weights,
    the versions of the code that runs them, and ``predict``, each span of a label scored at or
    above ``threshold``, as a ``dict`` of ``start``, ``end``, ``text`` and ``score``."""

    provider: str
    model_id: str
    revision: str
    runtime: Mapping[str, str]

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]: ...


class SpanBinding:
    """``binding(batch, seconds=s)`` over one loaded model, safe to call concurrently."""

    def __init__(self, model: SpanModel, *, threshold: float = THRESHOLD) -> None:
        self._model = model
        self._threshold = threshold
        self._lock = threading.Lock()  # one inference at a time, across threads and loops
        self._elicitation = f"span extraction, one request per question, threshold {threshold}"

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        deadline = time.monotonic() + seconds  # once, with no margin of its own
        values: dict[str, Any] = {}
        lines: list[str] = []
        requests: list[RequestProvenance] = []
        for call in batch:
            for ref in call.refs:
                values[ref] = _unsupported(ref, "the model provides no t: ref", lines)
            image = any(image_need(value) is not None for value in call.needs.values())
            for ref, question in call.questions.items():
                reason = _refusal(question, image)
                if reason is not None:
                    values[ref] = _unsupported(ref, reason, lines)
                    continue
                assert question.text is not None
                assert isinstance(question.wording, str)
                text = call.needs[question.text]
                label = question.wording  # what the model receives, as the record keeps it
                try:
                    spans = await asyncio.to_thread(self._predict, text, label, deadline)
                except Exception as error:  # never BaseException: a cancellation propagates
                    lines.append(f"{ref}: raised: {type(error).__name__}: {error}")
                    values[ref] = Unavailable(Cause("raised"))
                    requests.append(self._request(label, failed=True))
                    continue
                if spans is None:  # at the call's end: ommat has timed out the whole call
                    lines.append(f"{ref}: timed-out: the model was busy until the call's end")
                    values[ref] = Unavailable(Cause("timed-out"))
                    continue
                requests.append(self._request(label))
                values[ref] = _answer(ref, question, text, spans, lines)
        warnings = bounded_warnings(lines)
        return call_result(values, requests, precision="unrounded", warnings=warnings)

    def _request(self, label: str, *, failed: bool = False) -> RequestProvenance:
        """One model request's provenance: what ran it and how, the arguments it took, and why
        each detail the model cannot give is null; a request that failed reports no model."""
        model = self._model
        return RequestProvenance(
            provider=model.provider,
            model_requested=model.model_id,
            model_reported=None if failed else model.model_id,
            revision=model.revision,
            runtime=dict(model.runtime),
            elicitation=self._elicitation,
            prompt=prompt_text({"labels": [label], "threshold": self._threshold}),
            # The model's own word, not checked.
            claimed=("revision",) if failed else ("model_reported", "revision"),
            attempts=1,
            missing={**NOT_GIVEN, "model_reported": "not-sent"} if failed else NOT_GIVEN,
        )

    def _predict(
        self, text: str, label: str, deadline: float
    ) -> list[tuple[int, int, str, float]] | None:
        """The model's spans as Python values, a torch or numpy scalar made an int or float;
        ``None`` when another inference held the model until ``deadline``."""
        left = deadline - time.monotonic()
        if left <= 0 or not self._lock.acquire(timeout=left):
            return None
        try:
            found = self._model.predict(text, [label], self._threshold)
        finally:
            self._lock.release()
        spans = [(int(s["start"]), int(s["end"]), str(s["text"]), float(s["score"])) for s in found]
        return sorted(spans)


def _refusal(question: RenderedQuestion, image: bool) -> str | None:
    """Why the model cannot answer ``question``, or ``None`` when it can."""
    if question.kind != "extraction":
        return f"the model answers extractions, not a {question.kind}"
    if question.value_type != "text":
        return f"the model reads text spans, not a {question.value_type}"
    if not isinstance(question.wording, str):
        return "the model takes its label as a string, not a structured wording"
    if image:
        return "the source needs an image, which the model cannot read"
    return None


def _unsupported(ref: str, reason: str, lines: list[str]) -> Unavailable:
    lines.append(f"{ref}: unsupported: {reason}")
    return Unavailable(Cause("unsupported"))


def _answer(
    ref: str,
    question: RenderedQuestion,
    text: str,
    spans: list[tuple[int, int, str, float]],
    lines: list[str],
) -> object:
    """The extraction answer, or ``invalid-answer`` when a span does not read its value."""
    for start, end, found, _ in spans:
        if text[start:end] != found:
            lines.append(f"{ref}: invalid-answer: span [{start}, {end}] does not read {found!r}")
            return Unavailable(Cause("invalid-answer"))
    if question.is_list:  # every span found, and no one confidence for the list
        found_all = [found for _, _, found, _ in spans]
        return extraction_value(found_all, None, None, None, [(s, e) for s, e, _, _ in spans])
    if not spans:  # the model found none: a null value, not a marker
        return extraction_value(None, None, None, None, None)
    start, end, found, score = max(spans, key=lambda s: s[3])
    return extraction_value(found, None, None, score, (start, end))
