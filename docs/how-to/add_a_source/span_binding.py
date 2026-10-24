"""An oracle binding over a local span model run in-process.

The model loads once, before decide, and each call runs its inference in a worker thread, one at
a time. The binding answers extraction questions of value type ``text`` and marks every other
ref ``unsupported``, with its reason as a line of ``warnings``.
"""

import asyncio
import threading
import time
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


class SpanModel(Protocol):
    """The shape the binding asks of a model: names for its weights and the code that runs
    them, and ``predict``, each span of a label scored at or above ``threshold``, as a ``dict``
    of ``start``, ``end``, ``text`` and ``score``."""

    model_id: str
    revision: str
    runtime: str

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]: ...


class SpanBinding:
    """``binding(batch, seconds=s)`` over one loaded model, safe to call concurrently."""

    def __init__(self, model: SpanModel, *, threshold: float = THRESHOLD) -> None:
        self._model = model
        self._threshold = threshold
        self._lock = threading.Lock()  # one inference at a time, across threads and loops
        self._snapshot = f"{model.model_id}@{model.revision} on {model.runtime}"
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
                    requests.append(self._request(label))
                    continue
                if spans is None:  # at the call's end: ommat has timed out the whole call
                    lines.append(f"{ref}: timed-out: the model was busy until the call's end")
                    values[ref] = Unavailable(Cause("timed-out"))
                    continue
                requests.append(self._request(label))
                values[ref] = _answer(ref, question, text, spans, lines)
        return call_result(values, requests, precision=None, warnings=bounded_warnings(lines))

    def _request(self, label: str) -> RequestProvenance:
        """One model request's provenance: the model, the method and the arguments it took."""
        return RequestProvenance(
            snapshot=self._snapshot,
            elicitation=self._elicitation,
            prompt=prompt_text({"labels": [label], "threshold": self._threshold}),
            attempts=1,
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
