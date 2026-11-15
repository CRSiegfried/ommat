"""An oracle binding over a chat-completions endpoint, through a transport the caller passes.

It asks one question per request for verbalised probabilities as JSON, answers ``noul`` and
``choice`` questions, and marks every other ref ``unsupported``, with its reason as a line of
``warnings``. It imports only ``ommat`` and ``ommat.bindings``: the transport is any async
callable that posts a request body and returns the reply's JSON, such as a function over an
HTTP client.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol

from ommat import CallResult, Cause, RenderedQuestion, SourceCall, Unavailable
from ommat.bindings import (
    USAGE_KEYS,
    RequestProvenance,
    bounded_warnings,
    call_result,
    image_need,
    json_value,
    prompt_text,
)

PLACES = 2
"""The decimal places the model is asked for, and the answers' ``precision``."""

SYSTEM = (
    "Answer the question about the needs. Reply with JSON alone, "
    '{"probabilities": {"<label>": <probability>}}, a probability from 0 to 1 for each label, '
    f"each with at most {PLACES} decimal places."
)
ELICITATION = f"verbalized: JSON probabilities to {PLACES} places, one request per question"

SENT = {"revision": "not-pinned", "runtime": "not-applicable"}  # an alias, over the wire
FAILED = {
    **SENT,
    "model_reported": "not-sent",
    "request_id": "not-sent",
    "usage.input_tokens": "not-sent",
    "usage.output_tokens": "not-sent",
}
"""Why each detail is null on a request that answered, and on one that did not."""

Asked = tuple[str, object, RequestProvenance, str | None]
"""One request's ref, its value, its provenance and its line of ``warnings``, if any."""


class Post(Protocol):
    """``await post(body, seconds=s)``: send one request body and return the reply's JSON,
    raising ``TimeoutError`` when ``s`` seconds pass first."""

    def __call__(
        self, body: Mapping[str, Any], *, seconds: float
    ) -> Awaitable[Mapping[str, Any]]: ...


class ChatBinding:
    """``binding(batch, seconds=s)`` over one provider and one model, safe to call
    concurrently: it keeps each call's state in locals."""

    def __init__(self, post: Post, *, provider: str, model: str, timeout: float = 30.0) -> None:
        self._post = post
        self._provider = provider
        self._model = model
        self._timeout = timeout  # one request's own limit, within the call's seconds

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        deadline = time.monotonic() + seconds  # once, with no margin of its own
        values: dict[str, object] = {}
        lines: list[str] = []
        asks: list[Awaitable[Asked]] = []
        for call in batch:
            for ref in call.refs:
                values[ref] = _unsupported(ref, "the endpoint provides no t: ref", lines)
            needs, reason = _needs(call)
            for ref, question in call.questions.items():
                why = reason or _refusal(question)
                if why is not None:
                    values[ref] = _unsupported(ref, why, lines)
                    continue
                asks.append(self._ask(ref, question, needs, deadline))
        requests: list[RequestProvenance] = []
        for ref, value, request, line in await asyncio.gather(*asks):  # in the order given
            values[ref] = value
            requests.append(request)
            if line is not None:
                lines.append(line)
        return call_result(values, requests, precision=PLACES, warnings=bounded_warnings(lines))

    async def _ask(
        self, ref: str, question: RenderedQuestion, needs: object, deadline: float
    ) -> Asked:
        """Put one question to the endpoint; a failed request marks its own ref."""
        asked = {"question": question.wording, "labels": _labels(question)}
        body = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps({"needs": needs, **asked})},
            ],
        }
        prompt = prompt_text({"system": SYSTEM, **asked})  # the needs are in the record already
        try:
            left = min(self._timeout, deadline - time.monotonic())
            reply = await self._post(body, seconds=left)
        except TimeoutError as error:  # the request's own limit passed before the call's end
            line = f"{ref}: timed-out: {error}"
            return ref, Unavailable(Cause("timed-out")), self._failed(prompt), line
        except Exception as error:  # never BaseException: a cancellation propagates
            line = f"{ref}: raised: {type(error).__name__}: {error}"
            return ref, Unavailable(Cause("raised")), self._failed(prompt), line
        value, line = _answer(ref, question, reply)
        return ref, value, self._sent(prompt, reply), line

    def _sent(self, prompt: str | None, reply: Mapping[str, Any]) -> RequestProvenance:
        """A request that answered: what the reply says, and why each detail it leaves out is
        null."""
        usage: Mapping[str, Any] = reply.get("usage") or {}
        theirs = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}
        tokens = {key: usage[theirs[key]] for key in USAGE_KEYS if theirs[key] in usage}
        model, request_id = reply.get("model"), reply.get("id")
        missing = {
            **SENT,
            **{f"usage.{key}": "not-reported" for key in USAGE_KEYS if key not in tokens},
            **({} if model else {"model_reported": "not-reported"}),
            **({} if request_id else {"request_id": "not-reported"}),
        }
        return RequestProvenance(
            provider=self._provider,
            model_requested=self._model,
            model_reported=model,
            elicitation=ELICITATION,
            prompt=prompt,
            claimed=("model_reported",) if model else (),  # the provider's word, not checked
            usage=tokens or None,
            attempts=1,
            request_id=request_id,
            missing=missing,
            extras={"processing_ms": reply["processing_ms"]} if "processing_ms" in reply else None,
        )

    def _failed(self, prompt: str | None) -> RequestProvenance:
        """A request that did not answer: no model reported, no usage, no id."""
        return RequestProvenance(
            provider=self._provider,
            model_requested=self._model,
            elicitation=ELICITATION,
            prompt=prompt,
            attempts=1,
            missing=FAILED,
        )


def _needs(call: SourceCall) -> tuple[object, str | None]:
    """The source's needs as JSON keyed by ref, in the order given, or why they cannot go."""
    if any(image_need(value) is not None for value in call.needs.values()):
        return None, "the source needs an image, and this endpoint is sent text"
    try:
        return {ref: json_value(value) for ref, value in call.needs.items()}, None
    except TypeError as error:
        return None, f"a need cannot be sent as JSON: {error}"


def _refusal(question: RenderedQuestion) -> str | None:
    """Why the binding will not ask ``question``, or ``None`` when it will."""
    if question.kind not in ("noul", "choice"):
        return f"the binding asks noul and choice questions, not a {question.kind}"
    return None


def _labels(question: RenderedQuestion) -> list[object]:
    """Each label's name, with its description where it has one."""
    pairs = zip(question.labels, question.descriptions, strict=True)
    return [name if d is None else {"name": name, "description": d} for name, d in pairs]


def _unsupported(ref: str, reason: str, lines: list[str]) -> Unavailable:
    lines.append(f"{ref}: unsupported: {reason}")
    return Unavailable(Cause("unsupported"))


def _answer(
    ref: str, question: RenderedQuestion, reply: Mapping[str, Any]
) -> tuple[object, str | None]:
    """The answer, each label's probability; ``refused`` when the provider declined, and
    ``invalid-answer`` when the reply is not the JSON asked for."""
    try:
        message = reply["choices"][0]["message"]
        if message.get("refusal"):
            return Unavailable(Cause("refused")), f"{ref}: refused: {message['refusal']}"
        found: dict[str, Any] = json.loads(message["content"])["probabilities"]
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
        return _invalid(ref, f"the reply is not the JSON asked for: {error!r}")
    if type(found) is not dict or sorted(found) != sorted(question.labels):
        return _invalid(ref, f"{found!r} does not hold exactly the question's labels")
    answer = {name: found[name] for name in question.labels}  # in the question's order
    for name, p in answer.items():
        if type(p) not in (int, float) or not 0 <= p <= 1 or round(p, PLACES) != p:
            return _invalid(ref, f"{name}: {p!r} is not a probability to {PLACES} places")
    return answer, None


def _invalid(ref: str, why: str) -> tuple[object, str]:
    return Unavailable(Cause("invalid-answer")), f"{ref}: invalid-answer: {why}"
