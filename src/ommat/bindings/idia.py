"""The binding to idia: an oracle binding that puts a batch's questions to a provider through
an idia ``Dispatcher`` over one leaf.

Install it with the ``idia`` extra, after idia with the extras for its providers,
as the README shows:
``pip install "ommat[idia] @ git+https://github.com/CRSiegfried/ommat@v0.4.0"``.
``import ommat`` never imports it. Each oracle names one leaf, so a design that asks label kinds and
extractions of different models binds one dispatcher per oracle.

The binding is a map between the two libraries' shapes. Each source of a batch is one idia
``Request``: its questions in the order received, keyed by their refs, each label carrying its
description; a source asking an extraction sends the text its spans index as the state's first part,
its reference date in the request's options, and a source asking only label kinds sends its needs as
one JSON part keyed by ommat's refs, or, when the binding is built with ``state`` naming the
source's one need besides its images, that need's value as the part; each object goes in the order
the binding receives it, since a model reads the order. An image need is an image part after either.
The sources sent run concurrently under one deadline, the call's ``seconds``. Each answer is mapped
to the value ommat takes; a provider's refusal is ``refused``, a fault ``invalid-answer``, and each
ref of a wire request that did not answer ``timed-out`` when its error is idia's
``DeadlineExceeded``, else ``raised``. A source the binding cannot send is ``unsupported``, with its
reason a line of the call's warnings; so is a question alone whose described noul or structured
instruction the provider's capabilities do not carry, or that idia's ``Question`` refuses, a choice
rendered with fewer than two labels among them, and the source's other questions are sent. Each
outcome's provenance is written under ommat's names, idia's ``snapshot`` as ``revision``,
``versions`` as ``runtime``, ``prompt_sent`` as ``prompt`` and ``client_versions`` as ``clients``,
with idia's reasons and claims under the same names.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

try:
    import idia
except ModuleNotFoundError as error:
    if error.name != "idia":
        raise  # spec: The idia binding: any other import failure propagates
    raise ModuleNotFoundError(  # spec: The idia binding: a missing extra is named
        'ommat.bindings.idia needs the "idia" extra: pip install "ommat[idia]"',
        name=error.name,
    ) from error

from ommat import AnswerKind, CallResult, Cause, Failed, RenderedQuestion, SourceCall, Unavailable
from ommat._provenance import USAGE_KEYS
from ommat.bindings import (
    OracleBinding,
    RequestProvenance,
    bounded_warnings,
    call_result,
    extraction_value,
    image_need,
    json_value,
    prompt_text,
)

__all__ = ["oracle_binding"]

_MAX_DECIMALS = 17
"""The most decimal places a call's ``precision`` names."""


def oracle_binding(provider: object, *, state: str | None = None) -> OracleBinding:
    """An oracle binding over ``provider``, an idia ``Dispatcher`` over one leaf.

    ``state`` names a need a label source sends as the whole JSON part, its value unwrapped, in
    place of a part keyed by refs, so a caller sends the state shape it measured; a label source
    that needs it beside another need, images aside, is ``unsupported``.

    It refuses, with ``ValueError``, any other provider, whose policies idia does not show;
    a dispatcher whose re-ask is on, which would give a question a second answer; one whose
    overflow policy is not ``Fail``, which could trim the state the model saw; and one whose leaf
    declares probability decimals outside 0 to 17, which no call's ``precision`` can hold."""
    if not isinstance(provider, idia.Dispatcher):
        raise ValueError(  # spec: The idia binding: one leaf per oracle, its policies shown
            f"the idia binding takes a Dispatcher over one leaf, not a {type(provider).__name__}"
        )
    if provider.reask:
        raise ValueError(  # spec: The idia binding: a re-ask is a second answer to one question
            "the idia binding takes a Dispatcher with the re-ask off"
        )
    if not isinstance(provider.overflow, idia.Fail):
        raise ValueError(  # spec: The idia binding: a trimmed state is not what the record holds
            "the idia binding takes a Dispatcher whose overflow policy is Fail"
        )
    decimals = provider.capabilities.rounding.probability_decimals
    if decimals is not None and not 0 <= decimals <= _MAX_DECIMALS:
        raise ValueError(  # spec: The idia binding: a precision past 0 to 17 fails every call
            f"the idia binding takes a leaf whose probability decimals are 0 to "
            f"{_MAX_DECIMALS}, not {decimals}"
        )
    if state is not None and (type(state) is not str or not state):
        raise ValueError(  # spec: The idia binding: state names a need
            f"the idia binding's state names a need's ref, not {state!r}"
        )
    return _Binding(provider, state)


@dataclass(frozen=True, slots=True)
class _Binding:
    """The oracle binding ``oracle_binding`` returns: ``await binding(batch, seconds=s)``. Built
    only by ``oracle_binding``, which makes its refusals."""

    provider: idia.Dispatcher
    state: str | None = None

    async def __call__(
        self, batch: tuple[SourceCall, ...], /, *, seconds: float
    ) -> CallResult | Failed:
        try:
            deadline = idia.Deadline.after(seconds)
        except ValueError as error:
            # spec: The idia binding: a deadline idia cannot hold sends nothing
            refused = Unavailable(Cause("unsupported"))
            refs = (ref for source in batch for ref in (*source.refs, *source.questions))
            return CallResult(dict.fromkeys(refs, refused), warnings=bounded_warnings([str(error)]))
        values: dict[str, object] = {}
        lines: list[str] = []
        requests: dict[str, idia.Request] = {}
        for source in batch:
            reason = _refusal(self, source, deadline, requests, values, lines)
            if reason is not None:
                refused = Unavailable(Cause("unsupported"))  # spec: The idia binding
                values.update(dict.fromkeys((*source.refs, *source.questions), refused))
                lines.append(f"{source.id}: unsupported: {reason}")
        async with asyncio.TaskGroup() as group:
            tasks = {
                sid: group.create_task(_answer(self.provider, request))
                for sid, request in requests.items()
            }
        outcomes: list[idia.Outcome] = []
        for source in batch:
            if source.id not in tasks:
                continue
            response = tasks[source.id].result()
            outcomes += response.outcomes
            lines += _lines(source.id, response)
            failed = {key: o.error for o in response.outcomes if o.error for key in o.keys}
            for ref in (q.key for q in requests[source.id].questions):  # the questions sent
                error = failed.get(ref)
                if error is None:
                    values[ref] = _value(source.id, ref, response.answers[ref], lines)
                elif isinstance(error, idia.DeadlineExceeded):
                    values[ref] = Unavailable(Cause("timed-out"))  # spec: The binding protocol
                else:
                    values[ref] = Unavailable(Cause("raised"))  # spec: The binding protocol
        sent = [o for o in outcomes if o.attempts]
        try:
            return call_result(
                values,
                [_provenance(o) for o in outcomes],
                precision=_precision(sent[0].rounding) if sent else None,
                warnings=bounded_warnings(lines),
            )
        except ValueError:  # spec: The idia binding: one package at two versions in a call
            return Failed(Cause("invalid-return"))


# ---------------------------------------------------------------------------------------------
# Mapping in


class _Unsendable(Exception):
    """A source one idia request cannot hold; the message is the reason."""


def _refusal(
    binding: _Binding,
    source: SourceCall,
    deadline: idia.Deadline,
    requests: dict[str, idia.Request],
    values: dict[str, object],
    lines: list[str],
) -> str | None:
    """The reason the source is not sent, or ``None`` when it is sent (its request is then
    kept in ``requests``) or asks nothing. A question idia's ``Question`` refuses, a choice
    of fewer than two labels among them, is ``unsupported`` alone in ``values``, its reason a
    line, and the source's other questions are sent; a source left with none sends nothing."""
    if source.refs:
        return f"the idia binding asks q: refs only, not {', '.join(source.refs)}"
    questions: list[idia.Question] = []
    capabilities = binding.provider.capabilities
    for ref, rendered in source.questions.items():
        uncarried = _uncarried(rendered, capabilities)
        if uncarried is not None:
            values[ref] = Unavailable(Cause("unsupported"))  # spec: The idia binding
            lines.append(f"{source.id}: unsupported {ref}: the provider does not carry {uncarried}")
            continue
        try:
            questions.append(_question(ref, rendered))
        except ValueError as refused:
            values[ref] = Unavailable(Cause("unsupported"))  # spec: The idia binding
            lines.append(f"{source.id}: unsupported {ref}: {refused}")
    if not questions:
        return None
    try:
        request = _request(source, tuple(questions), deadline, binding.state)
        binding.provider.plan(request)
    except idia.DoesNotFit as unfit:
        return unfit.message
    except (_Unsendable, ValueError) as unsendable:
        return str(unsendable)
    requests[source.id] = request
    return None


def _request(
    source: SourceCall,
    questions: tuple[idia.Question, ...],
    deadline: idia.Deadline,
    whole: str | None,
) -> idia.Request:
    needs = dict(source.needs)
    images: list[idia.Part] = []
    for key in sorted(needs):
        image = image_need(needs[key])
        if image is not None:
            del needs[key]
            images.append(idia.ImagePart(data=image[1], media_type=image[0]))
    sent = [source.questions[q.key] for q in questions]
    extractions = [q for q in sent if q.kind is AnswerKind.extraction]
    options: dict[str, idia.JsonValue] = {}
    if extractions:
        # An extraction question's text is its needs key and its date a string or None.
        texts = sorted({cast(str, q.text) for q in extractions})
        dates = sorted({q.reference_date for q in extractions if q.reference_date is not None})
        if len(texts) > 1:
            raise _Unsendable(  # spec: The idia binding: one request holds one text its spans index
                f"its extractions read different texts, {', '.join(texts)}"
            )
        if len(dates) > 1:
            raise _Unsendable(  # spec: The idia binding: one request takes one reference date
                f"its extractions name different reference dates, {', '.join(dates)}"
            )
        if dates:
            options["reference_date"] = dates[0]
        # spec: The idia binding: the spans index the text, the state's part 0
        first: idia.Part = idia.TextPart(text=source.needs[texts[0]])
    else:
        if whole in needs and len(needs) > 1:
            others = ", ".join(sorted(set(needs) - {whole}))
            raise _Unsendable(  # spec: The idia binding: the state need is the whole JSON part
                f"it needs {whole}, sent as the whole state, beside {others}"
            )
        sent_as: object = needs[whole] if whole is not None and whole in needs else needs
        try:
            written = json_value(sent_as)
        except TypeError as unwritable:
            raise _Unsendable(str(unwritable)) from None  # spec: The idia binding: no bytes in JSON
        first = idia.JsonPart(value=_tuples(written))
    state = idia.State(parts=(first, *images))
    return idia.Request(state=state, questions=questions, deadline=deadline, options=options)


def _uncarried(rendered: RenderedQuestion, capabilities: idia.Capabilities) -> str | None:
    """What the provider's declared capabilities do not carry of a question: a noul's
    described yes and no, or a structured instruction; ``None`` when it carries both."""
    described = rendered.kind is AnswerKind.noul and any(
        d is not None for d in rendered.descriptions
    )
    if described and not capabilities.carries_noul_descriptions:
        return "a noul's yes and no descriptions"
    if not isinstance(rendered.wording, str) and not capabilities.carries_structured_instructions:
        return "a structured instruction"
    return None


def _question(ref: str, rendered: RenderedQuestion) -> idia.Question:
    """The question as idia asks it, keyed by its ref, in the order received."""
    labels = tuple(
        idia.Label(name=name, description=rendered.descriptions[i])
        for i, name in enumerate(rendered.labels)
    )
    return idia.Question(
        key=ref,
        kind=cast(idia.Kind, rendered.kind.value),
        instruction=rendered.wording
        if isinstance(rendered.wording, str)
        else _tuples(rendered.wording),
        # A noul sends its yes and no only when one is described, as idia takes them.
        labels=()
        if rendered.kind is AnswerKind.noul and all(d is None for d in rendered.descriptions)
        else labels,
        value_type=None if rendered.value_type is None else rendered.value_type.value,
        is_list=rendered.is_list,
    )


def _tuples(value: object) -> idia.JsonValue:
    """JSON data with each list a tuple, as idia's JSON part holds it."""
    if isinstance(value, dict):
        return {k: _tuples(v) for k, v in cast("dict[str, object]", value).items()}
    if isinstance(value, list):
        return tuple(_tuples(v) for v in cast("list[object]", value))
    return cast(idia.JsonValue, value)


async def _answer(provider: idia.Dispatcher, request: idia.Request) -> idia.Response:
    """The source's response, or the response a ``RequestFailed`` carries."""
    try:
        return await provider.answer(request)
    except idia.RequestFailed as failure:
        return failure.response


# ---------------------------------------------------------------------------------------------
# Mapping out


def _value(sid: str, ref: str, answer: idia.Answer, lines: list[str]) -> object:
    """An answer as ommat takes it, or its marker."""
    if isinstance(answer, idia.Refusal):
        return Unavailable(Cause("refused"))  # spec: The idia binding
    if isinstance(answer, idia.Fault):
        rules = ", ".join(dict.fromkeys(str(f.rule) for f in answer.failures))
        lines.append(f"{sid}: invalid-answer {ref}: {rules}")
        # spec: The idia binding: a fault is its question's alone, a value found unusable
        return Unavailable(Cause("invalid-answer"))
    if isinstance(answer, idia.NoulAnswer):
        # spec: The idia binding: no is idia's complement
        return {"yes": answer.probability, "no": idia.noul_complement(answer.probability)}
    if isinstance(answer, idia.ExtractionAnswer):
        try:
            return extraction_value(
                answer.value, answer.guess, answer.p_null, answer.confidence, _spans(answer.span)
            )
        except ValueError as unwritable:
            lines.append(f"{sid}: invalid-answer {ref}: {unwritable}")
            # spec: The idia binding: a value or span the text cannot hold
            return Unavailable(Cause("invalid-answer"))
    return dict(cast(idia.LabelAnswer, answer).probabilities)


def _spans(span: idia.Span | tuple[idia.Span, ...] | None) -> object:
    """The span as offsets into the text, the state's part 0."""
    if span is None:
        return None
    if isinstance(span, tuple):
        return [_pair(s) for s in span]
    return _pair(span)


def _pair(span: idia.Span) -> tuple[int, int]:
    if span.part != 0:
        raise ValueError(  # spec: The idia binding: a span indexes the text, part 0
            f"a span into part {span.part}, not the text"
        )
    return (span.start, span.end)


def _lines(sid: str, response: idia.Response) -> list[str]:
    """The source's lines: each failed attempt, each failed wire request with its status and
    message, then each warning, in the order idia gives them."""
    lines: list[str] = []
    for outcome in response.outcomes:
        for attempt in outcome.failed_attempts:
            lines.append(f"{sid}: failed-attempt: {_error(attempt.error)}")
        if outcome.error is not None:
            keys = ", ".join(outcome.keys)
            lines.append(f"{sid}: no answer for {keys}: {_error(outcome.error)}")
    for warning in (*response.warnings, *(w for o in response.outcomes for w in o.warnings)):
        field = "" if warning.field is None else f" {warning.field}"
        lines.append(f"{sid}: {warning.code}{field}: {warning.message}")
    return lines


def _error(error: idia.ProviderError) -> str:
    status = "" if error.status_code is None else f" status {error.status_code}"
    request = "" if error.request_id is None else f" request {error.request_id}"
    return f"{type(error).__name__}{status}{request}: {error.message}"


_NAMES = {"snapshot": "revision", "versions": "runtime", "prompt_sent": "prompt"}
"""idia's name for a profile detail, where ommat's differs."""

_UNRECORDED = frozenset({"calibrated"})
"""idia's details ommat does not record: a provider's own claim that its probabilities are
calibrated is unchecked, so it is dropped from ``claimed`` and ``missing``."""


def _detail(name: str) -> str:
    return _NAMES.get(name, name)


def _precision(rounding: idia.Rounding) -> int | str:
    """An outcome's rounding as ``precision``: its probability decimals, or ``unrounded``."""
    decimals = rounding.probability_decimals
    return "unrounded" if decimals is None else decimals


def _provenance(outcome: idia.Outcome) -> RequestProvenance:
    """One wire request's provenance under ommat's names: the profile's details, the prompt as
    text, the usage figures it reports, idia's reasons and claims, the client versions, and its
    keys when idia warns the provider cut it short."""
    usage = outcome.usage
    figures = (
        None
        if usage is None
        else {name: getattr(usage, name) for name in USAGE_KEYS if getattr(usage, name) is not None}
    )
    cut = any(w.code == idia.WARNING_PROVIDER_TRUNCATED for w in outcome.warnings)
    return RequestProvenance(
        provider=outcome.provider,
        model_requested=outcome.model_requested,
        model_reported=outcome.model_reported,
        revision=outcome.snapshot,
        runtime=dict(outcome.versions) or None,
        elicitation=outcome.elicitation,
        prompt=prompt_text(outcome.prompt_sent),
        claimed=tuple(_detail(d) for d in outcome.claimed if d not in _UNRECORDED),
        usage=figures or None,
        attempts=outcome.attempts,
        request_id=outcome.request_id,
        missing={_detail(d): r for d, r in outcome.unreported.items() if d not in _UNRECORDED}
        or None,
        clients=dict(outcome.client_versions) or None,
        truncated=tuple(outcome.keys) if cut else (),
    )
