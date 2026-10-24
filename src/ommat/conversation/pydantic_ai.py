"""The Pydantic AI renderer: a ``transcript`` view as Pydantic AI messages, outside the
compatibility promise while it is validated. Install it with ``ommat[pydantic-ai]``.

``to_model_messages(view)`` gives each source call and decide as a ``ModelResponse`` of tool
calls and a ``ModelRequest`` of their returns, each message marked as ommat's bookkeeping in its
``metadata``, which Pydantic AI never sends to a model; then the pending steps as a trailing
``ModelResponse``, unmarked, since those are real actions. A history ending in those calls, run
with no new user prompt, has the agent run them with its own tools. ``is_bookkeeping(message)``
reads the mark, and ``drop_earlier_bookkeeping`` is a history processor that keeps the latest
record's bookkeeping and drops every earlier record's, each return with its call.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from ommat.conversation._exchanges import BOOKKEEPING, exchanges

__all__ = ["drop_earlier_bookkeeping", "is_bookkeeping", "to_model_messages"]

_MARK = "ommat"


def to_model_messages(
    view: dict[str, Any] | None,
    *,
    model_name: str | None = None,
    provider_name: str | None = None,
) -> list[ModelMessage]:
    """A ``transcript`` view as Pydantic AI messages, ``[]`` for ``None``: per group of calls a
    ``ModelResponse`` of ``ToolCallPart``s, then a ``ModelRequest`` of a ``ToolReturnPart`` per
    call with a result, a failed source call's with ``outcome="failed"`` and decide's with
    ``metadata`` ``{"record_hash", "design_hash"}``. The source calls' and decide's messages
    carry ``metadata`` ``{"ommat": "bookkeeping", "record_hash"}``; the pending calls' response
    carries none.
    Every response carries ``model_name`` and ``provider_name``. Raises ``ValueError`` for a
    tool name no provider takes, as ``to_chat_completions`` does."""
    if view is None:
        return []
    stamp = {"record_hash": view["record_hash"], "design_hash": view["design"]["hash"]}
    messages: list[ModelMessage] = []
    for group in exchanges(view):
        kept = all(call.result is not None for call in group)
        calls = [
            ToolCallPart(call.name, deepcopy(call.arguments), tool_call_id=call.id)
            for call in group
        ]
        messages.append(
            ModelResponse(
                parts=calls,
                model_name=model_name,
                provider_name=provider_name,
                metadata=_marked(view["record_hash"]) if kept else None,
            )
        )
        returns = [
            ToolReturnPart(
                call.name,
                deepcopy(call.result),
                tool_call_id=call.id,
                metadata=dict(stamp) if call.decide else None,
                outcome="failed" if call.failed else "success",
            )
            for call in group
            if call.result is not None
        ]
        if returns:
            messages.append(ModelRequest(parts=returns, metadata=_marked(view["record_hash"])))
    return messages


def is_bookkeeping(message: ModelMessage) -> bool:
    """Whether ``message`` is one ``to_model_messages`` marked as ommat's bookkeeping: a source
    call or decide, or its return. A pending call's response, and anything the agent or the
    model wrote, is not."""
    metadata = message.metadata
    return type(metadata) is dict and metadata.get(_MARK) == BOOKKEEPING


def drop_earlier_bookkeeping(messages: list[ModelMessage]) -> list[ModelMessage]:
    """A Pydantic AI history processor: the messages with every bookkeeping call of a record
    other than the latest left out, and each return answering one of those calls, wherever it
    stands. The latest record's bookkeeping stays, since the model answers from decide's result.
    A return goes only with its call, so the history never holds one without the other."""
    marked = [m for m in messages if isinstance(m, ModelResponse) and is_bookkeeping(m)]
    if not marked:
        return list(messages)
    latest = _record(marked[-1])
    gone = {
        part.tool_call_id
        for m in marked
        if _record(m) != latest
        for part in m.parts
        if isinstance(part, ToolCallPart)
    }
    out: list[ModelMessage] = []
    for m in messages:
        if isinstance(m, ModelResponse):
            if is_bookkeeping(m) and _record(m) != latest:
                continue
        else:
            parts = [p for p in m.parts if not _answers(p, gone)]
            if not parts:
                continue
            if len(parts) != len(m.parts):
                m = replace(m, parts=parts)
        out.append(m)
    return out


def _answers(part: object, gone: set[str]) -> bool:
    return isinstance(part, ToolReturnPart) and part.tool_call_id in gone


def _record(message: ModelMessage) -> object:
    return cast("dict[str, Any]", message.metadata).get("record_hash")


def _marked(record_hash: str) -> dict[str, Any]:
    return {_MARK: BOOKKEEPING, "record_hash": record_hash}
