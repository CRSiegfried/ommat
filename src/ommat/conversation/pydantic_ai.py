"""The Pydantic AI renderer: a record's turn history as Pydantic AI messages, outside the
compatibility promise. Install it with ``ommat[pydantic-ai]``.

``to_model_messages(record, design)`` gives each source call and decide as a ``ModelResponse``
of tool calls and a ``ModelRequest`` of their returns, each marked as ommat's bookkeeping in its
``metadata``, which Pydantic AI never sends to a model. ``is_bookkeeping(message)``
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

from ommat._design import LoadedDesign
from ommat._ledger import Record
from ommat.conversation._exchanges import BOOKKEEPING, check_turn, history

__all__ = ["drop_earlier_bookkeeping", "is_bookkeeping", "to_model_messages"]

_MARK = "ommat"


def to_model_messages(
    record: Record,
    design: LoadedDesign | None = None,
    *,
    model_name: str | None = None,
    provider_name: str | None = None,
    turn: str | int | None = None,
) -> list[ModelMessage]:
    """A record's turn history as Pydantic AI messages, with the board ``to_chat_completions``
    gives and ``[]`` where it gives none: per group of calls a ``ModelResponse`` of
    ``ToolCallPart``s, then a ``ModelRequest`` of a ``ToolReturnPart`` per call, a failed source
    call's with ``outcome="failed"`` and decide's with ``metadata``
    ``{"record_hash", "design_hash"}``. Every message carries ``metadata``
    ``{"ommat": "bookkeeping", "record_hash"}``, with ``turn`` when given. Every response carries
    ``model_name`` and ``provider_name``. ``turn`` stems every id, and is held in the mark, as
    ``to_chat_completions`` says. Raises ``ValueError`` as ``to_chat_completions`` does."""
    check_turn(turn)
    turn_history = history(record, design, turn)
    record_hash = turn_history.record_hash
    stamp = {"record_hash": record_hash, "design_hash": turn_history.design_hash}
    messages: list[ModelMessage] = []
    for group in turn_history.groups:
        calls = [
            ToolCallPart(call.name, deepcopy(call.arguments), tool_call_id=call.id)
            for call in group
        ]
        messages.append(
            ModelResponse(
                parts=calls,
                model_name=model_name,
                provider_name=provider_name,
                metadata=_marked(record_hash, turn),
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
        ]
        messages.append(ModelRequest(parts=returns, metadata=_marked(record_hash, turn)))
    return messages


def is_bookkeeping(message: ModelMessage) -> bool:
    """Whether ``message`` is one ``to_model_messages`` marked as ommat's bookkeeping: a source
    call or decide, or its return. Anything the agent or the model wrote is not."""
    metadata = message.metadata
    if type(metadata) is not dict or metadata.get(_MARK) != BOOKKEEPING:
        return False
    if isinstance(message, ModelRequest):  # a request merged with other parts is not ours
        return all(isinstance(part, ToolReturnPart) for part in message.parts)
    return all(isinstance(part, ToolCallPart) for part in message.parts)


def drop_earlier_bookkeeping(messages: list[ModelMessage]) -> list[ModelMessage]:
    """A Pydantic AI history processor: the messages with every bookkeeping call of a record
    other than the latest left out, and each return answering one of those calls, wherever it
    stands. The latest record's bookkeeping stays, since the model answers from decide's result.
    A return goes only with its call, so the history never holds one without the other."""
    marked = [
        m
        for m in messages
        if isinstance(m, ModelResponse) and is_bookkeeping(m) and type(_record(m)[0]) is str
    ]
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
    earlier = {id(m) for m in marked if _record(m) != latest}
    out: list[ModelMessage] = []
    for m in messages:
        if isinstance(m, ModelResponse):
            if id(m) in earlier:
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


def _record(message: ModelMessage) -> tuple[object, object]:
    """A marked message's record and turn, which together tell one rendering from another."""
    metadata = cast("dict[str, Any]", message.metadata)
    return metadata.get("record_hash"), metadata.get("turn")


def _marked(record_hash: str, turn: object) -> dict[str, Any]:
    mark: dict[str, Any] = {_MARK: BOOKKEEPING, "record_hash": record_hash}
    if turn is not None:
        mark["turn"] = turn
    return mark
