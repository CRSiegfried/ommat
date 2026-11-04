"""The conversation renderers: a view rendered for an LLM conversation, outside the
compatibility promise while they are validated.

``to_chat_completions(view)`` renders a ``transcript`` view as Chat Completions messages: each
source call and decide as a tool call and its result. A renderer words nothing.
``is_bookkeeping`` and ``drop_earlier_bookkeeping`` tell ommat's bookkeeping, its source calls
and decide, from the model's own messages, by the transcript's ids.
``ommat.conversation.pydantic_ai`` renders a transcript as Pydantic AI messages. ``import ommat``
does not import this package, and this module takes nothing beyond the standard library.
"""

from typing import Any, cast

from ommat._canonical import ordered
from ommat.conversation._exchanges import BOOKKEEPING_ID, check_turn, exchanges

__all__ = ["drop_earlier_bookkeeping", "is_bookkeeping", "to_chat_completions"]


def to_chat_completions(
    view: dict[str, Any] | None, *, turn: str | int | None = None
) -> list[dict[str, Any]]:
    """A ``transcript`` view as Chat Completions messages, ``[]`` for ``None``: per group of calls
    an assistant message with null ``content`` and its ``tool_calls``, then a ``tool`` message per
    call. Every argument and result is in the ordered encoding, each object in the order the
    record holds it, so a record renders to the same bytes each time. Raises ``ValueError`` for a
    tool name no provider takes: a source named ``decide`` or a name over 64 characters. ``turn``,
    exactly a ``str`` or an ``int`` the canonical encoding takes, else ``ValueError``, gives every
    id a stem hashed from the record hash and the turn, so one record rendered for two turns of a
    conversation repeats no id; ``None`` keeps the transcript's ids."""
    check_turn(turn)
    if view is None:
        return []
    messages: list[dict[str, Any]] = []
    for group in exchanges(view, turn):
        calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": _json(call.arguments)},
            }
            for call in group
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in group:
            content = _json(call.result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return messages


def is_bookkeeping(message: dict[str, Any]) -> bool:
    """Whether a Chat Completions ``message`` is ommat's bookkeeping, a source call or decide or
    its result, read from the transcript's ids: an assistant message with no text whose every
    tool call has one, or a ``tool`` message answering one."""
    if message.get("role") == "tool":
        return _bookkept(message.get("tool_call_id"))
    if message.get("role") != "assistant" or message.get("content") not in (None, ""):
        return False
    calls = message.get("tool_calls")
    if type(calls) is not list or not calls:
        return False
    return all(_bookkept(_call_id(c)) for c in cast("list[Any]", calls))


def drop_earlier_bookkeeping(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Chat Completions messages with every bookkeeping call message of a record other than
    the latest left out, and each ``tool`` message answering one of those calls. The latest
    record's bookkeeping stays; a result goes only with its call."""
    calls = [m for m in messages if m.get("role") == "assistant" and is_bookkeeping(m)]
    if not calls:
        return list(messages)
    latest = _stem(calls[-1])
    gone = {
        _call_id(c) for m in calls if _stem(m) != latest for c in cast("list[Any]", m["tool_calls"])
    }
    earlier = {id(m) for m in calls if _stem(m) != latest}
    return [
        m
        for m in messages
        if id(m) not in earlier and not (m.get("role") == "tool" and _answers(m, gone))
    ]


def _answers(message: dict[str, Any], gone: set[object]) -> bool:
    call_id = message.get("tool_call_id")
    return type(call_id) is str and call_id in gone


def _call_id(call: object) -> object:
    return cast("dict[str, Any]", call).get("id") if type(call) is dict else None


def _stem(message: dict[str, Any]) -> str:
    """The record part of a bookkeeping call message's first id: ``om_`` and 24 hex digits."""
    return cast(str, _call_id(message["tool_calls"][0]))[:27]


def _bookkept(call_id: object) -> bool:
    return type(call_id) is str and BOOKKEEPING_ID.fullmatch(call_id) is not None


def _json(value: object) -> str:
    return ordered(value)
