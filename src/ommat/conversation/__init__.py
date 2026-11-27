"""The conversation renderers: a record rendered as its turn history for the next model in an
LLM conversation, outside the compatibility promise.

``to_chat_completions(record, design, refs=...)`` renders a turn as Chat Completions messages:
each source call and decide as a tool call and its result, minimal, decide's result the entries
at the caller's ``refs``. A renderer words nothing.
``is_bookkeeping`` and ``drop_earlier_bookkeeping`` tell ommat's bookkeeping, its source calls
and decide, from the model's own messages, by their ids.
``ommat.conversation.pydantic_ai`` renders a turn as Pydantic AI messages. ``import ommat``
does not import this package, and this module takes nothing beyond the standard library.
"""

from collections.abc import Sequence
from typing import Any, cast

from ommat._canonical import ordered
from ommat._design import LoadedDesign
from ommat._ledger import Record
from ommat.conversation._exchanges import BOOKKEEPING_ID, check_turn, history

__all__ = ["drop_earlier_bookkeeping", "is_bookkeeping", "to_chat_completions"]


def to_chat_completions(
    record: Record,
    design: LoadedDesign | None = None,
    *,
    refs: Sequence[str],
    turn: str | int | None = None,
) -> list[dict[str, Any]]:
    """A record's turn history as Chat Completions messages, ``[]`` for a record before format 8
    or one with no entry at one of ``refs``: per group of calls an assistant message with null
    ``content`` and its ``tool_calls``, then a ``tool`` message per call. ``refs``, a non-empty
    ``list`` or ``tuple`` of distinct ``d:`` refs of the record's rules, names the entries the
    caller acts on: decide's result is ``{ref: {"value": v}}`` or ``{ref: {"undecided": {"kind",
    "root_cause", "refs"}}}`` per ref, in their order, and decide's arguments are every other
    ``d:`` entry, cut with ``design``, the record's loaded design, to those the rules of ``refs``
    read and every marker. Every argument and result is in the ordered encoding, so a record
    renders to the same bytes each time. Raises ``ValueError`` for a ``record`` that is not
    exactly a ``Record``, a ``design`` that is not its ``LoadedDesign``, ``refs`` that are not
    such refs, or a tool name no provider takes: a source named ``decide`` or a name over 64
    characters. ``turn``, exactly a ``str`` or an ``int`` the canonical encoding takes, else
    ``ValueError``, gives every id a stem hashed from the record hash and the turn, so one record
    rendered for two turns of a conversation repeats no id; ``None`` keeps the record's ids."""
    check_turn(turn)
    messages: list[dict[str, Any]] = []
    for group in history(record, design, refs, turn).groups:
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
    its result, read from their ids: an assistant message with no text whose every
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
