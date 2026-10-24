"""The conversation renderers: a view rendered for an LLM conversation, outside the
compatibility promise while they are validated.

``handoff_text(handoff)`` renders a ``handoff`` view as plain text, for the caller to place as
instructions, a system or developer message, or fenced user content. ``to_chat_completions(view)``
renders a ``transcript`` view as Chat Completions messages: each source call and decide as a tool
call and its result, then the pending steps as calls with no result. A renderer words nothing:
each sentence it gives is a plan's ``reason`` or ``reply`` or a context entry's text.
``is_bookkeeping`` and ``drop_earlier_bookkeeping`` tell ommat's bookkeeping, its source calls
and decide, from the pending steps and the model's own messages, by the transcript's ids.
``ommat.conversation.pydantic_ai`` renders a transcript as Pydantic AI messages. ``import ommat``
does not import this package, and this module takes nothing beyond the standard library.
"""

from typing import Any, cast

from ommat._canonical import encode
from ommat.conversation._exchanges import BOOKKEEPING_ID, exchanges

__all__ = ["drop_earlier_bookkeeping", "handoff_text", "is_bookkeeping", "to_chat_completions"]


def handoff_text(handoff: dict[str, Any] | None) -> str:
    """A ``handoff`` view as labelled lines, ``""`` for ``None`` or a view that gives none:
    ``Plan:``; ``Verdict:``, ``Reason:`` and ``Reply:`` where not null; ``Step <n>:`` with its
    tool and tagged arguments as JSON; each entry's text, or ``<ref>: <kind> (<cause code>)``
    for a marker entry; ``Design:`` with its name, where there is one, and hash; ``Record:``.
    Texts are written as they are, so a text holding a newline can read as a line of its own:
    fence the whole as data where a slot or argument reads free text."""
    if handoff is None:
        return ""
    lines: list[str] = []
    plan = cast("dict[str, Any] | None", handoff.get("plan"))
    if plan is not None:
        if "kind" in plan:
            lines.append(f"Plan: {plan['kind']}")
            if plan["verdict"] is not None:
                lines.append(f"Verdict: {plan['verdict']}")
            if plan["reason"] is not None:
                lines.append(f"Reason: {plan['reason']}")
            if plan["reply"] is not None:
                lines.append(f"Reply: {plan['reply']}")
        for n, step in enumerate(cast("list[dict[str, Any]]", plan.get("steps", [])), 1):
            lines.append(f"Step {n}: {step['tool']} {_json(step['arguments'])}")
    for entry in cast("list[dict[str, Any]]", handoff["context"]):
        marker = entry["marker"]
        if marker is None:
            lines.append(cast(str, entry["text"]))
        else:
            lines.append(f"{entry['ref']}: {marker['kind']} ({marker['cause']['code']})")
    if "design" in handoff:
        design = cast("dict[str, Any]", handoff["design"])
        named = design["hash"] if design["name"] is None else f"{design['name']} {design['hash']}"
        lines.append(f"Design: {named}")
        lines.append(f"Record: {handoff['record_hash']}")
    return "\n".join(lines)


def to_chat_completions(view: dict[str, Any] | None) -> list[dict[str, Any]]:
    """A ``transcript`` view as Chat Completions messages, ``[]`` for ``None``: per group of
    calls an assistant message with null ``content`` and its ``tool_calls``, then a ``tool``
    message per call with a result. Every argument and result is canonical JSON, so a record
    renders to the same bytes each time. Raises ``ValueError`` for a tool name no provider
    takes: a source or pending tool named ``decide``, a pending tool outside ``[A-Za-z0-9_-]``,
    or a name over 64 characters."""
    if view is None:
        return []
    messages: list[dict[str, Any]] = []
    for group in exchanges(view):
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
            if call.result is not None:
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": _json(call.result)}
                )
    return messages


def is_bookkeeping(message: dict[str, Any]) -> bool:
    """Whether a Chat Completions ``message`` is ommat's bookkeeping, a source call or decide or
    its result, read from the transcript's ids: an assistant message with no text whose every
    tool call has one, or a ``tool`` message answering one. A pending call is not bookkeeping."""
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
    return encode(value).decode("utf-8")
