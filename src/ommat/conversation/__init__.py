"""The conversation renderers: a view rendered for an LLM conversation, outside the
compatibility promise while they are validated.

``handoff_text(handoff)`` renders a ``handoff`` view as plain text, for the caller to place as
instructions, a system or developer message, or fenced user content. ``to_chat_completions(view)``
renders a ``transcript`` view as Chat Completions messages: each source call and decide as a tool
call and its result, then the pending steps as calls with no result. A renderer words nothing:
each sentence it gives is a plan's ``reason`` or ``reply`` or a context entry's text.
``ommat.conversation.pydantic_ai`` renders a transcript as Pydantic AI messages. ``import ommat``
does not import this package, and this module takes nothing beyond the standard library.
"""

from typing import Any, cast

from ommat._canonical import encode
from ommat.conversation._exchanges import exchanges

__all__ = ["handoff_text", "to_chat_completions"]


def handoff_text(handoff: dict[str, Any] | None) -> str:
    """A ``handoff`` view as labelled lines, ``""`` for ``None`` or a view that gives none:
    ``Plan:``; ``Verdict:``, ``Reason:`` and ``Reply:`` where not null; ``Step <n>:`` with its
    tool and tagged arguments as JSON; each entry's text, or ``<ref>: <kind> (<cause code>)``
    for a marker entry; ``Design:`` with its name, where there is one, and hash; ``Record:``."""
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
            kind = cast(str, marker["kind"]).lower()
            lines.append(f"{entry['ref']}: {kind} ({marker['cause']['code']})")
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


def _json(value: object) -> str:
    return encode(value).decode("utf-8")
