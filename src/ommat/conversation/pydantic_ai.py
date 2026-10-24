"""The Pydantic AI renderer: a ``transcript`` view as Pydantic AI messages, outside the
compatibility promise while it is validated. Install it with ``ommat[pydantic-ai]``.

``to_model_messages(view)`` gives each source call and decide as a ``ModelResponse`` of tool
calls and a ``ModelRequest`` of their returns, then the pending steps as a trailing
``ModelResponse``. A history ending in those calls, run with no new user prompt, has the agent
run them with its own tools. Every ``ModelResponse`` carries the given ``model_name`` and
``provider_name``, so a caller can keep these messages out of the state it builds.
"""

from copy import deepcopy
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from ommat.conversation._exchanges import exchanges

__all__ = ["to_model_messages"]


def to_model_messages(
    view: dict[str, Any] | None,
    *,
    model_name: str | None = None,
    provider_name: str | None = None,
) -> list[ModelMessage]:
    """A ``transcript`` view as Pydantic AI messages, ``[]`` for ``None``: per group of calls a
    ``ModelResponse`` of ``ToolCallPart``s, then a ``ModelRequest`` of a ``ToolReturnPart`` per
    call with a result, a failed source call's with ``outcome="failed"`` and decide's with
    ``metadata`` ``{"record_hash", "design_hash"}``. Raises ``ValueError`` for a tool name no
    provider takes, as ``to_chat_completions`` does."""
    if view is None:
        return []
    stamp = {"record_hash": view["record_hash"], "design_hash": view["design"]["hash"]}
    messages: list[ModelMessage] = []
    for group in exchanges(view):
        calls = [
            ToolCallPart(call.name, deepcopy(call.arguments), tool_call_id=call.id)
            for call in group
        ]
        messages.append(
            ModelResponse(parts=calls, model_name=model_name, provider_name=provider_name)
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
            messages.append(ModelRequest(parts=returns))
    return messages
