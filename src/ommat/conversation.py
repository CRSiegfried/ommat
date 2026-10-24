"""The conversation renderers: a turn rendered for an LLM conversation, outside the
compatibility promise while they are validated.

``to_messages(handoff, *, role)`` renders a ``handoff`` view as one plain-text message;
``role`` has no default, since an entry's text may carry the input's own text and the caller
picks the role for it. ``to_tool_calls(decision)`` renders a passed ``execute`` plan as
OpenAI-shape tool calls, each id the step's idempotency key, ending before the first step with
a binding argument; a caller with such a plan runs it with ``run_plan``, whose keys match.
``import ommat`` does not import this module, and it takes nothing beyond the standard library.
"""

from typing import Any, cast

from ommat._canonical import encode
from ommat._decide import Decision
from ommat._markers import PlanKind, VerdictKind
from ommat._resolve import board_entries, board_value
from ommat._values import type_name

__all__ = ["to_messages", "to_tool_calls"]


def to_messages(handoff: dict[str, Any] | None, *, role: str) -> list[dict[str, str]]:
    """One ``{"role", "content"}`` message from a ``handoff`` view, or ``[]`` for ``None``. Its
    lines, in order: the plan's kind, then its reason and reply where not null; a line per
    step, its tool and tagged arguments as JSON; a line per entry, its text, or for a marker
    entry ``<ref>: not settled (<kind>: <cause code>)``; and the design and record hash."""
    if handoff is None:
        return []
    lines: list[str] = []
    plan = cast("dict[str, Any] | None", handoff.get("plan"))
    if plan is not None:
        if "kind" in plan:
            lines.append(f"Plan: {plan['kind']}")
            if plan["reason"] is not None:
                lines.append(f"Reason: {plan['reason']}")
            if plan["reply"] is not None:
                lines.append(f"Reply: {plan['reply']}")
        for n, step in enumerate(cast("list[dict[str, Any]]", plan.get("steps", [])), 1):
            arguments = encode(step["arguments"]).decode("utf-8")
            lines.append(f"Step {n}: {step['tool']} {arguments}")
    for entry in cast("list[dict[str, Any]]", handoff["context"]):
        marker = entry["marker"]
        if marker is None:
            lines.append(cast(str, entry["text"]))
        else:
            kind = cast(str, marker["kind"]).lower()
            lines.append(f"{entry['ref']}: not settled ({kind}: {marker['cause']['code']})")
    if "design" in handoff:
        design = cast("dict[str, Any]", handoff["design"])
        named = design["hash"] if design["name"] is None else f"{design['name']} ({design['hash']})"
        lines.append(f"Decided by design {named}; record {handoff['record_hash']}")
    return [{"role": role, "content": "\n".join(lines)}]


def to_tool_calls(decision: Decision) -> list[dict[str, Any]]:
    """OpenAI-shape tool calls for a passed ``execute`` plan, ``[]`` for any other plan or a
    refused verdict: ``{"id", "type": "function", "function": {"name", "arguments"}}``, the
    arguments a JSON string of the values resolved from the record and the id the step's
    idempotency key; the calls end before the first step with a binding argument."""
    if type(decision) is not Decision:
        raise ValueError(f"to_tool_calls takes a Decision, not {type_name(decision)}")
    if (
        decision.verdict.kind is not VerdictKind.passed
        or decision.plan.kind is not PlanKind.execute
    ):
        return []
    record = decision.record
    log = record.log
    item = next(i for i in log if i["item"] == "plan")
    entries = board_entries(log)
    calls: list[dict[str, Any]] = []
    for n, step in enumerate(cast("list[dict[str, Any]]", item["steps"]), 1):
        origins = cast("dict[str, dict[str, Any]]", step["arguments"])
        if any("step" in origin for origin in origins.values()):
            break  # a binding resolves only from an earlier tool's result
        arguments: dict[str, Any] = {}
        for name, origin in origins.items():
            if "literal" in origin:
                arguments[name] = origin["literal"]
                continue
            ref = cast(str, origin["ref"])
            try:
                arguments[name] = board_value(entries, ref)
            except (KeyError, TypeError):
                raise ValueError(
                    f"step {n}'s argument {name!r} names {ref!r}, which holds no value in the log"
                ) from None
        calls.append(
            {
                "id": f"{record.record_hash}:{n}",
                "type": "function",
                "function": {"name": step["tool"], "arguments": encode(arguments).decode("utf-8")},
            }
        )
    return calls
