"""A transcript view as the tool exchanges both renderers give: per round the round's source
calls and their results, then decide's call and result, then the pending steps' calls with
no result. Each tool name is checked here, so the renderers refuse the same names."""

import re
from dataclasses import dataclass
from typing import Any, cast

_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_DECIDE = "decide"
BOOKKEEPING = "bookkeeping"
BOOKKEEPING_ID = re.compile(r"om_[0-9a-f]{24}_(?:c[1-9][0-9]*|d)")
"""A source call's or decide's id, never a pending step's, by the transcript's id form."""


@dataclass(frozen=True, slots=True)
class Call:
    """One tool call and its result: ``result`` is ``None`` for a pending step, which the
    caller runs, and ``failed`` is true for a source call whose outcome is not ``returned``."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    failed: bool = False
    decide: bool = False


def exchanges(view: dict[str, Any]) -> list[list[Call]]:
    """The view's calls in groups, each group one assistant turn: a round's calls, decide,
    then the pending steps when there are any."""
    groups: list[list[Call]] = []
    for calls in cast("list[list[dict[str, Any]]]", view["rounds"]):
        group: list[Call] = []
        for call in calls:
            name = _source_name(cast(str, call["source"]))
            group.append(
                Call(
                    call["id"],
                    name,
                    {"asks": call["asks"]},
                    call["answers"],
                    failed=call["outcome"] != "returned",
                )
            )
        groups.append(group)
    decided = cast("dict[str, Any]", view["decide"])
    result = {key: value for key, value in decided.items() if key != "id"}
    groups.append([Call(decided["id"], _DECIDE, {"design": view["design"]}, result, decide=True)])
    pending = cast("list[dict[str, Any]]", view["pending"])
    if pending:
        groups.append(
            [Call(s["id"], _step_name(cast(str, s["tool"])), s["arguments"], None) for s in pending]
        )
    return groups


def _source_name(source: str) -> str:
    """A source's tool name: its name with each ``/`` as ``-``, which no ommat name holds."""
    name = source.replace("/", "-")
    if name == _DECIDE:
        raise ValueError(f"source {source!r} renders as {_DECIDE!r}, the name decide's call takes")
    if len(name) > 64:
        raise ValueError(f"source {source!r} renders as a tool name over 64 characters")
    return name


def _step_name(tool: str) -> str:
    """A step's tool name as the design declares it, which must be the caller's tool's."""
    if tool == _DECIDE:
        raise ValueError(f"step tool {tool!r} is the name decide's call takes")
    if _TOOL_NAME.fullmatch(tool) is None:
        raise ValueError(
            f"step tool {tool!r} is not a tool name a provider takes: [A-Za-z0-9_-], 1 to 64"
        )
    return tool
