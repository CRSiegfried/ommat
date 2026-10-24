"""The Pydantic AI renderer over the four paths of ``transcript_turns``, skipped when the
``pydantic-ai`` extra is not installed. The agent runs here use ``FunctionModel``, so no model
is called."""

import asyncio
import json
import re
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ommat import transcript
from ommat.conversation.pydantic_ai import to_model_messages
from transcript_turns import PATHS, turn


def view(path: str) -> dict[str, Any]:
    out = transcript(turn(path).record)
    assert out is not None
    return out


def calls(messages: list[ModelMessage]) -> list[ToolCallPart]:
    return [
        p
        for m in messages
        if isinstance(m, ModelResponse)
        for p in m.parts
        if isinstance(p, ToolCallPart)
    ]


def returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]


@pytest.mark.parametrize("path", PATHS)
def test_every_path_renders_as_responses_of_calls_and_requests_of_returns(path: str) -> None:
    v = view(path)
    messages = to_model_messages(v, model_name="ommat", provider_name="ommat")
    kinds = [type(m).__name__ for m in messages]
    rounds = len(v["rounds"])
    expected = ["ModelResponse", "ModelRequest"] * (rounds + 1)
    if v["pending"]:
        expected.append("ModelResponse")
    assert kinds == expected
    made = calls(messages)
    got = (
        [c["id"] for r in v["rounds"] for c in r]
        + [v["decide"]["id"]]
        + [s["id"] for s in v["pending"]]
    )
    assert [c.tool_call_id for c in made] == got
    assert [r.tool_call_id for r in returns(messages)] == got[: len(got) - len(v["pending"])]
    for m in messages:
        if isinstance(m, ModelResponse):
            assert (m.model_name, m.provider_name) == ("ommat", "ommat")


def test_the_parts_carry_the_view_s_names_arguments_and_results() -> None:
    v = view("clear")
    messages = to_model_messages(v)
    made = calls(messages)
    assert [c.tool_name for c in made] == [
        "ask-again",
        "mail-router",
        "lookup",
        "confirm",
        "decide",
        "archive",
        "notify",
    ]
    assert made[1].args == {"asks": ["q:route"]}
    assert made[4].args == {"design": v["design"]}
    assert made[5].args == {"folder": "old", "order": "o-1"}
    back = returns(messages)
    assert back[1].content == v["rounds"][0][1]["answers"]
    assert back[4].content == {k: x for k, x in v["decide"].items() if k != "id"}
    assert back[4].metadata == {"record_hash": v["record_hash"], "design_hash": v["design"]["hash"]}
    assert [r.metadata for r in back[:4]] == [None] * 4
    assert {r.outcome for r in back} == {"success"}


def test_a_failed_call_s_return_is_its_markers_with_outcome_failed() -> None:
    v = view("down")
    back = returns(to_model_messages(v))
    assert [r.outcome for r in back] == ["failed", "failed", "success", "success"]
    assert back[1].content == v["rounds"][0][1]["answers"]


def test_the_messages_do_not_share_the_view_s_objects() -> None:
    v = view("clear")
    made = calls(to_model_messages(v))
    made[0].args["asks"].append("x")  # type: ignore[index, union-attr]
    assert v["rounds"][0][0]["asks"] == ["q:again"]


def test_the_messages_survive_pydantic_ai_s_own_serialization() -> None:
    messages = to_model_messages(view("down"), model_name="ommat")
    again = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert [(type(m).__name__, len(m.parts)) for m in again] == [
        (type(m).__name__, len(m.parts)) for m in messages
    ]
    assert returns(list(again))[0].outcome == "failed"


def test_none_renders_as_no_messages() -> None:
    assert to_model_messages(None) == []


def test_a_tool_name_no_provider_takes_is_refused() -> None:
    v = json.loads(json.dumps(view("clear")))
    v["pending"][0]["tool"] = "mail/send"
    with pytest.raises(ValueError, match=re.escape("not a tool name a provider takes")):
        to_model_messages(v)


def test_an_agent_runs_the_pending_calls_with_its_own_tools_and_continues() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("Archived.")])

    agent = Agent(FunctionModel(front))
    ran: list[tuple[str, dict[str, Any]]] = []

    @agent.tool_plain
    def archive(order: str, folder: str) -> dict[str, str]:
        ran.append(("archive", {"order": order, "folder": folder}))
        return {"receipt": "r-1"}

    @agent.tool_plain
    def notify(receipt: str) -> None:
        ran.append(("notify", {"receipt": receipt}))

    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Archive the newsletters")])]
    history += to_model_messages(view("clear"), model_name="ommat")
    result = asyncio.run(agent.run(None, message_history=history))
    assert ran == [("archive", {"order": "o-1", "folder": "old"}), ("notify", {"receipt": "done"})]
    assert result.output == "Archived."
    last = seen[-1][-1]
    assert isinstance(last, ModelRequest)
    assert [p.tool_name for p in last.parts if isinstance(p, ToolReturnPart)] == [
        "archive",
        "notify",
    ]
