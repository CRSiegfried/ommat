"""The Pydantic AI renderer over the four paths of ``transcript_turns``, skipped when the
``pydantic-ai`` extra is not installed. The agent runs here use ``FunctionModel``, so no model
is called."""

import asyncio
import json
import re
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
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
from ommat.conversation.pydantic_ai import (
    drop_earlier_bookkeeping,
    is_bookkeeping,
    to_model_messages,
)
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


# ---------------------------------------------------------------------------------------------
# Bookkeeping


def test_the_source_calls_and_decide_are_bookkeeping_and_the_pending_calls_are_not() -> None:
    messages = to_model_messages(view("clear"), model_name="ommat")
    marks = [is_bookkeeping(m) for m in messages]
    assert marks == [True] * 6 + [False]
    assert messages[-1].metadata is None
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].model_name == "ommat"
    record_hash = view("clear")["record_hash"]
    assert messages[0].metadata == {"ommat": "bookkeeping", "record_hash": record_hash}


def test_the_mark_survives_pydantic_ai_s_serialization() -> None:
    messages = to_model_messages(view("clear"))
    again = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert [is_bookkeeping(m) for m in again] == [is_bookkeeping(m) for m in messages]


def test_nothing_the_model_or_the_agent_wrote_is_bookkeeping() -> None:
    assert not is_bookkeeping(ModelRequest(parts=[UserPromptPart("hi")]))
    assert not is_bookkeeping(ModelResponse(parts=[TextPart("hi")], metadata={"ommat": "x"}))


def _paired(messages: list[ModelMessage]) -> bool:
    called = [c.tool_call_id for c in calls(messages)]
    answered = [r.tool_call_id for r in returns(messages)]
    return sorted(called) == sorted(answered)


def test_a_history_processor_drops_earlier_turns_bookkeeping_over_two_turns() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])

    @agent.tool_plain
    def archive(order: str, folder: str) -> dict[str, str]:
        return {"receipt": "r-1"}

    @agent.tool_plain
    def notify(receipt: str) -> None:
        return None

    first: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Archive the newsletters")])]
    first += to_model_messages(view("clear"), model_name="ommat")
    one = asyncio.run(agent.run(None, message_history=first))
    assert any(is_bookkeeping(m) for m in seen[-1])  # the turn's own decide is shown
    stored = ModelMessagesTypeAdapter.validate_json(
        ModelMessagesTypeAdapter.dump_json(one.all_messages())
    )
    second = [*stored, ModelRequest(parts=[UserPromptPart("Are you sure?")])]
    second += to_model_messages(view("unsure"), model_name="ommat")
    asyncio.run(agent.run(None, message_history=second))
    shown = seen[-1]
    assert _paired(shown)
    names = [c.tool_name for c in calls(shown)]
    assert names == ["archive", "notify", "ask-again", "mail-router", "lookup", "confirm", "decide"]
    assert (
        sum(
            isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)
            for m in shown
        )
        == 2
    )


def test_the_processor_keeps_a_lone_record_whole() -> None:
    messages = to_model_messages(view("down"))
    assert drop_earlier_bookkeeping(messages) == messages


def _two_turns(*, merged: bool = False) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Turn one ends at decide's return; turn two's prompt follows it, merged into that request
    when ``merged``, as Pydantic AI 2.49 merges consecutive requests before a processor runs."""
    earlier = [ModelRequest(parts=[UserPromptPart("first")]), *to_model_messages(view("unsure"))]
    prompt = UserPromptPart("second")
    if merged:
        last = earlier[-1]
        assert isinstance(last, ModelRequest)
        earlier[-1] = ModelRequest(parts=[*last.parts, prompt])  # the merge keeps no metadata
    else:
        earlier.append(ModelRequest(parts=[prompt]))
    return earlier, to_model_messages(view("down"))


@pytest.mark.parametrize("merged", [False, True])
def test_an_earlier_record_leaves_whole_even_from_a_merged_request(merged: bool) -> None:
    earlier, current = _two_turns(merged=merged)
    kept = drop_earlier_bookkeeping(earlier + current)
    assert _paired(kept)
    assert kept[-len(current) :] == current
    prompts = [
        p
        for m in kept
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, UserPromptPart)
    ]
    assert [p.content for p in prompts] == ["first", "second"]
    assert not any(is_bookkeeping(m) for m in kept[: -len(current)])


def test_the_latest_record_stays_wherever_the_prompt_stands() -> None:
    current = to_model_messages(view("unsure"))
    after = [*current, ModelRequest(parts=[UserPromptPart("late")])]
    assert drop_earlier_bookkeeping(after) == after
    returned = ModelRequest(
        parts=[ToolReturnPart("archive", "r", tool_call_id="x"), UserPromptPart("an image")]
    )
    assert drop_earlier_bookkeeping([*current, returned]) == [*current, returned]


def test_a_return_whose_call_lost_its_mark_stays_with_it() -> None:
    earlier, current = _two_turns()
    earlier[1] = replace(earlier[1], metadata=None)
    kept = drop_earlier_bookkeeping(earlier + current)
    assert _paired(kept)
    assert earlier[1] in kept


def test_an_agent_sees_pairs_when_a_turn_ends_at_decide_s_return() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("first")])]
    history += to_model_messages(view("unsure"), model_name="ommat")
    history.append(ModelRequest(parts=[UserPromptPart("second")]))  # no reply between
    history += to_model_messages(view("down"), model_name="ommat")
    asyncio.run(agent.run(None, message_history=history))
    shown = seen[-1]
    assert _paired(shown)
    assert [c.tool_name for c in calls(shown)] == ["ask-again", "mail-router", "lookup", "decide"]


def test_a_marked_response_with_no_record_hash_does_not_become_the_latest() -> None:
    messages = to_model_messages(view("unsure"))
    foreign = ModelResponse(parts=[TextPart("x")], metadata={"ommat": "bookkeeping"})
    assert drop_earlier_bookkeeping([*messages, foreign]) == [*messages, foreign]


def test_a_marked_request_holding_a_prompt_is_not_bookkeeping() -> None:
    returns_ = to_model_messages(view("unsure"))[1]
    assert isinstance(returns_, ModelRequest)
    mixed = replace(returns_, parts=[*returns_.parts, UserPromptPart("hi")])
    assert is_bookkeeping(returns_)
    assert not is_bookkeeping(mixed)


@pytest.mark.parametrize(
    ("where", "name", "message"),
    [
        ("source", "decide", "the name decide's call takes"),
        ("source", "a/" + "b" * 63, "over 64 characters"),
        ("tool", "decide", "the name decide's call takes"),
    ],
)
def test_the_names_refused_are_the_chat_renderer_s(where: str, name: str, message: str) -> None:
    v = json.loads(json.dumps(view("clear")))
    if where == "source":
        v["rounds"][0][0]["source"] = name
    else:
        v["pending"][0]["tool"] = name
    with pytest.raises(ValueError, match=re.escape(message)):
        to_model_messages(v)


def test_a_marked_response_holding_text_is_not_bookkeeping() -> None:
    call = to_model_messages(view("unsure"))[0]
    assert isinstance(call, ModelResponse)
    assert not is_bookkeeping(replace(call, parts=[*call.parts, TextPart("x")]))


def test_five_agent_turns_repeating_one_record_repeat_no_id() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])

    @agent.tool_plain
    def archive(order: str, folder: str) -> dict[str, str]:
        return {"receipt": "r-1"}

    @agent.tool_plain
    def notify(receipt: str) -> None:
        return None

    history: list[ModelMessage] = []
    for n, path in enumerate(["clear", "unsure", "clear", "down", "clear"]):
        history.append(ModelRequest(parts=[UserPromptPart(f"turn {n}")]))
        history += to_model_messages(view(path), model_name="ommat", turn=n)
        result = asyncio.run(agent.run(None, message_history=history))
        history = list(
            ModelMessagesTypeAdapter.validate_json(
                ModelMessagesTypeAdapter.dump_json(result.all_messages())
            )
        )
        shown = seen[-1]
        assert _paired(shown)
        ids_ = [c.tool_call_id for c in calls(shown)]
        assert len(ids_) == len(set(ids_))
        marked = {m.metadata["turn"] for m in shown if is_bookkeeping(m) and m.metadata}
        assert marked == {n}
