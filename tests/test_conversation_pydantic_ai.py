"""The Pydantic AI renderer over the four paths of ``transcript_turns``, skipped when the
``pydantic-ai`` extra is not installed. The agent runs here use ``FunctionModel``, so no model
is called."""

import asyncio
import re
from dataclasses import replace
from typing import Any, cast

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

from ommat import CallResult, Question, Record, Source, Text, decide_sync, load
from ommat.conversation.pydantic_ai import (
    drop_earlier_bookkeeping,
    is_bookkeeping,
    to_model_messages,
)
from transcript_turns import PATHS, SOURCES, STATE, YES, confirming, design, tools, turn

LOADED = load(design())
REFS = ["d:act"]
"""The ref the caller acts on: the policy's."""


def rec(path: str) -> Record:
    return turn(path, LOADED).record


def rendered(path: str, **kw: Any) -> list[ModelMessage]:
    return to_model_messages(rec(path), LOADED, **kw, refs=REFS)


def named(name: str) -> Record:
    """A clear turn with one more source, ``name``, asked through the ``model`` oracle."""
    source = Source(
        {"q:extra": Question("noul", Text("Extra?"))}, needs=["t:state.message"], oracle="model"
    )

    async def answers(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({"q:route": YES, "q:again": YES, "q:extra": YES})

    ld = load(design(sources={**SOURCES, name: source}))
    bindings = {"default": confirming(YES), "model": answers, "tools": tools}
    return decide_sync(ld, STATE, bindings).record


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
    record = rec(path)
    messages = to_model_messages(
        record, LOADED, model_name="ommat", provider_name="ommat", refs=REFS
    )
    kinds = [type(m).__name__ for m in messages]
    rounds = len({i["round"] for i in record.log if i["item"] == "call"})
    assert kinds == ["ModelResponse", "ModelRequest"] * (rounds + 1)
    made = calls(messages)
    assert [r.tool_call_id for r in returns(messages)] == [c.tool_call_id for c in made]
    for m in messages:
        if isinstance(m, ModelResponse):
            assert (m.model_name, m.provider_name) == ("ommat", "ommat")


def test_the_parts_carry_the_history_s_names_arguments_and_results() -> None:
    record = rec("clear")
    messages = to_model_messages(record, LOADED, refs=REFS)
    made = calls(messages)
    assert [c.tool_name for c in made] == [
        "ask-again",
        "mail-router",
        "lookup",
        "confirm",
        "decide",
    ]
    assert [c.args for c in made[:4]] == [{}] * 4
    assert made[4].args == {"d:sure": "yes"}
    back = returns(messages)
    assert back[1].content == {"q:route": ["yes", 0.95]}
    assert back[4].content == {"d:act": {"value": "archive"}}
    stamp = {"record_hash": record.record_hash, "design_hash": record.design_hash}
    assert back[4].metadata == stamp
    assert [r.metadata for r in back[:4]] == [None] * 4
    assert {r.outcome for r in back} == {"success"}


def test_a_failed_call_s_return_is_its_markers_with_outcome_failed() -> None:
    back = returns(rendered("down"))
    assert [r.outcome for r in back] == ["failed", "failed", "success", "success"]
    assert back[1].content == {"q:route": {"marker": "Unavailable", "cause": "raised"}}


def test_the_messages_do_not_share_the_record_s_objects() -> None:
    record = rec("clear")
    back = returns(to_model_messages(record, LOADED, refs=REFS))
    cast("dict[str, Any]", back[2].content)["t:order"]["id"] = "changed"
    assert record.entries["t:order"]["value"] == {"id": "o-1"}
    assert returns(to_model_messages(record, LOADED, refs=REFS))[2].content == {
        "t:order": {"id": "o-1"}
    }


def test_the_messages_survive_pydantic_ai_s_own_serialization() -> None:
    messages = rendered("down", model_name="ommat")
    again = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))
    assert [(type(m).__name__, len(m.parts)) for m in again] == [
        (type(m).__name__, len(m.parts)) for m in messages
    ]
    assert returns(list(again))[0].outcome == "failed"


def test_what_is_not_a_record_is_refused() -> None:
    with pytest.raises(ValueError, match="a renderer takes a Record"):
        to_model_messages(None, refs=REFS)  # type: ignore[arg-type]


def test_refs_are_refused_as_the_chat_renderer_refuses_them() -> None:
    with pytest.raises(ValueError, match="at least one"):
        to_model_messages(rec("clear"), LOADED, refs=[])
    with pytest.raises(ValueError, match="no rule's ref"):
        to_model_messages(rec("clear"), LOADED, refs=["d:nothing"])


def test_an_agent_answers_from_the_history_ending_at_decide_s_return() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("Archived.")])

    agent = Agent(FunctionModel(front))
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Archive the newsletters")])]
    history += rendered("clear", model_name="ommat")
    result = asyncio.run(agent.run(None, message_history=history))
    assert result.output == "Archived."
    last = seen[-1][-1]
    assert isinstance(last, ModelRequest)
    assert [p.tool_name for p in last.parts if isinstance(p, ToolReturnPart)] == ["decide"]


# ---------------------------------------------------------------------------------------------
# Bookkeeping


def test_the_source_calls_and_decide_are_bookkeeping() -> None:
    messages = rendered("clear", model_name="ommat")
    marks = [is_bookkeeping(m) for m in messages]
    assert marks == [True] * 6
    record_hash = rec("clear").record_hash
    assert messages[0].metadata == {"ommat": "bookkeeping", "record_hash": record_hash}


def test_the_mark_survives_pydantic_ai_s_serialization() -> None:
    messages = rendered("clear")
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

    first: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Archive the newsletters")])]
    first += rendered("clear", model_name="ommat")
    one = asyncio.run(agent.run(None, message_history=first))
    assert any(is_bookkeeping(m) for m in seen[-1])  # the turn's own decide is shown
    stored = ModelMessagesTypeAdapter.validate_json(
        ModelMessagesTypeAdapter.dump_json(one.all_messages())
    )
    second = [*stored, ModelRequest(parts=[UserPromptPart("Are you sure?")])]
    second += rendered("unsure", model_name="ommat")
    asyncio.run(agent.run(None, message_history=second))
    shown = seen[-1]
    assert _paired(shown)
    names = [c.tool_name for c in calls(shown)]
    assert names == ["ask-again", "mail-router", "lookup", "confirm", "decide"]
    assert (
        sum(
            isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts)
            for m in shown
        )
        == 2
    )


def test_the_processor_keeps_a_lone_record_whole() -> None:
    messages = rendered("down")
    assert drop_earlier_bookkeeping(messages) == messages


def _two_turns(*, merged: bool = False) -> tuple[list[ModelMessage], list[ModelMessage]]:
    """Turn one ends at decide's return; turn two's prompt follows it, merged into that request
    when ``merged``, as Pydantic AI 2.49 merges consecutive requests before a processor runs."""
    earlier = [ModelRequest(parts=[UserPromptPart("first")]), *rendered("unsure")]
    prompt = UserPromptPart("second")
    if merged:
        last = earlier[-1]
        assert isinstance(last, ModelRequest)
        earlier[-1] = ModelRequest(parts=[*last.parts, prompt])  # the merge keeps no metadata
    else:
        earlier.append(ModelRequest(parts=[prompt]))
    return earlier, rendered("down")


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
    current = rendered("unsure")
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
    history += rendered("unsure", model_name="ommat")
    history.append(ModelRequest(parts=[UserPromptPart("second")]))  # no reply between
    history += rendered("down", model_name="ommat")
    asyncio.run(agent.run(None, message_history=history))
    shown = seen[-1]
    assert _paired(shown)
    assert [c.tool_name for c in calls(shown)] == ["ask-again", "mail-router", "lookup", "decide"]


def test_a_marked_response_with_no_record_hash_does_not_become_the_latest() -> None:
    messages = rendered("unsure")
    foreign = ModelResponse(parts=[TextPart("x")], metadata={"ommat": "bookkeeping"})
    assert drop_earlier_bookkeeping([*messages, foreign]) == [*messages, foreign]


def test_a_marked_request_holding_a_prompt_is_not_bookkeeping() -> None:
    returns_ = rendered("unsure")[1]
    assert isinstance(returns_, ModelRequest)
    mixed = replace(returns_, parts=[*returns_.parts, UserPromptPart("hi")])
    assert is_bookkeeping(returns_)
    assert not is_bookkeeping(mixed)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("decide", "the name decide's call takes"),
        ("a/" + "b" * 63, "over 64 characters"),
    ],
)
def test_the_names_refused_are_the_chat_renderer_s(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        to_model_messages(named(name), refs=REFS)


def test_a_marked_response_holding_text_is_not_bookkeeping() -> None:
    call = rendered("unsure")[0]
    assert isinstance(call, ModelResponse)
    assert not is_bookkeeping(replace(call, parts=[*call.parts, TextPart("x")]))


def test_five_agent_turns_repeating_one_record_repeat_no_id() -> None:
    seen: list[list[ModelMessage]] = []

    def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])

    history: list[ModelMessage] = []
    for n, path in enumerate(["clear", "unsure", "clear", "down", "clear"]):
        history.append(ModelRequest(parts=[UserPromptPart(f"turn {n}")]))
        history += rendered(path, model_name="ommat", turn=n)
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
