"""The how-to's Pydantic AI driver: two turns rendered as Pydantic AI messages. In the first the
decision is ``refund_order``, which the caller carries out itself before the agent answers; in
the second the decision is ``ask_if_refund``, and the caller appends its own question. A
stand-in function plays the model, so it runs with no model; it needs ``ommat[pydantic-ai]``."""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ommat import decide_sync, load
from ommat.conversation.pydantic_ai import (
    drop_earlier_bookkeeping,
    is_bookkeeping,
    to_model_messages,
)
from support_design import DESIGN
from turn_run import STATE, answering


def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A stand-in for the front model: it answers once the refund has run."""
    return ModelResponse(parts=[TextPart("Your refund is on its way.")])


agent = Agent(FunctionModel(front), capabilities=[ProcessHistory(drop_earlier_bookkeeping)])


def refund_order(order: str) -> dict[str, str]:
    """The caller's own action on the decision ``refund_order``."""
    print("refunding", order)
    return {"refund": f"R-{order}"}


def main() -> None:
    loaded = load(DESIGN)
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(str(STATE["message"]))])]
    history += to_model_messages(decision.record, loaded, refs=["d:policy"], model_name="ommat")
    if decision.record.entries["d:policy"].get("value") == "refund_order":
        refund_order(str(STATE["order"]["id"]))  # type: ignore[index]
    result = asyncio.run(agent.run(None, message_history=history))
    print(result.output)

    said = "Hmm, maybe. Not sure yet."
    again = {**STATE, "message": said}
    unsure = decide_sync(loaded, again, {"default": answering(0.6)})
    history = result.all_messages()
    history.append(ModelRequest(parts=[UserPromptPart(said)]))
    history += to_model_messages(unsure.record, loaded, refs=["d:policy"], model_name="ommat")
    if unsure.record.entries["d:policy"].get("value") == "ask_if_refund":
        reply = "Would you like a refund for this order?"
        history.append(ModelResponse(parts=[TextPart(reply)], model_name="ommat"))
        print(reply)
    kept = drop_earlier_bookkeeping(history)
    print(
        sum(map(is_bookkeeping, history)),
        "bookkeeping messages,",
        sum(map(is_bookkeeping, kept)),
        "kept",
    )


if __name__ == "__main__":
    main()
