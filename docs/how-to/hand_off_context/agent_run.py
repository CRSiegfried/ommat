"""The how-to's Pydantic AI driver: render a turn as Pydantic AI messages and let an agent run
its pending call with its own tool. A stand-in function plays the model, so it runs with no
model; it needs ``ommat[pydantic-ai]``."""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from handoff_run import STATE, answering
from ommat import decide_sync, load, transcript
from ommat.conversation.pydantic_ai import to_model_messages
from support_design import DESIGN


def front(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A stand-in for the front model: it answers once the refund has run."""
    return ModelResponse(parts=[TextPart("Your refund is on its way.")])


agent = Agent(FunctionModel(front))


@agent.tool_plain
def refund_order(order: str) -> dict[str, str]:
    print("refunding", order)
    return {"refund": f"R-{order}"}


def main() -> None:
    decision = decide_sync(load(DESIGN), STATE, {"default": answering(0.9)})
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(str(STATE["message"]))])]
    history += to_model_messages(transcript(decision.record), model_name="ommat")
    result = asyncio.run(agent.run(None, message_history=history))
    print(result.output)


if __name__ == "__main__":
    main()
