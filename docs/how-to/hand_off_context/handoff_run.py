"""The how-to's driver: decide two turns of the support design with a stand-in binding, and
print what each hands the next model and the turn as Chat Completions messages."""

from collections.abc import Awaitable, Callable

from ommat import CallResult, SourceCall, decide_sync, handoff, load, transcript
from ommat.conversation import handoff_text, to_chat_completions
from support_design import DESIGN

Binding = Callable[..., Awaitable[CallResult]]

STATE = {
    "message": "The blender arrived broken. I want my money back.",
    "email": "sam@example.com",
    "order": {"id": "A-1001", "status": "delivered"},
}


def answering(yes: float) -> Binding:
    """A stand-in for a model: it answers the refund question with ``yes`` at ``yes``."""

    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({"q:refund": {"yes": yes, "no": round(1 - yes, 2)}})

    return binding


def main() -> None:
    loaded = load(DESIGN)
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    print(handoff_text(handoff(decision.record)))

    for message in to_chat_completions(transcript(decision.record)):
        if message["role"] == "assistant":
            for call in message["tool_calls"]:
                print("call", call["id"], call["function"]["name"], call["function"]["arguments"])
        else:
            print("result", message["tool_call_id"], len(message["content"]), "bytes")

    unsure = decide_sync(loaded, STATE, {"default": answering(0.6)})
    print(handoff_text(handoff(unsure.record)))


if __name__ == "__main__":
    main()
