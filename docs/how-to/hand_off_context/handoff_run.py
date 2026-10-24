"""The how-to's driver: decide two turns of the support design with a stand-in binding, and
print what each hands the next model."""

from collections.abc import Awaitable, Callable

from ommat import CallResult, SourceCall, decide_sync, handoff, load
from ommat.conversation import to_messages, to_tool_calls
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
    (message,) = to_messages(handoff(decision.record), role="user")
    print(message["content"])
    for call in to_tool_calls(decision):
        print(call["function"])
        print(call["id"] == f"{decision.record.record_hash}:1")

    unsure = decide_sync(loaded, STATE, {"default": answering(0.6)})
    (message,) = to_messages(handoff(unsure.record), role="user")
    print(message["content"])


if __name__ == "__main__":
    main()
