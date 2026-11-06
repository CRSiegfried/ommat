"""The how-to's driver: decide three turns of the support design with ``fixed`` standing in
for the model, and print the first turn's calls as Chat Completions messages and the note each
turn hands on."""

from ommat import Decision, decide_sync, escalation, load
from ommat.bindings import OracleBinding, fixed
from ommat.conversation import to_chat_completions
from support_design import DESIGN

STATE = {
    "message": "The blender arrived broken. I want my money back.",
    "email": "sam@example.com",
    "order": {"id": "A-1001", "status": "delivered"},
}


def answering(yes: float) -> OracleBinding:
    """A stand-in for a model: it answers the refund question with ``yes`` at ``yes``."""
    return fixed({"q:refund": {"yes": yes, "no": round(1 - yes, 2)}})


SILENT = fixed({})
"""A stand-in for a model that cannot answer: the refund question comes back ``unsupported``."""


def next_model_note(decision: Decision) -> str:
    """The caller's own words for what the turn settled."""
    escalated = escalation(decision.record)
    if escalated is not None:
        return f"The turn is undecided: {escalated['root_cause']['code']}."
    return f"The turn decided {decision.estimate}."


def main() -> None:
    loaded = load(DESIGN)
    decision = decide_sync(loaded, STATE, {"default": answering(0.9)})
    print(next_model_note(decision))
    for message in to_chat_completions(decision.record, loaded):
        if message["role"] == "assistant":
            for call in message["tool_calls"]:
                print("call", call["id"], call["function"]["name"], call["function"]["arguments"])
        else:
            print("result", message["tool_call_id"], len(message["content"]), "bytes")

    unsure = decide_sync(loaded, STATE, {"default": answering(0.6)})
    print(next_model_note(unsure))

    broken = decide_sync(loaded, STATE, {"default": SILENT})
    print(next_model_note(broken))


if __name__ == "__main__":
    main()
