"""The how-to's driver: three turns through the gated design with ``fixed`` standing in for the
model, one per thing the gate can hold, printing what the refund source got."""

from gate_design import DESIGN
from ommat import decide_sync, load
from ommat.bindings import fixed

ANSWERS = {
    "I want my money back, 40 euros": {"refund": 0.9, "billing": 0.05, "other": 0.05},
    "Why was I charged twice?": {"refund": 0.1, "billing": 0.85, "other": 0.05},
    "Hello?": {"refund": 0.4, "billing": 0.35, "other": 0.25},
}


YES = {"yes": 0.9, "no": 0.1}


def main() -> None:
    loaded = load(DESIGN)
    for message in ANSWERS:
        model = fixed(
            {"q:ask/intent": ANSWERS[message], "q:ask/urgent": YES, "q:refund/amount_stated": YES}
        )
        decision = decide_sync(loaded, {"message": message}, {"default": model})
        entries = decision.record.entries
        gate = entries["d:intent_label"]
        held = gate.get("value", gate.get("marker", {}).get("kind"))
        asked = entries["q:refund/amount_stated"]
        if "marker" in asked:
            cause = asked["marker"]["cause"]
            outcome = f"{asked['marker']['kind']} ({cause['code']}: {', '.join(cause['refs'])})"
        else:
            outcome = "asked"
        print(f"{message!r}: gate {held}; refund source {outcome}; decision {decision.estimate}")


if __name__ == "__main__":
    main()
