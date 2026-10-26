"""The how-to's driver: three turns through the gated design with a stand-in binding, one
per thing the gate can hold, printing what the refund source got."""

from gate_design import DESIGN
from ommat import CallResult, SourceCall, decide_sync, load

ANSWERS = {
    "I want my money back, 40 euros": {"refund": 0.9, "billing": 0.05, "other": 0.05},
    "Why was I charged twice?": {"refund": 0.1, "billing": 0.85, "other": 0.05},
    "Hello?": {"refund": 0.4, "billing": 0.35, "other": 0.25},
}


def binding_for(message: str):
    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        values: dict[str, object] = {}
        for source in batch:
            for ref in source.questions:
                if ref == "q:ask/intent":
                    values[ref] = ANSWERS[message]
                else:
                    values[ref] = {"yes": 0.9, "no": 0.1}
        return CallResult(values)

    return binding


def main() -> None:
    loaded = load(DESIGN)
    for message in ANSWERS:
        decision = decide_sync(loaded, {"message": message}, {"default": binding_for(message)})
        entries = {i["ref"]: i for i in decision.record.log if i["item"] == "entry"}
        gate = entries["d:intent_label"]
        held = gate.get("value", gate.get("marker", {}).get("kind"))
        asked = entries["q:refund/amount_stated"]
        if "marker" in asked:
            cause = asked["marker"]["cause"]
            outcome = f"{asked['marker']['kind']} ({cause['code']}: {', '.join(cause['refs'])})"
        else:
            outcome = "asked"
        print(f"{message!r}: gate {held}; refund source {outcome}; plan {decision.plan.kind}")


if __name__ == "__main__":
    main()
