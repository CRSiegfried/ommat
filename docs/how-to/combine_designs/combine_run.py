"""The how-to's driver: two turns of the combined design with ``fixed`` standing in for the
model, printing each turn's calls and decision, then the email workflow decided alone."""

from combine_design import DESIGN, EMAIL
from ommat import decide_sync, load, replay_sync
from ommat.bindings import fixed
from ommat.helpers import standalone

YES = {"yes": 0.9, "no": 0.1}
NO = {"yes": 0.1, "no": 0.9}
ROUTE = {
    "File the invoice under Billing": {"email": 0.9, "other": 0.1},
    "What is the weather like?": {"email": 0.1, "other": 0.9},
}


def main() -> None:
    loaded = load(DESIGN)
    for message, route in ROUTE.items():
        model = fixed({"q:route": route, "q:urgent": NO, "q:email/folder_named": YES})
        decision = decide_sync(loaded, {"message": message}, {"default": model})
        calls = [item["sources"] for item in decision.record.log if item["item"] == "call"]
        _, findings = replay_sync(decision.record, loaded)
        print(f"{message!r}: calls {calls}; decision {decision.estimate}; findings {findings}")

    alone = load(standalone(EMAIL, {"d:email/chosen": True, "d:email/urgent": "no"}))
    model = fixed({"q:email/folder_named": NO})
    decision = decide_sync(alone, {"message": "Move it"}, {"default": model})
    print(f"email alone, chosen and not urgent: decision {decision.estimate}")


if __name__ == "__main__":
    main()
