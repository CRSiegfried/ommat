"""The how-to's driver: four messages decided with ``fixed`` standing in for the model and the
order system, each read back through ``read_refund``, then the check of the module's map."""

import refund_design
from ommat import decide_sync, load
from ommat.bindings import fixed
from refund_checks import map_disagrees
from refund_design import DESIGN, ORDERS, read_refund

YES = {"yes": 0.9, "no": 0.1}
TORN = {"yes": 0.55, "no": 0.45}
TURNS = {
    "The mug arrived broken, I want my money back": (YES, "damaged", 12),
    "I changed my mind about the lamp, refund please": (YES, "changed_mind", 3),
    "The chair from March is late, refund it": (YES, "late", 50),
    "Not sure the vase is what I wanted": (TORN, "changed_mind", 3),
}


def main() -> None:
    loaded = load(DESIGN)
    for message, (asks, why, age) in TURNS.items():
        reason = {label: 0.9 if label == why else 0.05 for label in refund_design.REASONS}
        model = fixed({"q:refund": asks, "q:reason": reason})
        orders = fixed({"t:order_age": age})
        state = {"message": message, "order_id": "A-1"}
        decision = decide_sync(loaded, state, {"default": model, ORDERS: orders})
        print(f"{message!r}: {read_refund(decision.record)}")
    print(f"map disagrees: {map_disagrees(refund_design, DESIGN)}")


if __name__ == "__main__":
    main()
