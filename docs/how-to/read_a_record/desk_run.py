"""The how-to's driver: decide three turns of the desk design, with ``fixed`` standing in for
the model and the clock, and print what each question of the guide reads from the records."""

import warnings
from typing import Any

from desk_design import DESIGN
from ommat import Decision, DefectWarning, decide_sync, defects, escalation, load
from ommat.bindings import fixed
from ommat.evaluation import answers

LOADED = load(DESIGN)
CLOCK = fixed({"t:today": "2026-11-04"})


def turn(message: str, ordered: str, refund: dict[str, float] | None) -> Decision:
    """One turn, the model answering ``refund``, or not answering when it is ``None``."""
    model = fixed({} if refund is None else {"q:refund": refund})
    state = {"message": message, "ordered": ordered}
    return decide_sync(LOADED, state, {"default": model, "clock": CLOCK})


def held(entry: dict[str, Any]) -> object:
    """An entry's value, or its marker's kind and cause code."""
    if "value" in entry:
        return entry["value"]
    return f"{entry['marker']['kind']} {entry['marker']['cause']['code']}"


def main() -> None:
    sure = turn("It arrived broken. Refund me.", "2026-10-30", {"yes": 0.9, "no": 0.1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DefectWarning)
        late = turn("It arrived broken. Refund me.", "last week", {"yes": 0.9, "no": 0.1})
    silent = turn("It arrived broken.", "2026-10-30", None)

    record = sure.record
    entry = record.entries["d:days_open"]
    print("d:days_open", held(entry))
    asked = record.entries["q:refund"]
    call = record.log[int(asked["call"])]
    print("q:refund", asked["value"]["probabilities"], "from", call["sources"], call["provider"])

    for found in defects(late.record) or []:
        print("defect", found["ref"], held(found), "estimate", late.estimate)
    print("tracebacks at", sorted(late.record.unhashed["exceptions"]))
    (warned,) = caught
    assert isinstance(warned.message, DefectWarning)
    print("warned", warned.category.__name__, [d["ref"] for d in warned.message.defects])

    escalated = escalation(silent.record)
    assert escalated is not None
    print("undecided", escalated["kind"], escalated["root_cause"]["code"])

    for row in answers([sure.record, late.record, silent.record], LOADED):
        marker = row["marker"]
        print("answer", row["ref"], row["top"] if marker is None else marker["cause"]["code"])


if __name__ == "__main__":
    main()
