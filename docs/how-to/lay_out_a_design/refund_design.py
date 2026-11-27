"""The refund desk as an ommat design: what to do with a customer's message about an order.

It decides ``refund``, ``store_credit``, ``decline`` or ``answer``, or stays ``Unsure`` and
hands the message to a person.

Sources:

- ``ask`` (the model): ``q:refund``, ``q:reason``.
- ``orders`` (the order system): ``t:order_age``, the order's age in days.

Rules, in graph order:

- ``d:refund``: ``yes`` when the customer asks for their money back, ``no`` when not.
- ``d:reason``: why, ``damaged``, ``late`` or ``changed_mind``.
- ``d:within_window``: whether the order is young enough to refund.
- ``d:outcome``: the decision.

Bars are set in ``BARS``; the wording is in ``QUESTIONS``.
"""

from dataclasses import dataclass
from typing import Any

from ommat import (
    Design,
    Parameter,
    Question,
    ReadView,
    Record,
    Rule,
    Source,
    Table,
    Text,
    Unsure,
    escalation,
)
from ommat import types as t
from ommat.helpers import determined, pick_with_margin

# --- constants and labels ------------------------------------------------------------------
ORDERS = "orders"
"""The oracle name the caller binds to the order system."""
REASONS = ("damaged", "late", "changed_mind")
OUTCOMES = ("refund", "store_credit", "decline", "answer")

# --- the questions -------------------------------------------------------------------------
QUESTIONS = {
    "refund": Question("noul", Text("Does the customer ask for their money back?")),
    "reason": Question("choice", Text("Why does the customer write?"), REASONS),
}

# --- types ---------------------------------------------------------------------------------
YES_NO_T = t.label("yes", "no")
REASON_T = t.label(*REASONS)
OUTCOME_T = t.label(*OUTCOMES)


# --- rule functions, in graph order --------------------------------------------------------
def refund(r: ReadView) -> Any:
    """``d:refund``: ``yes`` when the customer asks for their money back."""
    return pick_with_margin(r, "q:refund", r.param("bars/refund_at"), r.param("bars/refund_lead"))


def reason(r: ReadView) -> Any:
    """``d:reason``: why the customer writes."""
    return pick_with_margin(r, "q:reason", r.param("bars/reason_at"), r.param("bars/reason_lead"))


def within_window(r: ReadView) -> Any:
    """``d:within_window``: whether the order is at most ``bars/window_days`` old."""
    held = determined(r, "t:order_age")
    if not isinstance(held, tuple):
        return held  # a marker passes on, with its cause
    (age,) = held
    return age <= r.param("bars/window_days")


# --- bars ----------------------------------------------------------------------------------
BARS = {
    # Each bar is one decision's, so raising one moves only that decision.
    "bars/refund_at": Parameter(t.number(0, 1), 0.7),
    "bars/refund_lead": Parameter(t.number(0, 1), 0.2),
    "bars/reason_at": Parameter(t.number(0, 1), 0.6),
    # Starts at the refund's lead; it is its own bar so it can move alone.
    "bars/reason_lead": Parameter(t.number(0, 1), 0.2),
    # The shop's policy, not a measurement: thirty days from the order.
    "bars/window_days": Parameter(t.integer(0, 365), 30),
}

# --- tables --------------------------------------------------------------------------------
OUTCOME_INPUTS = {
    "refund": "d:refund",
    "within_window": "d:within_window",
    "reason": "d:reason",
}


def row(out: object, **cells: object) -> list[object]:
    """One row of ``d:outcome``, each cell named by its input, ``-`` where none is given."""
    unknown = cells.keys() - OUTCOME_INPUTS.keys()
    if unknown:
        raise KeyError(f"d:outcome has no input named {sorted(unknown)}")
    return [cells.get(name, "-") for name in OUTCOME_INPUTS] + [out]


OUTCOME = Table(
    "d:outcome",
    OUTCOME_T,
    "F",  # the first matching row wins
    list(OUTCOME_INPUTS.values()),
    [
        # A refund we could not settle goes to a person before any row reads its value.
        row(Unsure, refund="Undetermined"),
        row("answer", refund="no"),
        row("decline", refund="yes", within_window=False),
        row("refund", refund="yes", within_window=True, reason="{damaged, late}"),
        row("store_credit", refund="yes", within_window=True, reason="changed_mind"),
        row(Unsure),
    ],
)

# --- the design ----------------------------------------------------------------------------
DESIGN = Design(
    name="refund",
    state=t.record(message=t.string, order_id=t.string),
    sources={
        "ask": Source({f"q:{qid}": q for qid, q in QUESTIONS.items()}, needs=["t:state.message"]),
        "orders": Source(
            {"t:order_age": t.integer(0, 100_000)}, needs=["t:state.order_id"], oracle=ORDERS
        ),
    },
    parameters=BARS,
    rules={
        "refund": Rule(
            "d:refund",
            YES_NO_T,
            refund,
            version="1",
            reads=["q:refund"],
            params=["bars/refund_at", "bars/refund_lead"],
        ),
        "reason": Rule(
            "d:reason",
            REASON_T,
            reason,
            version="1",
            reads=["q:reason"],
            params=["bars/reason_at", "bars/reason_lead"],
        ),
        "within_window": Rule(
            "d:within_window",
            t.boolean,
            within_window,
            version="1",
            reads=["t:order_age"],
            params=["bars/window_days"],
        ),
        "outcome": OUTCOME,
    },
)


# --- reading the record back ---------------------------------------------------------------
@dataclass(frozen=True)
class RefundReading:
    """What a turn decided, in the caller's words: no ommat type crosses this line."""

    outcome: str | None
    reason: str | None
    defect: bool


def read_refund(record: Record) -> RefundReading:
    """The turn's outcome and reason; ``outcome`` is ``None`` when a person must look, and
    ``defect`` says the design, not the evidence, is at fault (the escalation view's test)."""
    entries = record.entries  # decoded afresh at each read, so read once
    held = escalation(record, "d:outcome")
    return RefundReading(
        outcome=_value(entries, "d:outcome"),
        reason=_value(entries, "d:reason"),
        defect=held is not None and held["defect"],
    )


def _value(entries: dict[str, Any], ref: str) -> Any:
    """The value of the entry at ``ref``, or ``None`` for a marker or no entry."""
    item = entries.get(ref)
    return item["value"] if item is not None and "value" in item else None
