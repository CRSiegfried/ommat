"""An extraction design for the extraction-kind tests: a source asking an amount owed, a
number, and the dates it is due by, a list of dates, from the state's ``body``, each resolving
a yearless date against the state's ``today``; rules reading them through ``extract``; and a
policy rule whose reply depends on both. Builders change one declaration."""

import dataclasses
from typing import Any

from designs import minimal, slot
from ommat import (
    Design,
    LoadedDesign,
    Parameter,
    Question,
    Rule,
    Source,
    Table,
    Text,
    Undetermined,
    load,
)
from ommat import types as t
from ommat.helpers import extract

STATE_T = t.record(body=t.string, today=t.string)
BODY = "Owe 120 by 2024-02-29"
STATE: dict[str, Any] = {"body": BODY, "today": "2024-01-01"}

AMOUNT: dict[str, Any] = {
    "value": 120,
    "guess": 120,
    "p_null": 0.1,
    "confidence": 0.9,
    "span": [4, 7],
}
DATES: dict[str, Any] = {
    "value": ["2024-02-29"],
    "guess": ["2024-02-29"],
    "p_null": 0.0,
    "confidence": 1,
    "span": [[11, 21]],
}


def amount(r: Any) -> Any:
    value = extract(r, "q:amount", r.param("amount/null"))
    if isinstance(value, Undetermined):
        return value
    return "big" if value >= r.param("amount/big") else "small"


def dates(r: Any) -> Any:
    return extract(r, "q:dates", 0.5)


def reply(r: Any) -> str:
    size, due = r.read("d:amount"), r.read("d:dates")
    shown = "unsure" if isinstance(size, Undetermined) else size
    when = "no date" if isinstance(due, Undetermined) else ", ".join(due)
    return f"{shown}; {when}"


def question(value_type: str = "number", is_list: bool = False, **changes: Any) -> Question:
    fields: dict[str, Any] = {
        "kind": "extraction",
        "wording": Text("What is owed in {body}?", [slot("body", ref="t:state.body")]),
        "value_type": value_type,
        "is_list": is_list,
        "text": {"ref": "t:state.body"},
        "reference_date": {"ref": "t:state.today"},
    }
    fields.update(changes)
    return Question(**fields)


def reader(**provides: Any) -> Source:
    """The source asking both questions, each as given or its default."""
    questions = {
        "q:amount": question(),
        "q:dates": question("date", True, wording=Text("When is it due?")),
    }
    return Source({**questions, **provides}, needs=["t:state.body"])


BOUNDARIES: dict[str, Any] = {
    "q:amount": {"null": ["amount/null"], "confidence": [], "value": ["amount/big"]}
}


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "state": STATE_T,
        "sources": {"read": reader()},
        "rules": {
            "amount": Rule(
                "d:amount",
                t.label("big", "small"),
                amount,
                version="1",
                reads=["q:amount"],
                params=["amount/null", "amount/big"],
                boundaries=BOUNDARIES,
            ),
            "dates": Rule("d:dates", t.list_(t.string), dates, version="1", reads=["q:dates"]),
            "reply": Rule("d:reply", t.string, reply, version="1", reads=["d:amount", "d:dates"]),
        },
        "parameters": {
            "amount/null": Parameter(t.number(0, 1), 0.5),
            "amount/big": Parameter(t.number(), 100),
        },
    }
    fields.update(changes)
    return minimal(**fields, alone=True)


def with_question(ref: str, q: Any, **changes: Any) -> Design:
    """``design`` with one question of the reader replaced."""
    return design(sources={"read": reader(**{ref: q})}, **changes)


def loaded(**changes: Any) -> LoadedDesign:
    return load(design(**changes))


def replaced(d: Design, **changes: Any) -> Design:
    return dataclasses.replace(d, **changes)


def dates_only(r: Any) -> str:
    due = r.read("d:dates")
    return "no date" if isinstance(due, Undetermined) else ", ".join(due)


DATES_ONLY = Rule("d:reply", t.string, dates_only, version="1", reads=["d:dates"])
"""The policy's rule reading ``d:dates`` alone."""


def dates_only_rules(d: Design) -> dict[str, Rule | Table]:
    """``d``'s rules with the policy's rule replaced by ``DATES_ONLY``."""
    return {**d.rules, "reply": DATES_ONLY}


def plain(**provides: Any) -> Design:
    """``design`` with no ``boundaries`` field, so ``q:amount`` may be asked as any question."""
    d = design(sources={"read": reader(**provides)})
    rule = dataclasses.replace(d.rules["amount"], boundaries=None)
    return replaced(d, rules={**d.rules, "amount": rule})
