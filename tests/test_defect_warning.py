"""The defect warning: a turn decide returns whose record holds a defect emits one
``DefectWarning`` once its record is closed; nothing else warns. This file runs under the
suite's ``error`` filter, so a stray warning fails it."""

import asyncio
import pickle
import warnings
from typing import Any

import pytest

from desk_design import DESIGN  # the read-a-record how-to's design
from ommat import (
    Decision,
    DefectWarning,
    Design,
    Rule,
    decide,
    decide_sync,
    defects,
    load,
    replay_sync,
)
from ommat import types as t
from ommat.bindings import fixed
from ommat.helpers import TurnFailure, decide_many

LOADED = load(DESIGN)
YES = {"q:refund": {"yes": 0.9, "no": 0.1}}
CLOCK = fixed({"t:today": "2026-11-04"})
GOOD = {"message": "It arrived broken. Refund me.", "ordered": "2026-10-30"}
BAD_DATE = {"message": "It arrived broken. Refund me.", "ordered": "last week"}


def bindings(answers: dict[str, Any] = YES) -> dict[str, Any]:
    return {"default": fixed(answers), "clock": CLOCK}


def test_a_turn_with_a_defect_warns_once_after_its_record_closes() -> None:
    with pytest.warns(DefectWarning) as caught:
        decision = decide_sync(LOADED, BAD_DATE, bindings())
    (warned,) = caught
    message = warned.message
    assert isinstance(message, DefectWarning)
    assert message.record is not None
    assert message.record.record_hash == decision.record.record_hash
    assert message.defects == defects(decision.record)
    assert [d["ref"] for d in message.defects] == ["d:days_open"]
    assert "end" in message.record.unhashed["turn"]  # closed before the warning
    text = str(message)
    assert text.startswith("the turn's record holds 1 defect:\n")
    assert "  d:days_open: Unavailable, raised naming days_open" in text
    assert "ValueError: Invalid isoformat string: 'last week'" in text
    assert decision.estimate == "refund_order"  # the turn still decided


@pytest.mark.parametrize(
    ("state", "answers"),
    [(GOOD, YES), (GOOD, {})],  # a clean turn, and an outage: the model never answered
    ids=["clean", "outage"],
)
def test_a_turn_with_no_defect_warns_nothing(state: dict[str, str], answers: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decide_sync(LOADED, state, bindings(answers))


def test_a_broken_entry_warns_once_with_the_codecs_traceback() -> None:
    def codec(state: object) -> object:
        raise LookupError("no such customer")

    with pytest.warns(DefectWarning) as caught:
        decision = decide_sync(LOADED, GOOD, bindings(), codec=codec)
    (warned,) = caught
    message = warned.message
    assert isinstance(message, DefectWarning)
    assert [d["ref"] for d in message.defects] == ["t:state", "d:policy"]
    assert "LookupError: no such customer" in str(message)
    assert decision.record.entries["d:policy"]["marker"]["cause"]["code"] == "raised"


def test_under_an_error_filter_decide_raises_it_with_the_whole_record() -> None:
    seen: list[int] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", DefectWarning)
        with pytest.raises(DefectWarning) as raised:
            decide_sync(LOADED, BAD_DATE, bindings(), observer=lambda i, _: seen.append(i))
    record = raised.value.record
    assert record is not None
    assert seen == list(range(len(record.log)))  # every item was written first
    assert record.entries["d:policy"]["value"] == "refund_order"


def test_the_warning_names_the_line_that_called_decide_or_decide_sync() -> None:
    with pytest.warns(DefectWarning) as caught:
        decide_sync(LOADED, BAD_DATE, bindings())
    assert caught[0].filename == __file__

    async def caller() -> Decision:
        return await decide(LOADED, BAD_DATE, bindings())

    with pytest.warns(DefectWarning) as caught:
        asyncio.run(caller())
    assert caught[0].filename == __file__


def test_a_turn_ended_by_the_observers_exception_warns_nothing() -> None:
    def observer(index: int, item: dict[str, Any]) -> None:
        if item.get("ref") == "d:policy":
            raise KeyError("observer down")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(KeyError):
            decide_sync(LOADED, BAD_DATE, bindings(), observer=observer)


def test_replaying_a_record_with_a_defect_warns_nothing() -> None:
    with pytest.warns(DefectWarning):
        record = decide_sync(LOADED, BAD_DATE, bindings()).record
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rerun, findings = replay_sync(record, LOADED)
    assert rerun is not None
    assert findings == []


def test_the_warning_pickles_with_its_record_and_defects() -> None:
    with pytest.warns(DefectWarning) as caught:
        decide_sync(LOADED, BAD_DATE, bindings())
    message = caught[0].message
    assert isinstance(message, DefectWarning)
    copied = pickle.loads(pickle.dumps(message))
    assert str(copied) == str(message)
    assert copied.record is not None
    assert message.record is not None
    assert copied.record.record_hash == message.record.record_hash
    assert copied.defects == message.defects
    assert issubclass(DefectWarning, UserWarning)


def test_each_turn_of_decide_many_with_a_defect_warns() -> None:
    states = [GOOD, BAD_DATE, BAD_DATE]
    with pytest.warns(DefectWarning) as caught:
        asyncio.run(decide_many(states, LOADED, bindings(), concurrency=2))
    assert len([w for w in caught if w.category is DefectWarning]) == 2


def test_under_an_error_filter_a_defect_turn_of_decide_many_is_a_turn_failure() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DefectWarning)
        results = asyncio.run(decide_many([GOOD, BAD_DATE], LOADED, bindings(), concurrency=2))
    good, bad = results
    assert isinstance(good, Decision)
    assert isinstance(bad, TurnFailure)
    assert isinstance(bad.error, DefectWarning)
    assert bad.record is bad.error.record


# ---------------------------------------------------------------------------------------------
# The byte scan that decides whether a closed record is worth decoding for defects


def may_hold_defect(record: Any) -> bool:
    from ommat._decide import _may_hold_defect  # pyright: ignore[reportPrivateUsage]

    return _may_hold_defect(record)


def test_the_scan_finds_a_raised_defect_in_a_rule_and_at_the_codec() -> None:
    with pytest.warns(DefectWarning):
        record = decide_sync(LOADED, BAD_DATE, bindings()).record
    assert may_hold_defect(record)
    assert [d["marker"]["cause"]["code"] for d in defects(record)] == ["raised"]

    def codec(state: object) -> object:
        raise LookupError("no such customer")

    with pytest.warns(DefectWarning):
        record = decide_sync(LOADED, GOOD, bindings(), codec=codec).record
    assert may_hold_defect(record)
    assert record.entries["t:state"]["marker"]["cause"]["code"] == "raised"


def test_the_scan_finds_an_invalid_return_at_the_state_entry() -> None:
    with pytest.warns(DefectWarning):
        record = decide_sync(LOADED, GOOD, bindings(), codec=lambda s: {"message": 1}).record
    assert may_hold_defect(record)
    assert record.entries["t:state"]["marker"]["cause"] == {
        "code": "invalid-return",
        "refs": ["t:state"],
    }


def reads_nothing(r: Any) -> str:
    r.read("t:nothing")
    return "ok"


def says_ok(r: Any) -> str:
    return "ok"


def one_rule(function: Any) -> Any:
    return load(
        Design(
            name="scan",
            state=t.record(text=t.string),
            sources={},
            rules={"act": Rule("d:act", t.string, function, version="1")},
            parameters={},
            policy="d:act",
        )
    )


def test_the_scan_finds_an_undeclared_read() -> None:
    with pytest.warns(DefectWarning):
        record = decide_sync(one_rule(reads_nothing), {"text": "x"}, {}).record
    assert may_hold_defect(record)
    assert record.entries["d:act"]["marker"]["cause"]["code"] == "undeclared-read"


def test_a_string_value_spelling_a_defect_code_is_no_defect() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        record = decide_sync(one_rule(says_ok), {"text": '"code":"raised"'}, {}).record
    assert not may_hold_defect(record)
    assert defects(record) == []
    assert record.entries["t:state"]["value"] == {"text": '"code":"raised"'}
