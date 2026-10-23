"""A noul's described yes and no, carried from the design to the binding."""

import json
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    CallResult,
    Design,
    InvalidDesign,
    InvalidRecord,
    Plan,
    Policy,
    Question,
    Record,
    Source,
    SourceCall,
    Text,
    decide_sync,
    from_record,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t

YES = {"name": "yes", "description": "The message names a sender"}
DESCRIBED = [YES, "no"]


def inform(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def design(labels: Any = None, wording: Any = None) -> Design:
    q = Question("noul", wording or Text("Does it name a sender?"), labels)
    return Design(
        name="wording",
        state=t.record(),
        sources={"ask": Source({"q:named": q})},
        policy=Policy(inform, version="1"),
    )


class Seen:
    def __init__(self) -> None:
        self.batches: list[tuple[SourceCall, ...]] = []

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        self.batches.append(batch)
        return CallResult(
            {ref: {"yes": 0.8, "no": 0.2} for call in batch for ref in call.questions}
        )


def run(d: Design, binding: Any = None) -> tuple[Record, Seen]:
    seen = Seen()
    record = decide_sync(load(d), {}, {"default": binding or seen}).record
    return record, seen


def call_item(record: Record) -> dict[str, Any]:
    (item,) = [i for i in record.log if i["item"] == "call"]
    return item


def test_a_described_noul_reaches_the_binding_with_its_descriptions() -> None:
    record, seen = run(design(DESCRIBED))
    q = seen.batches[0][0].questions["q:named"]
    assert q.labels == ("yes", "no")
    assert q.descriptions == ("The message names a sender", None)
    assert call_item(record)["questions"]["q:named"]["labels"] == DESCRIBED
    entry = next(i for i in record.log if i.get("ref") == "q:named")
    assert entry["value"]["labels"] == ["yes", "no"]


def test_an_undescribed_noul_keeps_null_labels_and_its_identity() -> None:
    record, seen = run(design())
    assert seen.batches[0][0].questions["q:named"].descriptions == (None, None)
    assert call_item(record)["questions"]["q:named"]["labels"] is None
    described, _ = run(design(DESCRIBED))
    ident = call_item(record)["questions"]["q:named"]["identity"]
    assert ident != call_item(described)["questions"]["q:named"]["identity"]


@pytest.mark.parametrize(
    ("labels", "says"),
    [
        (["yes", "no"], "declares yes and no with no description"),
        (["no", YES], "whose labels are not yes then no"),
        ([YES, "no", "maybe"], "whose labels are not yes then no"),
    ],
)
def test_a_nouls_labels_are_yes_then_no_with_one_described(labels: Any, says: str) -> None:
    with pytest.raises(InvalidDesign) as raised:
        load(design(labels))
    (failure,) = raised.value.failures
    assert failure.check == 10
    assert says in failure.message


def test_a_described_noul_replays_and_from_record_answers_it() -> None:
    d = design(DESCRIBED)
    record, _ = run(d)
    assert replay_sync(record, load(d))[1] == []
    live = decide_sync(load(d), {}, {"default": from_record(record, load(d))}).record
    entry = next(i for i in live.log if i.get("ref") == "q:named")
    assert "value" in entry
    other = design([{"name": "yes", "description": "Another wording"}, "no"])
    live = decide_sync(load(other), {}, {"default": from_record(record, load(other))}).record
    entry = next(i for i in live.log if i.get("ref") == "q:named")
    assert entry["marker"]["cause"]["code"] == "unsupported"


def test_a_described_noul_before_format_4_is_refused_in_a_document_and_a_record() -> None:
    loaded = load(design(DESCRIBED))
    doc = {**json.loads(loaded.document), "format": 3}
    with pytest.raises(InvalidDesign) as raised:
        load(json.dumps(doc).encode())
    assert "is a noul question with labels, which format 3 does not have" in str(raised.value)
    record, _ = run(design(DESCRIBED))
    stored = json.loads(record.document)
    stored["log"][0]["format"] = 3
    with pytest.raises(InvalidRecord, match="a noul's labels, which a format-3 record"):
        read_record(rehashed(stored))
