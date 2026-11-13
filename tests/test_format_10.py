"""The format-8 record a turn under a stored format-10 design wrote, read under its own rules: its
``settings`` item names its policy, an ordinary decision table, and holds class trees, and the
record ends at the policy's entry. The stored format-9 and format-10 designs are refused by
check 11, naming their formats (``test_format_11`` refuses the format-10 one). The design built
here is today's form of the stored one, defined here as it was."""

import json
from pathlib import Path
from typing import Any

import pytest

from ommat import (
    Design,
    FindingKind,
    InvalidDesign,
    Question,
    Rule,
    Source,
    Table,
    Text,
    escalation,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat.conversation import to_chat_completions
from ommat.helpers import pick_with_margin

FORMAT_9 = Path(__file__).parent / "format_9"
"""Named for its design's format: it holds a format-9 design and a format-7 record."""
FORMAT_10 = Path(__file__).parent / "format_10"
"""Named for its design's format: it holds a format-10 design and a format-8 record."""
FORMAT_10_DESIGN = "sha256:bb057cd9b02f94ba4b1b059b0c801298db7f6496a3402f8a02b9fb0b52671779"
FORMAT_8_RECORD = "sha256:1ac49cde45c8bf79bffbad22227f5d342180487b85a1033045638cbbfdcad213"


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


DESIGN = Design(
    state=t.record(subject=t.string),
    policy="d:act",
    name="format 10",
    sources={
        "ask": Source(
            {"q:spam": Question("noul", Text("Is it spam?"))},
            needs=["t:state.subject"],
            oracle="mail",
        )
    },
    rules={
        "route": Rule("d:route", t.label("no", "yes"), route, version="1", reads=["q:spam"]),
        "act": Table(
            "d:act",
            t.label("archive", "keep"),
            "F",
            ["d:route"],
            [
                ["yes", "archive"],
                ["-", "keep"],
            ],
        ),
    },
)


def test_check_11_refuses_the_stored_format_9_design_naming_its_format() -> None:
    data = (FORMAT_9 / "design.json").read_bytes()
    assert json.loads(data)["format"] == 9
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 9 is not 11, the one this engine reads" in str(raised.value)


def test_a_stored_format_8_record_reads_to_its_own_hash() -> None:
    data = (FORMAT_10 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_8_RECORD
    log = record.log
    assert log[0]["format"] == 8
    assert log[0]["hash"] == FORMAT_10_DESIGN
    assert log[1]["policy"] == "d:act"
    assert log[1]["classes"]["t:state"] == {"subject": "content"}
    assert "shapes" not in log[1]
    assert [i["item"] for i in log if i["item"] in ("plan", "verdict")] == []
    assert log[-1]["item"] == "round"
    act = next(i for i in log if i["item"] == "entry" and i["ref"] == "d:act")
    assert act["value"] == "archive"


FORMAT_7_RECORDS = {
    "format_9/record.json": (
        "sha256:e8a2363be41d51624a24d68e7e67698164b9fe010a6075e6ba91f9452a11d52d"
    ),
    "format_7_record/turn.record.json": (
        "sha256:a6a79f64c73cb347c2e003f92be7fdab0d19700672eb9f77e866839c7a2b6e10"
    ),
}
"""The two stored format-7 records, each with its record hash."""


@pytest.mark.parametrize("path", sorted(FORMAT_7_RECORDS))
def test_a_stored_format_7_record_reads_to_its_bytes_and_its_hash(path: str) -> None:
    data = (Path(__file__).parent / path).read_bytes()
    record = read_record(data)
    assert record.log[0]["format"] == 7
    assert record.document == data
    assert record.record_hash == FORMAT_7_RECORDS[path]


def test_a_stored_format_8_record_gives_its_decision() -> None:
    record = read_record((FORMAT_10 / "record.json").read_bytes())
    assert escalation(record) is None
    assert json.loads(to_chat_completions(record)[-1]["content"]) == {"decision": "archive"}


def test_a_stored_format_8_record_replays_under_todays_design_as_a_design_mismatch() -> None:
    record = read_record((FORMAT_10 / "record.json").read_bytes())
    loaded = load(DESIGN)
    rerun, findings = replay_sync(record, loaded)
    assert findings == [
        {
            "kind": FindingKind.design_mismatch,
            "recorded": FORMAT_10_DESIGN,
            "replayed": loaded.design_hash,
        },
        {"kind": FindingKind.helpers_mismatch, "recorded": 3, "replayed": 4},
    ]
    assert rerun is not None
    assert rerun.log[0]["format"] == 9
    entries = [i for i in record.log if i["item"] == "entry"]
    assert [i for i in rerun.log if i["item"] == "entry"] == entries


def test_a_stored_format_8_record_redacts_under_its_class_trees_as_a_floor() -> None:
    record = read_record((FORMAT_10 / "record.json").read_bytes())
    policy = {"t:state": "open", "d:route": "content", "d:act": "open"}
    view = json.loads(redact(record, bytes(32), policy))
    assert view["applied"] == {
        "d:act": "open",  # open in the policy and the stored tree
        "d:route": "content",  # the policy digests more than the stored open
        "q:spam": "content",  # named nothing: content, over the stored open
        "t:state": {"subject": "content"},  # the stored content holds over the policy's open
    }
    state = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == "t:state")
    assert state["value"]["subject"].startswith("hmac-sha256:")
