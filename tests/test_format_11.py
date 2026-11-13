"""A stored format-11 design and the format-9 record a turn under it wrote, read under their own
rules: the design declares no class block, and the record's ``settings`` item holds each ref's
shape tree under ``shapes``, which ``redact`` reads the caller's policy over. The stored
format-10 design is refused by check 11, naming its format. The design's one function is defined
here, so its source hash is this file's."""

import json
from pathlib import Path
from typing import Any

import pytest

from ommat import (
    CallResult,
    Design,
    InvalidDesign,
    LoadedDesign,
    Question,
    Rule,
    Source,
    SourceCall,
    Table,
    Text,
    escalation,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

FORMAT_10 = Path(__file__).parent / "format_10"
"""Named for its design's format: it holds a format-10 design and a format-8 record."""
FORMAT_11 = Path(__file__).parent / "format_11"
"""Named for its design's format: it holds a format-11 design and a format-9 record."""
FORMAT_11_DESIGN = "sha256:9468d04086afeab1bb7c51c3fdba2185ee0929f522dbeafabe31544d5f7fcf65"
FORMAT_9_RECORD = "sha256:bb677453dc1dae834ca3677270c051a9f183ba0246fd0b23affc0ad3512e93cc"


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


DESIGN = Design(
    state=t.record(subject=t.string),
    policy="d:act",
    name="format 11",
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


async def mail(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """The oracle the stored turn was decided under: spam, surely."""
    return CallResult({"q:spam": {"yes": 0.9, "no": 0.1}})


def stored() -> LoadedDesign:
    return load((FORMAT_11 / "design.json").read_bytes())


def test_a_stored_format_11_design_loads_to_its_own_hash() -> None:
    data = (FORMAT_11 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_11_DESIGN
    assert loaded.document == data
    doc = json.loads(data)
    assert doc["format"] == 11
    assert "classes" not in doc
    assert doc["policy"] == "d:act"


def test_the_design_built_here_is_the_stored_one() -> None:
    assert load(DESIGN).document == (FORMAT_11 / "design.json").read_bytes()


def test_check_11_refuses_the_stored_format_10_design_naming_its_format() -> None:
    data = (FORMAT_10 / "design.json").read_bytes()
    assert json.loads(data)["format"] == 10
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 10 is not 11, the one this engine reads" in str(raised.value)


def test_a_stored_format_9_record_reads_to_its_own_hash_and_holds_shapes() -> None:
    data = (FORMAT_11 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_9_RECORD
    log = record.log
    assert log[0]["format"] == 9
    assert log[0]["hash"] == FORMAT_11_DESIGN
    assert log[1]["shapes"] == {
        "d:act": "leaf",
        "d:route": "leaf",
        "q:spam": "leaf",
        "t:state": {"subject": "leaf"},
    }
    assert "classes" not in log[1]
    act = next(i for i in log if i["item"] == "entry" and i["ref"] == "d:act")
    assert act["value"] == "archive"


def test_a_stored_format_9_record_gives_its_decision_and_redacts_under_a_policy() -> None:
    record = read_record((FORMAT_11 / "record.json").read_bytes())
    assert escalation(record) is None
    view = json.loads(redact(record, bytes(32), {"d:act": "open"}))
    act = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == "d:act")
    assert act["value"] == "archive"
    route_entry = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == "d:route")
    assert route_entry["value"].startswith("hmac-sha256:")
    assert view["applied"]["t:state"] == {"subject": "content"}


def test_a_stored_format_9_record_replays_under_its_design_with_no_finding() -> None:
    record = read_record((FORMAT_11 / "record.json").read_bytes())
    rerun, findings = replay_sync(record, stored())
    assert findings == []
    assert rerun is not None
    assert rerun.log == record.log
