"""A stored format-5 design and record, as the format-5 engine wrote them, read under their own
rules: record fields spelled in code-point order, no order members, and values held sorted."""

import json
from pathlib import Path

from ommat import load, read_record, replay_sync

FORMAT_5 = Path(__file__).parent / "format_5"
FORMAT_5_DESIGN = "sha256:086a77f1639db43f063f77df2e89f703d93841e4564d71f9ea9993cfc2a1d07a"
FORMAT_5_RECORD = "sha256:e16cea78edb8022a7863fe6c9411121dd91a6bcbdca32e896cd353c331392292"


def test_a_stored_format_5_design_loads_to_its_own_hash_with_no_order_member() -> None:
    data = (FORMAT_5 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_5_DESIGN
    assert loaded.document == data
    doc = json.loads(data)
    assert doc["format"] == 5
    assert doc["state"] == "record{body: string, subject: string}"
    assert "payload_order" not in doc["sources"]["lookup"]
    assert "order" not in doc["parameters"]["order/weights"]
    assert "order" not in doc["sources"]["ask"]["provides"]["q:ask/tone"]["wording"]


def test_a_stored_format_5_record_reads_to_its_own_hash_its_values_sorted() -> None:
    data = (FORMAT_5 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_5_RECORD
    entries = {i["ref"]: i for i in record.log if i["item"] == "entry"}
    assert all("order" not in i for i in record.log)
    assert list(entries["t:state"]["value"]) == ["body", "subject"]
    assert list(entries["t:found"]["value"]) == ["alpha", "zeta"]


def test_a_stored_format_5_record_replays_under_its_design_with_no_finding() -> None:
    loaded = load((FORMAT_5 / "design.json").read_bytes())
    rerun, findings = replay_sync(read_record((FORMAT_5 / "record.json").read_bytes()), loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.log[0] == {"item": "design", "hash": FORMAT_5_DESIGN, "format": 6, "helpers": 2}
    state = next(i for i in rerun.log if i["item"] == "entry" and i["ref"] == "t:state")
    assert list(state["value"]) == ["body", "subject"]  # the format-5 design's spelling
