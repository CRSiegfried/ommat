"""A stored format-5 design and record, as the format-5 engine wrote them, read under their own
rules: record fields spelled in code-point order, no order members, and values held sorted."""

import json
from pathlib import Path

from ommat import FindingKind, load, read_record, replay_sync

FORMAT_5 = Path(__file__).parent / "format_5"
FORMAT_5_DESIGN = "sha256:9b9822d9596a2c25ac054f52b109b75db04958a2c0dcd0d32239a1f5ae34e9a9"
FORMAT_5_RECORD = "sha256:05dac70e9d88239b5d5e9c0f56178da66c5aeb27f98f30f1a4adc94b33f5b1a3"


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
    plan = next(i for i in record.log if i["item"] == "plan")
    assert plan["context"]["entries"][0]["text"] == 'Weights {"alpha":1,"zeta":2}'


def test_a_stored_format_5_record_replays_under_its_design_with_the_helper_version_alone() -> None:
    loaded = load((FORMAT_5 / "design.json").read_bytes())
    rerun, findings = replay_sync(read_record((FORMAT_5 / "record.json").read_bytes()), loaded)
    assert findings == [{"kind": FindingKind.helpers_mismatch, "recorded": 2, "replayed": 3}]
    assert rerun is not None
    assert rerun.log[0] == {"item": "design", "hash": FORMAT_5_DESIGN, "format": 6, "helpers": 3}
    state = next(i for i in rerun.log if i["item"] == "entry" and i["ref"] == "t:state")
    assert list(state["value"]) == ["body", "subject"]  # the format-5 design's spelling
    # The rule builds zeta first; the rerun renders the slot sorted, as the format-5 engine did.
    plan = next(i for i in rerun.log if i["item"] == "plan")
    assert plan["context"]["entries"][0]["text"] == 'Weights {"alpha":1,"zeta":2}'
