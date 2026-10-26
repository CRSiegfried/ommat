"""A stored format-6 design and record, as the format-6 engine wrote them, read under their own
rules: no ``targets`` block, and no ``target`` or ``latent_targets`` on a question or a rule."""

import json
from pathlib import Path

import pytest

from format_6_design import DESIGN
from ommat import FindingKind, InvalidDesign, load, read_record, replay_sync

FORMAT_6 = Path(__file__).parent / "format_6"
FORMAT_6_DESIGN = "sha256:887d741baf21dd2211f067b0efb741134e8bddd422ba9ac12f8c6ee83ff4d810"
FORMAT_6_RECORD = "sha256:71ae0e560ab712df18b5b65f94996ff44d3d012fa6d048c56ece91b3158bbd9c"


def test_a_stored_format_6_design_loads_to_its_own_hash_with_no_targets() -> None:
    data = (FORMAT_6 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_6_DESIGN
    assert loaded.document == data
    doc = json.loads(data)
    assert doc["format"] == 6
    assert "targets" not in doc
    assert "target" not in doc["sources"]["ask"]["provides"]["q:ask/urgent"]
    assert "latent_targets" not in doc["sources"]["ask"]["provides"]["q:ask/urgent"]
    assert "target" not in doc["rules"]["urgent"]


@pytest.mark.parametrize(
    ("place", "says"),
    [
        ("targets", 'the document has the member "targets"'),
        ("target", '"q:ask/urgent" has the member "target"'),
        ("latent_targets", '"q:ask/urgent" has the member "latent_targets"'),
        ("rule", 'rule "urgent" has the member "target"'),
    ],
)
def test_check_11_refuses_a_format_6_document_holding_a_target_member(
    place: str, says: str
) -> None:
    doc = json.loads((FORMAT_6 / "design.json").read_bytes())
    question = doc["sources"]["ask"]["provides"]["q:ask/urgent"]
    if place == "targets":
        doc["targets"] = {}
    elif place == "target":
        question["target"] = None
    elif place == "latent_targets":
        question["latent_targets"] = []
    else:
        doc["rules"]["urgent"]["target"] = None
    with pytest.raises(InvalidDesign) as raised:
        load(doc)
    assert [f.check for f in raised.value.failures] == [11]
    assert says in str(raised.value)


def test_a_stored_format_6_record_reads_to_its_own_hash() -> None:
    data = (FORMAT_6 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_6_RECORD


def test_a_stored_format_6_record_replays_under_its_design_with_no_finding() -> None:
    loaded = load((FORMAT_6 / "design.json").read_bytes())
    rerun, findings = replay_sync(read_record((FORMAT_6 / "record.json").read_bytes()), loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == FORMAT_6_RECORD


def test_a_format_6_record_under_the_current_design_differs_only_by_its_design() -> None:
    rerun, findings = replay_sync(
        read_record((FORMAT_6 / "record.json").read_bytes()), load(DESIGN)
    )
    assert [f["kind"] for f in findings] == [FindingKind.design_mismatch]
    assert rerun is not None
