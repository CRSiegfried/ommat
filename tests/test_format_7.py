"""A stored format-7 design and format-6 record, as the step-18 engine wrote them, read under their
own rules: a ``list<string>`` candidate set whose ``candidates`` member is ``true``, and a
``settings`` item holding ``grounded`` and no ``sets``."""

import json
from pathlib import Path

import pytest

from format_7_design import DESIGN
from ommat import FindingKind, InvalidDesign, load, read_record, replay_sync

FORMAT_7 = Path(__file__).parent / "format_7"
FORMAT_7_DESIGN = "sha256:730f1f041cc6c9a019f5619c9d828079663705194d5acb05f11205a2b92a0f17"
FORMAT_6_RECORD = "sha256:1615a18d3512a71062293d65f48876c4aa58ae55e87202155c66cdc33121bf07"


def test_a_stored_format_7_design_loads_to_its_own_hash() -> None:
    data = (FORMAT_7 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_7_DESIGN
    assert loaded.document == data
    doc = json.loads(data)
    assert doc["format"] == 7
    assert doc["rules"]["streets"]["candidates"] is True


@pytest.mark.parametrize("candidates", ["text", ""])
def test_check_11_refuses_a_format_7_document_naming_a_candidate_field(candidates: str) -> None:
    doc = json.loads((FORMAT_7 / "design.json").read_bytes())
    doc["rules"]["streets"]["candidates"] = candidates
    with pytest.raises(InvalidDesign) as raised:
        load(doc)
    assert [f.check for f in raised.value.failures] == [11]
    assert 'rule "streets" names a candidate field, which format 7 does not have' in str(
        raised.value
    )


def test_a_stored_format_6_record_reads_to_its_own_hash_with_no_sets() -> None:
    data = (FORMAT_7 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_6_RECORD
    assert record.log[1]["grounded"] == {"q:lamps/out": "d:streets"}
    assert "sets" not in record.log[1]


def test_a_stored_format_6_record_replays_under_its_design_with_no_finding() -> None:
    loaded = load((FORMAT_7 / "design.json").read_bytes())
    rerun, findings = replay_sync(read_record((FORMAT_7 / "record.json").read_bytes()), loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.log[0]["format"] == 7
    assert rerun.log[1]["sets"] == {"d:streets": None}


def test_a_format_6_record_under_the_current_design_differs_only_by_its_design() -> None:
    rerun, findings = replay_sync(
        read_record((FORMAT_7 / "record.json").read_bytes()), load(DESIGN)
    )
    assert [f["kind"] for f in findings] == [FindingKind.design_mismatch]
    assert rerun is not None
