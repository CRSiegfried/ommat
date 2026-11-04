"""A stored format-5 design and record, as the format-5 engine wrote them: the design refused by
check 11, naming its format, and the record read under its own rules: record fields spelled in
code-point order, no order members, and values held sorted."""

from pathlib import Path

import pytest

from designs import minimal
from ommat import (
    FindingKind,
    InvalidDesign,
    escalation,
    load,
    read_record,
    replay_sync,
    transcript,
)

FORMAT_5 = Path(__file__).parent / "format_5"
FORMAT_5_RECORD = "sha256:05dac70e9d88239b5d5e9c0f56178da66c5aeb27f98f30f1a4adc94b33f5b1a3"


def test_check_11_refuses_a_stored_format_5_design_naming_its_format() -> None:
    with pytest.raises(InvalidDesign) as raised:
        load((FORMAT_5 / "design.json").read_bytes())
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 5 is not 10" in str(raised.value)


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


def test_a_stored_format_5_record_has_no_view() -> None:
    record = read_record((FORMAT_5 / "record.json").read_bytes())
    assert escalation(record) is None
    assert transcript(record) is None


def test_a_stored_format_5_record_replays_as_the_wrong_design() -> None:
    record = read_record((FORMAT_5 / "record.json").read_bytes())
    assert replay_sync(record, load(minimal())) == (None, [{"kind": FindingKind.wrong_design}])
