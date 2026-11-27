"""A stored format-6 design and record, as the format-6 engine wrote them: the design refused by
check 11, naming its format, and the record read under its own rules."""

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
)
from ommat.conversation import to_chat_completions

FORMAT_6 = Path(__file__).parent / "format_6"
FORMAT_6_RECORD = "sha256:71ae0e560ab712df18b5b65f94996ff44d3d012fa6d048c56ece91b3158bbd9c"


def test_check_11_refuses_a_stored_format_6_design_naming_its_format() -> None:
    with pytest.raises(InvalidDesign) as raised:
        load((FORMAT_6 / "design.json").read_bytes())
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 6 is not 12" in str(raised.value)


def test_a_stored_format_6_record_reads_to_its_own_hash() -> None:
    data = (FORMAT_6 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_6_RECORD


def test_a_stored_format_6_record_has_no_view() -> None:
    record = read_record((FORMAT_6 / "record.json").read_bytes())
    assert escalation(record, "d:route") is None
    assert to_chat_completions(record, refs=["d:route"]) == []


def test_a_stored_format_6_record_replays_as_the_wrong_design() -> None:
    record = read_record((FORMAT_6 / "record.json").read_bytes())
    assert replay_sync(record, load(minimal())) == (None, [{"kind": FindingKind.wrong_design}])
