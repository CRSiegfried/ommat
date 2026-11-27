"""A stored format-7 design and format-6 record, as the step-18 engine wrote them: the design
refused by check 11, naming its format, and the record read under its own rules: a ``settings``
item holding ``grounded`` and no ``sets``."""

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

FORMAT_7 = Path(__file__).parent / "format_7"
FORMAT_6_RECORD = "sha256:1615a18d3512a71062293d65f48876c4aa58ae55e87202155c66cdc33121bf07"


def test_check_11_refuses_a_stored_format_7_design_naming_its_format() -> None:
    with pytest.raises(InvalidDesign) as raised:
        load((FORMAT_7 / "design.json").read_bytes())
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 7 is not 12" in str(raised.value)


def test_a_stored_format_6_record_reads_to_its_own_hash_with_no_sets() -> None:
    data = (FORMAT_7 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_6_RECORD
    assert record.log[1]["grounded"] == {"q:lamps/out": "d:streets"}
    assert "sets" not in record.log[1]


def test_a_stored_format_6_record_has_no_view() -> None:
    record = read_record((FORMAT_7 / "record.json").read_bytes())
    assert escalation(record, "d:route") is None
    assert to_chat_completions(record, refs=["d:route"]) == []


def test_a_stored_format_6_record_replays_as_the_wrong_design() -> None:
    record = read_record((FORMAT_7 / "record.json").read_bytes())
    assert replay_sync(record, load(minimal())) == (None, [{"kind": FindingKind.wrong_design}])
