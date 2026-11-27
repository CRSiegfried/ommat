"""A stored format-12 design, the first that names no policy, and the format-11 records turns
under it wrote, read under their own rules: a decided turn, whose ``settings`` item names no
policy, and a broken entry, whose log ends at its ``t:state`` entry. The design's one function is
defined here, so its source hash is this file's."""

import json
from pathlib import Path
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    CallResult,
    DefectWarning,
    Design,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Question,
    Rule,
    Source,
    SourceCall,
    Table,
    Text,
    decide_sync,
    defects,
    escalation,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

FORMAT_12 = Path(__file__).parent / "format_12"
"""Named for its design's format: it holds a format-12 design and two format-11 records."""
FORMAT_12_DESIGN = "sha256:cee84042c495bbe200ee4d3e2b08149431af9ec543a788074b7b0c9502f56e3c"
FORMAT_11_DECIDED = "sha256:9106b4ee1bc629aa5e7a72913866754efc840cba0d9231bcc4a1a0da5578a0ac"
FORMAT_11_BROKEN = "sha256:f96fabb8804742906995e97cf1333cdf23d4a69d93324bc9e62b7d5889b8dfa7"
MARKER = {"kind": "Unavailable", "cause": {"code": "invalid-return", "refs": ["t:state"]}}


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


DESIGN = Design(
    t.record(subject=t.string),
    name="format 12",
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
DECIDED_STATE = {"subject": "hello"}
BROKEN_STATE = {"subject": 5}


async def mail(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """The oracle the stored turns were decided under: spam, surely."""
    return CallResult({"q:spam": {"yes": 0.9, "no": 0.1}})


def stored(name: str) -> bytes:
    return (FORMAT_12 / f"{name}.json").read_bytes()


def loaded() -> LoadedDesign:
    return load(stored("design"))


def test_a_stored_format_12_design_loads_to_its_own_hash_and_names_no_policy() -> None:
    data = stored("design")
    design = load(data)
    assert design.design_hash == FORMAT_12_DESIGN
    assert design.document == data
    doc = json.loads(data)
    assert doc["format"] == 12
    assert "policy" not in doc


def test_the_design_built_here_is_the_stored_one() -> None:
    assert load(DESIGN).document == stored("design")


def test_a_format_12_document_holding_a_policy_is_refused_by_check_11() -> None:
    doc = {**json.loads(stored("design")), "policy": "d:act"}
    with pytest.raises(InvalidDesign) as raised:
        load(json.dumps(doc).encode())
    assert [f.check for f in raised.value.failures] == [11]
    assert 'the member "policy"' in str(raised.value)


def test_a_format_12_source_holding_a_feedback_flag_is_refused_by_check_11() -> None:
    doc = json.loads(stored("design"))
    doc["sources"]["ask"]["feedback"] = False
    with pytest.raises(InvalidDesign) as raised:
        load(json.dumps(doc).encode())
    assert [f.check for f in raised.value.failures] == [11]
    assert 'source "ask" has the member "feedback", which its form does not have' in str(
        raised.value
    )


@pytest.mark.parametrize(
    ("name", "digest"), [("record", FORMAT_11_DECIDED), ("broken.record", FORMAT_11_BROKEN)]
)
def test_a_stored_format_11_record_reads_to_its_own_bytes_and_hash(name: str, digest: str) -> None:
    data = stored(name)
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == digest
    assert record.log[0] == {
        "format": 11,
        "hash": FORMAT_12_DESIGN,
        "helpers": 4,
        "item": "design",
    }
    assert "policy" not in record.log[1]
    assert "certificate_cap" not in record.log[1]["settings"]
    assert record.log[1]["shapes"]["t:state"] == {"subject": "leaf"}


def test_a_stored_format_11_record_replays_under_its_design_with_no_finding() -> None:
    record = read_record(stored("record"))
    assert record.entries["d:act"]["value"] == "archive"
    assert escalation(record, "d:act") is None
    rerun, findings = replay_sync(record, loaded())
    assert findings == []
    assert rerun is not None
    assert rerun.log == record.log


def test_a_stored_format_11_broken_entry_ends_at_its_t_state_entry() -> None:
    record = read_record(stored("broken.record"))
    entries = [i for i in record.log if i["item"] == "entry"]
    assert [(e["ref"], e["marker"]) for e in entries] == [("t:state", MARKER)]
    assert record.log[-1] == entries[-1]
    assert record.cause == MARKER["cause"]
    listed = defects(record)
    assert listed is not None
    assert [d["ref"] for d in listed] == ["t:state"]
    assert escalation(record, "d:act") is None
    found = escalation(record, "t:state")
    assert found is not None
    assert (found["kind"], found["defect"]) == ("Unavailable", True)
    assert replay_sync(record, loaded()) == (None, [{"kind": FindingKind.nothing_to_replay}])


def test_the_stored_records_are_what_today_s_engine_writes() -> None:
    """Holds while the engine writes format 11; the stored records outlive it."""
    design = loaded()
    decided = decide_sync(design, DECIDED_STATE, {"mail": mail}).record
    assert decided.log == read_record(stored("record")).log
    with pytest.warns(DefectWarning):
        broken = decide_sync(design, BROKEN_STATE, {"mail": mail}).record
    assert broken.log == read_record(stored("broken.record")).log


def edited(name: str, change: Any) -> bytes:
    doc = json.loads(stored(name))
    change(doc["log"])
    return json.dumps(rehashed(doc)).encode()


def test_a_format_11_record_holding_a_policy_is_an_invalid_record() -> None:
    def named(log: list[dict[str, Any]]) -> None:
        log[1]["policy"] = "d:act"

    with pytest.raises(InvalidRecord, match="holds policy, which a format-11 record does not"):
        read_record(edited("record", named))


def test_a_format_11_broken_entry_followed_by_the_old_policy_entry_is_an_invalid_record() -> None:
    def followed(log: list[dict[str, Any]]) -> None:
        log.append({**log[-1], "ref": "d:act"})

    with pytest.raises(InvalidRecord, match="nothing comes after a t:state entry holding a marker"):
        read_record(edited("broken.record", followed))


def test_a_format_11_record_holding_a_certificate_cap_is_an_invalid_record() -> None:
    def capped(log: list[dict[str, Any]]) -> None:
        log[1]["settings"]["certificate_cap"] = None

    with pytest.raises(InvalidRecord, match="certificate_cap, which a format-11 record does not"):
        read_record(edited("record", capped))


def test_a_format_11_record_holding_a_certificate_item_is_an_invalid_record() -> None:
    def certified(log: list[dict[str, Any]]) -> None:
        log.insert(3, {"item": "certificate", "round": 1})

    with pytest.raises(InvalidRecord, match="a format-11 record holds no certificate item"):
        read_record(edited("record", certified))


def test_a_format_11_record_holding_the_settled_code_is_an_invalid_record() -> None:
    def settled(log: list[dict[str, Any]]) -> None:
        log[-1]["marker"]["cause"]["code"] = "settled"

    with pytest.raises(InvalidRecord, match="code 'settled' is not in format 11's cause set"):
        read_record(edited("broken.record", settled))
