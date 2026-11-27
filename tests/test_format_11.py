"""A stored format-11 design and the format-9 record a turn under it wrote: the design, the last
that names a policy, is refused by check 11, naming its format, as the stored format-10 one is;
the record is read under its own rules, its ``settings`` item holding each ref's shape tree under
``shapes``, which ``redact`` reads the caller's redaction policy over. The format-10 records, a
decided turn and a broken entry, are the last that name a policy, written under the same design
by the engine that wrote format 10. The design built here is today's form of the stored one; its
one function is defined here, so its source hash is this file's."""

import json
from pathlib import Path
from typing import Any

import pytest

from ledger_logs import rehashed
from ommat import (
    CallResult,
    DefectWarning,
    Design,
    EntryRefusal,
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
    redact,
    replay_sync,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

FORMAT_10 = Path(__file__).parent / "format_10"
"""Named for its design's format: it holds a format-10 design and a format-8 record."""
FORMAT_11 = Path(__file__).parent / "format_11"
"""Named for its design's format: it holds a format-11 design and a format-9 record."""
FORMAT_10_RECORD = Path(__file__).parent / "format_10_record"
"""Named for its records' format: it holds the format-11 design again and two format-10 records."""
FORMAT_11_DESIGN = "sha256:9468d04086afeab1bb7c51c3fdba2185ee0929f522dbeafabe31544d5f7fcf65"
FORMAT_9_RECORD = "sha256:bb677453dc1dae834ca3677270c051a9f183ba0246fd0b23affc0ad3512e93cc"
FORMAT_10_DECIDED = "sha256:be3e64859a5ff82dec01d54129c3948bb25ecc520adc6df166c6fdbb8f6ce2ac"
FORMAT_10_BROKEN = "sha256:836ccc026d3982cc96627633c4c3d1940c9fe19e3e7bf2c7b085ca36a4865a7c"


def route(r: Any) -> Any:
    return pick_with_margin(r, "q:spam", 0.5, 0.1)


DESIGN = Design(
    state=t.record(subject=t.string),
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


def today() -> LoadedDesign:
    return load(DESIGN)


def test_check_11_refuses_the_stored_format_11_design_naming_its_format() -> None:
    data = (FORMAT_11 / "design.json").read_bytes()
    doc = json.loads(data)
    assert (doc["format"], doc["policy"]) == (11, "d:act")
    assert "classes" not in doc
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 11 is not 12, the one this engine reads" in str(raised.value)


def test_the_design_built_here_is_the_stored_one_but_for_its_format_policy_and_feedback() -> None:
    stored = json.loads((FORMAT_11 / "design.json").read_bytes())
    del stored["policy"]
    for source in stored["sources"].values():
        del source["feedback"]
    assert json.loads(today().document) == {**stored, "format": 12}
    assert today().design_hash != FORMAT_11_DESIGN


def test_check_11_refuses_the_stored_format_10_design_naming_its_format() -> None:
    data = (FORMAT_10 / "design.json").read_bytes()
    assert json.loads(data)["format"] == 10
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 10 is not 12, the one this engine reads" in str(raised.value)


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
    assert escalation(record, "d:act") is None
    view = json.loads(redact(record, bytes(32), {"d:act": "open"}))
    act = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == "d:act")
    assert act["value"] == "archive"
    route_entry = next(i for i in view["log"] if i["item"] == "entry" and i["ref"] == "d:route")
    assert route_entry["value"].startswith("hmac-sha256:")
    assert view["applied"]["t:state"] == {"subject": "content"}


def mismatch(record_hash: str = FORMAT_11_DESIGN) -> list[dict[str, Any]]:
    return [
        {
            "kind": FindingKind.design_mismatch,
            "recorded": record_hash,
            "replayed": today().design_hash,
        }
    ]


def test_a_stored_format_9_record_replays_under_todays_design_as_a_design_mismatch() -> None:
    record = read_record((FORMAT_11 / "record.json").read_bytes())
    rerun, findings = replay_sync(record, today())
    assert findings == mismatch()
    assert rerun is not None
    # The rerun is written at today's format, 11, under today's design, naming no policy.
    assert rerun.log[0] == {**record.log[0], "format": 11, "hash": today().design_hash}
    assert "policy" not in rerun.log[1]
    assert rerun.log[2:] == record.log[2:]


def format_10(name: str) -> bytes:
    return (FORMAT_10_RECORD / f"{name}.json").read_bytes()


def test_the_format_10_records_design_is_the_stored_format_11_one() -> None:
    assert format_10("design") == (FORMAT_11 / "design.json").read_bytes()


@pytest.mark.parametrize(
    ("name", "digest"), [("record", FORMAT_10_DECIDED), ("broken.record", FORMAT_10_BROKEN)]
)
def test_a_stored_format_10_record_reads_to_its_own_bytes_and_hash(name: str, digest: str) -> None:
    data = format_10(name)
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == digest
    assert record.log[0] == {
        "format": 10,
        "hash": FORMAT_11_DESIGN,
        "helpers": 4,
        "item": "design",
    }
    assert record.log[1]["policy"] == "d:act"
    assert record.log[1]["shapes"]["t:state"] == {"subject": "leaf"}


def test_a_stored_format_10_record_is_decided_and_replays_as_a_design_mismatch() -> None:
    record = read_record(format_10("record"))
    assert record.entries["d:act"]["value"] == "archive"
    assert record.log[-1] == {"cost": 0, "deadline": False, "item": "round", "round": 1}
    assert escalation(record, "d:act") is None
    rerun, findings = replay_sync(record, today())
    assert findings == mismatch()
    assert rerun is not None
    assert rerun.log[2:] == record.log[2:]


def test_a_stored_format_10_broken_entry_ends_at_the_policy_s_entry() -> None:
    record = read_record(format_10("broken.record"))
    marker = {"kind": "Unavailable", "cause": {"code": "invalid-return", "refs": ["t:state"]}}
    entries = [i for i in record.log if i["item"] == "entry"]
    assert [(e["ref"], e["marker"]) for e in entries] == [("t:state", marker), ("d:act", marker)]
    assert record.log[-1] == entries[-1]
    listed = defects(record)
    assert listed is not None
    assert [d["ref"] for d in listed] == ["t:state", "d:act"]
    found = escalation(record, "d:act")
    assert found is not None
    assert (found["kind"], found["defect"]) == ("Unavailable", True)
    assert replay_sync(record, today()) == (None, [{"kind": FindingKind.nothing_to_replay}])


def test_todays_engine_writes_the_format_10_records_at_format_11_naming_no_policy() -> None:
    """Today's records under today's form of the design: the stored format-10 logs with format
    11, today's design hash and no ``policy``, and a broken entry ending at ``t:state``."""
    loaded = today()
    decided = decide_sync(loaded, {"subject": "hello"}, {"mail": mail}).record
    with pytest.warns(DefectWarning):
        broken = decide_sync(loaded, {"subject": 5}, {"mail": mail}).record
    for written, stored, kept in ((decided, "record", None), (broken, "broken.record", -1)):
        log = read_record(format_10(stored)).log
        assert written.log[0] == {**log[0], "format": 11, "hash": loaded.design_hash}
        settings = {k: v for k, v in log[1]["settings"].items() if k != "certificate_cap"}
        assert written.log[1] == {
            **{k: v for k, v in log[1].items() if k != "policy"},
            "settings": settings,
        }
        assert written.log[2:] == log[2:kept]


def capped(cap: object) -> Any:
    """The stored format-10 record with its run settings' ``certificate_cap`` set to ``cap``,
    or dropped for ``...``, rehashed."""
    doc = json.loads(format_10("record"))
    settings = doc["log"][1]["settings"]
    assert settings["certificate_cap"] is None
    if cap is ...:
        del settings["certificate_cap"]
    else:
        settings["certificate_cap"] = cap
    return json.dumps(rehashed(doc)).encode()


def test_a_format_10_record_with_a_certificate_cap_reads_and_replay_refuses_it() -> None:
    record = read_record(capped(1))
    assert record.log[1]["settings"]["certificate_cap"] == 1
    with pytest.raises(EntryRefusal, match="the recorded certificate cap 1 is not None"):
        replay_sync(record, today())


def test_a_format_10_record_lacking_a_certificate_cap_is_an_invalid_record() -> None:
    with pytest.raises(InvalidRecord, match="have no 'certificate_cap' member"):
        read_record(capped(...))
