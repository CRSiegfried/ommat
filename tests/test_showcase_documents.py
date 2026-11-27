"""The stored documents of the outline and lens tests, under tests/showcase/: the showcase design,
the plain design and the full design of tests/designs.py, each as load wrote it, and the records
of the showcase turns those tests decide (decided, undecided, at the urgency bar, broken) and of
the plain turn, each as decide wrote it with its unhashed times fixed. Every one names a policy,
so they keep the outline's and the lens's policy path under test though no design names one now:
load refuses each design by check 11, and today's engine writes each turn as
``outline_showcase.policy_free_record`` gives it."""

import json
from pathlib import Path

import pytest

import outline_showcase as showcase
from designs import full
from ommat import (
    DefectWarning,
    FindingKind,
    InvalidDesign,
    Record,
    decide_sync,
    defects,
    escalation,
    load,
    read_record,
    replay_sync,
)
from ommat._documents import design_key
from ommat.outline import design_outline, turn_outline

SHOWCASE = Path(__file__).parent / "showcase"
DESIGNS = {
    "design": "sha256:8f588d3e8ad16ba5a3a6d8525ca900a473c444146ad5a5056d19dd87e4193781",
    "plain.design": "sha256:6ac4384b9886b7d51c77f79dcb888c6b2ccc301d57a33ae0ea7e12e89a366600",
    "full.design": "sha256:41c33a44fe8bba76e2ac04b734b45c0a7ebeec1d983baee25778f7b928617281",
}
"""Each stored design by its design hash."""
RECORDS = {
    "decided": "design",
    "undecided": "design",
    "level": "design",
    "broken": "design",
    "plain": "plain.design",
}
"""Each stored record by the stored design it was decided under."""


def stored(name: str) -> bytes:
    return (SHOWCASE / f"{name}.json").read_bytes()


def record(name: str) -> Record:
    return read_record(stored(f"{name}.record"))


def live(name: str) -> Record:
    """The record the outline and lens tests decide live for ``name``."""
    loaded = showcase.loaded()
    if name == "decided":
        return showcase.turn(loaded, *showcase.DECIDED)
    if name == "undecided":
        return showcase.turn(loaded, *showcase.UNDECIDED)
    if name == "level":
        return showcase.turn(loaded, 0.6, showcase.DECIDED[1])
    if name == "plain":
        return showcase.plain_turn()
    state = {"subject": 5, "body": "Nothing loads since 9."}
    with pytest.warns(DefectWarning):
        return decide_sync(
            loaded, state, {"llm": showcase.llm(*showcase.DECIDED), "desk": showcase.desk}
        ).record


@pytest.mark.parametrize("name", list(DESIGNS))
def test_a_stored_design_is_refused_by_check_11_and_outlines_its_policy(name: str) -> None:
    data = stored(name)
    assert json.loads(data)["format"] == 11
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 11 is not 12" in str(raised.value)
    assert design_outline(data)["policy_node"] is not None
    assert design_key(data) == DESIGNS[name]


def test_the_designs_the_tests_build_are_the_stored_ones_with_no_policy() -> None:
    assert showcase.policy_free("design") == showcase.loaded().document
    assert showcase.policy_free("plain.design") == load(showcase.PLAIN).document
    assert showcase.policy_free("full.design") == load(full()).document


@pytest.mark.parametrize("name", list(RECORDS))
def test_a_stored_record_reads_to_its_own_bytes_and_pairs_with_its_design(name: str) -> None:
    data = stored(f"{name}.record")
    read = read_record(data)
    assert read.document == data
    assert read.log[0]["format"] == 10
    assert read.log[0]["hash"] == design_key(stored(RECORDS[name]))
    assert read.log[1]["policy"] in read.entries
    assert turn_outline(data, stored(RECORDS[name]))["paired"] is True


@pytest.mark.parametrize("name", list(RECORDS))
def test_the_turn_the_tests_decide_is_the_stored_one_as_a_policy_free_record(name: str) -> None:
    written = read_record(showcase.policy_free_record(name, RECORDS[name]))
    assert written.log[0]["format"] == 11
    assert written.log == live(name).log


def test_the_stored_records_hold_a_decision_an_undecided_turn_and_a_broken_entry() -> None:
    assert record("decided").entries["d:act"]["value"] == "escalate"
    assert record("level").entries["d:act"]["value"] == "escalate"
    assert record("plain").entries["d:verdict"]["value"] == "yes"
    assert record("undecided").entries["d:act"]["marker"]["kind"] == "Unsure"
    undecided = escalation(record("undecided"), "d:act")
    assert undecided is not None
    assert undecided["root_cause"]["code"] == "abstained"
    broken = record("broken")
    assert [i["ref"] for i in broken.log if i["item"] == "entry"] == ["t:state", "d:act"]
    listed = defects(broken)
    assert listed is not None
    assert [d["ref"] for d in listed] == ["t:state", "d:act"]


@pytest.mark.parametrize("name", list(RECORDS))
def test_a_stored_record_replays_under_todays_design_as_a_design_mismatch(name: str) -> None:
    loaded = load(showcase.policy_free(RECORDS[name]))
    rerun, findings = replay_sync(record(name), loaded)
    if name == "broken":
        assert (rerun, findings) == (None, [{"kind": FindingKind.nothing_to_replay}])
        return
    assert findings == [
        {
            "kind": FindingKind.design_mismatch,
            "recorded": DESIGNS[RECORDS[name]],
            "replayed": loaded.design_hash,
        }
    ]
    assert rerun is not None
    # The fixture gives back the answers, not the call's profile, so only the entries agree.
    assert rerun.entries == record(name).entries


def test_the_stored_idia_turn_names_its_policy_and_pairs_with_its_design() -> None:
    """The idia end-to-end turn as the engine wrote it while designs named a policy, kept for
    the outline's lineage tests; ``tests/idia_logs`` holds it as today's engine writes it."""
    data, design = stored("idia.record"), stored("idia.design")
    read = read_record(data)
    assert read.document == data
    assert (read.log[0]["format"], read.log[1]["policy"]) == (10, "d:route")
    assert json.loads(design)["format"] == 11
    assert read.log[0]["hash"] == design_key(design)
    assert turn_outline(data, design)["paired"] is True
    today = read_record((Path(__file__).parent / "idia_logs" / "turn.record.json").read_bytes())
    assert today.log[0]["format"] == 11
    assert "policy" not in today.log[1]
    assert "certificate_cap" not in today.log[1]["settings"]
    assert today.log[2:] == read.log[2:]
