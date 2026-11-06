"""The deprecated transcript view, over the four paths of ``transcript_turns``: clear, hold,
undecided on an unsure confirmation, and the oracle down. The turn history the renderers give is
tested in ``test_turn_history``."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from ommat import (
    CallResult,
    Record,
    Rule,
    Source,
    decide_sync,
    load,
    read_record,
    replay_sync,
    transcript,
)
from ommat import types as t
from transcript_turns import (
    PATHS,
    RULES,
    SOURCES,
    STATE,
    YES,
    confirming,
    design,
    model,
    turn,
)

pytestmark = pytest.mark.filterwarnings("ignore:transcript is deprecated:DeprecationWarning")

ID = re.compile(r"om_[0-9a-f]{24}_(c[1-9][0-9]*|d)")


def view(path: str, **kw: Any) -> dict[str, Any]:
    out = transcript(turn(path, **kw).record)
    assert out is not None
    return out


def calls(v: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for calls in v["rounds"] for call in calls]


def ids(v: dict[str, Any]) -> list[str]:
    return [c["id"] for c in calls(v)] + [v["decide"]["id"]]


# ---------------------------------------------------------------------------------------------
# The view


def test_the_view_gives_every_call_and_the_decision_in_order() -> None:
    decision = turn("clear")
    record = decision.record
    v = transcript(record)
    assert v is not None
    stem = "om_" + record.record_hash.removeprefix("sha256:")[:24]
    assert list(v) == ["record_hash", "design", "rounds", "decide"]
    assert v["record_hash"] == record.record_hash
    assert v["design"] == {"hash": record.design_hash}
    yes = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"no": 0.05, "yes": 0.95}}
    half = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"no": 0.5, "yes": 0.5}}
    assert v["rounds"] == [
        [
            {
                "id": f"{stem}_c1",
                "source": "ask/again",
                "asks": ["q:again"],
                "outcome": "returned",
                "answers": {"q:again": {"value": half}},
            },
            {
                "id": f"{stem}_c2",
                "source": "mail/router",
                "asks": ["q:route"],
                "outcome": "returned",
                "answers": {"q:route": {"value": yes}},
            },
            {
                "id": f"{stem}_c3",
                "source": "lookup",
                "asks": ["t:order"],
                "outcome": "returned",
                "answers": {"t:order": {"value": {"id": "o-1"}}},
            },
        ],
        [
            {
                "id": f"{stem}_c4",
                "source": "confirm",
                "asks": ["q:confirm"],
                "outcome": "returned",
                "answers": {"q:confirm": {"value": yes}},
            }
        ],
    ]
    assert v["decide"] == {
        "id": f"{stem}_d",
        "board": {"d:route": {"value": "yes"}, "d:sure": {"value": "yes"}},
        "decision": {"value": "archive"},
    }


def test_two_sources_of_one_call_item_share_its_outcome_and_round() -> None:
    record = turn("clear").record
    items = [i for i in record.log if i["item"] == "call"]
    assert items[0]["sources"] == ["ask/again", "mail/router"]
    first = view("clear")["rounds"][0]
    assert [c["source"] for c in first[:2]] == items[0]["sources"]


def test_the_view_holds_no_state_provenance_prompt_or_derivation() -> None:
    text = json.dumps(view("clear"))
    for absent in ("t:state", "Archive the newsletters", "acct-7", "derivation", "identity"):
        assert absent not in text
    assert "Archive this?" not in text  # no question wording


def test_a_value_decision_is_tagged_and_the_board_leaves_out_the_policy_s_entry() -> None:
    v = view("hold")
    assert v["decide"]["decision"] == {"value": "hold"}
    assert "d:act" not in v["decide"]["board"]


def test_an_unsure_confirmation_leaves_the_decision_a_tagged_marker() -> None:
    v = view("unsure")
    assert v["decide"]["board"]["d:sure"] == {
        "marker": {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["q:confirm"]}}
    }
    assert v["decide"]["decision"] == {
        "marker": {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["q:confirm"]}}
    }


def test_a_failed_call_keeps_its_outcome_and_a_source_never_called_is_in_no_round() -> None:
    v = view("down")
    failed = {"failed": {"code": "raised", "refs": ["q:again", "q:route"]}}
    (first,) = v["rounds"]
    assert [c["outcome"] for c in first] == [failed, failed, "returned"]
    assert first[1]["answers"] == {
        "q:route": {"marker": {"kind": "Unavailable", "cause": failed["failed"]}}
    }
    assert "confirm" not in [c["source"] for c in calls(v)]


def test_a_value_with_a_marker_field_reads_as_a_value() -> None:
    lookup = Source(
        {"t:order": "record{id: string, marker: string}"},
        needs=["t:state.message"],
        oracle="tools",
    )

    async def marked(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({"t:order": {"id": "o-1", "marker": "m"}})

    ld = load(design(sources={**SOURCES, "lookup": lookup}))
    bindings = {"default": confirming(YES), "model": model(), "tools": marked}
    v = transcript(decide_sync(ld, STATE, bindings).record)
    assert v is not None
    assert calls(v)[2]["answers"] == {"t:order": {"value": {"id": "o-1", "marker": "m"}}}


@pytest.mark.parametrize("path", PATHS)
def test_ids_fit_the_pattern_are_unique_and_one_log_gives_the_same(path: str) -> None:
    record = turn(path).record
    v = transcript(record)
    assert v is not None
    got = ids(v)
    assert all(ID.fullmatch(i) and len(i) <= 40 for i in got)
    assert len(set(got)) == len(got)
    again = transcript(read_record(record.document))
    assert again == v
    replayed, findings = replay_sync(record, load(design()))
    assert findings == []
    assert replayed is not None
    assert replayed.record_hash == record.record_hash
    assert transcript(replayed) == v


OLD_RECORDS = {
    **{fmt: f"format_{fmt}/record.json" for fmt in range(1, 7)},
    7: "format_7_record/turn.record.json",
}
"""A stored record of each record format before 8, by its format."""


def old_record(fmt: int) -> Record:
    record = read_record((Path(__file__).parent / OLD_RECORDS[fmt]).read_bytes())
    assert record.log[0].get("format", 1) == fmt
    return record


@pytest.mark.parametrize("fmt", sorted(OLD_RECORDS))
def test_a_record_before_format_8_gives_no_transcript(fmt: int) -> None:
    assert transcript(old_record(fmt)) is None


def test_a_record_with_no_entry_at_the_policy_gives_none() -> None:
    def cancel(r: Any) -> str:
        raise asyncio.CancelledError

    rule = Rule("d:act", t.label("archive", "hold"), cancel, version="1", reads=["d:sure"])
    ld = load(design(policy="d:act", rules={**RULES, "act": rule}))
    with pytest.raises(asyncio.CancelledError) as caught:
        turn("clear", ld)
    record = cast("Record", getattr(caught.value, "record"))  # noqa: B009
    assert not any(i["item"] == "entry" and i["ref"] == "d:act" for i in record.log)
    assert transcript(record) is None


def test_the_view_refuses_what_is_not_a_record() -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        transcript(turn("clear"))  # type: ignore[arg-type]


def test_the_view_warns_that_it_is_deprecated() -> None:
    record = turn("clear").record
    with pytest.warns(DeprecationWarning, match="ommat.conversation renderer"):
        transcript(record)


def test_the_calls_of_one_call_item_share_no_outcome_object() -> None:
    first = view("down")["rounds"][0]
    assert first[0]["outcome"] == first[1]["outcome"]
    assert first[0]["outcome"] is not first[1]["outcome"]
