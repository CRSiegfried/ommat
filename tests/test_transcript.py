"""The transcript view and the Chat Completions renderer, over the four paths of
``transcript_turns``: clear, hold, undecided on an unsure confirmation, and the oracle down."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from ommat import (
    CallResult,
    Question,
    Record,
    Rule,
    Source,
    Text,
    decide_sync,
    load,
    read_record,
    replay_sync,
    transcript,
)
from ommat import types as t
from ommat.conversation import (
    drop_earlier_bookkeeping,
    is_bookkeeping,
    to_chat_completions,
)
from transcript_turns import (
    PATHS,
    RULES,
    SOURCES,
    STATE,
    YES,
    confirming,
    design,
    model,
    tools,
    turn,
)

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


# ---------------------------------------------------------------------------------------------
# The Chat Completions renderer


def test_chat_completions_gives_each_round_and_decide() -> None:
    v = view("clear")
    messages = to_chat_completions(v)
    roles = [(m["role"], len(m.get("tool_calls", []))) for m in messages]
    assert roles == [
        ("assistant", 3),
        ("tool", 0),
        ("tool", 0),
        ("tool", 0),
        ("assistant", 1),
        ("tool", 0),
        ("assistant", 1),
        ("tool", 0),
    ]
    first = messages[0]
    assert first["content"] is None
    assert [c["function"]["name"] for c in first["tool_calls"]] == [
        "ask-again",
        "mail-router",
        "lookup",
    ]
    assert first["tool_calls"][1] == {
        "id": calls(v)[1]["id"],
        "type": "function",
        "function": {"name": "mail-router", "arguments": '{"asks":["q:route"]}'},
    }
    assert [m["tool_call_id"] for m in messages[1:4]] == [c["id"] for c in calls(v)[:3]]
    assert json.loads(messages[2]["content"]) == calls(v)[1]["answers"]
    decide_call = messages[-2]["tool_calls"][0]
    assert decide_call["id"] == v["decide"]["id"]
    assert decide_call["function"]["name"] == "decide"
    assert json.loads(decide_call["function"]["arguments"]) == {"board": v["decide"]["board"]}
    result = json.loads(messages[-1]["content"])
    assert result == {k: x for k, x in v["decide"].items() if k not in ("id", "board")}
    assert result["decision"] == {"value": "archive"}


@pytest.mark.parametrize("path", PATHS)
def test_chat_completions_renders_every_path_and_ends_at_decide_s_result(
    path: str,
) -> None:
    v = view(path)
    messages = to_chat_completions(v)
    calls_made = [c for m in messages for c in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert [c["id"] for c in calls_made] == ids(v)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == v["decide"]["id"]
    assert results == [c["id"] for c in calls_made]


def test_a_failed_call_s_result_is_its_marker_answers_and_nothing_else() -> None:
    v = view("down")
    messages = to_chat_completions(v)
    router = next(m for m in messages if m.get("tool_call_id") == calls(v)[1]["id"])
    assert json.loads(router["content"]) == calls(v)[1]["answers"]
    assert "error" not in router["content"]
    assert "failed" not in router["content"]


@pytest.mark.parametrize("path", PATHS)
def test_chat_completions_bytes_are_stable_and_in_held_order(path: str) -> None:
    record = turn(path).record
    once = to_chat_completions(transcript(record))
    again = to_chat_completions(transcript(read_record(record.document)))
    assert json.dumps(once) == json.dumps(again)
    for m in once:
        strings = [m.get("content")] + [c["function"]["arguments"] for c in m.get("tool_calls", [])]
        for s in strings:
            if s is not None:
                # The canonical forms, each object's members in the order it holds.
                assert s == json.dumps(json.loads(s), separators=(",", ":"), ensure_ascii=False)


def test_none_renders_as_no_messages() -> None:
    assert to_chat_completions(None) == []


def _renamed(v: dict[str, Any], *, source: str | None = None) -> dict[str, Any]:
    v = json.loads(json.dumps(v))
    if source is not None:
        v["rounds"][0][0]["source"] = source
    return v


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source": "decide"}, "the name decide's call takes"),
        ({"source": "a/" + "b" * 63}, "over 64 characters"),
    ],
)
def test_a_tool_name_no_provider_takes_is_refused(change: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        to_chat_completions(_renamed(view("clear"), **change))


def test_a_source_s_slash_becomes_a_hyphen_and_a_long_name_within_64_passes() -> None:
    name = "a/" + "b" * 62
    messages = to_chat_completions(_renamed(view("clear"), source=name))
    assert messages[0]["tool_calls"][0]["function"]["name"] == "a-" + "b" * 62


# ---------------------------------------------------------------------------------------------
# Chat Completions bookkeeping


def test_chat_bookkeeping_is_the_source_calls_and_decide() -> None:
    messages = to_chat_completions(view("clear"))
    assert [is_bookkeeping(m) for m in messages] == [True] * 8


def test_a_message_of_the_caller_s_or_the_model_s_is_not_bookkeeping() -> None:
    assert not is_bookkeeping({"role": "user", "content": "hi"})
    assert not is_bookkeeping({"role": "assistant", "content": "ok"})
    assert not is_bookkeeping({"role": "tool", "tool_call_id": "call_1", "content": "{}"})
    own = {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    assert not is_bookkeeping({"role": "assistant", "content": None, "tool_calls": [own]})


def _chat_paired(messages: list[dict[str, Any]]) -> bool:
    called = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    return sorted(called) == sorted(m["tool_call_id"] for m in messages if m["role"] == "tool")


def _first_turn() -> list[dict[str, Any]]:
    first: list[dict[str, Any]] = [{"role": "user", "content": "Archive the newsletters"}]
    first += to_chat_completions(view("clear"))
    first += [{"role": "assistant", "content": "Archived."}]
    return first


def test_dropping_earlier_bookkeeping_over_two_turns_keeps_calls_paired() -> None:
    first = _first_turn()
    second = [*first, {"role": "user", "content": "Are you sure?"}]
    current = to_chat_completions(view("unsure"))
    kept = drop_earlier_bookkeeping(second + current)
    assert kept == [first[0], first[-1], second[-1], *current]
    assert _chat_paired(kept)


def test_chat_keeps_the_latest_record_wherever_the_prompt_stands() -> None:
    current = to_chat_completions(view("unsure"))
    late = [*current, {"role": "user", "content": "late"}]
    assert drop_earlier_bookkeeping(late) == late


def test_chat_takes_empty_content_as_none_and_keeps_a_result_with_a_kept_call() -> None:
    first = _first_turn()
    first[1] = {**first[1], "content": ""}
    assert is_bookkeeping(first[1])
    first[5] = {**first[5], "content": "thinking"}  # a call with text, not bookkeeping
    current = to_chat_completions(view("unsure"))
    kept = drop_earlier_bookkeeping([*first, *current])
    assert first[5] in kept
    assert first[6] in kept  # its result stays with it
    assert _chat_paired(kept)


# ---------------------------------------------------------------------------------------------
# Round-one review cases


def test_chat_filters_ignore_odd_members() -> None:
    odd: list[dict[str, Any]] = [
        {"role": "assistant", "content": None, "tool_calls": "x"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": ["om_x"], "content": "{}"},
    ]
    assert not any(is_bookkeeping(m) for m in odd)
    messages = [*to_chat_completions(view("down")), *odd, *to_chat_completions(view("unsure"))]
    kept = drop_earlier_bookkeeping(messages)
    assert kept[: len(odd)] == odd
    plain = [{"role": "user", "content": "hi"}]
    assert drop_earlier_bookkeeping(plain) == plain
    assert drop_earlier_bookkeeping(plain) is not plain


def test_the_calls_of_one_call_item_share_no_outcome_object() -> None:
    first = view("down")["rounds"][0]
    assert first[0]["outcome"] == first[1]["outcome"]
    assert first[0]["outcome"] is not first[1]["outcome"]


# ---------------------------------------------------------------------------------------------
# A record rendered for more than one turn


def test_a_turn_stems_every_id_within_the_pattern_and_none_keeps_the_view_s() -> None:
    v = view("clear")
    plain = to_chat_completions(v)
    assert plain == to_chat_completions(v, turn=None)
    turned = to_chat_completions(v, turn="retry-1")
    old = [c["id"] for m in plain for c in m.get("tool_calls", [])]
    new = [c["id"] for m in turned for c in m.get("tool_calls", [])]
    assert all(ID.fullmatch(i) and len(i) <= 40 for i in new)
    assert [i[27:] for i in new] == [i[27:] for i in old]
    assert set(new).isdisjoint(old)
    assert to_chat_completions(v, turn="retry-1") == turned
    assert to_chat_completions(v, turn=1) != to_chat_completions(v, turn="1")


@pytest.mark.parametrize("turn", [True, 1.5, b"x", ["a"], None.__class__])
def test_a_turn_of_another_type_is_refused(turn: Any) -> None:
    with pytest.raises(ValueError, match="turn must be a str or an int"):
        to_chat_completions(view("clear"), turn=turn)


def test_five_turns_repeating_one_record_repeat_no_id_and_keep_only_the_last() -> None:
    history: list[dict[str, Any]] = []
    for n, path in enumerate(["clear", "unsure", "clear", "down", "clear"]):
        history.append({"role": "user", "content": f"turn {n}"})
        messages = to_chat_completions(view(path), turn=n)
        history += messages
        history.append({"role": "assistant", "content": "ok"})
    every = [c["id"] for m in history for c in m.get("tool_calls", [])]
    assert len(every) == len(set(every))
    kept = drop_earlier_bookkeeping(history)
    assert _chat_paired(kept)
    stems = {c["id"][:27] for m in kept if is_bookkeeping(m) for c in m.get("tool_calls", [])}
    assert len(stems) == 1


# ---------------------------------------------------------------------------------------------
# Round-three mutation cases


def test_a_source_asking_two_refs_lists_them_in_code_point_order() -> None:
    pair = Source(
        {"q:zeta": Question("noul", Text("Z?")), "q:alpha": Question("noul", Text("A?"))},
        needs=["t:state.message"],
        oracle="model",
    )

    async def both(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({"q:route": YES, "q:again": YES, "q:zeta": YES, "q:alpha": YES})

    ld = load(design(sources={**SOURCES, "pair": pair}))
    bindings = {"default": confirming(YES), "model": both, "tools": tools}
    v = transcript(decide_sync(ld, STATE, bindings).record)
    assert v is not None
    (call,) = [c for c in calls(v) if c["source"] == "pair"]
    assert call["asks"] == ["q:alpha", "q:zeta"]
    assert list(call["answers"]) == ["q:alpha", "q:zeta"]


def test_a_call_message_mixing_ommat_and_other_ids_is_not_bookkeeping() -> None:
    ours = to_chat_completions(view("unsure"))[0]["tool_calls"][0]
    other = {"id": "call_abc", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    assert not is_bookkeeping({"role": "assistant", "content": None, "tool_calls": [ours, other]})


@pytest.mark.parametrize("suffix", ["c0", "c", "c01", "s1", "e"])
def test_only_a_source_call_s_or_decide_s_id_is_bookkeeping(suffix: str) -> None:
    assert not is_bookkeeping(
        {"role": "tool", "tool_call_id": f"om_{'a' * 24}_{suffix}", "content": "{}"}
    )


def test_records_whose_stems_differ_only_in_their_last_digit_are_two_records() -> None:
    def record(stem: str) -> list[dict[str, Any]]:
        call = {
            "id": f"{stem}_d",
            "type": "function",
            "function": {"name": "decide", "arguments": "{}"},
        }
        return [
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": f"{stem}_d", "content": "{}"},
        ]

    earlier, latest = record("om_" + "a" * 23 + "b"), record("om_" + "a" * 23 + "c")
    assert drop_earlier_bookkeeping(earlier + latest) == latest


@pytest.mark.parametrize(
    ("turn", "message"),
    [
        (2**53, "magnitude"),
        (-(2**53), "magnitude"),
        ("\ud800", "surrogate"),
        (True, "a str or an int"),
    ],
)
def test_a_turn_the_canonical_encoding_refuses_raises_even_with_no_view(
    turn: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        to_chat_completions(None, turn=turn)
    with pytest.raises(ValueError, match=message):
        to_chat_completions(view("clear"), turn=turn)


def test_the_largest_turn_the_encoding_takes_renders() -> None:
    assert to_chat_completions(view("clear"), turn=2**53 - 1)
