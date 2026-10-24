"""The transcript view and the Chat Completions renderer, over the four paths of
``transcript_turns``: clear, guard-refused, unsure then clarify, and the oracle down."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from ommat import (
    CallResult,
    Context,
    ContextEntry,
    Plan,
    Policy,
    Record,
    Requirement,
    Source,
    Text,
    decide_sync,
    handoff,
    load,
    read_record,
    replay_sync,
    transcript,
)
from ommat.conversation import (
    drop_earlier_bookkeeping,
    handoff_text,
    is_bookkeeping,
    to_chat_completions,
)
from transcript_turns import PATHS, SOURCES, STATE, YES, confirming, design, model, turn

ID = re.compile(r"om_[0-9a-f]{24}_(c[1-9][0-9]*|d|s[1-9][0-9]*)")


def view(path: str, **kw: Any) -> dict[str, Any]:
    out = transcript(turn(path, **kw).record)
    assert out is not None
    return out


def calls(v: dict[str, Any]) -> list[dict[str, Any]]:
    return [call for calls in v["rounds"] for call in calls]


def ids(v: dict[str, Any]) -> list[str]:
    return [c["id"] for c in calls(v)] + [v["decide"]["id"]] + [s["id"] for s in v["pending"]]


# ---------------------------------------------------------------------------------------------
# The view


def test_the_view_gives_every_call_and_the_conclusion_in_order() -> None:
    decision = turn("clear")
    record = decision.record
    v = transcript(record)
    assert v is not None
    stem = "om_" + record.record_hash.removeprefix("sha256:")[:24]
    assert list(v) == ["record_hash", "design", "rounds", "decide", "pending"]
    assert v["record_hash"] == record.record_hash
    assert v["design"] == {"name": "mail", "hash": record.design_hash}
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
    sure = [{"ref": "d:sure", "value": "yes"}]
    assert v["decide"] == {
        "id": f"{stem}_d",
        "board": {"d:route": {"value": "yes"}, "d:sure": {"value": "yes"}},
        "plan": {
            "kind": "execute",
            "reason": "the user asked to archive",
            "reply": None,
            "cause": None,
            "steps": [
                {
                    "tool": "archive",
                    "arguments": {
                        "folder": {"value": "old"},
                        "order": {"ref": "t:order.id", "value": "o-1"},
                    },
                    "requirements": sure,
                },
                {
                    "tool": "notify",
                    "arguments": {"receipt": {"value": "done"}},
                    "requirements": sure,
                },
            ],
        },
        "verdict": {"kind": "passed", "failures": []},
        "context": [{"marker": None, "ref": "d:route", "text": "Archive asked: yes"}],
    }
    assert v["pending"] == [
        {"id": f"{stem}_s1", "tool": "archive", "arguments": {"folder": "old", "order": "o-1"}},
        {"id": f"{stem}_s2", "tool": "notify", "arguments": {"receipt": "done"}},
    ]


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


def test_a_refused_verdict_is_given_as_recorded_and_nothing_is_pending() -> None:
    decision = turn("refused")
    v = transcript(decision.record)
    assert v is not None
    recorded = next(i for i in decision.record.log if i["item"] == "verdict")
    assert v["decide"]["verdict"] == {"kind": "refused", "failures": recorded["failures"]}
    assert v["decide"]["verdict"]["failures"]
    assert len(v["decide"]["plan"]["steps"]) == 2
    assert v["pending"] == []


def test_an_unsure_estimate_is_a_tagged_marker_and_the_plan_clarifies() -> None:
    v = view("unsure")
    assert v["decide"]["board"]["d:sure"] == {
        "marker": {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["q:confirm"]}}
    }
    assert v["decide"]["plan"]["kind"] == "clarify"
    assert v["decide"]["plan"]["reply"] == "Shall I archive them?"
    assert v["pending"] == []


def test_a_failed_call_keeps_its_outcome_and_a_source_never_called_is_in_no_round() -> None:
    v = view("down")
    failed = {"failed": {"code": "raised", "refs": ["q:again", "q:route"]}}
    (first,) = v["rounds"]
    assert [c["outcome"] for c in first] == [failed, failed, "returned"]
    assert first[1]["answers"] == {
        "q:route": {"marker": {"kind": "Unavailable", "cause": failed["failed"]}}
    }
    assert "confirm" not in [c["source"] for c in calls(v)]


def test_a_plan_with_a_binding_shows_it_tagged_and_leaves_nothing_pending() -> None:
    v = view("clear", bound=True)
    assert v["decide"]["plan"]["steps"][1]["arguments"] == {
        "receipt": {"step": 1, "path": "receipt"}
    }
    assert v["decide"]["verdict"]["kind"] == "passed"
    assert v["pending"] == []


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


def test_with_no_block_the_name_is_null_and_there_is_no_context() -> None:
    record = turn("clear", load(design(context=None))).record
    v = transcript(record)
    assert v is not None
    assert v["design"] == {"name": None, "hash": record.design_hash}
    assert "context" not in v["decide"]


def test_an_empty_block_gives_an_empty_context() -> None:
    v = transcript(turn("clear", load(design(context=Context([])))).record)
    assert v is not None
    assert v["decide"]["context"] == []


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
    replayed, findings = replay_sync(record, load(design(**_changes(path))))
    assert findings == []
    assert replayed is not None
    assert replayed.record_hash == record.record_hash
    assert transcript(replayed) == v


def _changes(path: str) -> dict[str, Any]:
    return {"requirements": [Requirement("d:route", "no")]} if path == "refused" else {}


@pytest.mark.parametrize("fmt", [1, 2, 3, 4])
def test_a_record_before_format_5_gives_no_name_and_no_context(fmt: int) -> None:
    data = (Path(__file__).parent / f"format_{fmt}" / "record.json").read_bytes()
    record = read_record(data)
    v = transcript(record)
    assert v is not None
    assert v["design"]["name"] is None
    assert "context" not in v["decide"]
    assert all(ID.fullmatch(i) for i in ids(v))


def test_a_record_with_no_plan_item_gives_none() -> None:
    def cancel(r: Any) -> Plan:
        raise asyncio.CancelledError

    ld = load(design(policy=Policy(cancel, version="1", reads=["d:sure"])))
    with pytest.raises(asyncio.CancelledError) as caught:
        turn("clear", ld)
    record = cast("Record", getattr(caught.value, "record"))  # noqa: B009
    assert not any(i["item"] == "plan" for i in record.log)
    assert transcript(record) is None


def test_the_view_refuses_what_is_not_a_record() -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        transcript(turn("clear"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The Chat Completions renderer


def test_chat_completions_gives_each_round_decide_and_the_pending_calls() -> None:
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
        ("assistant", 2),
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
    decide_call = messages[6]["tool_calls"][0]
    assert decide_call["id"] == v["decide"]["id"]
    assert decide_call["function"]["name"] == "decide"
    assert json.loads(decide_call["function"]["arguments"]) == {"design": v["design"]}
    result = json.loads(messages[7]["content"])
    assert result == {k: x for k, x in v["decide"].items() if k != "id"}
    pending = messages[8]["tool_calls"]
    assert [(c["id"], c["function"]["name"]) for c in pending] == [
        (s["id"], s["tool"]) for s in v["pending"]
    ]
    assert pending[0]["function"]["arguments"] == '{"folder":"old","order":"o-1"}'


@pytest.mark.parametrize("path", PATHS)
def test_chat_completions_renders_every_path_and_ends_after_decide_with_nothing_pending(
    path: str,
) -> None:
    v = view(path)
    messages = to_chat_completions(v)
    calls_made = [c for m in messages for c in m.get("tool_calls", [])]
    results = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert [c["id"] for c in calls_made] == ids(v)
    if path == "clear":
        assert messages[-1]["role"] == "assistant"
    else:
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
def test_chat_completions_bytes_are_stable_and_canonical(path: str) -> None:
    record = turn(path).record
    once = to_chat_completions(transcript(record))
    again = to_chat_completions(transcript(read_record(record.document)))
    assert json.dumps(once) == json.dumps(again)
    for m in once:
        strings = [m.get("content")] + [c["function"]["arguments"] for c in m.get("tool_calls", [])]
        for s in strings:
            if s is not None:
                assert s == json.dumps(
                    json.loads(s), sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )


def test_none_renders_as_no_messages() -> None:
    assert to_chat_completions(None) == []


def _renamed(
    v: dict[str, Any], *, source: str | None = None, tool: str | None = None
) -> dict[str, Any]:
    v = json.loads(json.dumps(v))
    if source is not None:
        v["rounds"][0][0]["source"] = source
    if tool is not None:
        v["pending"][0]["tool"] = tool
    return v


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source": "decide"}, "the name decide's call takes"),
        ({"source": "a/" + "b" * 63}, "over 64 characters"),
        ({"tool": "decide"}, "the name decide's call takes"),
        ({"tool": "mail/send"}, "not a tool name a provider takes"),
        ({"tool": "x" * 65}, "not a tool name a provider takes"),
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


def test_chat_bookkeeping_is_the_source_calls_and_decide_and_never_the_pending_calls() -> None:
    messages = to_chat_completions(view("clear"))
    assert [is_bookkeeping(m) for m in messages] == [True] * 8 + [False]


def test_a_message_of_the_caller_s_or_the_model_s_is_not_bookkeeping() -> None:
    assert not is_bookkeeping({"role": "user", "content": "hi"})
    assert not is_bookkeeping({"role": "assistant", "content": "ok"})
    assert not is_bookkeeping({"role": "tool", "tool_call_id": "call_1", "content": "{}"})
    step = to_chat_completions(view("clear"))[-1]["tool_calls"][0]
    mixed = {"role": "assistant", "content": None, "tool_calls": [step]}
    assert not is_bookkeeping(mixed)


def _chat_paired(messages: list[dict[str, Any]]) -> bool:
    called = [c["id"] for m in messages for c in m.get("tool_calls", [])]
    return sorted(called) == sorted(m["tool_call_id"] for m in messages if m["role"] == "tool")


def _first_turn() -> list[dict[str, Any]]:
    first: list[dict[str, Any]] = [{"role": "user", "content": "Archive the newsletters"}]
    first += to_chat_completions(view("clear"))
    step_ids = [c["id"] for c in first[-1]["tool_calls"]]
    first += [{"role": "tool", "tool_call_id": i, "content": "{}"} for i in step_ids]
    first += [{"role": "assistant", "content": "Archived."}]
    return first


def test_dropping_earlier_bookkeeping_over_two_turns_keeps_calls_paired() -> None:
    first = _first_turn()
    second = [*first, {"role": "user", "content": "Are you sure?"}]
    current = to_chat_completions(view("unsure"))
    kept = drop_earlier_bookkeeping(second + current)
    assert kept == [first[0], *first[-4:], second[-1], *current]
    assert _chat_paired(kept)


def test_chat_keeps_the_latest_record_wherever_the_prompt_stands() -> None:
    current = to_chat_completions(view("unsure"))
    late = [*current, {"role": "user", "content": "late"}]
    assert drop_earlier_bookkeeping(late) == late


def test_chat_takes_empty_content_as_none_and_keeps_a_result_with_a_kept_call() -> None:
    first = _first_turn()
    first[1] = {**first[1], "content": ""}
    assert is_bookkeeping(first[1])
    first[5] = {**first[5], "content": "thinking"}  # decide's call, no longer bookkeeping
    current = to_chat_completions(view("unsure"))
    kept = drop_earlier_bookkeeping([*first, *current])
    assert first[5] in kept
    assert first[6] in kept  # its result stays with it
    assert _chat_paired(kept)


# ---------------------------------------------------------------------------------------------
# Round-one review cases


def _with_step_over(record: Record, ref: str) -> Record:
    """The record with a step whose argument names ``ref``, rehashed, as no decide writes."""
    from ledger_logs import rehashed

    doc = json.loads(record.document)
    plan = next(i for i in doc["log"] if i["item"] == "plan")
    plan["steps"] = [{"tool": "archive", "arguments": {"order": {"ref": ref}}, "requirements": []}]
    return read_record(json.dumps(rehashed(doc)).encode())


def test_a_step_argument_with_no_value_in_the_log_raises_in_both_views() -> None:
    edited = _with_step_over(turn("unsure").record, "d:sure")  # a marker, no value
    with pytest.raises(ValueError, match="holds no value in the log"):
        transcript(edited)
    with pytest.raises(ValueError, match="holds no value in the log"):
        handoff(edited)


def test_the_flagged_ref_s_marker_wins_over_an_earlier_slot_s() -> None:
    entry = ContextEntry(
        "d:sure",
        Text(
            "{r} then {s}",
            [
                {"name": "r", "origin": {"ref": "d:route"}},
                {"name": "s", "origin": {"ref": "d:sure"}},
            ],
        ),
    )
    v = transcript(turn("down", load(design(context=Context([entry])))).record)
    assert v is not None
    (got,) = v["decide"]["context"]
    assert got["marker"]["cause"]["code"] == "need-unavailable"  # d:sure's, not d:route's


def test_handoff_text_writes_an_unavailable_marker_as_recorded() -> None:
    lines = handoff_text(handoff(turn("down").record)).splitlines()
    assert "d:route: Unavailable (raised)" in lines


def test_handoff_text_names_the_design_by_hash_alone_with_no_name() -> None:
    record = turn("clear").record
    view = handoff(record)
    assert view is not None
    view["design"]["name"] = None
    assert f"Design: {record.design_hash}" in handoff_text(view).splitlines()


@pytest.mark.parametrize("fmt", [1, 2, 3, 4])
def test_a_record_before_format_5_hands_off_nothing(fmt: int) -> None:
    data = (Path(__file__).parent / f"format_{fmt}" / "record.json").read_bytes()
    assert handoff(read_record(data)) is None


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
