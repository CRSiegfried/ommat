"""The turn history and the Chat Completions renderer, over the four paths of
``transcript_turns``: clear, hold, undecided on an unsure confirmation, and the oracle down."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

import extraction
from ommat import (
    CallResult,
    Question,
    Record,
    Rule,
    Source,
    Text,
    Unsure,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t
from ommat.bindings import fixed
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
LOADED = load(design())
REFS = ["d:act"]
"""The ref the caller acts on: the policy's."""


def rec(path: str) -> Record:
    return turn(path, LOADED).record


def chat(path: str, **kw: Any) -> list[dict[str, Any]]:
    return to_chat_completions(rec(path), LOADED, **kw, refs=REFS)


def call_ids(messages: list[dict[str, Any]]) -> list[str]:
    return [c["id"] for m in messages for c in m.get("tool_calls", [])]


def results(messages: list[dict[str, Any]]) -> list[Any]:
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


def decide_arguments(messages: list[dict[str, Any]]) -> Any:
    return json.loads(messages[-2]["tool_calls"][0]["function"]["arguments"])


# ---------------------------------------------------------------------------------------------
# The turn history


def test_each_round_s_calls_then_decide_with_empty_arguments_and_compact_answers() -> None:
    record = rec("clear")
    messages = to_chat_completions(record, LOADED, refs=REFS)
    stem = "om_" + record.record_hash.removeprefix("sha256:")[:24]
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
    assert messages[0]["content"] is None
    assert messages[0]["tool_calls"][1] == {
        "id": f"{stem}_c2",
        "type": "function",
        "function": {"name": "mail-router", "arguments": "{}"},
    }
    names = [c["function"]["name"] for m in messages for c in m.get("tool_calls", [])]
    assert names == ["ask-again", "mail-router", "lookup", "confirm", "decide"]
    assert call_ids(messages) == [f"{stem}_c{n}" for n in range(1, 5)] + [f"{stem}_d"]
    # A tie at the top gives the first tied label in declared order; a t: value is whole.
    assert results(messages) == [
        {"q:again": ["yes", 0.5]},
        {"q:route": ["yes", 0.95]},
        {"t:order": {"id": "o-1"}},
        {"q:confirm": ["yes", 0.95]},
        {"d:act": {"value": "archive"}},
    ]


def test_the_design_cuts_the_board_to_the_refs_reads_and_without_it_the_board_is_whole() -> None:
    record = rec("clear")
    assert decide_arguments(to_chat_completions(record, LOADED, refs=REFS)) == {"d:sure": "yes"}
    assert decide_arguments(to_chat_completions(record, refs=REFS)) == {
        "d:route": "yes",
        "d:sure": "yes",
    }


def test_the_board_keeps_every_marker_the_refs_do_not_read() -> None:
    assert decide_arguments(chat("down")) == {
        "d:route": {"marker": "Unavailable", "cause": "raised"},
        "d:sure": {"marker": "Unavailable", "cause": "need-unavailable"},
    }


def test_a_failed_call_s_result_is_each_ref_s_marker_kind_and_cause_code() -> None:
    first = results(chat("down"))[:3]
    assert first == [
        {"q:again": {"marker": "Unavailable", "cause": "raised"}},
        {"q:route": {"marker": "Unavailable", "cause": "raised"}},
        {"t:order": {"id": "o-1"}},
    ]
    assert "confirm" not in json.dumps(chat("down"))  # a source never called is in no round


def test_an_undecided_turn_ends_at_its_kind_root_cause_and_refs() -> None:
    assert results(chat("unsure"))[-1] == {
        "d:act": {"undecided": {"kind": "Unsure", "root_cause": "abstained", "refs": ["q:confirm"]}}
    }


def test_the_undecided_reason_follows_a_required_cause_to_its_root() -> None:
    def act(r: Any) -> str:
        r.read("t:order.id")
        return "archive" if r.require("d:sure") == "yes" else "hold"

    rule = Rule("d:act", t.label("archive", "hold"), act, version="1", reads=RULES["act"].reads)
    ld = load(design(rules={**RULES, "act": rule}))
    record = turn("unsure", ld).record
    policy = next(i for i in record.log if i["item"] == "entry" and i["ref"] == "d:act")
    assert policy["marker"]["cause"] == {"code": "required", "refs": ["d:sure"]}
    assert results(to_chat_completions(record, ld, refs=REFS))[-1] == {
        "d:act": {"undecided": {"kind": "Unsure", "root_cause": "abstained", "refs": ["q:confirm"]}}
    }


def test_a_decided_turn_whose_board_holds_a_marker_ends_at_its_decision() -> None:
    def act(r: Any) -> str:
        r.read("t:order.id")
        return "hold" if isinstance(r.read("d:sure"), Unsure) else "archive"

    rule = Rule("d:act", t.label("archive", "hold"), act, version="1", reads=RULES["act"].reads)
    ld = load(design(rules={**RULES, "act": rule}))
    messages = to_chat_completions(turn("unsure", ld).record, ld, refs=REFS)
    assert decide_arguments(messages) == {"d:sure": {"marker": "Unsure", "cause": "abstained"}}
    assert results(messages)[-1] == {"d:act": {"value": "hold"}}


def test_an_extraction_answer_is_its_value_and_confidence() -> None:
    ld = extraction.loaded()
    answers = fixed({"q:amount": extraction.AMOUNT, "q:dates": extraction.DATES})
    record = decide_sync(ld, extraction.STATE, {"default": answers}).record
    assert results(to_chat_completions(record, ld, refs=["d:reply"]))[0] == {
        "q:amount": [120, 0.9],
        "q:dates": [["2024-02-29"], 1],
    }


def test_an_extraction_with_no_value_gives_null_and_leaves_its_guess_out() -> None:
    ld = extraction.loaded()
    none = {"value": None, "guess": 120, "p_null": 0.8, "confidence": None, "span": None}
    answers = fixed({"q:amount": none, "q:dates": extraction.DATES})
    record = decide_sync(ld, extraction.STATE, {"default": answers}).record
    assert results(to_chat_completions(record, ld, refs=["d:reply"]))[0]["q:amount"] == [None, None]


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
    messages = to_chat_completions(decide_sync(ld, STATE, bindings).record, ld, refs=REFS)
    assert results(messages)[2] == {"t:order": {"id": "o-1", "marker": "m"}}


def test_the_history_holds_no_state_provenance_prompt_or_derivation() -> None:
    text = json.dumps(chat("clear"))
    for absent in ("t:state", "Archive the newsletters", "acct-7", "derivation", "identity"):
        assert absent not in text
    assert "Archive this?" not in text  # no question wording


def test_a_source_asking_two_refs_gives_them_in_code_point_order() -> None:
    pair = Source(
        {"q:zeta": Question("noul", Text("Z?")), "q:alpha": Question("noul", Text("A?"))},
        needs=["t:state.message"],
        oracle="model",
    )

    async def both(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({"q:route": YES, "q:again": YES, "q:zeta": YES, "q:alpha": YES})

    ld = load(design(sources={**SOURCES, "pair": pair}))
    bindings = {"default": confirming(YES), "model": both, "tools": tools}
    messages = to_chat_completions(decide_sync(ld, STATE, bindings).record, ld, refs=REFS)
    names = [c["function"]["name"] for c in messages[0]["tool_calls"]]
    assert list(results(messages)[names.index("pair")]) == ["q:alpha", "q:zeta"]


@pytest.mark.parametrize("path", PATHS)
def test_ids_fit_the_pattern_are_unique_and_one_log_gives_the_same(path: str) -> None:
    record = rec(path)
    messages = to_chat_completions(record, LOADED, refs=REFS)
    got = call_ids(messages)
    assert all(ID.fullmatch(i) and len(i) <= 40 for i in got)
    assert len(set(got)) == len(got)
    assert to_chat_completions(read_record(record.document), LOADED, refs=REFS) == messages
    replayed, findings = replay_sync(record, LOADED)
    assert findings == []
    assert replayed is not None
    assert to_chat_completions(replayed, LOADED, refs=REFS) == messages


@pytest.mark.parametrize("path", PATHS)
def test_every_path_ends_at_decide_s_result_each_call_answered_in_order(path: str) -> None:
    messages = chat(path)
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == call_ids(messages)[-1]
    answered = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert answered == call_ids(messages)


@pytest.mark.parametrize("path", PATHS)
def test_the_bytes_are_stable_and_in_the_ordered_encoding(path: str) -> None:
    record = rec(path)
    once = to_chat_completions(record, LOADED, refs=REFS)
    again = to_chat_completions(read_record(record.document), LOADED, refs=REFS)
    assert json.dumps(once) == json.dumps(again)
    for m in once:
        strings = [m.get("content")] + [c["function"]["arguments"] for c in m.get("tool_calls", [])]
        for s in strings:
            if s is not None:
                assert s == json.dumps(json.loads(s), separators=(",", ":"), ensure_ascii=False)


def test_the_default_rendering_of_one_record_is_pinned() -> None:
    """A change to the history's form changes these bytes, so it is made on purpose."""
    got = [
        (m["role"], m.get("content"), [c["function"] for c in m.get("tool_calls", [])])
        for m in chat("unsure")
    ]
    assert got == [
        (
            "assistant",
            None,
            [
                {"name": "ask-again", "arguments": "{}"},
                {"name": "mail-router", "arguments": "{}"},
                {"name": "lookup", "arguments": "{}"},
            ],
        ),
        ("tool", '{"q:again":["yes",0.5]}', []),
        ("tool", '{"q:route":["yes",0.95]}', []),
        ("tool", '{"t:order":{"id":"o-1"}}', []),
        ("assistant", None, [{"name": "confirm", "arguments": "{}"}]),
        ("tool", '{"q:confirm":["yes",0.5]}', []),
        (
            "assistant",
            None,
            [{"name": "decide", "arguments": '{"d:sure":{"marker":"Unsure","cause":"abstained"}}'}],
        ),
        (
            "tool",
            '{"d:act":{"undecided":{"kind":"Unsure","root_cause":"abstained",'
            '"refs":["q:confirm"]}}}',
            [],
        ),
    ]


# ---------------------------------------------------------------------------------------------
# What a renderer refuses, and a record with no history


def test_decide_s_result_is_each_named_ref_in_order_and_the_board_leaves_them_out() -> None:
    messages = to_chat_completions(rec("clear"), LOADED, refs=["d:route", "d:act"])
    assert messages[-1]["content"] == '{"d:route":{"value":"yes"},"d:act":{"value":"archive"}}'
    assert decide_arguments(messages) == {"d:sure": "yes"}  # the act rule's read
    whole = to_chat_completions(rec("clear"), refs=["d:route", "d:act"])
    assert decide_arguments(whole) == {"d:sure": "yes"}  # every d: entry but the named


def test_the_design_cuts_the_board_to_the_union_of_the_refs_reads_and_every_marker() -> None:
    messages = to_chat_completions(rec("unsure"), LOADED, refs=["d:sure"])
    assert results(messages)[-1] == {
        "d:sure": {
            "undecided": {"kind": "Unsure", "root_cause": "abstained", "refs": ["q:confirm"]}
        }
    }
    assert decide_arguments(messages) == {"d:act": {"marker": "Unsure", "cause": "abstained"}}
    assert decide_arguments(to_chat_completions(rec("clear"), LOADED, refs=["d:sure"])) == {}


@pytest.mark.parametrize(
    ("refs", "message"),
    [
        ("d:act", "a list or a tuple"),
        ([], "at least one"),
        (["q:route"], "d: refs"),
        ([1], "d: refs"),
        (["d:act.x"], "not a rule's ref"),
        (["d:act", "d:act"], "repeat"),
        (["d:nothing"], "no rule's ref in the record's design"),
    ],
)
def test_refs_that_are_not_distinct_rule_refs_are_refused(refs: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        to_chat_completions(rec("clear"), refs=refs)


def test_refs_are_required() -> None:
    with pytest.raises(TypeError, match="refs"):
        to_chat_completions(rec("clear"))  # type: ignore[call-arg]


OLD_RECORDS = {
    **{fmt: f"format_{fmt}/record.json" for fmt in range(1, 7)},
    7: "format_7_record/turn.record.json",
}
"""A stored record of each record format before 8, by its format."""


@pytest.mark.parametrize("fmt", sorted(OLD_RECORDS))
def test_a_record_before_format_8_renders_no_messages(fmt: int) -> None:
    record = read_record((Path(__file__).parent / OLD_RECORDS[fmt]).read_bytes())
    assert record.log[0].get("format", 1) == fmt
    assert to_chat_completions(record, refs=REFS) == []


def test_a_record_with_no_entry_at_a_named_ref_renders_no_messages() -> None:
    def cancel(r: Any) -> str:
        raise asyncio.CancelledError

    rule = Rule("d:act", t.label("archive", "hold"), cancel, version="1", reads=["d:sure"])
    ld = load(design(rules={**RULES, "act": rule}))
    with pytest.raises(asyncio.CancelledError) as caught:
        turn("clear", ld)
    record = cast("Record", getattr(caught.value, "record"))  # noqa: B009
    assert to_chat_completions(record, ld, refs=REFS) == []


@pytest.mark.parametrize("record", [None, {"log": []}, "record"])
def test_what_is_not_a_record_is_refused(record: Any) -> None:
    with pytest.raises(ValueError, match="a renderer takes a Record"):
        to_chat_completions(record, refs=REFS)


def test_a_design_that_is_not_a_loaded_design_is_refused() -> None:
    with pytest.raises(ValueError, match="design must be a LoadedDesign or None"):
        to_chat_completions(rec("clear"), design(), refs=REFS)  # type: ignore[arg-type]


def test_another_design_s_loaded_design_is_refused() -> None:
    other = load(design(name="other"))
    with pytest.raises(ValueError, match="is not the record's"):
        to_chat_completions(rec("clear"), other, refs=REFS)


def _named(name: str) -> list[dict[str, Any]]:
    source = Source(
        {"q:extra": Question("noul", Text("Extra?"))}, needs=["t:state.message"], oracle="model"
    )

    async def answers(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({"q:route": YES, "q:again": YES, "q:extra": YES})

    ld = load(design(sources={**SOURCES, name: source}))
    bindings = {"default": confirming(YES), "model": answers, "tools": tools}
    return to_chat_completions(decide_sync(ld, STATE, bindings).record, ld, refs=REFS)


@pytest.mark.parametrize(
    ("name", "message"),
    [("decide", "the name decide's call takes"), ("a/" + "b" * 63, "over 64 characters")],
)
def test_a_tool_name_no_provider_takes_is_refused(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _named(name)


def test_a_source_s_slash_becomes_a_hyphen_and_a_long_name_within_64_passes() -> None:
    names = [c["function"]["name"] for c in _named("a/" + "b" * 62)[0]["tool_calls"]]
    assert "a-" + "b" * 62 in names


# ---------------------------------------------------------------------------------------------
# Chat Completions bookkeeping


def test_chat_bookkeeping_is_the_source_calls_and_decide() -> None:
    assert [is_bookkeeping(m) for m in chat("clear")] == [True] * 8


def test_a_message_of_the_caller_s_or_the_model_s_is_not_bookkeeping() -> None:
    assert not is_bookkeeping({"role": "user", "content": "hi"})
    assert not is_bookkeeping({"role": "assistant", "content": "ok"})
    assert not is_bookkeeping({"role": "tool", "tool_call_id": "call_1", "content": "{}"})
    own = {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    assert not is_bookkeeping({"role": "assistant", "content": None, "tool_calls": [own]})


def _chat_paired(messages: list[dict[str, Any]]) -> bool:
    called = call_ids(messages)
    return sorted(called) == sorted(m["tool_call_id"] for m in messages if m["role"] == "tool")


def _first_turn() -> list[dict[str, Any]]:
    first: list[dict[str, Any]] = [{"role": "user", "content": "Archive the newsletters"}]
    first += chat("clear")
    first += [{"role": "assistant", "content": "Archived."}]
    return first


def test_dropping_earlier_bookkeeping_over_two_turns_keeps_calls_paired() -> None:
    first = _first_turn()
    second = [*first, {"role": "user", "content": "Are you sure?"}]
    current = chat("unsure")
    kept = drop_earlier_bookkeeping(second + current)
    assert kept == [first[0], first[-1], second[-1], *current]
    assert _chat_paired(kept)


def test_chat_keeps_the_latest_record_wherever_the_prompt_stands() -> None:
    late = [*chat("unsure"), {"role": "user", "content": "late"}]
    assert drop_earlier_bookkeeping(late) == late


def test_chat_takes_empty_content_as_none_and_keeps_a_result_with_a_kept_call() -> None:
    first = _first_turn()
    first[1] = {**first[1], "content": ""}
    assert is_bookkeeping(first[1])
    first[5] = {**first[5], "content": "thinking"}  # a call with text, not bookkeeping
    kept = drop_earlier_bookkeeping([*first, *chat("unsure")])
    assert first[5] in kept
    assert first[6] in kept  # its result stays with it
    assert _chat_paired(kept)


def test_chat_filters_ignore_odd_members() -> None:
    odd: list[dict[str, Any]] = [
        {"role": "assistant", "content": None, "tool_calls": "x"},
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": ["om_x"], "content": "{}"},
    ]
    assert not any(is_bookkeeping(m) for m in odd)
    kept = drop_earlier_bookkeeping([*chat("down"), *odd, *chat("unsure")])
    assert kept[: len(odd)] == odd
    plain = [{"role": "user", "content": "hi"}]
    assert drop_earlier_bookkeeping(plain) == plain
    assert drop_earlier_bookkeeping(plain) is not plain


def test_a_call_message_mixing_ommat_and_other_ids_is_not_bookkeeping() -> None:
    ours = chat("unsure")[0]["tool_calls"][0]
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


# ---------------------------------------------------------------------------------------------
# A record rendered for more than one turn


def test_a_turn_stems_every_id_within_the_pattern_and_none_keeps_the_record_s() -> None:
    record = rec("clear")
    plain = to_chat_completions(record, LOADED, refs=REFS)
    assert plain == to_chat_completions(record, LOADED, turn=None, refs=REFS)
    turned = to_chat_completions(record, LOADED, turn="retry-1", refs=REFS)
    old, new = call_ids(plain), call_ids(turned)
    assert all(ID.fullmatch(i) and len(i) <= 40 for i in new)
    assert [i[27:] for i in new] == [i[27:] for i in old]
    assert set(new).isdisjoint(old)
    assert to_chat_completions(record, LOADED, turn="retry-1", refs=REFS) == turned
    assert to_chat_completions(record, turn=1, refs=REFS) != to_chat_completions(
        record, turn="1", refs=REFS
    )


@pytest.mark.parametrize("turn", [True, 1.5, b"x", ["a"], None.__class__])
def test_a_turn_of_another_type_is_refused(turn: Any) -> None:
    with pytest.raises(ValueError, match="turn must be a str or an int"):
        to_chat_completions(rec("clear"), turn=turn, refs=REFS)


def test_five_turns_repeating_one_record_repeat_no_id_and_keep_only_the_last() -> None:
    history: list[dict[str, Any]] = []
    for n, path in enumerate(["clear", "unsure", "clear", "down", "clear"]):
        history.append({"role": "user", "content": f"turn {n}"})
        history += chat(path, turn=n)
        history.append({"role": "assistant", "content": "ok"})
    every = call_ids(history)
    assert len(every) == len(set(every))
    kept = drop_earlier_bookkeeping(history)
    assert _chat_paired(kept)
    stems = {c["id"][:27] for m in kept if is_bookkeeping(m) for c in m.get("tool_calls", [])}
    assert len(stems) == 1


@pytest.mark.parametrize(
    ("turn", "message"),
    [
        (2**53, "magnitude"),
        (-(2**53), "magnitude"),
        ("\ud800", "surrogate"),
        (True, "a str or an int"),
    ],
)
def test_a_turn_the_canonical_encoding_refuses_raises_before_the_record_is_read(
    turn: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        to_chat_completions(None, turn=turn, refs=REFS)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        to_chat_completions(rec("clear"), turn=turn, refs=REFS)


def test_the_largest_turn_the_encoding_takes_renders() -> None:
    assert to_chat_completions(rec("clear"), turn=2**53 - 1, refs=REFS)
