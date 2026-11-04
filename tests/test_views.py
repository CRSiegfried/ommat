# pyright: reportPrivateUsage=false
"""The record views: the escalation's four members, the root cause its ``required`` chain
ends at, and when a record holds none; the transcript of a decided turn; and
the redacted view, its digests recomputed here independently with ``hmac`` and the
``rfc8785`` oracle, over records decide wrote and over the synthetic logs and every prefix of
them."""

import copy
import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

import pytest
import rfc8785

from designs import minimal
from ledger_logs import (
    BROKEN_8,
    CANCEL,
    CANCELLED_8,
    DESIGN,
    ENTRY_CAUSE,
    LOGS,
    NORMAL,
    NORMAL_8,
    SETTINGS,
    STATE,
    call,
    entry,
    fresh,
    rehashed,
    round_item,
    unhashed_of,
)
from ommat import (
    Cause,
    CauseCode,
    InvalidRecord,
    LoadedDesign,
    Record,
    Rule,
    RunSettings,
    Unsure,
    decide_sync,
    escalation,
    load,
    read_record,
    redact,
    transcript,
)
from ommat import types as t
from ommat._canonical import decode, encode
from ommat._refs import grounded_ref
from ommat._values import MAX_DOCUMENT_DEPTH

ROOT = bytes(range(32))


def key_of(record_hash: str, root: bytes = ROOT) -> bytes:
    return hmac.digest(root, record_hash.encode("ascii") + b"\x01", "sha256")


def digest(value: Any, record: Record, root: bytes = ROOT) -> str:
    """A keyed digest computed apart from the code under test."""
    mac = hmac.new(key_of(record.record_hash, root), rfc8785.dumps(value), hashlib.sha256)
    return "hmac-sha256:" + mac.hexdigest()


def read(log: list[dict[str, Any]], unhashed: dict[str, Any] | None = None) -> Record:
    """The record of ``log``, which may share a marker between two entries, as a document
    holding no list or object at two places."""
    unshared = json.loads(json.dumps(log))
    return read_record(rehashed({"log": unshared, "unhashed": unhashed or {}}))


def view(record: Record, root: bytes = ROOT) -> dict[str, Any]:
    out = redact(record, root)
    assert encode(decode(out), MAX_DOCUMENT_DEPTH) == out  # canonical bytes
    return decode(out)


# ---------------------------------------------------------------------------------------------
# Turns decide wrote: the policy ``d:act`` reads ``d:mid``, which reads ``d:gone``


def send(r: Any) -> str:
    return "send"


def requiring(r: Any) -> str:
    r.require("d:mid")
    return "send"


def flag(r: Any) -> bool:
    return False


def gone(r: Any) -> Any:
    raise RuntimeError("no value")


def mid(r: Any) -> bool:
    r.require("d:gone")
    return True


def loaded_with(policy: Callable[[Any], str], **changes: Any) -> LoadedDesign:
    fields: dict[str, Any] = {
        "state": t.record(subject=t.string),
        "rules": {
            "flag": Rule("d:flag", t.boolean, flag, version="1"),
            "gone": Rule("d:gone", t.boolean, gone, version="1"),
            "mid": Rule("d:mid", t.boolean, mid, version="1", reads=["d:gone"]),
            "act": Rule(
                "d:act", t.label("send", "hold"), policy, version="1", reads=["d:flag", "d:mid"]
            ),
        },
        "policy": "d:act",
    }
    fields.update(changes)
    return load(minimal(**fields))


def turn(policy: Callable[[Any], str], **changes: Any) -> Record:
    return decide_sync(loaded_with(policy, **changes), {"subject": "hi"}, {}).record


# ---------------------------------------------------------------------------------------------
# The escalation


def test_a_value_at_the_policys_ref_holds_no_escalation() -> None:
    assert escalation(turn(send)) is None


def test_the_root_cause_follows_required_across_two_hops() -> None:
    record = turn(requiring)
    found = escalation(record)
    assert found == {
        "kind": "Unavailable",
        "cause": {"code": "required", "refs": ["d:mid"]},
        "root_cause": {"code": "raised", "refs": ["gone"]},  # d:act -> d:mid -> d:gone
        "record_hash": record.record_hash,
    }


def test_a_cause_not_required_is_its_own_root() -> None:
    def raising(r: Any) -> str:
        raise RuntimeError("the policy failed")

    record = turn(raising)
    cause = {"code": "raised", "refs": ["act"]}
    assert escalation(record) == {
        "kind": "Unavailable",
        "cause": cause,
        "root_cause": cause,
        "record_hash": record.record_hash,
    }


def test_a_broken_entry_escalates_with_its_entry_cause() -> None:
    record = decide_sync(loaded_with(send), {}, {}).record  # the state does not conform
    found = escalation(record)
    assert found is not None
    assert found["kind"] == "Unavailable"
    assert found["cause"] == found["root_cause"]
    assert found["cause"]["code"] == "invalid-return"


def test_the_root_cause_stops_at_a_ref_with_no_marker_entry_and_at_a_loop() -> None:
    log = fresh(NORMAL_8)
    required = {"kind": "Unsure", "cause": {"code": "required", "refs": ["d:tags.a"]}}
    log[9] = entry("d:route", 2, None, marker=required)  # d:tags holds a value
    assert escalation(read(log))["root_cause"] == required["cause"]  # type: ignore[index]
    looped = {"kind": "Unsure", "cause": {"code": "required", "refs": ["d:route"]}}
    log[9] = entry("d:route", 2, None, marker=looped)
    assert escalation(read(log))["root_cause"] == looped["cause"]  # type: ignore[index]
    log[9] = entry("d:route", 2, None, marker=required)
    log[10] = entry("d:tags", 2, None, marker=looped)  # d:route -> d:tags -> d:route
    assert escalation(read(log))["root_cause"] == required["cause"]  # type: ignore[index]


def whole(r: Any) -> Any:
    raise RuntimeError("no record")


def requiring_a_field(r: Any) -> str:
    r.require("d:whole.f")
    return "send"


def test_the_root_cause_hops_from_a_field_path_to_the_marker_at_its_entry() -> None:
    rules = {
        "whole": Rule("d:whole", t.record(f=t.boolean), whole, version="1"),
        "act": Rule(
            "d:act", t.label("send", "hold"), requiring_a_field, version="1", reads=["d:whole.f"]
        ),
    }
    record = turn(requiring_a_field, rules=rules)
    found = escalation(record)
    assert found is not None
    assert found["cause"] == {"code": "required", "refs": ["d:whole.f"]}
    assert found["root_cause"] == {"code": "raised", "refs": ["whole"]}


AWKWARD = 'A"b'
"""A candidate whose grounded ref escapes a character, as RFC 8785 serializes it."""


def names(r: Any) -> Any:
    return r.read("t:state.names")


def fit_or_raise(r: Any) -> bool:
    if r.read("candidate") == AWKWARD:
        raise RuntimeError("no fit")
    return True


def returning_a_grounded_cause(r: Any) -> Any:
    r.read("d:fit")
    return Unsure(Cause(CauseCode.required, [grounded_ref("d:fit", AWKWARD)]))


def test_the_root_cause_hops_from_a_grounded_ref_to_the_entry_the_ledger_wrote() -> None:
    loaded = load(
        minimal(
            state=t.record(names=t.list_(t.string)),
            rules={
                "people": Rule(
                    "d:people",
                    t.list_(t.string),
                    names,
                    version="1",
                    reads=["t:state.names"],
                    candidates=True,
                ),
                "fit": Rule(
                    "d:fit",
                    t.boolean,
                    fit_or_raise,
                    version="1",
                    reads=["candidate"],
                    grounded="d:people",
                ),
                "act": Rule(
                    "d:act",
                    t.label("send", "hold"),
                    returning_a_grounded_cause,
                    version="1",
                    reads=["d:fit"],
                ),
            },
            policy="d:act",
        )
    )
    record = decide_sync(loaded, {"names": ["ok", AWKWARD]}, {}).record
    written = grounded_ref("d:fit", AWKWARD)
    assert written in {i["ref"] for i in record.log if i["item"] == "entry"}
    found = escalation(record)
    assert found is not None
    assert found["cause"] == {"code": "required", "refs": [written]}
    assert found["root_cause"] == {"code": "raised", "refs": ["fit"]}


def test_the_escalation_is_plain_data_of_exactly_four_members() -> None:
    found = escalation(turn(requiring))
    assert found is not None
    assert type(found) is dict
    assert list(found) == ["kind", "cause", "root_cause", "record_hash"]
    cause: object = found["cause"]
    assert type(cause) is dict
    assert type(found["cause"]["refs"]) is list


@pytest.mark.parametrize("name", ["normal", "broken", "cancelled", "extraction", "described"])
def test_a_record_before_format_8_holds_no_escalation(name: str) -> None:
    assert escalation(read(fresh(LOGS[name]))) is None


def test_a_record_with_no_entry_at_the_policys_ref_holds_no_escalation() -> None:
    assert escalation(read(fresh(CANCELLED_8))) is None
    for n in range(1, 10):  # every prefix before the policy's entry
        assert escalation(read(fresh(NORMAL_8[:n]))) is None


def test_the_escalation_reads_the_markers_and_the_hash() -> None:
    record = read(fresh(BROKEN_8))
    assert escalation(record) == {
        "kind": "Unavailable",
        "cause": ENTRY_CAUSE["cause"],
        "root_cause": ENTRY_CAUSE["cause"],
        "record_hash": record.record_hash,
    }


class Untouchable:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the record was read: {name}")


class SubRecord(Record):
    pass


# Built in the test, since pytest reads a parameter at collection.
NOT_RECORDS: list[Callable[[], Any]] = [lambda: NORMAL, lambda: None, Untouchable]
VIEWS = [escalation, transcript]


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
@pytest.mark.parametrize("of", VIEWS, ids=["escalation", "transcript"])
def test_each_view_refuses_what_is_not_a_record_before_reading_it(
    record: Callable[[], Any], of: Callable[[Record], Any]
) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        of(record())


@pytest.mark.parametrize("of", VIEWS, ids=["escalation", "transcript"])
def test_each_view_refuses_a_record_subclass_by_its_exact_type(
    of: Callable[[Record], Any],
) -> None:
    sub = object.__new__(SubRecord)  # never initialised: refused before anything is read
    with pytest.raises(ValueError, match="takes a Record"):
        of(sub)


# ---------------------------------------------------------------------------------------------
# The transcript


def test_the_transcript_ends_at_decides_call_the_board_and_its_result_the_decision() -> None:
    record = read(fresh(NORMAL_8))
    stem = "om_" + record.record_hash.removeprefix("sha256:")[:24]
    found = transcript(record)
    assert found is not None
    assert found["record_hash"] == record.record_hash
    assert found["design"] == {"hash": NORMAL_8[0]["hash"]}
    assert [[c["source"] for c in r] for r in found["rounds"]] == [["fetch_mail"], ["ask_spam"]]
    assert found["rounds"][1][0]["answers"] == {"q:spam": {"value": NORMAL_8[8]["value"]}}
    assert found["decide"] == {
        "id": f"{stem}_d",
        "board": {"d:tags": {"value": [{"a": "x"}, None]}},  # the policy's own entry left out
        "decision": {"value": "archive"},
    }


def test_the_transcript_of_a_decided_turn_holds_its_board_and_decision() -> None:
    record = turn(requiring, name="mail")
    found = transcript(record)
    assert found is not None
    assert found["design"] == {"hash": record.design_hash}
    assert found["rounds"] == []
    decided = found["decide"]
    assert set(decided["board"]) == {"d:flag", "d:gone", "d:mid"}
    assert decided["decision"] == {
        "marker": {"kind": "Unavailable", "cause": {"code": "required", "refs": ["d:mid"]}}
    }
    assert set(decided) == {"id", "board", "decision"}


def test_no_policys_entry_or_a_record_before_format_8_gives_no_transcript() -> None:
    assert transcript(read(fresh(NORMAL))) is None
    assert transcript(read(fresh(CANCELLED_8))) is None
    assert transcript(read(fresh(NORMAL_8[:9]))) is None


def test_a_broken_entrys_transcript_holds_no_rounds_and_the_entry_cause() -> None:
    found = transcript(read(fresh(BROKEN_8)))
    assert found is not None
    assert found["rounds"] == []
    assert found["decide"]["board"] == {}
    assert found["decide"]["decision"] == {"marker": ENTRY_CAUSE}


def test_a_record_cancelled_after_the_policy_fired_still_shows_its_decision() -> None:
    record = read(fresh([*NORMAL_8, CANCEL]))
    assert escalation(record) is None
    found = transcript(record)
    assert found is not None
    assert found["decide"]["decision"] == {"value": "archive"}


# ---------------------------------------------------------------------------------------------
# The redacted view over the synthetic normal log


def test_redact_digests_by_class_and_passes_open_values() -> None:
    record = read(fresh(NORMAL))
    out = view(record)
    log = out["log"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert log[:2] == NORMAL[:2]  # the design and settings items are open
    assert log[2]["value"] == {"subject": d("hello"), "to": d("dana@example.com")}
    assert log[3] == NORMAL[3]  # a call item with no prompt
    assert log[4]["value"] == d("text of the mail")
    assert log[5]["value"] == d("hello")
    assert log[7]["prompt"] == d("Is this spam?")
    assert log[7]["questions"] == NORMAL[7]["questions"]
    assert log[8] == NORMAL[8]  # an answer is open
    assert log[9] == NORMAL[9]
    assert log[10]["value"] == [{d("a"): "x"}, None]  # a mapping's keys, under every tree
    plan = log[12]
    assert plan["steps"][0]["arguments"] == {
        "folder": {"literal": d("spam")},
        "what": {"ref": "d:route"},
    }
    assert plan["steps"][0]["requirements"] == NORMAL[12]["steps"][0]["requirements"]
    assert (plan["reason"], plan["reply"]) == (None, None)
    assert log[13] == NORMAL[13]
    assert out["unhashed"] == {}
    assert out["redacted"] == record.record_hash
    assert set(out) == {"log", "unhashed", "redacted"}


def test_a_requirement_value_takes_its_refs_tree_in_the_plan_and_the_verdict() -> None:
    log = fresh(NORMAL)
    log[1]["classes"]["d:route"] = "content"
    log[13] = {
        "item": "verdict",
        "kind": "refused",
        "failures": [
            {"step": 1, "requirement": {"ref": "d:route", "value": "archive"}, "ref": "d:route"}
        ],
    }
    record = read(log)
    out = view(record)["log"]
    hidden = {"ref": "d:route", "value": digest("archive", record)}
    assert out[12]["steps"][0]["requirements"] == [hidden]
    assert out[13]["failures"][0]["requirement"] == hidden
    assert out[13]["failures"][0]["ref"] == "d:route"


def test_a_requirement_whose_ref_is_outside_the_grammar_is_digested_whole() -> None:
    log = fresh(NORMAL)
    failure = {"step": 1, "requirement": {"ref": "garbage", "value": "x"}, "ref": "garbage"}
    log[13] = {"item": "verdict", "kind": "refused", "failures": [failure]}
    record = read(log)
    assert view(record)["log"][13]["failures"][0]["requirement"]["value"] == digest("x", record)


def test_equal_values_show_equal_digests_within_one_record() -> None:
    log = view(read(fresh(NORMAL)))["log"]
    assert log[2]["value"]["subject"] == log[5]["value"]


def test_the_key_is_the_records_own_and_one_log_has_one_key() -> None:
    plain = read(fresh(NORMAL))
    with_text = read(fresh(NORMAL), {"engine": "0.0.1", "reads": {"9": "t:x"}})
    assert plain.record_hash == with_text.record_hash
    assert view(plain)["log"] == view(with_text)["log"]
    other = view(plain, bytes(32))["log"]
    assert other[4]["value"] == digest("text of the mail", plain, bytes(32))
    assert other[4]["value"] != view(plain)["log"][4]["value"]


def test_the_redacted_view_is_refused_by_read_record() -> None:
    with pytest.raises(InvalidRecord, match="redacted"):
        read_record(redact(read(fresh(NORMAL)), ROOT))


# ---------------------------------------------------------------------------------------------
# The grain

TREES: dict[str, Any] = {
    "d:words": "content",
    "d:recs": {"k": "content", "n": "open"},
    "d:map": ["content"],
    "d:mapo": ["open"],
    "d:mapr": [{"k": "content"}],
    "d:nums": "open",
    "d:who": "identifier",
    "d:note": "content",
    "d:deep": "open",
}


def grain(values: dict[str, Any]) -> tuple[Record, dict[str, Any]]:
    """A round-0 log holding each value at its ref under ``TREES``."""
    settings = copy.deepcopy(SETTINGS)
    settings["providers"] = {ref: ref[2:] for ref in TREES}
    settings["classes"] = {"t:state": SETTINGS["classes"]["t:state"], **TREES}
    log = [DESIGN, settings, fresh([STATE])[0]]
    log += [entry(ref, 0, None, values[ref]) for ref in sorted(values)]
    record = read(log)
    redacted = view(record)["log"]
    return record, {item["ref"]: item.get("value") for item in redacted[3:]}


def test_a_list_or_mapping_with_no_open_and_no_record_is_one_digest() -> None:
    record, out = grain({"d:words": ["a", None, "b"], "d:map": {"x": "y"}})
    assert out["d:words"] == digest(["a", None, "b"], record)  # the null inside included
    assert out["d:map"] == digest({"x": "y"}, record)


def test_any_other_list_or_mapping_is_opened() -> None:
    record, out = grain(
        {
            "d:recs": [{"k": "a", "n": 1}, None],
            "d:mapo": {"x": "y"},
            "d:mapr": {"x": {"k": "a"}},
            "d:nums": [1, 2, [3]],
        }
    )

    def d(value: Any) -> str:
        return digest(value, record)

    assert out["d:recs"] == [{"k": d("a"), "n": 1}, None]
    assert out["d:mapo"] == {d("x"): "y"}
    assert out["d:mapr"] == {d("x"): {"k": d("a")}}
    assert out["d:nums"] == [1, 2, [3]]


def test_a_null_passes_under_every_class() -> None:
    _, out = grain({"d:note": None, "d:who": None, "d:words": None})
    assert out == {"d:note": None, "d:who": None, "d:words": None}


def test_identifier_and_content_digest_alike() -> None:
    record, out = grain({"d:who": "dana", "d:note": "dana"})
    assert out["d:who"] == out["d:note"] == digest("dana", record)


def test_a_value_64_deep_redacts() -> None:
    deep: Any = 1
    for _ in range(63):
        deep = [deep]
    _, out = grain({"d:deep": deep})
    assert out["d:deep"] == deep


# ---------------------------------------------------------------------------------------------
# Values outside entries


def plan_with(arguments: dict[str, Any], **members: Any) -> dict[str, Any]:
    step: dict[str, Any] = {"tool": "archive", "arguments": arguments, "requirements": []}
    return {
        "item": "plan",
        "kind": "execute",
        "steps": [step],
        "reason": None,
        "reply": None,
        "cause": None,
        **members,
    }


def test_a_literal_takes_the_default_of_its_json_form() -> None:
    arguments = {
        "text": {"literal": "spam"},
        "n": {"literal": 3},
        "yes": {"literal": True},
        "none": {"literal": None},
        "list": {"literal": ["x", 1, None]},
        "object": {"literal": {"a": ["x", 2]}},
        "bound": {"step": 1, "path": None},
    }
    record = read([*fresh(NORMAL[:12]), plan_with(arguments)])
    out = view(record)["log"][12]["steps"][0]["arguments"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert out == {
        "text": {"literal": d("spam")},
        "n": {"literal": 3},
        "yes": {"literal": True},
        "none": {"literal": None},
        "list": {"literal": [d("x"), 1, None]},
        "object": {"literal": {d("a"): [d("x"), 2]}},
        "bound": {"step": 1, "path": None},
    }


@pytest.mark.parametrize("set_", ["d:route", "t:mail/body", "a set", None])
def test_read_record_refuses_an_origin_carrying_set(set_: Any) -> None:
    # Nothing checks a candidate-value literal's set in this engine, so redact could find no
    # tree for it; a null set is a set key all the same.
    arguments = {"folder": {"literal": "x", "set": set_}}
    with pytest.raises(InvalidRecord, match="carries set"):
        read([*fresh(NORMAL[:12]), plan_with(arguments)])


def test_a_plans_reason_and_reply_are_content() -> None:
    plan = plan_with({}, kind="escalate", steps=[], reason="why")
    record = read([*fresh(NORMAL[:12]), plan])
    out = view(record)["log"][12]
    assert out["reason"] == digest("why", record)
    assert out["reply"] is None


UNHASHED: dict[str, Any] = {
    "turn": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:00:01Z"},
    "engine": "0.0.1",
    "codec": "encode",
    "calls": {
        "3": {
            "start": "2026-01-01T00:00:00Z",
            "latency": 0.5,
            "usage": {"input": 3},
            "attempts": 1,
            "unexpected": ["t:x", "t:y"],
            "invalid": {"snapshot": "b'x'"},
        },
        "7": {
            "start": "2026-01-01T00:00:00Z",
            "latency": 0.5,
            "usage": None,
            "attempts": None,
            "sums": {"q:spam": 1.8},
        },
    },
    "exceptions": {"9": {"type": "ValueError", "message": "bad", "traceback": "Traceback..."}},
    "reads": {"9": "t:secret"},
    "later": {"a": [1]},
}


def test_unhashed_text_is_content_and_its_names_keys_and_usage_open() -> None:
    record = read(fresh(NORMAL), copy.deepcopy(UNHASHED))
    out = view(record)["unhashed"]

    def d(value: Any) -> str:
        return digest(value, record)

    assert out == {
        "turn": UNHASHED["turn"],
        "engine": "0.0.1",
        "codec": "encode",
        "calls": {
            "3": {
                "start": "2026-01-01T00:00:00Z",
                "latency": 0.5,
                "usage": {"input": 3},
                "attempts": 1,
                "unexpected": d(["t:x", "t:y"]),  # digested as one value
                "invalid": {"snapshot": d("b'x'")},
            },
            "7": UNHASHED["calls"]["7"],  # a sums key and its number open
        },
        "exceptions": {
            "9": {"type": "ValueError", "message": d("bad"), "traceback": d("Traceback...")}
        },
        "reads": {"9": d("t:secret")},
        "later": d({"a": [1]}),  # a member the engine does not know, whole
    }


# ---------------------------------------------------------------------------------------------
# The root key


class Key(bytes):
    pass


class LongKey(bytes):
    def __len__(self) -> int:
        raise AssertionError("the caller's __len__ was called")


@pytest.mark.parametrize(
    "root",
    [bytes(31), bytes(33), b"", "k" * 32, bytearray(32), memoryview(bytes(32)), None, 32],
    ids=["31 bytes", "33 bytes", "empty", "str", "bytearray", "memoryview", "None", "int"],
)
def test_a_bad_root_key_is_refused_before_the_record_is_read(root: Any) -> None:
    with pytest.raises(ValueError, match="root key"):
        redact(read(fresh(NORMAL)), root)
    # The record is tested first: a non-Record with a bad key takes the record's refusal,
    # and nothing of it is read.
    with pytest.raises(ValueError, match="takes a Record"):
        redact(Untouchable(), root)  # type: ignore[arg-type]


def test_a_bytes_subclass_root_key_is_taken_by_its_bytes_alone() -> None:
    record = read(fresh(NORMAL))
    assert redact(record, Key(ROOT)) == redact(record, ROOT)
    assert redact(record, LongKey(ROOT)) == redact(record, ROOT)
    with pytest.raises(ValueError, match="root key"):
        redact(record, LongKey(bytes(31)))


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
def test_redact_refuses_what_is_not_a_record_with_a_good_key(record: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        redact(record(), ROOT)
    with pytest.raises(ValueError, match="takes a Record"):
        redact(object.__new__(SubRecord), ROOT)


# ---------------------------------------------------------------------------------------------
# Every closed record


PREFIXES = [(name, n) for name, log in LOGS.items() for n in range(1, len(log) + 1)]


@pytest.mark.parametrize(("name", "n"), PREFIXES, ids=[f"{name}-{n}" for name, n in PREFIXES])
def test_every_prefix_of_the_test_logs_redacts_and_escalates(name: str, n: int) -> None:
    record = read(fresh(LOGS[name][:n]), unhashed_of(name, n))
    out = view(record)
    for key, keyed in record.unhashed.get("calls", {}).items():
        if "truncated" in keyed:  # declared refs, open
            assert out["unhashed"]["calls"][key]["truncated"] == keyed["truncated"]
        if "warnings" in keyed:  # the binding's text, one digest
            warned = out["unhashed"]["calls"][key]["warnings"]
            assert warned == digest(keyed["warnings"], record)
    assert out["redacted"] == record.record_hash
    assert out.get("hash") == (record.record_hash if record.log[0]["format"] >= 3 else None)
    assert len(out["log"]) == n
    escalation(record)


def test_a_cause_item_passes_through_and_a_design_only_record_has_nothing_digested() -> None:
    cancelled = read(fresh(LOGS["cancelled"]))
    assert view(cancelled)["log"][-1] == LOGS["cancelled"][-1]
    assert view(read(fresh([DESIGN])))["log"] == [DESIGN]


def test_a_redacted_turn_decide_wrote_digests_its_state() -> None:
    loaded = loaded_with(send, name="mail")
    record = decide_sync(loaded, {"subject": "secret"}, {}, RunSettings()).record
    out = view(record)
    assert out["log"][2]["value"] == {"subject": digest("secret", record)}
    assert out["unhashed"]["engine"] == record.unhashed["engine"]


def test_a_round_item_and_a_call_item_without_prompt_pass() -> None:
    log = [*fresh(NORMAL[:3]), call(1, "fetch_mail"), round_item(1)]
    log.insert(4, entry("t:mail/body", 1, 3, None))
    log.insert(5, entry("t:mail/subject", 1, 3, "s"))
    out = view(read(log))["log"]
    assert out[3] == log[3]
    assert out[-1] == log[-1]


def test_a_declared_class_tree_is_the_one_the_ledger_fits_and_redact_reads() -> None:
    # The settings item carries load's tree in full, the declared tree over the default, so the
    # ledger fits each value to it and redact digests by it.
    classes = {"t:state": {"subject": "open"}, "d:flag": "identifier"}
    loaded = loaded_with(send, classes=classes)
    record = decide_sync(loaded, {"subject": "secret"}, {}, RunSettings()).record
    assert record.log[1]["classes"]["t:state"] == {"subject": "open"}
    assert record.log[1]["classes"]["d:flag"] == "identifier"
    out = view(record)
    assert out["log"][2]["value"] == {"subject": "secret"}
    (index,) = [i for i, item in enumerate(record.log) if item.get("ref") == "d:flag"]
    assert out["log"][index]["value"] == digest(False, record)


def test_a_decided_policy_entry_is_digested_by_its_class_and_a_marker_passes() -> None:
    loaded = loaded_with(send, classes={"d:act": "identifier"})
    record = decide_sync(loaded, {"subject": "hi"}, {}, RunSettings()).record
    (index,) = [i for i, item in enumerate(record.log) if item.get("ref") == "d:act"]
    assert view(record)["log"][index]["value"] == digest("send", record)

    undecided = decide_sync(
        loaded_with(requiring, classes={"d:act": "identifier"}), {"subject": "hi"}, {}
    ).record
    (index,) = [i for i, item in enumerate(undecided.log) if item.get("ref") == "d:act"]
    policy_entry = undecided.log[index]
    assert policy_entry["marker"]["kind"] == "Unavailable"
    assert view(undecided)["log"][index] == policy_entry
