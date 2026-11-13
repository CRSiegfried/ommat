# pyright: reportPrivateUsage=false
"""The record views: the escalation's five members, the root cause its ``required`` chain
ends at, whether that is a defect, and when a record holds none; the transcript of a decided
turn; and the redacted view, its digests recomputed here independently with ``hmac`` and the
``rfc8785`` oracle, over records decide wrote and over the synthetic logs and every prefix of
them. A candidate-valued answer's labels are tested in ``test_label_sets.py``; a grounded ref's
candidate, ``sums`` and ``truncated`` in ``test_per_candidate.py``, ``test_grounded_rules.py``,
``test_grounded_sources.py``, ``test_record_sets.py`` and ``test_extraction_turn.py``; a set
literal in ``test_ledger.py``."""

import copy
import hashlib
import hmac
import json
import re
import types
from collections.abc import Callable
from typing import Any, cast

import pytest
import rfc8785

from designs import minimal
from ledger_logs import (
    ANSWER,
    BROKEN_8,
    CANCEL,
    CANCELLED_8,
    DESIGN,
    DESIGN_9,
    ENTRY_CAUSE,
    LOGS,
    NORMAL,
    NORMAL_8,
    NORMAL_9,
    SETTINGS,
    SETTINGS_9,
    STATE,
    call,
    entry,
    fresh,
    rehashed,
    round_item,
    shape_of,
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
    defects,
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
from ommat._views import _defect

pytestmark = [
    pytest.mark.filterwarnings("ignore::ommat.DefectWarning"),  # defects are the point here
    pytest.mark.filterwarnings("ignore:transcript is deprecated:DeprecationWarning"),
]

ROOT = bytes(range(32))


def key_of(view: dict[str, Any], root: bytes = ROOT) -> bytes:
    """The per-turn key a verifier recomputes from a redacted view alone: HKDF-Expand under the
    root key, its info the record hash and then the hash of the applied policy."""
    applied = "sha256:" + hashlib.sha256(rfc8785.dumps(view["applied"])).hexdigest()
    info = view["redacted"].encode("ascii") + applied.encode("ascii")
    return hmac.digest(root, info + b"\x01", "sha256")


def digest(value: Any, view: dict[str, Any], root: bytes = ROOT) -> str:
    """A keyed digest computed apart from the code under test, under ``view``'s key."""
    mac = hmac.new(key_of(view, root), rfc8785.dumps(value), hashlib.sha256)
    return "hmac-sha256:" + mac.hexdigest()


def redacted(record: Record, policy: Any = None, root: bytes = ROOT) -> dict[str, Any]:
    """The redacted view of ``record`` under ``policy``, none named by default, decoded."""
    return cast("dict[str, Any]", decode(redact(record, root, {} if policy is None else policy)))


def read(log: list[dict[str, Any]], unhashed: dict[str, Any] | None = None) -> Record:
    """The record of ``log``, which may share a marker between two entries, as a document
    holding no list or object at two places."""
    unshared = json.loads(json.dumps(log))
    return read_record(rehashed({"log": unshared, "unhashed": unhashed or {}}))


def view(record: Record, root: bytes = ROOT, policy: Any = None) -> dict[str, Any]:
    out = redact(record, root, {} if policy is None else policy)
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
        "defect": True,
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
        "defect": True,
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


@pytest.mark.parametrize(
    ("root", "defect"),
    [
        ({"code": "raised", "refs": ["route"]}, True),  # a rule
        ({"code": "invalid-return", "refs": ["tagger"]}, True),
        ({"code": "undeclared-read", "refs": ["route", "d:tags"]}, True),
        ({"code": "raised", "refs": ["codec"]}, True),  # the codec
        ({"code": "invalid-return", "refs": ["t:state"]}, True),
        ({"code": "raised", "refs": ["q:spam"]}, False),  # an oracle's question
        ({"code": "invalid-return", "refs": ["q:spam"]}, False),
        ({"code": "raised", "refs": ["t:mail/body"]}, False),
        ({"code": "undeclared-read", "refs": ["fetch_mail", "t:state.to"]}, False),  # a when
        ({"code": "abstained", "refs": ["route"]}, False),
        ({"code": "timed-out", "refs": ["q:spam"]}, False),
        ({"code": "raised", "refs": []}, False),
    ],
)
def test_a_root_cause_is_a_defect_when_its_code_is_and_it_names_a_rule_or_the_codec(
    root: dict[str, Any], defect: bool
) -> None:
    assert _defect(fresh(NORMAL_8), root) is defect


def test_the_escalation_is_plain_data_of_exactly_five_members() -> None:
    found = escalation(turn(requiring))
    assert found is not None
    assert type(found) is dict
    assert list(found) == ["kind", "cause", "root_cause", "defect", "record_hash"]
    assert type(found["defect"]) is bool
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
        "defect": True,
        "record_hash": record.record_hash,
    }


class Untouchable:
    def __getattribute__(self, name: str) -> Any:
        raise AssertionError(f"the record was read: {name}")


class SubRecord(Record):
    pass


# Built in the test, since pytest reads a parameter at collection.
NOT_RECORDS: list[Callable[[], Any]] = [lambda: NORMAL, lambda: None, Untouchable]
VIEWS = [escalation, defects, transcript]
VIEW_IDS = ["escalation", "defects", "transcript"]


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
@pytest.mark.parametrize("of", VIEWS, ids=VIEW_IDS)
def test_each_view_refuses_what_is_not_a_record_before_reading_it(
    record: Callable[[], Any], of: Callable[[Record], Any]
) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        of(record())


@pytest.mark.parametrize("of", VIEWS, ids=VIEW_IDS)
def test_each_view_refuses_a_record_subclass_by_its_exact_type(
    of: Callable[[Record], Any],
) -> None:
    sub = object.__new__(SubRecord)  # never initialised: refused before anything is read
    with pytest.raises(ValueError, match="takes a Record"):
        of(sub)


# ---------------------------------------------------------------------------------------------
# The defects


def test_a_rule_that_raised_on_a_decided_turn_is_a_defect_the_escalation_cannot_show() -> None:
    record = turn(send)
    assert escalation(record) is None
    found = defects(record)
    assert found is not None
    assert [item["ref"] for item in found] == ["d:gone"]  # d:mid's required cause is not one
    assert found[0] == record.entries["d:gone"]


@pytest.mark.parametrize("policy", [requiring, send], ids=["undecided", "decided"])
def test_the_defects_hold_the_root_cause_of_an_escalation_that_is_a_defect(
    policy: Callable[[Any], str],
) -> None:
    record = turn(policy)
    found = escalation(record)
    listed = [item["marker"]["cause"] for item in defects(record) or []]
    if found is not None:
        assert found["defect"] is (found["root_cause"] in listed)


def test_a_broken_entrys_defect_is_its_entry_cause_at_the_state_and_the_policy() -> None:
    record = read(fresh(BROKEN_8))
    found = defects(record) or []
    assert [item["ref"] for item in found] == ["t:state", record.log[1]["policy"]]
    assert all(item["marker"]["cause"] == ENTRY_CAUSE["cause"] for item in found)


def test_a_turn_with_no_defect_lists_none() -> None:
    assert defects(read(fresh(NORMAL_8))) == []


@pytest.mark.parametrize("name", ["normal", "broken", "cancelled", "extraction", "described"])
def test_a_record_before_format_8_holds_no_defects(name: str) -> None:
    assert defects(read(fresh(LOGS[name]))) is None


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
# The redacted view under the caller's policy, over the synthetic format-9 log


def filled(shape: Any, name: str) -> Any:
    """A shape tree with the class ``name`` at every leaf."""
    if isinstance(shape, dict):
        return {k: filled(v, name) for k, v in cast("dict[str, Any]", shape).items()}
    if isinstance(shape, list):
        return [filled(cast("list[Any]", shape)[0], name)]
    return name


DIGEST = re.compile(r"hmac-sha256:[0-9a-f]{64}")


def digests_in(out: dict[str, Any]) -> set[str]:
    return set(re.findall(r"hmac-sha256:[0-9a-f]{64}", json.dumps(out)))


def test_redact_digests_every_value_the_policy_names_nothing_for() -> None:
    record = read(fresh(NORMAL_9))
    out = view(record)
    log = out["log"]

    def d(value: Any) -> str:
        return digest(value, out)

    assert log[:2] == NORMAL_9[:2]  # the design and settings items are open
    assert log[2]["value"] == {"subject": d("hello"), "to": d("dana@example.com")}
    assert log[3] == record.log[3]  # a call item with no prompt
    assert log[4]["value"] == d("text of the mail")
    assert log[7]["prompt"] == d("Is this spam?")
    assert log[7]["questions"] == record.log[7]["questions"]
    assert log[8]["value"] == d(ANSWER)  # an answer, whole
    assert log[9]["value"] == d("archive")
    assert log[10]["value"] == d([{"a": "x"}, None])  # a list under content: one digest
    assert {k: v for k, v in log[9].items() if k != "value"} == {
        k: v for k, v in record.log[9].items() if k != "value"
    }
    assert out["applied"] == {
        ref: filled(shape, "content") for ref, shape in SETTINGS_9["shapes"].items()
    }
    assert out["unhashed"] == {}
    assert out["redacted"] == out["hash"] == record.record_hash
    assert set(out) == {"log", "unhashed", "redacted", "applied", "hash"}


def test_the_policy_names_each_class_and_the_rest_stays_content() -> None:
    record = read(fresh(NORMAL_9))
    policy = {
        "t:state": {"subject": "open", "to": "identifier"},
        "q:spam": "open",
        "d:tags": "open",
    }
    out = view(record, policy=policy)
    log = out["log"]

    def d(value: Any) -> str:
        return digest(value, out)

    assert log[2]["value"] == {"subject": "hello", "to": d("dana@example.com")}
    assert log[8]["value"] == ANSWER
    assert log[10]["value"] == [{d("a"): "x"}, None]  # a mapping's keys, under every tree
    assert log[4]["value"] == d("text of the mail")
    assert log[9]["value"] == d("archive")
    assert out["applied"]["t:state"] == {"subject": "open", "to": "identifier"}
    assert out["applied"]["d:tags"] == ["open"]
    assert out["applied"]["d:route"] == "content"


def test_a_field_path_key_names_a_field_and_overrides_a_shorter_key_whatever_their_order() -> None:
    record = read(fresh(NORMAL_9))
    out = view(record, policy={"t:state": "open", "t:state.to": "identifier"})
    assert out["applied"]["t:state"] == {"subject": "open", "to": "identifier"}
    assert out["log"][2]["value"] == {"subject": "hello", "to": digest("dana@example.com", out)}
    assert view(record, policy={"t:state.to": "identifier", "t:state": "open"}) == out


def test_the_policy_is_required() -> None:
    with pytest.raises(TypeError, match="policy"):
        redact(read(fresh(NORMAL_9)), ROOT)  # type: ignore[call-arg]
    record = read(fresh(NORMAL_9))
    assert redact(record, ROOT, policy={}) == redact(record, ROOT, {})


@pytest.mark.parametrize(
    ("policy", "says"),
    [
        ({"d:none": "open"}, "names no ref of the record"),
        ({"t:state.nofield": "open"}, "names a field the record's shape does not hold"),
        ({"t:mail/body.x": "open"}, "names a field the record's shape does not hold"),
        ({"T:x": "open"}, "is not a ref"),
        ({'t:mail/body["a"]': "open"}, "has a grounded bracket"),
        ({1: "open"}, "is not a string"),
        ({"t:state": "secret"}, "not open, identifier or content"),
        ({"t:state": {"subject": "secret"}}, "not open, identifier or content"),
        ({"t:state": {"nofield": "open"}}, "which is not a field there"),
        ({"t:state": {"subject": {"x": "open"}}}, "where the record's shape is not a record"),
        ({"d:route": {}}, "where the record's shape is not a record"),
        ({"d:tags": {"a": "open"}}, "where the record's shape is not a record"),
        ({"t:state": None}, "not a class name or an object"),
        ({"t:state": ["open"]}, "is a list, where the record's shape is no mapping layer"),
        ({"d:tags": ["open", "open"]}, "is a list, where the record's shape is no mapping layer"),
        ({"t:state": 1}, "not a class name or an object"),
        (["t:state"], "not a mapping"),
        (None, "not a mapping"),
    ],
)
def test_a_policy_of_another_form_or_naming_what_the_record_lacks_is_refused(
    policy: Any, says: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(says)):
        redact(read(fresh(NORMAL_9)), ROOT, policy)


class OddKey(str):
    """A key equal to nothing but itself, so a dict holds it beside its string data."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


def test_a_policy_with_two_keys_of_one_string_data_is_refused() -> None:
    policy = {OddKey("t:state"): "open", "t:state": "content"}
    with pytest.raises(ValueError, match="two keys of the string data 't:state'"):
        redact(read(fresh(NORMAL_9)), ROOT, policy)


@pytest.mark.parametrize("first", [0, 1], ids=["subclass first", "str first"])
def test_a_policy_object_with_two_fields_of_one_string_data_is_refused(first: int) -> None:
    keys = [OddKey("subject"), "subject"]
    fields = {keys[first]: "open", keys[1 - first]: "content"}
    with pytest.raises(ValueError, match="at 't:state' has two keys of the string data 'subject'"):
        redact(read(fresh(NORMAL_9)), ROOT, {"t:state": fields})


def test_a_policy_may_be_any_mapping() -> None:
    record = read(fresh(NORMAL_9))
    proxy = types.MappingProxyType({"t:state": types.MappingProxyType({"subject": "open"})})
    assert view(record, policy=proxy) == view(record, policy={"t:state": {"subject": "open"}})


def test_a_tuple_at_a_mapping_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="at 'd:tags' is <tuple>, not a class name or an object"):
        redact(read(fresh(NORMAL_9)), ROOT, {"d:tags": ("open",)})


def test_two_policies_over_one_record_share_no_digest() -> None:
    record = read(fresh(NORMAL_9))
    plain, opened = view(record), view(record, policy={"q:spam": "open"})
    assert digests_in(plain)
    assert digests_in(opened)
    assert not digests_in(plain) & digests_in(opened)
    assert view(record, policy={"q:spam": "open"}) == opened
    # A policy naming what the default gives is the policy the default applies: one key.
    assert view(record, policy={"d:route": "content"}) == plain


def test_a_policy_that_digests_differently_changes_the_key_of_every_digest() -> None:
    record = read(fresh(NORMAL_9))
    plain = view(record)
    identifier = view(record, policy={"q:spam": "identifier"})
    # Each digest is of the same value, under another key.
    assert plain["log"][4]["value"] != identifier["log"][4]["value"]
    assert identifier["log"][4]["value"] == digest("text of the mail", identifier)
    assert identifier["log"][8]["value"] == digest(ANSWER, identifier)


# ---------------------------------------------------------------------------------------------
# A record before format 9: its stored class trees are a floor under the policy


@pytest.mark.parametrize("log", [NORMAL, NORMAL_8], ids=["format 1", "format 8"])
def test_an_empty_policy_over_a_stored_class_tree_leaves_no_data_in_clear(
    log: list[dict[str, Any]],
) -> None:
    record = read(fresh(log))
    out = view(record)
    redacted_log = out["log"]

    def d(value: Any) -> str:
        return digest(value, out)

    assert out["applied"] == {
        ref: filled(shape_of(tree), "content") for ref, tree in SETTINGS["classes"].items()
    }
    assert redacted_log[:2] == record.log[:2]  # the design and settings items are open
    assert redacted_log[2]["value"] == {"subject": d("hello"), "to": d("dana@example.com")}
    assert redacted_log[4]["value"] == d("text of the mail")
    assert redacted_log[5]["value"] == d("hello")
    assert redacted_log[7]["prompt"] == d("Is this spam?")
    assert redacted_log[8]["value"] == d(ANSWER)  # the stored open does not open it
    assert redacted_log[9]["value"] == d("archive")
    assert redacted_log[10]["value"] == d([{"a": "x"}, None])
    assert not [
        leaf
        for item in redacted_log
        if item["item"] == "entry"
        for leaf in clear_leaves(item.get("value"))
    ]


def clear_leaves(value: Any) -> list[Any]:
    """The leaves of a value, field by field and member by member, that are neither a null nor
    a keyed digest."""
    if isinstance(value, dict):
        members = cast("dict[str, Any]", value)
        return [leaf for v in members.values() for leaf in clear_leaves(v)]
    if isinstance(value, list):
        return [leaf for v in cast("list[Any]", value) for leaf in clear_leaves(v)]
    if value is None or (isinstance(value, str) and DIGEST.fullmatch(value)):
        return []
    return [value]


@pytest.mark.parametrize("log", [NORMAL, NORMAL_8], ids=["format 1", "format 8"])
def test_a_policy_opens_only_where_the_stored_class_tree_is_open_too(
    log: list[dict[str, Any]],
) -> None:
    record = read(fresh(log))
    policy = {"q:spam": "open", "d:route": "open", "t:state": "open", "t:mail/body": "open"}
    out = view(record, policy=policy)
    applied = out["applied"]
    assert (applied["q:spam"], applied["d:route"]) == ("open", "open")  # both open
    assert applied["t:mail/body"] == "content"  # the stored content holds
    assert applied["t:state"] == {"subject": "content", "to": "identifier"}
    assert applied["d:tags"] == ["content"]  # named nothing: content over the stored open
    assert out["log"][8]["value"] == ANSWER
    assert out["log"][9]["value"] == "archive"
    assert out["log"][4]["value"] == digest("text of the mail", out)


def test_a_malformed_stored_class_tree_no_entry_is_held_to_is_content() -> None:
    log = fresh(NORMAL[:3])
    log[1]["classes"]["d:unused"] = "secret"
    log[1]["providers"]["d:unused"] = "other"
    out = view(read(log))
    assert out["applied"]["d:unused"] == "content"
    with pytest.raises(ValueError, match="where the record's shape is not a record"):
        redact(read(log), ROOT, {"d:unused": {"a": "open"}})


@pytest.mark.parametrize("log", [NORMAL, NORMAL_8], ids=["format 1", "format 8"])
def test_a_policy_digests_more_than_a_stored_class_tree_never_less(
    log: list[dict[str, Any]],
) -> None:
    record = read(fresh(log))
    policy = {
        "t:mail/body": "open",  # stored content: not opened
        "q:spam": "content",  # stored open: digested
        "d:route": "open",  # stored open: opened
        "t:state": {"subject": "identifier", "to": "open"},
    }
    out = view(record, policy=policy)
    applied = out["applied"]
    assert applied["t:mail/body"] == "content"
    assert applied["q:spam"] == "content"
    # content is above identifier, and the stored identifier above open.
    assert applied["t:state"] == {"subject": "content", "to": "identifier"}
    assert applied["d:route"] == "open"
    assert out["log"][4]["value"] == digest("text of the mail", out)
    assert out["log"][8]["value"] == digest(ANSWER, out)
    assert out["log"][9]["value"] == "archive"


@pytest.mark.parametrize("log", [NORMAL, NORMAL_8], ids=["format 1", "format 8"])
def test_the_applied_policy_of_a_stored_class_tree_applies_itself(
    log: list[dict[str, Any]],
) -> None:
    record = read(fresh(log))
    out = view(record, policy={"d:tags": "content", "t:state": {"to": "open"}})
    assert out["applied"]["d:tags"] == ["content"]
    assert out["applied"]["t:state"] == {"subject": "content", "to": "identifier"}
    assert view(record, policy=out["applied"]) == out


def test_the_plan_of_a_record_before_format_8_redacts_under_the_applied_trees() -> None:
    record = read(fresh(NORMAL))
    out = view(record)
    plan = out["log"][12]
    assert plan["steps"][0]["arguments"] == {
        "folder": {"literal": digest("spam", out)},
        "what": {"ref": "d:route"},
    }
    assert plan["steps"][0]["requirements"] == [{"ref": "d:route", "value": digest("archive", out)}]
    opened = view(record, policy={"d:route": "open"})["log"][12]  # open in the stored tree too
    assert opened["steps"][0]["requirements"] == NORMAL[12]["steps"][0]["requirements"]
    assert (plan["reason"], plan["reply"]) == (None, None)
    assert out["log"][13] == NORMAL[13]
    assert set(out) == {"log", "unhashed", "redacted", "applied"}


def test_a_requirement_value_takes_its_refs_tree_in_the_plan_and_the_verdict() -> None:
    log = fresh(NORMAL)
    log[13] = {
        "item": "verdict",
        "kind": "refused",
        "failures": [
            {"step": 1, "requirement": {"ref": "d:route", "value": "archive"}, "ref": "d:route"}
        ],
    }
    record = read(log)
    out = view(record)
    hidden = {"ref": "d:route", "value": digest("archive", out)}
    assert out["log"][12]["steps"][0]["requirements"] == [hidden]
    assert out["log"][13]["failures"][0]["requirement"] == hidden
    assert out["log"][13]["failures"][0]["ref"] == "d:route"
    opened = view(record, policy={"d:route": "open"})["log"]
    assert opened[13]["failures"][0]["requirement"] == {"ref": "d:route", "value": "archive"}


@pytest.mark.parametrize("ref", ["garbage", "t:state.to"], ids=["outside", "field path"])
def test_a_requirement_at_a_ref_with_no_tree_of_its_own_is_digested_whole(ref: str) -> None:
    log = fresh(NORMAL)
    failure = {"step": 1, "requirement": {"ref": ref, "value": "x"}, "ref": ref}
    log[13] = {"item": "verdict", "kind": "refused", "failures": [failure]}
    record = read(log)
    for policy in ({}, {"t:state": "open"}, {"t:state.to": "open"}):
        out = view(record, policy=policy)
        assert out["log"][13]["failures"][0]["requirement"]["value"] == digest("x", out)


def test_equal_values_show_equal_digests_within_one_view() -> None:
    log = view(read(fresh(NORMAL_9)))["log"]
    assert log[2]["value"]["subject"] == log[5]["value"]


def test_the_key_is_the_records_own_and_one_log_has_one_key() -> None:
    plain = read(fresh(NORMAL_9))
    with_text = read(fresh(NORMAL_9), {"engine": "0.0.1", "reads": {"9": "t:x"}})
    assert plain.record_hash == with_text.record_hash
    assert view(plain)["log"] == view(with_text)["log"]
    other = view(plain, bytes(32))
    assert other["log"][4]["value"] == digest("text of the mail", other, bytes(32))
    assert other["log"][4]["value"] != view(plain)["log"][4]["value"]


def test_the_redacted_view_is_refused_by_read_record() -> None:
    for log in (NORMAL, NORMAL_9):
        with pytest.raises(InvalidRecord, match="redacted"):
            read_record(redact(read(fresh(log)), ROOT, {}))


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
POLICY: dict[str, Any] = {
    **TREES,
    "d:map": "content",
    "d:mapo": "open",
    "d:mapr": {"k": "content"},
}
"""``TREES`` as a policy names them, with no node at a mapping layer."""
FORMATS = [1, 9]
"""The grain under ``POLICY``, over ``TREES`` stored as class trees in a format-1 record and
over a format-9 record's shape trees."""


def grain(values: dict[str, Any], fmt: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """A round-0 log holding each value at its ref under ``TREES``: the view and each value."""
    settings = copy.deepcopy(SETTINGS_9 if fmt >= 9 else SETTINGS)
    settings["providers"] = {ref: ref[2:] for ref in TREES}
    state = SETTINGS["classes"]["t:state"]
    if fmt >= 9:
        settings["shapes"] = {
            "t:state": shape_of(state),
            **{r: shape_of(t) for r, t in TREES.items()},
        }
        settings["policy"] = "d:words"
    else:
        settings["classes"] = {"t:state": state, **TREES}
    log = [DESIGN_9 if fmt >= 9 else DESIGN, settings, fresh([STATE])[0]]
    log += [entry(ref, 0, None, values[ref]) for ref in sorted(values)]
    out = view(read(log), policy=POLICY)  # over format 1, the stored floor is the same trees
    assert out["applied"] == {"t:state": out["applied"]["t:state"], **TREES}
    return out, {item["ref"]: item.get("value") for item in out["log"][3:]}


def test_the_applied_policy_is_a_policy_that_applies_itself() -> None:
    record = read(fresh(NORMAL_9))
    out = view(record, policy={"t:state.subject": "open", "d:tags": ["open"]})
    assert out["applied"]["d:tags"] == ["open"]
    assert view(record, policy=out["applied"]) == out


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_list_or_mapping_with_no_open_and_no_record_is_one_digest(fmt: int) -> None:
    view_, out = grain({"d:words": ["a", None, "b"], "d:map": {"x": "y"}}, fmt)
    assert out["d:words"] == digest(["a", None, "b"], view_)  # the null inside included
    assert out["d:map"] == digest({"x": "y"}, view_)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_list_or_mapping_whose_tree_opens_a_value_is_opened(fmt: int) -> None:
    view_, out = grain(
        {
            "d:recs": [{"k": "a", "n": 1}, None],
            "d:mapo": {"x": "y"},
            "d:mapr": {"x": {"k": "a"}},
            "d:nums": [1, 2, [3]],
        },
        fmt,
    )

    def d(value: Any) -> str:
        return digest(value, view_)

    assert out["d:recs"] == [{"k": d("a"), "n": 1}, None]
    assert out["d:mapo"] == {d("x"): "y"}
    assert out["d:mapr"] == {d("x"): {"k": d("a")}}
    assert out["d:nums"] == [1, 2, [3]]


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_null_passes_under_every_class(fmt: int) -> None:
    _, out = grain({"d:note": None, "d:who": None, "d:words": None, "d:nums": None}, fmt)
    assert out == {"d:note": None, "d:who": None, "d:words": None, "d:nums": None}


def test_a_null_passes_where_the_policy_names_nothing() -> None:
    out = view(read([*fresh(NORMAL_9[:9]), entry("d:route", 2, None, None, derivation=[])]))
    assert out["log"][9]["value"] is None


@pytest.mark.parametrize("fmt", FORMATS)
def test_identifier_and_content_digest_alike(fmt: int) -> None:
    view_, out = grain({"d:who": "dana", "d:note": "dana"}, fmt)
    assert out["d:who"] == out["d:note"] == digest("dana", view_)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_value_64_deep_redacts(fmt: int) -> None:
    deep: Any = 1
    for _ in range(63):
        deep = [deep]
    _, out = grain({"d:deep": deep}, fmt)
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


def test_a_literal_is_content_whole_whatever_the_policy() -> None:
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
    for policy in ({}, {"d:route": "open", "t:state": "open"}):
        out = view(record, policy=policy)

        def d(value: Any, out: dict[str, Any] = out) -> str:
            return digest(value, out)

        assert out["log"][12]["steps"][0]["arguments"] == {
            "text": {"literal": d("spam")},
            "n": {"literal": d(3)},
            "yes": {"literal": d(True)},
            "none": {"literal": None},
            "list": {"literal": d(["x", 1, None])},
            "object": {"literal": d({"a": ["x", 2]})},
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
    out = view(read([*fresh(NORMAL[:12]), plan]))["log"]
    assert out[12]["reason"] == digest("why", view(read([*fresh(NORMAL[:12]), plan])))
    assert out[12]["reply"] is None


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


@pytest.mark.parametrize(
    "policy", [{}, {"t:state": "open", "q:spam": "open"}], ids=["none", "open"]
)
def test_unhashed_text_is_content_whatever_the_policy_and_its_names_keys_and_usage_open(
    policy: dict[str, Any],
) -> None:
    record = read(fresh(NORMAL), copy.deepcopy(UNHASHED))
    out = view(record, policy=policy)

    def d(value: Any) -> str:
        return digest(value, out)

    assert out["unhashed"] == {
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
    assert out["log"][7]["prompt"] == d("Is this spam?")  # the policy never names it


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
        redact(read(fresh(NORMAL)), root, {})
    # The record is tested first: a non-Record with a bad key takes the record's refusal,
    # and nothing of it is read.
    with pytest.raises(ValueError, match="takes a Record"):
        redact(Untouchable(), root, {})  # type: ignore[arg-type]
    # The key is tested before the policy.
    with pytest.raises(ValueError, match="root key"):
        redact(read(fresh(NORMAL)), root, None)  # type: ignore[arg-type]


def test_a_bytes_subclass_root_key_is_taken_by_its_bytes_alone() -> None:
    record = read(fresh(NORMAL))
    assert redact(record, Key(ROOT), {}) == redact(record, ROOT, {})
    assert redact(record, LongKey(ROOT), {}) == redact(record, ROOT, {})
    with pytest.raises(ValueError, match="root key"):
        redact(record, LongKey(bytes(31)), {})


@pytest.mark.parametrize("record", NOT_RECORDS, ids=["a log", "None", "untouchable"])
def test_redact_refuses_what_is_not_a_record_with_a_good_key(record: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        redact(record(), ROOT, {})
    with pytest.raises(ValueError, match="takes a Record"):
        redact(object.__new__(SubRecord), ROOT, {})


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
            assert warned == digest(keyed["warnings"], out)
    assert out["redacted"] == record.record_hash
    assert out.get("hash") == (record.record_hash if record.log[0]["format"] >= 3 else None)
    assert len(out["log"]) == n
    escalation(record)
    defects(record)


def test_a_cause_item_passes_through_and_a_design_only_record_has_nothing_digested() -> None:
    cancelled = read(fresh(LOGS["cancelled"]))
    assert view(cancelled)["log"][-1] == LOGS["cancelled"][-1]
    design_only = view(read(fresh([DESIGN])))
    assert design_only["log"] == [DESIGN]
    assert design_only["applied"] == {}
    with pytest.raises(ValueError, match="names no ref of the record"):
        redact(read(fresh([DESIGN])), ROOT, {"t:state": "open"})


def test_a_round_item_and_a_call_item_without_prompt_pass() -> None:
    log = [*fresh(NORMAL[:3]), call(1, "fetch_mail"), round_item(1)]
    log.insert(4, entry("t:mail/body", 1, 3, None))
    log.insert(5, entry("t:mail/subject", 1, 3, "s"))
    out = view(read(log))["log"]
    assert out[3] == log[3]
    assert out[-1] == log[-1]


# ---------------------------------------------------------------------------------------------
# Turns decide wrote


def at(record: Record, ref: str) -> int:
    (index,) = [i for i, item in enumerate(record.log) if item.get("ref") == ref]
    return index


def test_a_decided_turn_holds_its_shapes_and_digests_what_the_policy_names_nothing_for() -> None:
    loaded = loaded_with(send, name="mail")
    record = decide_sync(loaded, {"subject": "secret"}, {}, RunSettings()).record
    assert record.log[0]["format"] == 9
    assert record.log[1]["shapes"]["t:state"] == {"subject": "leaf"}
    assert "classes" not in record.log[1]
    out = view(record)
    assert out["log"][2]["value"] == {"subject": digest("secret", out)}
    assert out["log"][at(record, "d:flag")]["value"] == digest(False, out)  # a boolean too
    assert out["log"][at(record, "d:act")]["value"] == digest("send", out)
    assert out["unhashed"]["engine"] == record.unhashed["engine"]


def test_a_decided_turns_policy_names_a_class_over_its_shapes() -> None:
    record = decide_sync(loaded_with(send), {"subject": "secret"}, {}, RunSettings()).record
    out = view(record, policy={"t:state": {"subject": "open"}, "d:flag": "identifier"})
    assert out["log"][2]["value"] == {"subject": "secret"}
    assert out["log"][at(record, "d:flag")]["value"] == digest(False, out)
    assert out["applied"]["t:state"] == {"subject": "open"}
    assert out["applied"]["d:flag"] == "identifier"


def test_a_decided_policy_entry_is_digested_unless_opened_and_a_marker_passes() -> None:
    record = decide_sync(loaded_with(send), {"subject": "hi"}, {}, RunSettings()).record
    out = view(record)
    assert out["log"][at(record, "d:act")]["value"] == digest("send", out)
    assert view(record, policy={"d:act": "open"})["log"][at(record, "d:act")]["value"] == "send"

    undecided = decide_sync(loaded_with(requiring), {"subject": "hi"}, {}).record
    policy_entry = undecided.log[at(undecided, "d:act")]
    assert policy_entry["marker"]["kind"] == "Unavailable"
    assert view(undecided)["log"][at(undecided, "d:act")] == policy_entry
