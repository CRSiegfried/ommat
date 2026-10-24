"""Format 5's provenance profile: what interpret writes on the call item and in its ``calls``
entry, what it refuses, how the ledger reads each format, how ``call_result`` folds a call's
requests, the conformance suite's ``provenance`` check, and the redacted view of the new
members."""

import copy
import enum
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, cast

import pytest
import rfc8785

from counting import SmallBounds
from ledger_logs import before_profile, rehashed
from ommat import (
    CallResult,
    Design,
    InvalidRecord,
    Plan,
    Policy,
    Question,
    ReadView,
    Record,
    Source,
    SourceCall,
    Text,
    decide_sync,
    from_record,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._provenance import DETAILS
from ommat.bindings import RequestProvenance, call_result
from ommat.bindings.conformance import check
from test_views import key_of

Batch = tuple[SourceCall, ...]


def plan(r: ReadView) -> Plan:
    return Plan("inform", reply="asked")


DESIGN = Design(
    name="provenance",
    state=t.record(text=t.string),
    sources={"a": Source({"q:a/yes": Question("noul", Text("Is it so?"))}, oracle="o")},
    policy=Policy(plan, version="1", reads=[]),
)
LOADED = load(DESIGN)
YES = {"yes": 0.8, "no": 0.2}
ROOT = b"k" * 32


def digest(value: Any, record: Record) -> str:
    """A keyed digest computed apart from the code under test."""
    mac = hmac.new(key_of(record.record_hash, ROOT), rfc8785.dumps(value), hashlib.sha256)
    return "hmac-sha256:" + mac.hexdigest()


PROFILE: dict[str, Any] = {
    "provider": "gliner",
    "model_requested": "fastino/gliner2.5-small-v1",
    "model_reported": "fastino/gliner2.5-small-v1",
    "revision": "7132dc4",
    "runtime": {"python": "3.11.9", "torch": "2.5.1"},
    "elicitation": "passthrough",
    "precision": "unrounded",
    "prompt": "the prompt",
    "claimed": ("revision", "model_reported"),
    "usage": {"input_tokens": 12},
    "attempts": 1,
    "request_id": None,
    "calibrated": False,
    "missing": {
        "request_id": "not-reported",
        "usage.output_tokens": "not-applicable",
        "usage.passes": "not-reported",
    },
    "clients": {"huggingface_hub": "0.26.0", "idia": "0.1.0"},
}


def returning(**members: Any) -> Any:
    async def binding(batch: Batch, *, seconds: float) -> object:
        return CallResult({"q:a/yes": YES}, **members)

    return binding


def turn(**members: Any) -> Record:
    return decide_sync(LOADED, {"text": "a"}, {"o": returning(**members)}).record


def call_item(record: Record) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "call")


def calls_entry(record: Record) -> dict[str, Any]:
    (entry,) = record.unhashed["calls"].values()
    return entry


# ---------------------------------------------------------------------------------------------
# What interpret writes


def test_a_call_item_holds_the_profiles_hashed_details_and_its_claims_sorted() -> None:
    record = turn(**PROFILE)
    call = call_item(record)
    for name in ("provider", "model_requested", "model_reported", "revision", "elicitation"):
        assert call[name] == PROFILE[name]
    assert call["runtime"] == PROFILE["runtime"]
    assert call["precision"] == "unrounded"
    assert call["claimed"] == ["model_reported", "revision"]
    assert "snapshot" not in call
    entry = calls_entry(record)
    assert entry["calibrated"] is False
    assert entry["missing"] == PROFILE["missing"]
    assert entry["clients"] == PROFILE["clients"]
    assert "request_id" not in entry  # null is not written
    assert read_record(record.document).log == record.log


def test_a_call_that_sent_no_request_holds_no_provenance() -> None:
    call = call_item(turn())
    for name in ("provider", "model_requested", "model_reported", "revision", "runtime"):
        assert call[name] is None
    assert (call["precision"], call["claimed"]) == (None, [])
    entry = calls_entry(turn())
    assert not {"request_id", "calibrated", "missing", "clients"} & set(entry)


@pytest.mark.parametrize(
    "precision", [0, 17, "unrounded", {"step": "0.05"}, {"step": "0.25"}, {"step": "0.5"}]
)
def test_each_form_of_precision_is_written(precision: object) -> None:
    assert call_item(turn(precision=precision))["precision"] == precision


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("provider", 3),
        ("revision", "\ud800"),
        ("runtime", {}),
        ("runtime", {"torch": 2}),
        ("precision", "Unrounded"),
        ("precision", {"step": "0.1"}),
        ("precision", {"step": "0.050"}),
        ("precision", {"step": 0.05}),
        ("precision", {"step": "0.05", "of": "p"}),
        ("precision", 18),
        ("claimed", ("nothing",)),
        ("claimed", ("provider",)),  # names a detail that holds no value
        ("claimed", ["revision"]),
    ],
)
def test_a_hashed_detail_off_its_form_fails_the_call(member: str, value: object) -> None:
    members: dict[str, Any] = {"revision": "r1", member: value}
    record = turn(**members)
    call = call_item(record)
    assert call["outcome"] == {"failed": {"code": "invalid-return", "refs": ["q:a/yes"]}}
    assert (call["revision"], call["claimed"]) == (None, [])
    assert member in calls_entry(record)["invalid"]


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("request_id", 3),
        ("calibrated", 1),
        ("clients", {}),
        ("missing", {"snapshot": "not-pinned"}),
        ("missing", {"provider": "lost"}),
        ("missing", {"revision": "not-pinned"}),  # the revision holds a value
        ("missing", {"usage.input_tokens": "not-reported"}),  # so does that figure
    ],
)
def test_an_unhashed_detail_off_its_form_fails_nothing(member: str, value: object) -> None:
    record = turn(revision="r1", usage={"input_tokens": 3}, **{member: value})
    assert call_item(record)["outcome"] == "returned"
    entry = calls_entry(record)
    assert member not in entry
    assert member in entry["invalid"]


def test_an_empty_missing_is_neither_written_nor_invalid() -> None:
    entry = calls_entry(turn(missing={}))
    assert "missing" not in entry
    assert "invalid" not in entry


class Name(enum.StrEnum):
    gliner = "gliner"


def test_a_string_subclass_detail_is_written_by_its_string_data() -> None:
    call = call_item(turn(provider=Name.gliner, runtime={Name.gliner: "1"}))
    assert type(call["provider"]) is str
    assert call["provider"] == "gliner"
    assert call["runtime"] == {"gliner": "1"}


def test_a_claim_on_a_malformed_unhashed_detail_fails_nothing() -> None:
    record = turn(request_id=3, claimed=("request_id",), usage=[1])
    assert call_item(record)["outcome"] == "returned"
    assert set(calls_entry(record)["invalid"]) == {"request_id", "usage"}


def test_an_unexpected_key_outranks_a_bad_provenance_detail() -> None:
    async def both(batch: Batch, *, seconds: float) -> object:
        return CallResult({"q:a/yes": YES, "t:nope": 1}, provider=cast(Any, 3))

    record = decide_sync(LOADED, {"text": "a"}, {"o": both}).record
    assert call_item(record)["outcome"]["failed"]["code"] == "unexpected-ref"


def test_a_failed_call_writes_no_missing() -> None:
    async def faulted(batch: Batch, *, seconds: float) -> object:
        return CallResult({"q:a/yes": YES, "t:nope": 1}, missing={"revision": "not-pinned"})

    record = decide_sync(LOADED, {"text": "a"}, {"o": faulted}).record
    assert call_item(record)["outcome"]["failed"]["code"] == "unexpected-ref"
    assert "missing" not in calls_entry(record)


# ---------------------------------------------------------------------------------------------
# The ledger, format by format


def doc_of(record: Record) -> dict[str, Any]:
    doc = json.loads(record.document)
    del doc["hash"]
    return doc


def refused(doc: dict[str, Any], says: str) -> None:
    with pytest.raises(InvalidRecord, match=says):
        read_record(rehashed(doc))


def test_a_format_5_call_item_holding_snapshot_or_lacking_a_profile_member_is_refused() -> None:
    doc = doc_of(turn(**PROFILE))
    with_snapshot = copy.deepcopy(doc)
    with_snapshot["log"][3]["snapshot"] = None
    refused(with_snapshot, "holds snapshot, which a format-5 record does not")
    for name in ("provider", "revision", "runtime", "claimed"):
        lacking = copy.deepcopy(doc)
        del lacking["log"][3][name]
        refused(lacking, f"has no '{name}' member")


def test_a_format_4_record_holds_snapshot_and_none_of_the_profile() -> None:
    doc = doc_of(turn(**PROFILE))
    old = before_profile(copy.deepcopy(doc), 4)
    assert read_record(rehashed(old)).log[3]["snapshot"] == "7132dc4"
    holding = copy.deepcopy(old)
    holding["log"][3]["provider"] = "gliner"
    refused(holding, "holds provider, which a format-4 record does not")
    unrounded = copy.deepcopy(old)
    unrounded["log"][3]["precision"] = "unrounded"
    refused(unrounded, "precision is not an integer or null, which a format-4 record's is")
    for name, value in (("request_id", "r"), ("calibrated", True), ("clients", {"idia": "0"})):
        entry = copy.deepcopy(old)
        next(iter(entry["unhashed"]["calls"].values()))[name] = value
        refused(entry, f"holds {name}, which a format-4 record does not")


def test_a_format_4_calls_entry_holds_none_of_the_new_members_as_a_key_or_an_invalid_key() -> None:
    old = before_profile(doc_of(turn(**PROFILE)), 4)
    for name in ("missing", "request_id", "provider", "runtime"):
        changed = copy.deepcopy(old)
        entry = next(iter(changed["unhashed"]["calls"].values()))
        if name == "missing":
            entry["missing"] = {"revision": "not-pinned"}
        else:
            entry["invalid"] = {name: "3"}
        refused(changed, f"holds {name}, which a format-4 record does not")
    current = doc_of(turn(**PROFILE))
    entry = next(iter(current["unhashed"]["calls"].values()))
    entry["invalid"] = {"revision": "3", "claimed": "()"}
    read_record(rehashed(current))


def test_a_format_3_or_4_precision_is_read_by_its_json_kind_alone() -> None:
    for fmt in (3, 4):
        old = before_profile(doc_of(turn(precision=2)), fmt)
        old["log"][3]["precision"] = 18  # past 17, but an integer as the format-3 rules read
        assert read_record(rehashed(old)).log[3]["precision"] == 18


def test_a_request_id_or_clients_past_the_size_bounds_is_refused_at_the_read(
    small_bounds: SmallBounds,
) -> None:
    for name, big in (("request_id", "x" * 300), ("clients", {f"p{i}": "1" for i in range(70)})):
        doc = doc_of(turn(**PROFILE))
        next(iter(doc["unhashed"]["calls"].values()))[name] = big
        with pytest.raises(InvalidRecord, match=name):
            read_record(rehashed(doc))


def test_a_calls_entrys_missing_is_read_against_the_profiles_lists() -> None:
    doc = doc_of(turn(**PROFILE))
    for missing, says in (
        ({"snapshot": "not-pinned"}, "names no detail"),
        ({"revision": "lost"}, "is not a reason of the provenance profile"),
        ({}, "is empty"),
    ):
        changed = copy.deepcopy(doc)
        next(iter(changed["unhashed"]["calls"].values()))["missing"] = missing
        refused(changed, says)
    with_invalid = copy.deepcopy(doc)
    next(iter(with_invalid["unhashed"]["calls"].values()))["invalid"] = {"snapshot": "3"}
    refused(with_invalid, "holds snapshot, which a format-5 record does not")


# ---------------------------------------------------------------------------------------------
# call_result's fold


def sent(**members: Any) -> RequestProvenance:
    return RequestProvenance(attempts=1, **members)


def test_the_fold_reads_only_requests_that_were_sent() -> None:
    result = call_result(
        {}, [RequestProvenance(provider="p", missing={"revision": "not-sent"})], precision=2
    )
    assert (result.provider, result.precision, result.missing, result.attempts) == (
        None,
        None,
        None,
        None,
    )


def test_the_fold_merges_runtimes_and_refuses_a_package_at_two_versions() -> None:
    result = call_result(
        {},
        [sent(runtime={"torch": "2.5"}, clients={"idia": "0.1"}), sent(runtime={"python": "3.11"})],
        precision=None,
    )
    assert result.runtime == {"torch": "2.5", "python": "3.11"}
    assert result.clients == {"idia": "0.1"}
    with pytest.raises(ValueError, match="runtime names torch at two versions"):
        call_result(
            {}, [sent(runtime={"torch": "2.5"}), sent(runtime={"torch": "2.4"})], precision=0
        )


def test_the_fold_unions_claims_and_reads_calibrated_from_those_that_say() -> None:
    result = call_result(
        {},
        [sent(claimed=("revision",), calibrated=True), sent(claimed=("model_reported",))],
        precision=None,
    )
    assert result.claimed == ("model_reported", "revision")
    assert result.calibrated is True
    either = [sent(calibrated=True), sent(calibrated=False)]
    assert call_result({}, either, precision=None).calibrated is False
    assert call_result({}, [sent(), sent()], precision=None).calibrated is None


def test_a_detail_is_missing_only_when_every_sent_request_lacks_it() -> None:
    first = sent(missing={"revision": "not-sent", "prompt": "hidden", "request_id": "not-sent"})
    second = sent(revision="r", missing={"prompt": "not-sent", "request_id": "not-reported"})
    result = call_result({}, [first, second], precision=1)
    assert result.revision == "r"
    assert result.missing == {"prompt": "hidden", "request_id": "not-reported"}


def test_precisions_reason_comes_from_the_requests_and_goes_with_a_value() -> None:
    unknown = sent(missing={"precision": "not-exposed"})
    assert call_result({}, [unknown], precision=None).missing == {"precision": "not-exposed"}
    assert call_result({}, [unknown], precision=3).missing is None


# ---------------------------------------------------------------------------------------------
# The conformance suite's provenance check


ONE: list[Batch] = [
    (
        SourceCall(
            "a",
            None,
            {},
            ("t:a/n",),
            {},
        ),
    )
]


def answering(**members: Any) -> Any:
    async def binding(batch: Batch, *, seconds: float) -> object:
        values: dict[str, object] = {ref: 1 for call in batch for ref in call.refs}
        return CallResult(values, **members)

    return binding


def test_a_sent_call_owes_a_value_or_a_reason_for_each_detail() -> None:
    every = dict.fromkeys(sorted(DETAILS), "not-reported")
    assert check(answering(attempts=1, missing=every), batches=ONE) == []
    failures = check(answering(attempts=1), batches=ONE)
    assert "provenance: batch 0: revision holds no value and no reason" in failures
    assert len(failures) == len(DETAILS)
    both = {**every}
    del both["provider"]
    both["revision"] = "not-pinned"
    assert check(answering(attempts=1, provider="p", revision="r", missing=both), batches=ONE) == [
        "provenance: batch 0: revision holds a value and a reason"
    ]


def test_a_claim_on_a_null_detail_and_a_reason_off_the_list_fail_the_check() -> None:
    every = dict.fromkeys(sorted(DETAILS), "not-reported")
    assert check(answering(attempts=1, missing=every, claimed=("revision",)), batches=ONE) == [
        "provenance: batch 0: claimed names 'revision', no detail holding a value"
    ]
    odd = {**every, "revision": "lost"}
    assert check(answering(attempts=1, missing=odd), batches=ONE) == [
        "provenance: batch 0: missing gives revision 'lost', no reason"
    ]


def test_a_call_that_sent_nothing_owes_no_reason() -> None:
    assert check(answering(), batches=ONE) == []


def test_a_binding_that_gives_a_new_request_id_each_call_is_still_deterministic() -> None:
    every: dict[str, str] = dict.fromkeys(sorted(DETAILS - {"request_id"}), "not-reported")
    counter = iter(range(100))

    async def fresh(batch: Batch, *, seconds: float) -> object:
        values: dict[str, object] = {ref: 1 for call in batch for ref in call.refs}
        return CallResult(values, attempts=1, request_id=f"r-{next(counter)}", missing=every)

    assert check(fresh, batches=ONE, deterministic=True) == []


# ---------------------------------------------------------------------------------------------
# The redacted view


def test_a_request_id_is_digested_and_the_other_new_members_are_open() -> None:
    missing = {k: v for k, v in PROFILE["missing"].items() if k != "request_id"}
    record = turn(**{**PROFILE, "request_id": "caller-case-42", "missing": missing})
    view = json.loads(redact(record, ROOT))
    entry = next(iter(view["unhashed"]["calls"].values()))
    assert entry["request_id"] == digest("caller-case-42", record)
    assert (entry["calibrated"], entry["missing"], entry["clients"]) == (
        False,
        missing,
        PROFILE["clients"],
    )
    call = next(i for i in view["log"] if i["item"] == "call")
    assert call["prompt"] == digest("the prompt", record)
    for name in ("provider", "model_requested", "model_reported", "revision", "elicitation"):
        assert call[name] == PROFILE[name]
    assert (call["runtime"], call["precision"], call["claimed"]) == (
        PROFILE["runtime"],
        "unrounded",
        ["model_reported", "revision"],
    )


def test_a_format_4_record_redacts_with_its_snapshot_open_and_replays_at_format_5() -> None:
    old = read_record(rehashed(before_profile(doc_of(turn(**PROFILE)), 4)))
    view = json.loads(redact(old, ROOT))
    call = next(i for i in view["log"] if i["item"] == "call")
    assert call["snapshot"] == "7132dc4"
    rerun, findings = replay_sync(old, LOADED)
    assert findings == []
    assert rerun is not None
    assert rerun.log[0]["format"] == 5
    assert call_item(rerun)["revision"] is None  # replay calls no model


def test_from_record_copies_none_of_the_stored_provenance() -> None:
    stored = turn(**PROFILE)
    again = decide_sync(LOADED, {"text": "a"}, {"o": from_record(stored, LOADED)}).record
    call = call_item(again)
    for name in ("provider", "model_requested", "model_reported", "revision", "runtime"):
        assert call[name] is None
    assert (call["precision"], call["claimed"]) == (None, [])
    assert not {"request_id", "calibrated", "missing", "clients"} & set(calls_entry(again))


# ---------------------------------------------------------------------------------------------
# A stored format-4 record, as the format-4 engine wrote it

FORMAT_4 = Path(__file__).parent / "format_4"
FORMAT_4_DESIGN = "sha256:2ebb72436429d162a946568d00f92217e402d348777dcf5e00d30f8545eeb451"
FORMAT_4_RECORD = "sha256:2cb40daf65396a4572086299bc284dd35c4911f28b47e4347254b4658170ee69"


def test_a_stored_format_4_record_reads_to_its_own_hash_and_keeps_its_snapshot() -> None:
    data = (FORMAT_4 / "record.json").read_bytes()
    record = read_record(data)
    assert record.document == data
    assert record.record_hash == FORMAT_4_RECORD
    call = call_item(record)
    assert (call["snapshot"], call["precision"]) == ("m-1", 2)
    assert "revision" not in call
    view = json.loads(redact(record, ROOT))
    assert call_item_of(view)["snapshot"] == "m-1"


def test_a_stored_format_4_design_loads_to_its_own_hash_and_its_record_replays() -> None:
    data = (FORMAT_4 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_4_DESIGN
    assert loaded.document == data
    rerun, findings = replay_sync(read_record((FORMAT_4 / "record.json").read_bytes()), loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.log[0] == {"item": "design", "hash": FORMAT_4_DESIGN, "format": 5, "helpers": 2}


def call_item_of(view: dict[str, Any]) -> dict[str, Any]:
    return next(i for i in view["log"] if i["item"] == "call")
