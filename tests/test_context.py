"""The context block, the handoff view and the conversation renderers: every switch on and off,
a marker entry, a ``t:state`` path and a parameter slot, replay byte for byte, ``redact``, the
escalation unchanged, the load checks, the ledger's checks at the read, and the async observer
refused at entry."""

# pyright: reportPrivateUsage=false

import asyncio
import copy
import functools
import json
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from ommat import (
    Binding,
    Context,
    ContextEntry,
    Design,
    EntryRefusal,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Origin,
    Parameter,
    Plan,
    Policy,
    Question,
    Record,
    Requirement,
    Rule,
    Source,
    Step,
    Structured,
    Text,
    decide_sync,
    escalation,
    handoff,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import decode, encode
from ommat.conversation import handoff_text
from ommat.evaluation._diff import _context_alone
from ommat.helpers import pick_with_margin
from scripted import Scripted, returns

STATE_T = t.record(
    customer=t.string,
    account=t.string,
    order=t.record(id=t.string, status=t.string),
)
STATE: dict[str, Any] = {
    "customer": "I want my money back for order 42",
    "account": "acct-991",
    "order": {"id": "42", "status": "delivered"},
}
ASK = Source(
    {"q:refund": Question("noul", Text("Is this a refund request?"))},
    needs=["t:state.customer"],
)


def refund(r: Any) -> Any:
    return pick_with_margin(r, "q:refund", r.param("refund/bar"), 0.1)


def status(r: Any) -> Any:
    return r.read("t:state.order.status")


RULES = {
    "refund": Rule(
        "d:refund",
        t.label("no", "yes"),
        refund,
        version="1",
        reads=["q:refund"],
        params=["refund/bar"],
    ),
    "status": Rule("d:status", t.string, status, version="1", reads=["t:state.order.status"]),
}


def policy(r: Any) -> Plan:
    decided = r.read("d:refund")
    if decided == "yes":
        return Plan(
            "execute",
            [
                Step(
                    "refund_order",
                    {"order": Origin(ref="t:state.order.id"), "note": Origin(literal="asked")},
                    [Requirement("d:refund", "yes")],
                ),
                Step(
                    "notify", {"receipt": Binding(1, "receipt")}, [Requirement("d:refund", "yes")]
                ),
            ],
            reason="the customer asked for a refund",
        )
    if decided == "no":
        return Plan("inform", reply="No refund was asked for.")
    return Plan("escalate", reason="unsure whether a refund was asked for")


def slot(name: str, **origin: str) -> dict[str, Any]:
    return {"name": name, "origin": origin}


ENTRIES = [
    ContextEntry("d:refund", Text("Refund requested: {r}", [slot("r", ref="d:refund")])),
    ContextEntry(
        "t:state.order.status",
        Text(
            "Order {id} is {s}",
            [slot("id", ref="t:state.order.id"), slot("s", ref="t:state.order.status")],
        ),
    ),
    ContextEntry(
        "d:status",
        Text("Status {s} at bar {b}", [slot("s", ref="d:status"), slot("b", param="refund/bar")]),
    ),
]


def design(**changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "refunds",
        "state": STATE_T,
        "sources": {"ask": ASK},
        "rules": RULES,
        "policy": Policy(policy, version="1", reads=["d:refund", "t:state.order.id"]),
        "tools": ["refund_order", "notify"],
        "parameters": {"refund/bar": Parameter("number [0, 1]", 0.6)},
        "classes": {"t:state": {"account": "identifier"}},
        "context": Context(ENTRIES),
    }
    fields.update(changes)
    return Design(**fields)


def loaded(**changes: Any) -> LoadedDesign:
    return load(design(**changes))


YES = {"q:refund": {"yes": 0.9, "no": 0.1}}
NO = {"q:refund": {"yes": 0.1, "no": 0.9}}
UNSURE = {"q:refund": {"yes": 0.55, "no": 0.45}}


def run(answers: Mapping[str, Any] = YES, under: LoadedDesign | None = None, **kw: Any) -> Any:
    binding = Scripted({"ask": returns(dict(answers))})
    return decide_sync(under or loaded(), STATE, {"default": binding}, **kw)


def plan_item(record: Any) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "plan")


# ---------------------------------------------------------------------------------------------
# Rendering into the plan item


def test_compile_renders_each_entry_into_the_plan_item_in_declared_order() -> None:
    context = plan_item(run().record)["context"]
    assert context == {
        "design": "refunds",
        "entries": [
            {"ref": "d:refund", "text": "Refund requested: yes", "marker": None},
            {"ref": "t:state.order.status", "text": "Order 42 is delivered", "marker": None},
            {"ref": "d:status", "text": "Status delivered at bar 0.6", "marker": None},
        ],
        "plan": True,
        "steps": True,
        "record": True,
    }


def test_a_marker_in_the_flagged_ref_makes_a_marker_entry_with_no_text() -> None:
    entry = plan_item(run(UNSURE).record)["context"]["entries"][0]
    assert entry == {
        "ref": "d:refund",
        "text": None,
        "marker": {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["q:refund"]}},
    }


def test_a_marker_in_another_slot_makes_the_entry_a_marker_entry() -> None:
    entry = ContextEntry(
        "t:state.order.id",
        Text("Order {id}: {r}", [slot("id", ref="t:state.order.id"), slot("r", ref="d:refund")]),
    )
    context = plan_item(run(UNSURE, loaded(context=Context([entry]))).record)["context"]
    assert context["entries"][0]["text"] is None
    assert context["entries"][0]["marker"]["kind"] == "Unsure"


def test_no_block_writes_a_null_context_and_handoff_returns_none() -> None:
    record = run(under=loaded(context=None)).record
    assert plan_item(record)["context"] is None
    assert handoff(record) is None


def test_an_empty_block_hands_off_the_plan_and_record_with_no_entries() -> None:
    record = run(under=loaded(context=Context())).record
    view = handoff(record)
    assert view is not None
    assert view["context"] == []
    assert view["plan"]["kind"] == "execute"


def test_a_broken_entry_writes_the_block_with_no_entries() -> None:
    def codec(state: object) -> object:
        return {"wrong": True}

    record = run(under=loaded(), codec=codec).record
    assert plan_item(record)["kind"] == "broken"
    assert plan_item(record)["context"]["entries"] == []
    view = handoff(record)
    assert view is not None
    assert view["context"] == []
    assert view["plan"] == {
        "kind": "broken",
        "reply": None,
        "reason": None,
        "verdict": "passed",
        "steps": [],
    }


def test_the_context_is_rendered_before_the_policy_runs() -> None:
    def changing(r: Any) -> Plan:
        state = r.read("t:state.order")
        state["status"] = "changed by the policy"  # a convention broken, not a rule
        return Plan("inform", reply="ok")

    record = run(under=loaded(policy=Policy(changing, version="1", reads=["t:state.order"]))).record
    assert plan_item(record)["context"]["entries"][1]["text"] == "Order 42 is delivered"


# ---------------------------------------------------------------------------------------------
# The handoff view and its switches


def test_the_handoff_holds_the_design_the_plan_its_tagged_steps_and_the_entries() -> None:
    record = run().record
    view = handoff(record)
    assert view is not None
    assert list(view) == ["design", "record_hash", "plan", "context"]
    assert view["design"] == {"name": "refunds", "hash": record.design_hash}
    assert view["record_hash"] == record.record_hash
    assert view["plan"] == {
        "kind": "execute",
        "reply": None,
        "reason": "the customer asked for a refund",
        "verdict": "passed",
        "steps": [
            {
                "tool": "refund_order",
                "arguments": {
                    "note": {"value": "asked"},
                    "order": {"ref": "t:state.order.id", "value": "42"},
                },
            },
            {"tool": "notify", "arguments": {"receipt": {"step": 1, "path": "receipt"}}},
        ],
    }
    assert view["context"] == plan_item(record)["context"]["entries"]
    text = json.dumps(view)
    for absent in ("requirement", "failures", "probabilit", "0.9", "acct-991", "helpers"):
        assert absent not in text


@pytest.mark.parametrize(
    ("plan", "steps", "record", "keys", "plan_keys"),
    [
        (
            True,
            True,
            True,
            ["design", "record_hash", "plan", "context"],
            {"kind", "reply", "reason", "verdict", "steps"},
        ),
        (False, True, True, ["design", "record_hash", "plan", "context"], {"steps"}),
        (
            True,
            False,
            True,
            ["design", "record_hash", "plan", "context"],
            {"kind", "reply", "reason", "verdict"},
        ),
        (True, True, False, ["plan", "context"], {"kind", "reply", "reason", "verdict", "steps"}),
        (False, False, True, ["design", "record_hash", "context"], None),
        (False, False, False, ["context"], None),
    ],
)
def test_each_switch_turns_its_members_off(
    plan: bool, steps: bool, record: bool, keys: list[str], plan_keys: set[str] | None
) -> None:
    block = Context(ENTRIES, plan=plan, steps=steps, record=record)
    rec = run(under=loaded(context=block)).record
    view = handoff(rec)
    assert view is not None
    assert list(view) == keys
    assert (set(view["plan"]) if "plan" in view else None) == plan_keys
    assert plan_item(rec)["context"]["plan"] is plan
    assert len(view["context"]) == 3


def test_a_non_execute_plan_gives_no_steps_as_an_empty_list() -> None:
    view = handoff(run(NO).record)
    assert view is not None
    assert view["plan"] == {
        "kind": "inform",
        "reply": "No refund was asked for.",
        "reason": None,
        "verdict": "passed",
        "steps": [],
    }


def test_a_refused_verdict_keeps_the_kind_and_drops_the_steps() -> None:
    ld = loaded(requirements=[Requirement("d:status", "open")])
    decision = run(under=ld)
    assert decision.verdict.kind == "refused"
    view = handoff(decision.record)
    assert view is not None
    assert view["plan"]["kind"] == "execute"
    assert "steps" not in view["plan"]
    assert view["plan"]["verdict"] == "refused"
    assert "failures" not in json.dumps(view)
    only_steps = run(
        under=loaded(
            requirements=[Requirement("d:status", "open")], context=Context(ENTRIES, plan=False)
        )
    )
    assert "plan" not in handoff(only_steps.record)  # type: ignore[operator]


def test_a_record_with_no_plan_item_hands_off_nothing() -> None:
    def cancel(r: Any) -> Plan:
        raise asyncio.CancelledError

    ld = loaded(policy=Policy(cancel, version="1", reads=["d:refund"]))
    binding = Scripted({"ask": returns(dict(YES))})
    with pytest.raises(asyncio.CancelledError) as caught:
        decide_sync(ld, STATE, {"default": binding})
    record = cast("Record", getattr(caught.value, "record"))  # noqa: B009
    assert handoff(record) is None


def test_handoff_refuses_what_is_not_a_record() -> None:
    with pytest.raises(ValueError, match="takes a Record"):
        handoff({"log": []})  # type: ignore[arg-type]


def test_the_escalation_is_unchanged_by_the_block() -> None:
    with_block = escalation(run(UNSURE).record)
    without = escalation(run(UNSURE, loaded(context=None)).record)
    assert with_block is not None
    assert without is not None
    assert {k: v for k, v in with_block.items() if k != "record_hash"} == {
        k: v for k, v in without.items() if k != "record_hash"
    }


# ---------------------------------------------------------------------------------------------
# Replay, the record and redact


def test_replay_reproduces_the_plan_item_byte_for_byte() -> None:
    ld = loaded()
    record = run(under=ld).record
    again, findings = replay_sync(record, ld)
    assert findings == []
    assert again is not None
    assert encode(plan_item(again)) == encode(plan_item(record))
    assert read_record(record.document).log == record.log


def test_redact_digests_the_text_and_the_design_name_and_keeps_the_rest_open() -> None:
    record = run(UNSURE).record
    view = decode(redact(record, bytes(32)))
    context = next(i for i in view["log"] if i["item"] == "plan")["context"]
    assert context["design"].startswith("hmac-sha256:")
    texts = [e["text"] for e in context["entries"]]
    assert texts[0] is None
    assert all(x.startswith("hmac-sha256:") for x in texts[1:])
    assert [e["ref"] for e in context["entries"]] == [
        "d:refund",
        "t:state.order.status",
        "d:status",
    ]
    assert context["entries"][0]["marker"]["kind"] == "Unsure"
    assert (context["plan"], context["steps"], context["record"]) == (True, True, True)


def test_a_context_text_change_alone_is_no_plan_change_to_do_and_diff() -> None:
    mine = plan_item(run().record)
    theirs = copy.deepcopy(mine)
    theirs["context"]["entries"][0]["text"] = "something else"
    assert _context_alone({"kind": FindingKind.plan, "recorded": mine, "replayed": theirs})
    theirs["reason"] = "another"
    assert not _context_alone({"kind": FindingKind.plan, "recorded": mine, "replayed": theirs})


def _edited(record: Any, change: Any) -> bytes:
    """The record's document with its plan item changed, rehashed so only the ledger's
    shape checks can refuse it."""
    from ledger_logs import rehashed

    doc = json.loads(record.document)
    change(next(i for i in doc["log"] if i["item"] == "plan"))
    return json.dumps(rehashed(doc)).encode()


Change = Callable[[dict[str, Any]], None]


def _drop_context(item: dict[str, Any]) -> None:
    item.pop("context")


def _entry(**members: Any) -> Change:
    def change(item: dict[str, Any]) -> None:
        item["context"]["entries"][0].update(members)

    return change


def _block(**members: Any) -> Change:
    def change(item: dict[str, Any]) -> None:
        item["context"].update(members)

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_drop_context, "has no 'context' member"),
        (_entry(marker=None, text=None), "neither or both"),
        (_entry(ref="not a ref"), "is not a ref"),
        (_entry(ref="d:nobody"), "is not t:state or a d: ref"),
        (_entry(ref="t:state.nofield"), "names a field"),
        (_block(plan="yes"), "boolean"),
        (_block(design=3), "a string or null"),
    ],
)
def test_the_ledger_refuses_a_misfit_context_at_the_read(change: Change, message: str) -> None:
    record = run().record
    with pytest.raises(InvalidRecord, match=message):
        read_record(_edited(record, change))


def test_a_record_before_format_5_holding_a_context_is_refused() -> None:
    from ledger_logs import before_profile, rehashed

    doc = json.loads(run().record.document)
    before_profile(doc, 4)
    assert read_record(rehashed(doc)).log[0]["format"] == 4
    next(i for i in doc["log"] if i["item"] == "plan")["context"] = None
    with pytest.raises(InvalidRecord, match="holds context, which a format-4 record does not"):
        read_record(rehashed(doc))


# ---------------------------------------------------------------------------------------------
# Load


def failures(**changes: Any) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as caught:
        load(design(**changes))
    return [(f.check, f.message) for f in caught.value.failures]


def one(ref: str, text: Text | Any) -> Context:
    return Context([ContextEntry(ref, text)])


def test_the_document_holds_the_block_and_loads_back_to_its_own_hash() -> None:
    ld = loaded()
    doc = decode(ld.document)
    assert doc["format"] == 5
    assert doc["context"]["entries"][0] == {
        "ref": "d:refund",
        "text": {"template": "Refund requested: {r}", "slots": [slot("r", ref="d:refund")]},
    }
    again = load(ld.document, functions={"policy": policy, "refund": refund, "status": status})
    assert again.design_hash == ld.design_hash


def test_a_design_before_format_5_with_a_context_member_is_refused_at_check_11() -> None:
    doc = decode(loaded().document)
    doc["format"] = 4
    with pytest.raises(InvalidDesign, match='has the member "context"'):
        load(doc, functions={"policy": policy, "refund": refund, "status": status})


def test_two_entries_flagging_one_ref_are_refused_at_check_1() -> None:
    entry = ENTRIES[0]
    assert (1, 'two context entries flag "d:refund"') in failures(context=Context([entry, entry]))


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        (
            "q:refund",
            'the context entry for "q:refund" reads "q:refund", which is not at level d: or t:',
        ),
        (
            "t:other",
            'the context entry for "t:other" flags "t:other", which is not t:state or a d: ref',
        ),
    ],
)
def test_a_flagged_ref_of_another_kind_is_refused_at_check_4(ref: str, message: str) -> None:
    assert (4, message) in failures(context=one(ref, Text("{x}!", [slot("x", ref=ref)])))


def test_a_candidate_slot_is_refused_at_check_4() -> None:
    text = Text("{x} {c}", [slot("x", ref="d:refund"), slot("c", candidate="d:refund")])
    found = failures(context=one("d:refund", text))
    assert any(c == 4 and "has a candidate slot" in m for c, m in found)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (Text("Status {s}", [slot("s", ref="d:status")]), 'has no slot naming "d:refund"'),
        (Text("{r}", [slot("r", ref="d:refund")]), "no character that is not whitespace"),
        (Text("Refund {r} {x}", [slot("r", ref="d:refund")]), 'placeholder "x" names no slot'),
    ],
)
def test_a_text_is_checked_at_check_9(text: Text, message: str) -> None:
    found = failures(context=one("d:refund", text))
    assert any(c == 9 and message in m for c, m in found), found


def test_a_path_naming_no_field_is_refused_at_check_9() -> None:
    text = Text("Order {x}", [slot("x", ref="t:state.order.nofield")])
    found = failures(context=one("t:state.order.nofield", text))
    assert any(c == 9 for c, _ in found), found


@pytest.mark.parametrize("ref", ["t:state.account", "t:state"])
def test_a_slot_over_an_identifier_tree_is_refused_at_check_9(ref: str) -> None:
    flagged = Text("Account {a} {r}", [slot("a", ref=ref), slot("r", ref="d:refund")])
    found = failures(context=one("d:refund", flagged))
    shown = (
        f'the context entry for "d:refund"\'s text names "{ref}", whose class tree holds identifier'
    )
    assert (9, shown) in found


def test_a_d_ref_with_a_field_path_is_allowed() -> None:
    order = Rule("d:order", t.record(id=t.string), lambda r: {"id": "42"}, version="1")
    text = Text("Order {i}", [slot("i", ref="d:order.id")])
    rules = {**RULES, "order": order}
    record = run(under=loaded(rules=rules, context=one("d:order.id", text))).record
    assert plan_item(record)["context"]["entries"][0]["text"] == "Order 42"


def test_a_param_slot_names_a_declared_leaf_at_check_8() -> None:
    text = Text("Refund {r} {p}", [slot("r", ref="d:refund"), slot("p", param="no/leaf")])
    found = failures(context=one("d:refund", text))
    assert any(c == 8 and '"no/leaf"' in m for c, m in found)


def test_a_slot_with_no_provider_is_refused_at_check_3() -> None:
    text = Text("Refund {r} {n}", [slot("r", ref="d:refund"), slot("n", ref="d:nobody")])
    found = failures(context=one("d:refund", text))
    assert any(c == 3 and '"d:nobody"' in m for c, m in found)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        ("not a block", "the context block is not a Context"),
        (Context(["no entry"]), "holds an entry that is not a ContextEntry"),  # type: ignore[list-item]
        (one("d:refund", Structured({"a": "{r}"}, [slot("r", ref="d:refund")])), "is not a Text"),
        (Context(ENTRIES, plan="yes"), "plan switch is not a boolean"),  # type: ignore[arg-type]
    ],
)
def test_check_10_tests_the_block_s_form(context: Any, message: str) -> None:
    found = failures(context=context)
    assert any(c == 10 and message in m for c, m in found), found


# ---------------------------------------------------------------------------------------------
# The conversation renderers


def test_handoff_text_gives_the_view_s_lines_in_order() -> None:
    record = run().record
    assert handoff_text(handoff(record)).splitlines() == [
        "Plan: execute",
        "Verdict: passed",
        "Reason: the customer asked for a refund",
        'Step 1: refund_order {"note":{"value":"asked"},'
        '"order":{"ref":"t:state.order.id","value":"42"}}',
        'Step 2: notify {"receipt":{"path":"receipt","step":1}}',
        "Refund requested: yes",
        "Order 42 is delivered",
        "Status delivered at bar 0.6",
        f"Design: refunds {record.design_hash}",
        f"Record: {record.record_hash}",
    ]


def test_handoff_text_gives_a_marker_by_its_ref_kind_and_cause() -> None:
    assert "d:refund: Unsure (abstained)" in handoff_text(handoff(run(UNSURE).record)).splitlines()


def test_handoff_text_gives_a_refused_verdict_by_its_kind_and_no_steps() -> None:
    record = run(under=loaded(requirements=[Requirement("d:status", "open")])).record
    lines = handoff_text(handoff(record)).splitlines()
    assert lines[:3] == [
        "Plan: execute",
        "Verdict: refused",
        "Reason: the customer asked for a refund",
    ]
    assert not any(line.startswith("Step") for line in lines)


def test_handoff_text_gives_no_verdict_line_for_a_log_with_no_verdict() -> None:
    view: dict[str, Any] = {
        "plan": {"kind": "execute", "reply": None, "reason": None, "verdict": None},
        "context": [],
    }
    assert handoff_text(view) == "Plan: execute"


def test_handoff_text_is_empty_for_none_or_a_view_that_gives_no_line() -> None:
    assert handoff_text(None) == ""
    block = Context([], plan=False, steps=False, record=False)
    assert handoff_text(handoff(run(under=loaded(context=block)).record)) == ""


# ---------------------------------------------------------------------------------------------
# An async observer is refused at entry


async def async_observer(index: int, item: Any) -> None:
    return None


async def async_gen_observer(index: int, item: Any) -> Any:
    yield index


class AsyncCall:
    async def __call__(self, index: int, item: Any) -> None:
        return None

    async def method(self, index: int, item: Any) -> None:
        return None


class SyncCall:
    def __call__(self, index: int, item: Any) -> None:
        return None


@pytest.mark.parametrize(
    "observer",
    [
        async_observer,
        async_gen_observer,
        functools.partial(async_observer),
        functools.partial(functools.partial(async_observer)),
        AsyncCall(),
        AsyncCall().method,
    ],
)
def test_an_async_observer_is_refused_by_decide_and_replay(observer: Any) -> None:
    ld = loaded()
    with pytest.raises(EntryRefusal, match="async"):
        run(under=ld, observer=observer)
    record = run(under=ld).record
    with pytest.raises(EntryRefusal, match="async"):
        replay_sync(record, ld, observer=observer)


def sync_observer(index: int, item: Any) -> None:
    return None


@pytest.mark.parametrize("observer", [sync_observer, SyncCall()])
def test_a_sync_observer_is_accepted(observer: Any) -> None:
    assert run(observer=observer).plan.kind == "execute"
