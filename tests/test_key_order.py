# pyright: reportPrivateUsage=false
"""Key order (§key-order): a JSON object a model reads keeps the order it was declared or
returned in. A state record's fields reach a binding's needs, the board and a slot's text in
declared order; a record holds each entry, plan and verdict item's ``order``, reads it back and
replays it; the ordered encoding and the order object; design format 6's payload, parameter
and structured-template orders; the redacted view's ``order``; and ``from_record``'s need
comparison."""

import json
import random
from typing import Any, cast

import pytest

from designs import document_before_key_order
from ledger_logs import before_key_order, rehashed, reordered
from ommat import (
    CallResult,
    Design,
    FindingKind,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Parameter,
    Plan,
    Policy,
    Question,
    Record,
    Rule,
    Source,
    SourceCall,
    Structured,
    Text,
    decide_sync,
    from_record,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._canonical import (
    _CanonicalError,
    decode,
    encode,
    hash_value,
    order_of,
    ordered,
    with_order,
)

Batch = tuple[SourceCall, ...]

# Declared out of code-point order at both depths.
STATE_TYPE = t.record(zeta=t.string, alpha=t.integer(), mid=t.record(y=t.string, x=t.string))
STATE = {"zeta": "z", "alpha": 1, "mid": {"y": "why", "x": "ex"}}
DECLARED = ["zeta", "alpha", "mid"]


def inform(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def state_keys(r: Any) -> list[str]:
    """The board's state fields in the order the rule reads them."""
    state = r.read("t:state")
    return [*state, *state["mid"]]


def weight_keys(r: Any) -> list[str]:
    return list(r.read("t:weights"))


def param_keyed(r: Any) -> dict[str, int]:
    """A mapping whose key order is the parameter's: the design decides it."""
    return dict.fromkeys(r.param("keys"), 1)


FUNCTIONS = {"policy": inform, "state_keys": state_keys, "weight_keys": weight_keys}


def state_design() -> Design:
    ask = Question("noul", Text("Is {s} fine?", [{"name": "s", "origin": {"ref": "t:state"}}]))
    return Design(
        name="order",
        state=STATE_TYPE,
        sources={"ask": Source({"q:fine": ask}, needs=["t:state"])},
        rules={
            "state_keys": Rule(
                "d:state_keys", t.list_(t.string), state_keys, version="1", reads=["t:state"]
            )
        },
        policy=Policy(inform, version="1"),
    )


class Seen:
    """A binding answering every question yes and every t: ref from ``values``, keeping the
    batches it was given."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values = values or {}
        self.batches: list[Batch] = []

    async def __call__(self, batch: Batch, *, seconds: float) -> object:
        self.batches.append(batch)
        found: dict[str, Any] = {}
        for call in batch:
            for ref in call.questions:
                found[ref] = {"yes": 0.9, "no": 0.1}
            for ref in call.refs:
                found[ref] = self.values[ref]
        return CallResult(found)


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i.get("ref") == ref)


# ---------------------------------------------------------------------------------------------
# A state record declared out of code-point order


def test_a_state_record_reaches_needs_the_board_and_a_slot_in_declared_order() -> None:
    loaded = load(state_design())
    seen = Seen()
    # Given in another order: the turn holds it as declared.
    record = decide_sync(
        loaded, {"mid": {"x": "ex", "y": "why"}, "alpha": 1, "zeta": "z"}, {"default": seen}
    ).record
    (call,) = seen.batches[0]
    assert list(call.needs["t:state"]) == DECLARED
    assert list(call.needs["t:state"]["mid"]) == ["y", "x"]
    assert call.questions["q:fine"].wording == (
        'Is {"zeta":"z","alpha":1,"mid":{"y":"why","x":"ex"}} fine?'
    )
    assert entry(record, "d:state_keys")["value"] == [*DECLARED, "y", "x"]
    assert list(entry(record, "t:state")["value"]) == DECLARED


def test_a_state_record_keeps_declared_order_under_replay() -> None:
    loaded = load(state_design())
    record = decide_sync(loaded, STATE, {"default": Seen()}).record
    again, findings = replay_sync(read_record(record.document), loaded)
    assert findings == []
    assert again is not None
    assert list(entry(again, "t:state")["value"]) == DECLARED
    assert entry(again, "d:state_keys")["value"] == [*DECLARED, "y", "x"]
    assert ordered(entry(again, "t:state")) == ordered(entry(record, "t:state"))


# ---------------------------------------------------------------------------------------------
# A mapping a binding returns out of canonical order

RETURNED = {"b": 1, "a": 2, "c": 3}


def weights_design() -> Design:
    return Design(
        name="weights",
        state=t.record(),
        sources={"look": Source({"t:weights": t.mapping(t.number())})},
        rules={
            "weight_keys": Rule(
                "d:weight_keys", t.list_(t.string), weight_keys, version="1", reads=["t:weights"]
            )
        },
        policy=Policy(inform, version="1"),
    )


def weights_record(loaded: LoadedDesign) -> Record:
    return decide_sync(loaded, {}, {"default": Seen({"t:weights": dict(RETURNED)})}).record


def test_a_mapping_returned_out_of_order_is_recorded_with_its_order() -> None:
    record = weights_record(load(weights_design()))
    stored = json.loads(record.document)
    item = next(i for i in stored["log"] if i.get("ref") == "t:weights")
    # The document is canonical; the item's order says the value's object held b, a, c.
    assert list(item["value"]) == ["a", "b", "c"]
    assert item["order"] == {"1": [1, 0, 2]}
    assert list(entry(record, "t:weights")["value"]) == ["b", "a", "c"]
    assert entry(record, "d:weight_keys")["value"] == ["b", "a", "c"]


def test_the_observer_sees_each_item_with_its_order_as_the_closed_record_gives_it() -> None:
    seen: list[dict[str, Any]] = []
    loaded = load(weights_design())
    returned = Seen({"t:weights": dict(RETURNED)})
    record = decide_sync(
        loaded, {}, {"default": returned}, observer=lambda i, item: seen.append(item)
    ).record
    observed = next(i for i in seen if i.get("ref") == "t:weights")
    assert list(observed["value"]) == ["b", "a", "c"]
    assert observed["order"] == {"1": [1, 0, 2]}
    assert observed == entry(record, "t:weights")
    assert list(observed) == list(entry(record, "t:weights"))


def test_use_answers_refuses_a_state_held_in_another_order_as_another_turn() -> None:
    from ommat.evaluation import EvaluationRefusal, RefusalReason, use_answers_sync

    design = Design(
        name="mapped state",
        state=t.record(tags=t.mapping(t.integer())),
        policy=Policy(inform, version="1"),
    )
    loaded = load(design)
    one = decide_sync(loaded, {"tags": {"b": 1, "a": 2}}, {}).record
    other = decide_sync(loaded, {"tags": {"a": 2, "b": 1}}, {}).record
    with pytest.raises(EvaluationRefusal) as refused:
        use_answers_sync(one, loaded, other)
    assert refused.value.reason is RefusalReason.different_turn


def test_a_mapping_reads_back_in_its_order_and_replays_with_no_finding() -> None:
    loaded = load(weights_design())
    record = weights_record(loaded)
    back = read_record(record.document)
    assert back.document == record.document
    assert list(entry(back, "t:weights")["value"]) == ["b", "a", "c"]
    again, findings = replay_sync(back, loaded)
    assert findings == []
    assert again is not None
    assert list(entry(again, "t:weights")["value"]) == ["b", "a", "c"]


def test_a_copy_whose_mapping_order_differs_gives_an_entry_finding() -> None:
    loaded = load(weights_design())
    record = weights_record(loaded)
    doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
    item = next(i for i in doc["log"] if i.get("ref") == "t:weights")
    item["value"] = {"c": 3, "a": 2, "b": 1}  # the same members, another order
    reordered(item)
    assert item["order"] == {"1": [2, 0, 1]}
    copy_ = read_record(rehashed(doc))
    assert list(entry(copy_, "t:weights")["value"]) == ["c", "a", "b"]
    _, findings = replay_sync(copy_, loaded)
    assert [(f["kind"], f.get("ref")) for f in findings] == [(FindingKind.entry, "d:weight_keys")]
    (found,) = findings
    assert found["recorded"]["value"] == ["b", "a", "c"]
    assert found["replayed"]["value"] == ["c", "a", "b"]


def test_a_replay_that_rebuilds_another_order_is_a_finding() -> None:
    def keyed(keys: list[str]) -> LoadedDesign:
        return load(
            Design(
                name="keyed",
                state=t.record(),
                rules={
                    "keyed": Rule(
                        "d:keyed",
                        t.mapping(t.integer()),
                        param_keyed,
                        version="1",
                        params=["keys"],
                    )
                },
                parameters={"keys": Parameter(t.list_(t.string), keys)},
                policy=Policy(inform, version="1"),
            )
        )

    recorded_under = keyed(["b", "a"])
    record = decide_sync(recorded_under, {}, {}).record
    assert list(entry(record, "d:keyed")["value"]) == ["b", "a"]
    # The same members, built in the other order by the design replayed under.
    _, findings = replay_sync(record, keyed(["a", "b"]))
    kinds = [(f["kind"], f.get("ref")) for f in findings]
    assert (FindingKind.entry, "d:keyed") in kinds
    found = next(f for f in findings if f.get("ref") == "d:keyed")
    assert found["recorded"]["value"] == found["replayed"]["value"]  # equal as data
    assert found["recorded"]["order"] == {"1": [1, 0]}
    assert found["replayed"]["order"] == {}
    # Under the design it was recorded under, the same order: no finding.
    assert replay_sync(record, recorded_under)[1] == []


# ---------------------------------------------------------------------------------------------
# Records before format 6, and an order that does not fit


def built(r: Any) -> dict[str, int]:
    return {"b": 1, "a": 2}


def test_a_format_5_record_replays_with_no_finding_though_the_replay_holds_order() -> None:
    loaded = load(
        Design(
            name="built",
            state=t.record(),
            sources={"look": Source({"t:weights": t.mapping(t.number())})},
            rules={"built": Rule("d:built", t.mapping(t.integer()), built, version="1")},
            policy=Policy(inform, version="1"),
        )
    )
    record = decide_sync(loaded, {}, {"default": Seen({"t:weights": dict(RETURNED)})}).record
    old = read_record(rehashed(before_key_order({"log": record.log, "unhashed": record.unhashed})))
    assert old.log[0]["format"] == 5
    assert all("order" not in i for i in old.log)
    # A record before format 6 holds its objects in canonical order.
    assert list(entry(old, "d:built")["value"]) == ["a", "b"]
    again, findings = replay_sync(old, loaded)
    assert findings == []
    assert again is not None
    assert again.log[0]["format"] == 6
    assert entry(again, "d:built")["order"] == {"1": [1, 0]}


def test_read_record_refuses_an_order_that_does_not_fit() -> None:
    record = weights_record(load(weights_design()))
    orders: list[object] = [{"1": [0, 1]}, {"1": [0, 0, 1]}, {"7": [0]}, {"1": "x"}, []]
    for order in orders:
        doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
        index, item = next((n, i) for n, i in enumerate(doc["log"]) if i.get("ref") == "t:weights")
        item["order"] = order
        with pytest.raises(InvalidRecord, match=f"item {index}'s order does not fit it"):
            read_record(rehashed(doc, reorder=False))


def test_read_record_refuses_an_order_of_an_items_own_members() -> None:
    # A valid permutation of the item's own members: the ledger keeps them in canonical order,
    # so such an order is never derived, and two documents must not read as one record.
    record = weights_record(load(weights_design()))
    doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
    index, item = next((n, i) for n, i in enumerate(doc["log"]) if i.get("ref") == "t:weights")
    own: list[int] = list(range(len(item) - 1))[::-1]
    item["order"] = {**item["order"], "0": own}
    with pytest.raises(InvalidRecord, match=f"item {index}'s order orders its own members"):
        read_record(rehashed(doc, reorder=False))


@pytest.mark.parametrize("kind", [["entry"], {"a": 1}, 3])
def test_read_record_refuses_an_item_kind_that_is_not_a_string_as_invalid(kind: object) -> None:
    record = weights_record(load(weights_design()))
    doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
    next(i for i in doc["log"] if i.get("ref") == "t:weights")["item"] = kind
    with pytest.raises(InvalidRecord):
        read_record(rehashed(doc, reorder=False))


class _Loud(str):
    """A key whose comparisons raise, so a read that compares a key by its own code fails."""

    __hash__ = str.__hash__

    def __eq__(self, other: object) -> bool:
        raise AssertionError("a key's code ran")

    def __ne__(self, other: object) -> bool:
        raise AssertionError("a key's code ran")


def test_read_record_runs_no_code_of_a_str_subclass_key() -> None:
    record = weights_record(load(weights_design()))
    doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
    item = next(i for i in doc["log"] if i.get("ref") == "t:weights")
    rebuilt = {(_Loud(k) if k in ("item", "order", "value") else k): v for k, v in item.items()}
    doc["log"][doc["log"].index(item)] = rebuilt
    doc["hash"] = record.record_hash
    read = read_record(doc)
    assert read.record_hash == record.record_hash


def test_read_record_refuses_an_order_in_a_format_5_record() -> None:
    record = weights_record(load(weights_design()))
    doc = before_key_order({"log": record.log, "unhashed": record.unhashed})
    index, item = next((n, i) for n, i in enumerate(doc["log"]) if i.get("ref") == "t:weights")
    item["order"] = {}
    with pytest.raises(
        InvalidRecord, match=f"item {index} \\(entry\\) has the member 'order', which its kind"
    ):
        read_record(rehashed(doc))


def test_a_format_6_item_without_its_order_is_refused() -> None:
    record = weights_record(load(weights_design()))
    doc: dict[str, Any] = {"log": record.log, "unhashed": record.unhashed}
    index, item = next((n, i) for n, i in enumerate(doc["log"]) if i.get("ref") == "t:weights")
    del item["order"]
    with pytest.raises(InvalidRecord, match=f"item {index} has no 'order' member"):
        read_record(rehashed(doc, reorder=False))


# ---------------------------------------------------------------------------------------------
# The ordered encoding and the order object

# U+FFFF sorts after U+10000 by UTF-16 code units (a surrogate pair from 0xD800) though before
# it by code point: the canonical order is UTF-16's.
HIGH, ASTRAL = "\uffff", "\U00010000"


_CANONICAL: list[Any] = [
    None,
    [1, 2.5, "x", True],
    {"a": 1, "b": [{"c": None, "d": {}}]},
    {ASTRAL: 1, HIGH: 2},
    {"": 0, "z": [[], {}], "\u00e9": "\u00e9"},
]


@pytest.mark.parametrize("value", _CANONICAL)
def test_the_ordered_encoding_is_the_canonical_one_when_every_object_is_in_canonical_order(
    value: Any,
) -> None:
    assert order_of(value) == {}
    assert ordered(value).encode("utf-8") == encode(value)


def test_the_ordered_encoding_keeps_each_objects_held_order() -> None:
    value = {"b": 1, "a": {"y": [{"q": 1, "p": 2}], "x": 1.0}, HIGH: 0, ASTRAL: 0}
    assert ordered(value) == ('{"b":1,"a":{"y":[{"q":1,"p":2}],"x":1},"\uffff":0,"\U00010000":0}')
    assert encode(value) == (
        '{"a":{"x":1,"y":[{"p":2,"q":1}]},"b":1,"\U00010000":0,"\uffff":0}'.encode()
    )
    # Pre-order as the canonical encoding writes it: 0 the outer object, 1 a, 2 y's object.
    assert order_of(value) == {"0": [1, 0, 3, 2], "1": [1, 0], "2": [1, 0]}


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), {1: "x"}, {"s": {1, 2}}, b"bytes", 2**64, object()]
)
def test_the_ordered_encoding_raises_where_encode_raises(value: Any) -> None:
    with pytest.raises(_CanonicalError):
        encode(value)
    with pytest.raises(_CanonicalError):
        ordered(value)


_KEYS = ["a", "b", "B", "", "\u00e9", "z0", HIGH, ASTRAL, "\U0001f600", "\ud7ff"]


def _random_value(rng: random.Random, depth: int) -> Any:
    roll = rng.random()
    if depth <= 0 or roll < 0.3:
        return rng.choice([None, True, 0, -1.5, "s", ""])
    if roll < 0.55:
        return [_random_value(rng, depth - 1) for _ in range(rng.randint(0, 3))]
    keys = rng.sample(_KEYS, rng.randint(0, 5))
    return {k: _random_value(rng, depth - 1) for k in keys}


def _canonical(value: Any) -> Any:
    return decode(encode(value))


@pytest.mark.parametrize("seed", range(40))
def test_order_of_and_with_order_round_trip(seed: int) -> None:
    value = _random_value(random.Random(seed), 4)
    order = order_of(value)
    rebuilt = with_order(_canonical(value), order)
    assert ordered(rebuilt) == ordered(value)
    assert order_of(rebuilt) == order
    assert rebuilt == value
    # Each listed object is out of canonical order, its positions a permutation.
    for positions in order.values():
        assert sorted(positions) == list(range(len(positions)))
        assert positions != sorted(positions)


@pytest.mark.parametrize(
    "order",
    [
        {"0": [0, 0]},
        {"0": [1, 0, 2]},
        {"0": [0, 1]},  # canonical order is never listed
        {"9": [1, 0]},
        {"0": ["1", "0"]},
        {"0": (1, 0)},
        [],
    ],
)
def test_with_order_refuses_an_order_that_does_not_fit(order: Any) -> None:
    with pytest.raises(ValueError, match="order"):
        with_order({"a": 1, "b": 2}, order)


# ---------------------------------------------------------------------------------------------
# Design format 6: the payload's, a parameter value's and a structured template's order

PAYLOAD = {"model": "m", "options": {"top_p": 1, "temperature": 0}, "max": 5}
WEIGHTS = {"zeta": 1, "alpha": 2}
TEMPLATE = {"task": "Is it {w}?", "context": {"weights": "{w}"}, "answer": "yes or no"}
IN_ORDER = {"a": "Is it?", "b": "yes or no"}


def check_param(r: Any) -> list[str]:
    return list(r.param("weights"))


def ordered_design(template: Any = TEMPLATE) -> Design:
    slots = [{"name": "w", "origin": {"param": "weights"}}] if template is TEMPLATE else []
    return Design(
        name="ordered",
        state=t.record(),
        sources={
            "ask": Source({"q:it": Question("noul", Structured(template, slots))}, payload=PAYLOAD)
        },
        rules={
            "param_keys": Rule(
                "d:param_keys", t.list_(t.string), check_param, version="1", params=["weights"]
            )
        },
        parameters={"weights": Parameter(t.mapping(t.number()), WEIGHTS)},
        policy=Policy(inform, version="1"),
    )


def check_param_functions() -> dict[str, Any]:
    return {"policy": inform, "param_keys": check_param}


def test_a_document_from_format_6_writes_each_order_and_loads_back_to_its_own_hash() -> None:
    loaded = load(ordered_design())
    doc = decode(loaded.document)
    assert doc["format"] == 7
    assert doc["sources"]["ask"]["payload_order"] == {"0": [1, 2, 0], "1": [1, 0]}
    assert doc["parameters"]["weights"]["order"] == {"0": [1, 0]}
    assert doc["sources"]["ask"]["provides"]["q:it"]["wording"]["order"] == {"0": [2, 1, 0]}
    again = load(loaded.document, functions=check_param_functions())
    assert again.design_hash == loaded.design_hash
    assert again.document == loaded.document
    twice = load(json.loads(again.document), functions=check_param_functions())
    assert twice.document == loaded.document


@pytest.mark.parametrize("from_document", [False, True])
def test_each_order_reaches_the_call_the_rule_and_the_wording(from_document: bool) -> None:
    loaded = load(ordered_design())
    if from_document:
        loaded = load(loaded.document, functions=check_param_functions())
    seen = Seen()
    record = decide_sync(loaded, {}, {"default": seen}).record
    (call,) = seen.batches[0]
    assert list(call.payload) == ["model", "options", "max"]
    assert list(call.payload["options"]) == ["top_p", "temperature"]
    wording = call.questions["q:it"].wording
    assert isinstance(wording, dict)
    assert list(wording) == ["task", "context", "answer"]
    assert wording["task"] == 'Is it {"zeta":1,"alpha":2}?'
    assert entry(record, "d:param_keys")["value"] == ["zeta", "alpha"]


@pytest.mark.parametrize(
    ("leaf_type", "value"),
    [
        (t.mapping(t.bytes_), {"zeta": b"\x00z", "alpha": b"a"}),
        (t.mapping(t.string), {"zeta": "z", "alpha": "a"}),
    ],
    ids=["bytes", "string"],
)
def test_a_mapping_leaf_of_any_scalar_keeps_its_order_through_its_document(
    leaf_type: str, value: Any
) -> None:
    design = Design(
        name="leaf",
        state=t.record(),
        parameters={"weights": Parameter(leaf_type, value)},
        policy=Policy(inform, version="1"),
    )
    loaded = load(design)
    assert decode(loaded.document)["parameters"]["weights"]["order"] == {"0": [1, 0]}
    again = load(loaded.document, functions={"policy": inform})
    assert again.design_hash == loaded.design_hash
    assert list(again._leaf_value("weights")) == ["zeta", "alpha"]  # pyright: ignore[reportPrivateUsage]


def test_a_template_of_a_list_of_objects_keeps_each_objects_order() -> None:
    template = [{"z": "first", "a": {"y": "x", "b": "c"}}, {"q": "second", "c": "d"}]
    design = Design(
        name="list",
        state=t.record(),
        sources={"ask": Source({"q:it": Question("noul", Structured(template))})},
        policy=Policy(inform, version="1"),
    )
    loaded = load(design)
    again = load(loaded.document, functions={"policy": inform})
    assert again.design_hash == loaded.design_hash
    seen = Seen()
    decide_sync(again, {}, {"default": seen})
    (call,) = seen.batches[0]
    wording = call.questions["q:it"].wording
    assert isinstance(wording, list)
    assert [list(o) for o in wording] == [["z", "a"], ["q", "c"]]
    assert list(wording[0]["a"]) == ["y", "b"]


def test_an_in_order_structured_template_writes_no_order_and_keeps_its_identity() -> None:
    loaded = load(ordered_design(IN_ORDER))
    wording = decode(loaded.document)["sources"]["ask"]["provides"]["q:it"]["wording"]
    assert wording == {"template": IN_ORDER, "slots": []}
    record = decide_sync(loaded, {}, {"default": Seen()}).record
    call = next(i for i in record.log if i["item"] == "call")
    assert call["questions"]["q:it"]["identity"] == hash_value(
        {"kind": "noul", "wording": {"template": IN_ORDER, "slots": []}, "labels": None}
    )


def test_a_documents_empty_wording_order_is_refused() -> None:
    doc = decode(load(ordered_design(IN_ORDER)).document)
    doc["sources"]["ask"]["provides"]["q:it"]["wording"]["order"] = {}
    with pytest.raises(InvalidDesign, match="writes an empty order, which it leaves out"):
        load(doc, functions=check_param_functions())


def test_a_documents_order_that_does_not_fit_is_refused_at_check_11() -> None:
    doc = decode(load(ordered_design()).document)
    doc["parameters"]["weights"]["order"] = {"0": [0, 1]}
    with pytest.raises(InvalidDesign) as raised:
        load(doc, functions=check_param_functions())
    ((check, message),) = [(f.check, f.message) for f in raised.value.failures]
    assert check == 11
    assert 'parameter "weights"\'s order does not fit its value' in message


def test_check_11_refuses_a_format_5_document_holding_payload_order() -> None:
    doc = document_before_key_order(decode(load(ordered_design(IN_ORDER)).document))
    load(doc, functions=check_param_functions())  # without it, it loads at format 5
    doc["sources"]["ask"]["payload_order"] = {}
    with pytest.raises(InvalidDesign) as raised:
        load(doc, functions=check_param_functions())
    assert [(f.check, f.message) for f in raised.value.failures] == [
        (11, 'source "ask" has the member "payload_order", which its form does not have')
    ]


def test_check_10_refuses_a_format_5_document_whose_record_type_is_out_of_order() -> None:
    doc = decode(load(state_design()).document)
    assert doc["state"] == "record{zeta: string, alpha: integer, mid: record{y: string, x: string}}"
    old = document_before_key_order(json.loads(json.dumps(doc)))
    assert old["state"] == "record{alpha: integer, mid: record{x: string, y: string}, zeta: string}"
    functions = {**FUNCTIONS}
    load(old, functions=functions)  # in code-point order it loads at format 5
    old["state"] = doc["state"]
    with pytest.raises(InvalidDesign) as raised:
        load(old, functions=functions)
    ((check, message),) = [(f.check, f.message) for f in raised.value.failures]
    assert check == 10
    assert "the state declaration" in message


# ---------------------------------------------------------------------------------------------
# The redacted view's order


def test_a_redacted_views_order_fits_its_own_items() -> None:
    record = weights_record(load(weights_design()))
    view = json.loads(redact(record, b"\x07" * 32))
    held = [i for i in view["log"] if i["item"] in ("entry", "plan", "verdict")]
    assert held
    for item in held:
        rest = {k: v for k, v in item.items() if k != "order"}
        assert order_of(with_order(rest, item["order"])) == item["order"]
    # The mapping's keys are digests now, so its order is derived over them afresh: in the
    # order held, the values are b's, a's and c's, as returned.
    (weights,) = [i for i in held if i.get("ref") == "t:weights"]
    rest = {k: v for k, v in weights.items() if k != "order"}
    rebuilt = cast("dict[str, Any]", with_order(rest, weights["order"]))
    assert list(cast("dict[str, int]", rebuilt["value"]).values()) == [1, 2, 3]
    assert weights["order"] != entry(record, "t:weights")["order"]


# ---------------------------------------------------------------------------------------------
# from_record compares a need by the order it holds from format 6


def asked(r: Any) -> Plan:
    return Plan("inform", reply="asked")


def needing_design() -> LoadedDesign:
    return load(
        Design(
            name="needing",
            state=t.record(),
            sources={
                "look": Source({"t:weights": t.mapping(t.number())}, oracle="look"),
                "ask": Source(
                    {"q:fine": Question("noul", Text("Fine?"))}, needs=["t:weights"], oracle="ask"
                ),
            },
            policy=Policy(asked, version="1"),
        )
    )


def needing_record(loaded: LoadedDesign) -> Record:
    binds = {"look": Seen({"t:weights": dict(RETURNED)}), "ask": Seen()}
    return decide_sync(loaded, {}, binds).record


def answered_from(record: object, loaded: LoadedDesign, weights: dict[str, int]) -> Record:
    """A live turn with ``look`` returning ``weights`` and ``ask`` answered from ``record``."""
    binds = {"look": Seen({"t:weights": weights}), "ask": from_record(record, loaded)}
    return decide_sync(loaded, {}, binds).record


def marker_code(record: Record, ref: str) -> str | None:
    found = entry(record, ref)
    return found["marker"]["cause"]["code"] if "marker" in found else None


def test_from_record_answers_a_need_in_the_order_the_record_holds() -> None:
    loaded = needing_design()
    record = needing_record(loaded)
    again = answered_from(record, loaded, dict(RETURNED))
    assert marker_code(again, "q:fine") is None


def test_from_record_refuses_a_need_whose_key_order_differs_from_format_6() -> None:
    loaded = needing_design()
    record = needing_record(loaded)
    again = answered_from(record, loaded, {"a": 2, "b": 1, "c": 3})  # equal as data
    assert marker_code(again, "q:fine") == "unsupported"
    log = again.log
    index = next(n for n, i in enumerate(log) if i["item"] == "call" and "ask" in i["sources"])
    lines = again.unhashed["calls"][str(index)]["warnings"]
    assert "source 'ask': needs differ from the stored turn's: 't:weights'" in lines


def test_from_record_compares_a_need_by_canonical_bytes_before_format_6() -> None:
    loaded = needing_design()
    record = needing_record(loaded)
    old = rehashed(before_key_order({"log": record.log, "unhashed": record.unhashed}))
    for weights in ({"a": 2, "b": 1, "c": 3}, dict(RETURNED)):
        assert marker_code(answered_from(old, loaded, weights), "q:fine") is None
    assert marker_code(answered_from(old, loaded, {"a": 2, "b": 1, "c": 4}), "q:fine") == (
        "unsupported"
    )


# ---------------------------------------------------------------------------------------------
# A record before format 6 beside the engine's own


def built_mapping(r: Any) -> dict[str, int]:
    return {"b": 1, "a": 2}


def informed(r: Any) -> Plan:
    return Plan("inform", reply="ok")


def context_design() -> LoadedDesign:
    from ommat import Context, ContextEntry

    slot = [{"name": "m", "origin": {"ref": "d:built"}}]
    return load(
        Design(
            name="context",
            state=t.record(),
            sources={
                "ask": Source(
                    {"q:fine": Question("noul", Text("Is {m} fine?", slot))}, needs=["d:built"]
                )
            },
            rules={"built": Rule("d:built", t.mapping(t.integer()), built_mapping, version="1")},
            policy=Policy(informed, version="1"),
            context=Context([ContextEntry("d:built", Text("M={m}", slot))]),
        )
    )


def test_slot_text_renders_a_mapping_in_the_order_the_rule_built_it() -> None:
    record = decide_sync(context_design(), {}, {"default": Seen()}).record
    plan = next(i for i in record.log if i["item"] == "plan")
    assert plan["context"]["entries"][0]["text"] == 'M={"b":1,"a":2}'


def test_a_format_5_record_whose_slot_text_was_sorted_replays_with_no_finding() -> None:
    loaded = context_design()
    record = decide_sync(loaded, {}, {"default": Seen()}).record
    doc = before_key_order({"log": record.log, "unhashed": record.unhashed})
    for item in doc["log"]:  # as the format-5 engine rendered it: sorted
        if item["item"] == "plan":
            item["context"]["entries"][0]["text"] = 'M={"a":2,"b":1}'
    old = read_record(rehashed(doc))
    rerun, findings = replay_sync(old, loaded)
    assert findings == []
    assert rerun is not None
    plan = next(i for i in rerun.log if i["item"] == "plan")
    assert plan["context"]["entries"][0]["text"] == 'M={"a":2,"b":1}'
    # The flag is the replay's alone: a live turn after it renders in held order again.
    again = decide_sync(loaded, {}, {"default": Seen()}).record
    assert (
        next(i for i in again.log if i["item"] == "plan")["context"]["entries"][0]["text"]
        == 'M={"b":1,"a":2}'
    )


def test_use_answers_takes_a_format_5_and_a_format_6_record_of_one_turn() -> None:
    from ommat.evaluation import use_answers_sync

    loaded = load(weights_design())
    six = weights_record(loaded)
    five = read_record(rehashed(before_key_order({"log": six.log, "unhashed": six.unhashed})))
    # No pairing is refused as another turn: an item before format 6 is compared without the
    # other's order. Findings may follow, since a format-5 record holds its mappings sorted.
    for record, other in ((five, six), (six, five), (five, five), (six, six)):
        use_answers_sync(record, loaded, other)
    assert use_answers_sync(six, loaded, six)["findings"] == []
