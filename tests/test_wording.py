# pyright: reportPrivateUsage=false
"""A noul's described yes and no, carried from the design to the binding."""

import json
from typing import Any

import pytest

from ledger_logs import before_profile, rehashed
from ommat import (
    CallResult,
    Design,
    InvalidDesign,
    InvalidRecord,
    Question,
    Record,
    Rule,
    Source,
    SourceCall,
    Text,
    decide_sync,
    from_record,
    load,
    read_record,
    replay_sync,
)
from ommat import types as t

YES = {"name": "yes", "description": "The message names a sender"}
DESCRIBED = [YES, "no"]


def inform(r: Any) -> str:
    return "ok"


ACT = Rule("d:act", t.string, inform, version="1")


def design(labels: Any = None, wording: Any = None) -> Design:
    q = Question("noul", wording or Text("Does it name a sender?"), labels)
    return Design(
        name="wording",
        state=t.record(),
        sources={"ask": Source({"q:named": q})},
        rules={"act": ACT},
    )


class Seen:
    def __init__(self) -> None:
        self.batches: list[tuple[SourceCall, ...]] = []

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        self.batches.append(batch)
        return CallResult(
            {ref: {"yes": 0.8, "no": 0.2} for call in batch for ref in call.questions}
        )


def run(d: Design, binding: Any = None) -> tuple[Record, Seen]:
    seen = Seen()
    record = decide_sync(load(d), {}, {"default": binding or seen}).record
    return record, seen


def call_item(record: Record) -> dict[str, Any]:
    (item,) = [i for i in record.log if i["item"] == "call"]
    return item


def test_a_described_noul_reaches_the_binding_with_its_descriptions() -> None:
    record, seen = run(design(DESCRIBED))
    q = seen.batches[0][0].questions["q:named"]
    assert q.labels == ("yes", "no")
    assert q.descriptions == ("The message names a sender", None)
    assert call_item(record)["questions"]["q:named"]["labels"] == DESCRIBED
    entry = next(i for i in record.log if i.get("ref") == "q:named")
    assert entry["value"]["labels"] == ["yes", "no"]


def test_an_undescribed_noul_keeps_null_labels_and_its_identity() -> None:
    record, seen = run(design())
    assert seen.batches[0][0].questions["q:named"].descriptions == (None, None)
    assert call_item(record)["questions"]["q:named"]["labels"] is None
    described, _ = run(design(DESCRIBED))
    ident = call_item(record)["questions"]["q:named"]["identity"]
    assert ident != call_item(described)["questions"]["q:named"]["identity"]


@pytest.mark.parametrize(
    ("labels", "says"),
    [
        (["yes", "no"], "declares yes and no with no description"),
        (["no", YES], "whose labels are not yes then no"),
        ([YES, "no", "maybe"], "whose labels are not yes then no"),
    ],
)
def test_a_nouls_labels_are_yes_then_no_with_one_described(labels: Any, says: str) -> None:
    with pytest.raises(InvalidDesign) as raised:
        load(design(labels))
    (failure,) = raised.value.failures
    assert failure.check == 10
    assert says in failure.message


def test_a_described_noul_replays_and_from_record_answers_it() -> None:
    d = design(DESCRIBED)
    record, _ = run(d)
    assert replay_sync(record, load(d))[1] == []
    live = decide_sync(load(d), {}, {"default": from_record(record, load(d))}).record
    entry = next(i for i in live.log if i.get("ref") == "q:named")
    assert "value" in entry
    other = design([{"name": "yes", "description": "Another wording"}, "no"])
    live = decide_sync(load(other), {}, {"default": from_record(record, load(other))}).record
    entry = next(i for i in live.log if i.get("ref") == "q:named")
    assert entry["marker"]["cause"]["code"] == "unsupported"


def test_a_described_noul_before_format_4_is_refused_in_a_record() -> None:
    record, _ = run(design(DESCRIBED))
    stored = json.loads(record.document)
    before_profile(stored, 3)
    with pytest.raises(InvalidRecord, match="a noul's labels, which a format-3 record"):
        read_record(rehashed(stored))


# ---------------------------------------------------------------------------------------------
# A structured instruction

from ommat import Structured  # noqa: E402


def slotted(template: Any, *slots: dict[str, Any]) -> Design:
    q = Question("noul", Structured(template, list(slots)), DESCRIBED)
    return Design(
        name="structured",
        state=t.record(body=t.string),
        sources={"ask": Source({"q:named": q})},
        rules={"act": ACT},
    )


BODY = {"name": "body", "origin": {"ref": "t:state.body"}}


def asked_wording(d: Design, state: Any) -> Any:
    seen = Seen()
    decide_sync(load(d), state, {"default": seen})
    return seen.batches[0][0].questions["q:named"].wording


def test_a_structured_wordings_string_values_render_and_its_keys_stay() -> None:
    d = slotted({"task": "Does {body} name a sender?", "{keep}": ["literal {{x}}", 3, None]}, BODY)
    wording = asked_wording(d, {"body": "Hi from Ana"})
    expected = {"task": "Does Hi from Ana name a sender?", "{keep}": ["literal {x}", 3, None]}
    assert wording == expected


def test_a_structured_wording_renders_in_its_template_order() -> None:
    d = slotted({"b": "two", "a": "one", "c": {"y": 1, "x": 2}})
    wording = asked_wording(d, {"body": ""})
    assert list(wording) == ["b", "a", "c"]
    assert list(wording["c"]) == ["y", "x"]


def test_changing_the_template_after_load_reaches_no_turn() -> None:
    template: dict[str, Any] = {"task": "Does it name a sender?"}
    loaded = load(slotted(template))
    template["task"] = "changed"
    seen = Seen()
    decide_sync(loaded, {"body": ""}, {"default": seen})
    assert seen.batches[0][0].questions["q:named"].wording == {"task": "Does it name a sender?"}


@pytest.mark.parametrize(
    ("template", "slots", "check", "says"),
    [
        ("plain", (), 10, "template is not a JSON object or array"),
        ({"a": {1, 2}}, (), 10, "outside the value domain"),
        ({"task": "{nope}"}, (), 9, 'placeholder "nope" names no slot'),
        ({"task": "fine"}, (BODY,), 9, 'slot "body" is not in its template'),
        ({"task": ["ok", "a { b"]}, (), 9, 'at ["task"][1] has a {'),
        ({}, (), 9, "holds no character that is not whitespace"),
        ([" ", ["  "]], (), 9, "holds no character that is not whitespace"),
    ],
)
def test_a_structured_wording_is_checked(template: Any, slots: Any, check: int, says: str) -> None:
    with pytest.raises(InvalidDesign) as raised:
        load(slotted(template, *slots))
    failures = [f for f in raised.value.failures if f.check == check]
    assert any(says in f.message for f in failures), raised.value.failures


def test_a_structured_wording_has_its_own_identity_and_a_text_keeps_its_own() -> None:
    def identity(d: Design) -> str:
        record, _ = run(d)
        return call_item(record)["questions"]["q:named"]["identity"]

    text = design(DESCRIBED, Text("Does it name a sender?"))
    structured = design(DESCRIBED, Structured({"task": "Does it name a sender?"}))
    assert identity(text) != identity(structured)
    assert identity(text) == identity(design(DESCRIBED))


def test_a_documents_string_template_reads_as_text_and_an_object_as_structured() -> None:
    from ommat._load import _text_of

    assert isinstance(_text_of({"template": "x", "slots": []}), Text)
    assert isinstance(_text_of({"template": {"a": "x"}, "slots": []}), Structured)


def test_a_structured_wording_loads_from_its_document_to_its_hash() -> None:
    loaded = load(slotted({"task": "Does {body} name a sender?"}, BODY))
    again = load(loaded.document, functions={"act": inform})
    assert again.design_hash == loaded.design_hash


# ---------------------------------------------------------------------------------------------
# End to end through the idia binding over idia's scripted leaf


def idia_turn(**caps: bool) -> tuple[Record, Any, Any]:
    import idia
    from idia import Dispatcher, RetryPolicy
    from idia.providers.scripted import Scripted

    from ommat.bindings.idia import oracle_binding
    from test_idia_binding import answer_all
    from test_idia_binding import caps as capabilities

    leaf = Scripted(capabilities=capabilities(**caps), script=answer_all)
    binding = oracle_binding(Dispatcher(leaf, retry=RetryPolicy.without_delay()))
    d = slotted({"task": "Does {body} name a sender?", "options": ["yes", "no"]}, BODY)
    loaded = load(d)
    record = decide_sync(loaded, {"body": "Hi from Ana"}, {"default": binding}).record
    assert idia is not None
    return record, leaf, loaded


def test_a_described_noul_with_a_structured_wording_runs_through_idia_and_replays() -> None:
    record, leaf, loaded = idia_turn()
    (q,) = [q for c in leaf.calls for q in c.wire_request.questions]
    assert dict(q.instruction) == {
        "task": "Does Hi from Ana name a sender?",
        "options": ("yes", "no"),
    }
    assert [(label.name, label.description) for label in q.labels] == [
        ("yes", "The message names a sender"),
        ("no", None),
    ]
    entry = next(i for i in record.log if i.get("ref") == "q:named")
    assert "value" in entry
    assert replay_sync(record, loaded)[1] == []


def test_a_provider_without_structured_instructions_leaves_the_question_unsupported() -> None:
    record, leaf, _ = idia_turn(carries_structured_instructions=False)
    assert leaf.calls == []
    entry = next(i for i in record.log if i.get("ref") == "q:named")
    assert entry["marker"]["cause"]["code"] == "unsupported"
