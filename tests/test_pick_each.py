"""The per-candidate picks: ``pick_each_with_margin`` and ``pick_each_with_thresholds`` over a
per-candidate question's declared ref, each candidate's pick, abstention or marker in the set's
order, and the record a rule passing one candidate's abstention on writes, a flagged context
entry included."""

from typing import Any, cast

import pytest

from designs import slot
from ommat import (
    CallResult,
    Context,
    ContextEntry,
    Design,
    Plan,
    Policy,
    Question,
    ReadView,
    Record,
    Rule,
    Source,
    SourceCall,
    Text,
    Unavailable,
    Unsure,
    decide_sync,
    load,
    read_record,
    redact,
    replay_sync,
)
from ommat import types as t
from ommat._refs import grounded_ref
from ommat.helpers import pick_each_with_margin, pick_each_with_thresholds

OUT = "q:lamps/out"
SET = "d:streets"
STREETS = ["Quay Street", "Elm Row", "Bridge Lane", "Back Lane"]

PICKS: list[object] = []


def streets(r: ReadView) -> object:
    return r.read("t:state.streets")


def lit(r: ReadView) -> object:
    picks = pick_each_with_margin(r, OUT, 0.8, 0.2)
    PICKS.append(picks)
    if not isinstance(picks, dict):
        return picks
    return [street for street, pick in picks.items() if pick == "yes"]


def first_doubt(r: ReadView) -> object:
    """The first street's pick, its abstention passed on whole."""
    picks = pick_each_with_thresholds(r, OUT, {"yes": 0.8, "no": 0.8}, 0.2)
    if not isinstance(picks, dict) or not picks:
        return "no"
    return next(iter(picks.values()))


def escalate(r: ReadView) -> Plan:
    return Plan("escalate", reason="checked")


DESIGN = Design(
    name="lamps",
    state=t.record(text=t.string, streets=t.list_(t.string)),
    sources={
        "lamps": Source(
            {OUT: Question("noul", Text("Is a lamp out on {s}?", [slot("s", candidate=SET)]))},
            needs=["t:state.text"],
            grounded=SET,
        )
    },
    rules={
        "streets": Rule(
            SET, t.list_(t.string), streets, version="1", reads=["t:state.streets"], candidates=True
        ),
        "lit": Rule("d:lit", t.list_(t.string), lit, version="1", reads=[OUT]),
        "doubt": Rule("d:doubt", t.label("yes", "no"), first_doubt, version="1", reads=[OUT]),
    },
    policy=Policy(escalate, version="1", reads=["d:lit", "d:doubt"]),
    context=Context([ContextEntry("d:doubt", Text("Doubt: {d}", [slot("d", ref="d:doubt")]))]),
)
LOADED = load(DESIGN)

ANSWERS: dict[str, dict[str, float]] = {
    "Quay Street": {"yes": 0.95, "no": 0.05},
    "Elm Row": {"yes": 0.55, "no": 0.45},
    "Bridge Lane": {"yes": 0.05, "no": 0.95},
}
"""Back Lane has no answer, so its entry is a marker."""


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    asked = {ref for s in batch for ref in s.questions}
    out: dict[str, Any] = {}
    for street, probabilities in ANSWERS.items():
        if grounded_ref(OUT, street) in asked:
            out[grounded_ref(OUT, street)] = dict(probabilities)
    return CallResult(out)


def decided(names: list[str]) -> Record:
    PICKS.clear()
    state = {"text": "Lamps out", "streets": names}
    return decide_sync(LOADED, state, {"default": answer}).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def test_each_candidate_gets_its_pick_its_abstention_or_its_marker_in_the_set_s_order() -> None:
    record = decided(["Elm Row", "Quay Street", "Back Lane", "Bridge Lane"])
    (held,) = PICKS
    assert isinstance(held, dict)
    picks = cast("dict[str, object]", held)
    assert list(picks) == ["Elm Row", "Quay Street", "Back Lane", "Bridge Lane"]
    assert picks["Quay Street"] == "yes"
    assert picks["Bridge Lane"] == "no"
    elm = picks["Elm Row"]
    assert isinstance(elm, Unsure)
    assert elm.cause.code.value == "abstained"
    assert list(elm.cause.refs) == [grounded_ref(OUT, "Elm Row")]
    assert isinstance(picks["Back Lane"], Unavailable)
    assert entries(record)["d:lit"]["value"] == ["Quay Street"]


def test_the_derivation_names_the_declared_ref_alone() -> None:
    record = decided(STREETS)
    (call,) = entries(record)["d:lit"]["derivation"]
    assert call == {"helper": "pick_each_with_margin", "refs": [OUT], "rested_on": [OUT]}


def test_a_rule_passing_one_candidate_s_abstention_on_into_the_context_records_cleanly() -> None:
    record = decided(["Elm Row", "Quay Street"])
    doubt = entries(record)["d:doubt"]
    assert doubt["marker"]["cause"] == {
        "code": "abstained",
        "refs": [grounded_ref(OUT, "Elm Row")],
    }
    (plan,) = [i for i in record.log if i["item"] == "plan"]
    (context_entry,) = plan["context"]["entries"]
    assert context_entry["marker"] == doubt["marker"]
    # The record reads back, replays and redacts with the candidate digested in the cause.
    stored = read_record(record.document)
    assert replay_sync(stored, LOADED)[1] == []
    redacted = redact(record, b"k" * 32).decode("utf-8")
    assert "Elm Row" not in redacted


def test_an_empty_set_gives_an_empty_dict() -> None:
    record = decided([])
    assert PICKS == [{}]
    assert entries(record)["d:lit"]["value"] == []


@pytest.mark.parametrize("helper", [pick_each_with_margin, pick_each_with_thresholds])
def test_a_read_that_is_not_per_candidate_raises(helper: Any) -> None:
    def misuse(r: ReadView) -> object:
        if helper is pick_each_with_margin:
            return helper(r, "q:lamps/urgent", 0.8, 0.2)
        return helper(r, "q:lamps/urgent", {"yes": 0.8}, 0.2)

    globals()["misuse"] = misuse
    misuse.__qualname__ = "misuse"
    source = Source(
        {
            OUT: Question("noul", Text("Is a lamp out on {s}?", [slot("s", candidate=SET)])),
            "q:lamps/urgent": Question("noul", Text("Is it urgent?")),
        },
        needs=["t:state.text"],
        grounded=SET,
    )
    rules = dict(DESIGN.rules)
    rules["lit"] = Rule("d:lit", t.list_(t.string), misuse, version="1", reads=["q:lamps/urgent"])
    loaded = load(
        Design(
            name="lamps",
            state=DESIGN.state,
            sources={"lamps": source},
            rules=rules,
            policy=DESIGN.policy,
        )
    )

    async def both(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({ref: {"yes": 0.9, "no": 0.1} for s in batch for ref in s.questions})

    state = {"text": "Lamps out", "streets": ["Quay Street"]}
    record = decide_sync(loaded, state, {"default": both}).record
    lit_entry = entries(record)["d:lit"]
    assert lit_entry["marker"]["cause"]["code"] == "raised"


def test_thresholds_refuse_a_label_an_answered_candidate_lacks() -> None:
    def lacking(r: ReadView) -> object:
        return pick_each_with_thresholds(r, OUT, {"maybe": 0.5}, 0.0)

    globals()["lacking"] = lacking
    lacking.__qualname__ = "lacking"
    rules = dict(DESIGN.rules)
    rules["lit"] = Rule("d:lit", t.list_(t.string), lacking, version="1", reads=[OUT])
    fields = {f: getattr(DESIGN, f) for f in ("name", "state", "sources", "policy", "context")}
    loaded = load(Design(rules=rules, **fields))
    state = {"text": "Lamps out", "streets": ["Quay Street"]}
    record = decide_sync(loaded, state, {"default": answer}).record
    assert entries(record)["d:lit"]["marker"]["cause"]["code"] == "raised"
