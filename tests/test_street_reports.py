"""The street-reports design: docs/tutorials/street_reports, shared by the binding's tests and the
tutorial. It loads, its lints pass, and fixed answers reach each row of its policy's table."""

import json
from collections.abc import Callable, Mapping
from typing import Any

import pytest

import street_reports_design as works
from ommat import (
    CallResult,
    Decision,
    LoadedDesign,
    PlanKind,
    SourceCall,
    decide_sync,
    load,
    replay_sync,
)
from ommat._binding import OracleBinding
from ommat.helpers import dead_questions, unsure_to_execute

FILED_ON = "2026-03-10"
LIGHT = "The street light outside 14 Alder Lane has flickered since 3 March and is dark by 9 pm."
DRAIN = "A drain cover is missing on Mill Road by the school gate, and children walk past it."
SHELTER = "The bench in the bus shelter on Quay Street is broken."
POTHOLE = "There is a deep pothole on Station Road near the library."
UNDATED = "The give-way sign at the end of Birch Close has been knocked flat."
VAGUE = "Something near the park looks wrong and should be checked."

Answers = dict[str, Any]
QUESTIONS = [
    "q:details/noticed",
    "q:details/places",
    "q:timing/seen_at",
    "q:triage/asset",
    "q:triage/hazard",
    "q:triage/severity",
]
ORACLES = ("classifier", "extractor", "registry")


@pytest.fixture(scope="module")
def loaded() -> LoadedDesign:
    return load(works.DESIGN)


def found(text: str, value: str, at: str | None = None) -> dict[str, Any]:
    """An extraction answer whose span is where ``at`` (default ``value``) sits in ``text``."""
    start = text.index(at or value)
    end = start + len(at or value)
    return {"value": value, "guess": None, "p_null": 0.05, "confidence": 0.9, "span": [start, end]}


def listed(text: str, *values: str) -> dict[str, Any]:
    """A list extraction answer with one span per item."""
    spans = [[text.index(v), text.index(v) + len(v)] for v in values]
    return {"value": list(values), "guess": None, "p_null": 0.05, "confidence": 0.9, "span": spans}


def none_found(p_null: float) -> dict[str, Any]:
    return {"value": None, "guess": None, "p_null": p_null, "confidence": 0.8, "span": None}


def noul(yes: float) -> dict[str, float]:
    return {"yes": yes, "no": 1 - yes}


def top(labels: list[Any], pick: str) -> dict[str, float]:
    """A distribution putting 0.9 on ``pick`` and sharing the rest."""
    names = [label if isinstance(label, str) else label["name"] for label in labels]
    rest = [name for name in names if name != pick]
    return {pick: 0.9} | dict.fromkeys(rest, 0.1 / len(rest))


def answers(
    text: str,
    *,
    hazard: float = 0.1,
    asset: str = "road",
    severity: str = "moderate",
    noticed: dict[str, Any] | None = None,
    places: dict[str, Any] | None = None,
    seen_at: dict[str, Any] | None = None,
) -> Answers:
    return {
        "q:triage/hazard": noul(hazard),
        "q:triage/asset": top(works.ASSETS, asset),
        "q:triage/severity": top(works.SEVERITY, severity),
        "q:details/noticed": noticed or none_found(0.9),
        "q:details/places": places or none_found(0.9),
        "q:timing/seen_at": seen_at or none_found(0.9),
    }


def fixed(
    given: Answers, open_orders: list[str], seen: list[tuple[SourceCall, ...]]
) -> OracleBinding:
    """One binding for every oracle: the models' answers, or the registry's list."""

    async def binding(calls: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        seen.append(calls)
        values: dict[str, Any] = {}
        for call in calls:
            for ref in (*call.refs, *call.questions):
                values[ref] = open_orders if ref == "t:open_orders" else given[ref]
        return CallResult(values)

    return binding


def decide(
    loaded: LoadedDesign,
    report: str,
    given: Answers,
    open_orders: list[str] | None = None,
    seen: list[tuple[SourceCall, ...]] | None = None,
) -> Decision:
    calls = seen if seen is not None else []
    binding = fixed(given, open_orders or [], calls)
    state = {"report": report, "filed_on": FILED_ON}
    return decide_sync(loaded, state, dict.fromkeys(ORACLES, binding))


def entry(decision: Decision, ref: str) -> dict[str, Any]:
    return next(i for i in decision.record.log if i["item"] == "entry" and i["ref"] == ref)


TURNS: dict[str, tuple[str, Callable[[], Answers], list[str]]] = {
    "execute": (
        LIGHT,
        lambda: answers(
            LIGHT,
            asset="lighting",
            noticed=found(LIGHT, "2026-03-03", at="3 March"),
            places=listed(LIGHT, "Alder Lane"),
            seen_at=found(LIGHT, "21:00:00", at="9 pm"),
        ),
        [],
    ),
    "execute-undated": (
        UNDATED,
        lambda: answers(UNDATED, asset="signage", places=listed(UNDATED, "Birch Close")),
        [],
    ),
    "escalate-hazard": (
        DRAIN,
        lambda: answers(DRAIN, hazard=0.95, asset="drainage", places=listed(DRAIN, "Mill Road")),
        [],
    ),
    "escalate-abstained": (VAGUE, lambda: answers(VAGUE, places=none_found(0.8)), []),
    "inform-other": (
        SHELTER,
        lambda: answers(SHELTER, asset="other", places=listed(SHELTER, "Quay Street")),
        [],
    ),
    "inform-open": (
        POTHOLE,
        lambda: answers(POTHOLE, severity="serious", places=listed(POTHOLE, "Station Road")),
        ["Station Road"],
    ),
}


def run(loaded: LoadedDesign, name: str) -> Decision:
    report, given, open_orders = TURNS[name]
    return decide(loaded, report, given(), open_orders)


# --- the design ---------------------------------------------------------------------------


def test_the_document_reloads_to_the_same_hash_and_bytes(loaded: LoadedDesign) -> None:
    again = load(loaded.document)
    assert again.design_hash == loaded.design_hash
    assert again.document == loaded.document


def test_every_function_component_imports_from_the_design_module(loaded: LoadedDesign) -> None:
    document = json.loads(loaded.document)
    components = [*document["rules"].values(), document["policy"]]
    assert len(components) == len(works.RULES) + 1
    for component in components:
        assert component["module"] == "street_reports_design"
        assert "<" not in component["qualname"]
        assert component["source"] is not None


def test_it_asks_each_kind_the_binding_must_handle(loaded: LoadedDesign) -> None:
    seen: list[tuple[SourceCall, ...]] = []
    report, given, open_orders = TURNS["execute"]
    decide(loaded, report, given(), open_orders, seen)
    by_id = {tuple(call.id for call in batch): batch for batch in seen}
    assert sorted(by_id) == [("details", "timing"), ("registry",), ("triage",)]
    details, timing = by_id[("details", "timing")]
    (triage,) = by_id[("triage",)]
    (registry,) = by_id[("registry",)]
    assert list(details.questions) == ["q:details/noticed", "q:details/places"]
    assert list(timing.questions) == ["q:timing/seen_at"]
    assert list(triage.questions) == ["q:triage/asset", "q:triage/hazard", "q:triage/severity"]
    assert set(details.needs) == {"t:state.filed_on", "t:state.report"}
    assert set(timing.needs) == {"t:state.report"}
    assert set(triage.needs) == {"t:state.report"}
    assert set(registry.needs) == {"t:state"}
    asked = (details, timing, triage)
    kinds = {ref: q.kind for call in asked for ref, q in call.questions.items()}
    assert kinds == dict.fromkeys(QUESTIONS[:3], "extraction") | {
        "q:triage/asset": "choice",
        "q:triage/hazard": "noul",
        "q:triage/severity": "score",
    }
    time = timing.questions["q:timing/seen_at"]
    assert (time.value_type, time.is_list, time.reference_date) == ("time", False, None)
    date = details.questions["q:details/noticed"]
    assert (date.value_type, date.is_list, date.reference_date) == ("date", False, FILED_ON)
    assert date.text == "t:state.report"
    places = details.questions["q:details/places"]
    assert (places.value_type, places.is_list, places.reference_date) == ("text", True, None)
    assert date.descriptions == places.descriptions == ()
    assert triage.questions["q:triage/hazard"].descriptions == (None, None)
    asset = triage.questions["q:triage/asset"].descriptions
    assert asset[-1] is None
    assert all(d is not None for d in asset[:-1])
    assert all(d is not None for d in triage.questions["q:triage/severity"].descriptions)
    assert registry.refs == ("t:open_orders",)
    assert registry.questions == {}


@pytest.mark.slow
def test_its_lints_pass(loaded: LoadedDesign) -> None:
    inputs: Mapping[str, list[Any]] = {
        "q:details/noticed": ["2026-03-03"],
        "q:details/places": [["Alder Lane"]],
        "q:timing/seen_at": ["21:00:00"],
        "t:open_orders": [[]],
        "t:state.filed_on": [FILED_ON],
        "t:state.report": [LIGHT],
    }
    report = dead_questions(loaded, values=inputs)
    assert {ref: q["status"] for ref, q in report["questions"].items()} == dict.fromkeys(
        QUESTIONS, "live"
    )
    reads = {
        "d:noticed": ["2026-03-03"],
        "d:places": [["Alder Lane"]],
        "d:seen_at": ["21:00:00"],
        "t:open_orders": [[], ["Alder Lane"]],
    }
    assert unsure_to_execute(loaded, values=reads)["execute"] == []


# --- the policy's table, row by row -------------------------------------------------------


def test_a_settled_report_opens_a_work_order(loaded: LoadedDesign) -> None:
    decision = run(loaded, "execute")
    assert decision.plan.kind is PlanKind.execute
    assert decision.verdict.kind == "passed"
    (step,) = decision.plan.steps
    assert step.tool == "open_work_order"
    assert entry(decision, "d:noticed")["value"] == "2026-03-03"
    assert entry(decision, "d:places")["value"] == ["Alder Lane"]
    assert entry(decision, "d:seen_at")["value"] == "21:00:00"


def test_a_report_with_no_date_opens_an_order_without_one(loaded: LoadedDesign) -> None:
    decision = run(loaded, "execute-undated")
    assert decision.plan.kind is PlanKind.execute
    assert entry(decision, "d:noticed")["value"] is None


def test_a_missing_date_the_model_thinks_is_there_abstains(loaded: LoadedDesign) -> None:
    report, given, open_orders = TURNS["execute"]
    decision = decide(loaded, report, given() | {"q:details/noticed": none_found(0.1)}, open_orders)
    assert decision.plan.kind is PlanKind.escalate
    assert decision.plan.reason == "d:noticed Unsure: abstained on q:details/noticed"


def test_a_hazard_escalates(loaded: LoadedDesign) -> None:
    decision = run(loaded, "escalate-hazard")
    assert decision.plan.kind is PlanKind.escalate
    assert decision.plan.reason == "a hazard to people: send a crew now"


def test_an_abstaining_rule_escalates_naming_its_answer(loaded: LoadedDesign) -> None:
    decision = run(loaded, "escalate-abstained")
    assert decision.plan.kind is PlanKind.escalate
    assert decision.plan.reason == "d:places Unsure: abstained on q:details/places"
    marker = entry(decision, "d:places")["marker"]
    assert marker["cause"] == {"code": "abstained", "refs": ["q:details/places"]}


def test_an_asset_the_desk_does_not_keep_informs(loaded: LoadedDesign) -> None:
    decision = run(loaded, "inform-other")
    assert decision.plan.kind is PlanKind.inform
    assert decision.plan.reply == "Not a public-works asset; passed to the front desk."


def test_a_place_with_an_open_order_informs(loaded: LoadedDesign) -> None:
    decision = run(loaded, "inform-open")
    assert decision.plan.kind is PlanKind.inform
    assert decision.plan.reply == "A work order is already open at Station Road."


@pytest.mark.parametrize("name", list(TURNS))
def test_each_turn_replays_with_no_findings(loaded: LoadedDesign, name: str) -> None:
    decision = run(loaded, name)
    rerun, findings = replay_sync(decision.record, loaded)
    assert findings == []
    assert rerun is not None
    assert rerun.record_hash == decision.record.record_hash
