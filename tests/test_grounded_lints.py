# pyright: reportPrivateUsage=false
"""Both lints over grounded forms: the first search fills a grounded declared read's set as a
declared read and sets each grounded entry, not the declared ref, to ``Unsure``; the
abstention search reports a grounded rule's one-ref pick at the subject's candidate, with
``via`` the entry naming the subject's declared ref; the grid holds nothing for a cell over a
candidate read; the dead-question lint grounds a ``t:`` source's entries."""

from typing import Any

import test_grounded_plans as plans
import test_grounded_rules as rules
import test_grounded_sources as sources
from grounded_lint_design import ANA, BO, DESIGN, FIT, OK, SET
from ommat import load
from ommat._refs import grounded_ref
from ommat.helpers import dead_questions, unsure_to_execute

LOADED = load(DESIGN)
VALUES: dict[str, Any] = {SET: [[ANA, BO]], "t:state.people": [[ANA, BO]]}


def test_the_first_search_sets_each_grounded_entry_to_unsure() -> None:
    found = unsure_to_execute(LOADED, values=VALUES)["execute"]
    first = [f for f in found if f["read"].startswith(FIT)]
    unsure = {"kind": "Unsure", "cause": {"code": "abstained", "refs": [OK]}}
    assert first == [
        {
            "read": grounded_ref(FIT, "Ana"),
            "point": {FIT: {"Ana": unsure, "Bo": "yes"}, SET: [ANA, BO]},
        },
        {
            "read": grounded_ref(FIT, "Bo"),
            "point": {FIT: {"Ana": "yes", "Bo": unsure}, SET: [ANA, BO]},
        },
    ]


def test_an_undeclared_set_is_held_but_never_set_to_unsure() -> None:
    found = unsure_to_execute(LOADED, values=VALUES)["execute"]
    assert SET not in {f["read"] for f in found}


def test_the_abstention_search_reports_a_grounded_rules_pick_with_its_entry_as_via() -> None:
    report = unsure_to_execute(LOADED, values=VALUES)
    second = [f for f in report["execute"] if f["read"].startswith(OK)]
    assert [(f["read"], f["via"]) for f in second] == [
        (grounded_ref(OK, "Ana"), grounded_ref(FIT, "Ana")),
        (grounded_ref(OK, "Bo"), grounded_ref(FIT, "Bo")),
    ]
    assert second[0]["point"][grounded_ref(OK, "Ana")]["probabilities"] == {"yes": 0.0, "no": 0.0}


def test_the_first_search_over_a_grounded_plan_finds_no_execute_on_unsure() -> None:
    values = {SET: [[plans.ANA, plans.BO]], plans.ADDR: ["a@example.org"]}
    report = unsure_to_execute(plans.LOADED, values=values)
    assert report["execute"] == []
    # d:ready at two points crossed with the set and the two entries, each set in turn.
    assert report["points"] > 0


def test_a_cell_over_a_candidate_read_adds_nothing_to_the_grid() -> None:
    assert dict(rules.LOADED._grid) == {}


def test_the_dead_question_lint_grounds_a_t_sources_entries() -> None:
    values: dict[str, list[Any]] = {
        "t:state.people": [[sources.ANA, sources.BO]],
        sources.CITY: ["Oslo", None],
        sources.INFO: [{"city": "x"}],
    }
    report = dead_questions(sources.LOADED, values=values)
    assert report["questions"] == {sources.OUT: {"status": "not-on-grid", "witness": None}}
    assert report["points"] > 0


def test_the_first_search_sets_no_grounded_t_entry_to_unsure() -> None:
    import dataclasses

    from ommat import Plan, Policy, ReadView

    def policy(r: ReadView) -> Plan:
        r.read(sources.CITY)
        return Plan("execute")

    d = dataclasses.replace(
        sources.design(), policy=Policy(policy, version="1", reads=[sources.CITY])
    )
    values: dict[str, list[Any]] = {
        SET: [[sources.ANA, sources.BO]],
        sources.CITY: ["Oslo", None],
    }
    report = unsure_to_execute(load(d), values=values)
    assert not any(f["read"].startswith("t:") for f in report["execute"])


def test_the_abstention_search_finds_a_plain_question_a_grounded_rule_picks() -> None:
    import dataclasses

    from ommat import Question, ReadView, Rule, Source, Text
    from ommat import types as t
    from ommat.helpers import pick_with_margin

    plain = "q:plain/ok"

    def picked(r: ReadView) -> object:
        return pick_with_margin(r, plain, 0.6, 0.2)

    d = dataclasses.replace(
        DESIGN,
        sources={"plain": Source({plain: Question("noul", Text("Ok?"))}, needs=["t:state"])},
        rules={
            **DESIGN.rules,
            "fit": Rule(
                FIT,
                t.label("no", "yes"),
                picked,
                version="1",
                reads=[plain, "candidate"],
                grounded=SET,
            ),
        },
    )
    report = unsure_to_execute(load(d), values=VALUES)
    found = {f["read"]: f.get("via") for f in report["execute"]}
    assert found[plain] == grounded_ref(FIT, "Ana")
