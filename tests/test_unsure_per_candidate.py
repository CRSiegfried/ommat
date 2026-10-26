"""The ``Unsure``-to-execute lint over a per-candidate pick: a rule that drops one candidate's
abstention from its value lets the policy execute on it, so the lint runs the turn with each
grounded entry at the abstention point and reports each run that executes on it."""

from typing import Any

import pytest

from ommat import Design, Plan, Policy, Question, ReadView, Rule, Source, Text, load
from ommat import types as t
from ommat._markers import Undetermined
from ommat._refs import grounded_ref
from ommat.helpers import pick_each_with_margin, unsure_to_execute

SET = "t:folders"
OUT = "q:out"
ABSTAINED = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0, "no": 0}}


def kept(r: ReadView) -> object:
    """The folders picked yes; an abstaining folder is left out, its doubt dropped."""
    picks = pick_each_with_margin(r, OUT, 0.8, 0.2)
    if not isinstance(picks, dict):
        return picks
    return [c for c, pick in picks.items() if pick == "yes"]


def passed_on(r: ReadView) -> object:
    """The same, passing the first undecided folder's marker on."""
    picks = pick_each_with_margin(r, OUT, 0.8, 0.2)
    if not isinstance(picks, dict):
        return picks
    for pick in picks.values():
        if isinstance(pick, Undetermined):
            return pick
    return [c for c, pick in picks.items() if pick == "yes"]


def unpicked(r: ReadView) -> object:
    """The folders answered yes at 0.8, read with no pick."""
    answers: Any = r.read(OUT)
    if not isinstance(answers, dict):
        return answers
    return [
        c
        for c, a in answers.items()  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(a, Undetermined) and a.probabilities["yes"] >= 0.8  # pyright: ignore
    ]


def count(r: ReadView) -> object:
    folders = r.read("d:kept")
    return len(folders) if isinstance(folders, list) else folders  # pyright: ignore[reportUnknownArgumentType]


def act(r: ReadView) -> Plan:
    if isinstance(r.read("d:kept"), list):
        return Plan("execute")
    return Plan("escalate", reason="the folders are not settled")


def act_on_count(r: ReadView) -> Plan:
    if isinstance(r.read("d:count"), int):
        return Plan("execute")
    return Plan("escalate", reason="the count is not settled")


def act_on_nothing(r: ReadView) -> Plan:
    return Plan("execute") if r.read("t:state.go") else Plan("escalate", reason="no")


def design(
    rule: Any = kept, policy: Any = act, reads: tuple[str, ...] = ("d:kept",), go: bool = False
) -> Design:
    rules = {
        "kept": Rule("d:kept", t.list_(t.string), rule, version="1", reads=[OUT]),
        "count": Rule("d:count", t.integer(), count, version="1", reads=["d:kept"]),
    }
    return Design(
        state=t.record(go=t.boolean) if go else t.record(),
        sources={
            "look": Source({SET: {"type": t.list_(t.string), "candidates": True}}),
            "ask": Source(
                {
                    OUT: Question(
                        "noul",
                        Text("Is '{c}' new?", [{"name": "c", "origin": {"candidate": SET}}]),
                    )
                },
                grounded=SET,
            ),
        },
        rules=rules,
        policy=Policy(policy, version="1", reads=list(reads)),
    )


VALUES: dict[str, Any] = {SET: [["a", "b"]], "d:kept": [["a"]]}


def test_an_execute_on_a_candidate_s_abstention_dropped_by_a_pick_is_reported() -> None:
    report = unsure_to_execute(load(design()), values=VALUES)
    reads = [item["read"] for item in report["execute"]]
    assert reads == [grounded_ref(OUT, "a")] * 3 + [grounded_ref(OUT, "b")] * 3
    for item in report["execute"]:
        assert item["point"][item["read"]] == ABSTAINED
    # One policy call with d:kept Unsure, then three runs per candidate: the other's yes, no
    # and marker.
    assert report["points"] == 7


def test_a_decided_candidate_is_not_reported() -> None:
    report = unsure_to_execute(load(design()), values=VALUES)
    for item in report["execute"]:
        decided = [
            ref for ref, v in item["point"].items() if ref.startswith("q:") and v != ABSTAINED
        ]
        assert item["read"] not in decided
    passing = unsure_to_execute(load(design(rule=passed_on)), values=VALUES)
    assert passing == {"points": 7, "execute": []}


def test_a_pick_through_a_chain_of_rules_is_reported() -> None:
    loaded = load(design(policy=act_on_count, reads=("d:count",)))
    report = unsure_to_execute(loaded, values={SET: [["a", "b"]], "d:count": [1]})
    assert {item["read"] for item in report["execute"]} == {
        grounded_ref(OUT, "a"),
        grounded_ref(OUT, "b"),
    }


def test_a_rule_reading_the_answers_without_a_pick_is_not_reported() -> None:
    report = unsure_to_execute(load(design(rule=unpicked)), values=VALUES)
    assert report == {"points": 7, "execute": []}


def test_a_pick_off_the_policy_s_lineage_runs_no_turn() -> None:
    loaded = load(design(policy=act_on_nothing, reads=("t:state.go",), go=True))
    report = unsure_to_execute(loaded)
    assert report["points"] == 0
    assert report["execute"] == []


def test_a_read_the_per_candidate_search_needs_is_named_before_any_run() -> None:
    with pytest.raises(ValueError, match='unfilled: "t:folders"') as raised:
        unsure_to_execute(load(design()), values={"d:kept": [["a"]]})
    assert "a read the lint does not fill" not in str(raised.value)
    with pytest.raises(ValueError, match="a read the lint does not fill"):
        unsure_to_execute(load(design()), values={**VALUES, "t:nothing": [1]})
