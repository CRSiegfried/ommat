"""The ``Unsure``-to-execute lint over a per-candidate pick: a rule that drops one candidate's
abstention from its value lets the policy execute on it, so the lint runs the turn with each
grounded entry at the abstention point and reports each run that executes on it."""

import contextlib
from typing import Any

import pytest

from ommat import Design, Plan, Policy, Question, ReadView, Rule, Source, Text, load
from ommat import types as t
from ommat._markers import Undetermined
from ommat._refs import grounded_ref
from ommat.helpers import (
    determined,
    pick_each_with_margin,
    pick_each_with_thresholds,
    unsure_to_execute,
)

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
    assert reads == [grounded_ref(OUT, "a"), grounded_ref(OUT, "b")]  # one per target
    for item in report["execute"]:
        assert item["point"][item["read"]] == ABSTAINED
    # One policy call with d:kept Unsure, then each candidate's first run, which finds it.
    assert report["points"] == 1 + 1 + 1


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


# ---------------------------------------------------------------------------------------------
# What the search counts as a pick, and what it runs

OTHER = "q:other"


def wording(set_ref: str) -> Text:
    return Text("Is '{c}' new?", [{"name": "c", "origin": {"candidate": set_ref}}])


def kept_by_thresholds(r: ReadView) -> object:
    picks = pick_each_with_thresholds(r, OUT, {"yes": 0.8, "no": 0.8}, 0.2)
    if not isinstance(picks, dict):
        return picks
    return [c for c, pick in picks.items() if pick == "yes"]


def caught(r: ReadView) -> object:
    """A pick that raises on its bar, caught: it returned nothing, so it rests on nothing."""
    with contextlib.suppress(ValueError):
        pick_each_with_margin(r, OUT, 2.0, 0.0)
    return []


def by_determined(r: ReadView) -> object:
    got = determined(r, OUT)
    return got if not isinstance(got, tuple) else []


def kept_other(r: ReadView) -> object:
    """Reads both questions, picks over the other one only."""
    r.read(OUT)
    picks = pick_each_with_margin(r, OTHER, 0.8, 0.2)
    if not isinstance(picks, dict):
        return picks
    return [c for c, pick in picks.items() if pick == "yes"]


def spare(r: ReadView) -> object:
    picks = pick_each_with_margin(r, OUT, 0.8, 0.2)
    return [] if isinstance(picks, dict) else picks


def two(rule: Any, reads: tuple[str, ...] = (OUT,), extra: dict[str, Rule] | None = None) -> Design:
    """Two per-candidate questions over one set; the policy reads ``d:kept``."""
    rules = {"kept": Rule("d:kept", t.list_(t.string), rule, version="1", reads=list(reads))}
    rules.update(extra or {})
    return Design(
        state=t.record(),
        sources={
            "look": Source({SET: {"type": t.list_(t.string), "candidates": True}}),
            "ask": Source(
                {OUT: Question("noul", wording(SET)), OTHER: Question("noul", wording(SET))},
                grounded=SET,
            ),
        },
        rules=rules,
        policy=Policy(act, version="1", reads=["d:kept"]),
    )


def test_a_thresholds_pick_is_reported_as_a_margin_pick_is() -> None:
    report = unsure_to_execute(load(design(rule=kept_by_thresholds)), values=VALUES)
    assert len(report["execute"]) == 2


@pytest.mark.parametrize("rule", [caught, by_determined], ids=["a caught raise", "determined"])
def test_a_rule_that_returned_no_pick_over_the_question_is_not_reported(rule: Any) -> None:
    report = unsure_to_execute(load(design(rule=rule)), values=VALUES)
    assert report == {"points": 7, "execute": []}


def test_a_pick_over_another_question_reports_only_that_question_s_entries() -> None:
    report = unsure_to_execute(load(two(kept_other, reads=(OUT, OTHER))), values=VALUES)
    assert {item["read"] for item in report["execute"]} == {
        grounded_ref(OTHER, "a"),
        grounded_ref(OTHER, "b"),
    }


def test_a_pick_off_the_lineage_is_not_counted_for_a_rule_on_it() -> None:
    extra = {"spare": Rule("d:spare", t.list_(t.string), spare, version="1", reads=[OUT])}
    report = unsure_to_execute(load(two(unpicked, extra=extra)), values=VALUES)
    assert report["execute"] == []


def test_a_question_no_lineage_rule_reads_holds_its_first_point() -> None:
    # Without the hold, q:other's three points per candidate would cross every run: 7 to 55.
    report = unsure_to_execute(load(two(kept)), values=VALUES)
    assert report["points"] == 1 + 1 + 1
    assert len(report["execute"]) == 2


def fixed_folders(r: ReadView) -> object:
    return ["a", "b"]


def test_a_d_set_is_learned_by_a_probe_run_that_counts() -> None:
    d_set = Rule("d:folders", t.list_(t.string), fixed_folders, version="1", candidates=True)
    designed = Design(
        state=t.record(),
        sources={
            "ask": Source({OUT: Question("noul", wording("d:folders"))}, grounded="d:folders")
        },
        rules={
            "folders": d_set,
            "kept": Rule("d:kept", t.list_(t.string), kept, version="1", reads=[OUT]),
        },
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    report = unsure_to_execute(load(designed), values={"d:kept": [["a"]]})
    assert report["points"] == 1 + 1 + 2  # the policy call, the probe, a run per candidate
    assert len(report["execute"]) == 2


def kept_b(r: ReadView) -> object:
    picks = pick_each_with_margin(r, "q:second", 0.8, 0.2)
    if not isinstance(picks, dict):
        return picks
    return [c for c, pick in picks.items() if pick == "yes"]


def act_on_b(r: ReadView) -> Plan:
    settled = isinstance(r.read("d:kept"), list) and isinstance(r.read("d:kept_b"), list)
    return Plan("execute") if settled else Plan("escalate", reason="x")


def test_a_set_that_moves_with_the_entries_raises() -> None:
    chained = Design(
        state=t.record(),
        sources={
            "look": Source({SET: {"type": t.list_(t.string), "candidates": True}}),
            "ask": Source({OUT: Question("noul", wording(SET))}, grounded=SET),
            "again": Source({"q:second": Question("noul", wording("d:kept"))}, grounded="d:kept"),
        },
        rules={
            "kept": Rule(
                "d:kept", t.list_(t.string), kept, version="1", reads=[OUT], candidates=True
            ),
            "kept_b": Rule("d:kept_b", t.list_(t.string), kept_b, version="1", reads=["q:second"]),
        },
        policy=Policy(act_on_b, version="1", reads=["d:kept", "d:kept_b"]),
    )
    # The policy reads d:kept, so q:out's entries range, and d:kept, which grounds
    # q:second, moves with them.
    values = {SET: [["a", "b"]], "d:kept": [["a"]], "d:kept_b": [["a"]]}
    with pytest.raises(ValueError, match="changes within one assignment"):
        unsure_to_execute(load(chained), values=values)


@pytest.mark.parametrize(("folders", "runs"), [(["a", "b"], 2 * 3), (["a", "b", "c"], 3 * 5)])
def test_with_no_finding_each_entry_varies_the_others_one_at_a_time(
    folders: list[str], runs: int
) -> None:
    # Each of E entries in turn at the abstention point: one run with the others at their
    # first point, then one per other point of each other entry, 1 + 2 (E - 1) runs.
    loaded = load(design(rule=passed_on))
    report = unsure_to_execute(loaded, values={SET: [folders], "d:kept": [["a"]]})
    assert report == {"points": 1 + runs, "execute": []}
