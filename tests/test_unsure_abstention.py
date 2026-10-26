"""The ``Unsure``-to-execute lint's abstention search over a plain question: a rule on the
policy's lineage that turns a pick's or ``extract``'s ``Unsure`` into a value lets the policy
execute on it, so the lint runs the turn with each question a lineage rule reads at its
abstention point, the reads the lineage rests on over their points, and reports each run that
executes on it."""

from typing import Any

import pytest

from extraction import STATE_T
from extraction import question as extraction_question
from ommat import (
    Design,
    Plan,
    Policy,
    Question,
    ReadView,
    Rule,
    Source,
    SourceRequirement,
    Text,
    When,
    load,
)
from ommat import types as t
from ommat._markers import Undetermined
from ommat.helpers import (
    agree,
    determined,
    extract,
    pick_each_with_margin,
    pick_with_margin,
    pick_with_thresholds,
    unsure_to_execute,
)
from ommat.helpers._points import is_abstention

A, B, SPARE = "q:a", "q:b", "q:spare"
ZEROS = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0, "no": 0}}
VALUES: dict[str, Any] = {"d:kept": [["a"]]}


def picked(r: ReadView) -> dict[str, object]:
    return {ref: pick_with_margin(r, ref, 0.8, 0.2) for ref in (A, B)}


def dropped(r: ReadView) -> object:
    """The questions picked yes; an abstaining one is left out, its doubt dropped."""
    return [ref for ref, pick in picked(r).items() if pick == "yes"]


def passed_on(r: ReadView) -> object:
    picks = picked(r)
    for pick in picks.values():
        if isinstance(pick, Undetermined):
            return pick
    return [ref for ref, pick in picks.items() if pick == "yes"]


def unpicked(r: ReadView) -> object:
    r.read(A)
    r.read(B)
    return ["a"]


def agreed(r: ReadView) -> object:
    """A conflict the rule drops: the rule's judgment of its own evidence, out of scope."""
    got = agree(r, A, B)
    return [] if isinstance(got, Undetermined) else ["a"]


def mid(r: ReadView) -> object:
    """Passes a pick's marker on."""
    return pick_with_margin(r, A, 0.8, 0.2)


def via_determined(r: ReadView) -> object:
    """Drops the marker ``d:mid`` passed on, through ``determined``."""
    got = determined(r, "d:mid")
    return [] if not isinstance(got, tuple) else [got[0]]


def spare_pick(r: ReadView) -> object:
    pick_with_margin(r, A, 0.8, 0.2)
    return []


def act(r: ReadView) -> Plan:
    if isinstance(r.read("d:kept"), list):
        return Plan("execute")
    return Plan("escalate", reason="not settled")


def act_on_nothing(r: ReadView) -> Plan:
    return Plan("execute")


def design(
    rule: Any = dropped,
    reads: tuple[str, ...] = (A, B),
    extra: dict[str, Rule] | None = None,
    questions: tuple[str, ...] = (A, B, SPARE),
) -> Design:
    rules = {"kept": Rule("d:kept", t.list_(t.string), rule, version="1", reads=list(reads))}
    rules.update(extra or {})
    return Design(
        state=t.record(),
        sources={"ask": Source({q: Question("noul", Text("Is it so?")) for q in questions})},
        rules=rules,
        policy=Policy(act, version="1", reads=["d:kept"]),
    )


def test_a_dropped_pick_over_a_plain_question_is_reported() -> None:
    report = unsure_to_execute(load(design()), values=VALUES)
    assert [item["read"] for item in report["execute"]] == [A, B]  # one finding per target
    for item in report["execute"]:
        assert item["point"][item["read"]] == ZEROS
    # One policy call with d:kept Unsure, then each target's first run, which finds it.
    assert report["points"] == 1 + 1 + 1


@pytest.mark.parametrize(
    "rule", [passed_on, unpicked, agreed], ids=["passed on", "no pick", "a conflict"]
)
def test_an_abstention_the_rule_does_not_drop_is_not_reported(rule: Any) -> None:
    report = unsure_to_execute(load(design(rule=rule)), values=VALUES)
    assert report == {"points": 7, "execute": []}


def test_a_pick_off_the_lineage_is_not_counted() -> None:
    extra = {"spare": Rule("d:spare", t.list_(t.string), spare_pick, version="1", reads=[A])}
    report = unsure_to_execute(load(design(rule=unpicked, extra=extra)), values=VALUES)
    assert report == {"points": 7, "execute": []}


def test_a_pick_passed_on_and_dropped_through_determined_is_reported() -> None:
    extra = {"mid": Rule("d:mid", t.label("yes", "no"), mid, version="1", reads=[A])}
    report = unsure_to_execute(
        load(design(rule=via_determined, reads=("d:mid",), extra=extra)), values=VALUES
    )
    assert {item["read"] for item in report["execute"]} == {A}


def passes_ab(r: ReadView) -> object:
    for q in (A, B):
        pick = pick_with_margin(r, q, 0.8, 0.2)
        if isinstance(pick, Undetermined):
            return pick
    return []


def passes_abc(r: ReadView) -> object:
    for q in (A, B, "q:c"):
        pick = pick_with_margin(r, q, 0.8, 0.2)
        if isinstance(pick, Undetermined):
            return pick
    return []


@pytest.mark.parametrize(
    ("rule", "questions", "runs"),
    [(passes_ab, (A, B, SPARE), 2 * (1 + 2)), (passes_abc, (A, B, "q:c", SPARE), 3 * (1 + 2 + 2))],
    ids=["two targets", "three targets"],
)
def test_with_no_finding_each_target_varies_each_other_ranging_read_one_at_a_time(
    rule: Any, questions: tuple[str, ...], runs: int
) -> None:
    # Each target at its abstention point: one run with every other read at its first point,
    # then one per other point of each other target, never their cross product; q:spare held.
    reads = tuple(q for q in questions if q != SPARE)
    loaded = load(design(rule=rule, reads=reads, questions=questions))
    report = unsure_to_execute(loaded, values=VALUES)
    assert report == {"points": 1 + runs, "execute": []}


def opened(r: ReadView) -> object:
    return pick_with_margin(r, "q:door", 0.5, 0.0)


def test_a_question_gating_the_target_s_source_ranges() -> None:
    # q:door, read only by the gate rule d:open, decides through it whether q:a's source is
    # asked; at q:door's first point it is not, so holding q:door would hide the finding.
    designed = Design(
        state=t.record(),
        sources={
            "door": Source({"q:door": Question("noul", Text("Open?"))}),
            "ask": Source(
                {A: Question("noul", Text("Is it so?"))},
                requirement=SourceRequirement("d:open", ["yes"]),
            ),
        },
        rules={
            "open": Rule("d:open", t.label("yes", "no"), opened, version="1", reads=["q:door"]),
            "kept": Rule("d:kept", t.list_(t.string), dropped_a, version="1", reads=[A]),
        },
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    report = unsure_to_execute(load(designed), values=VALUES)
    on_a = [item for item in report["execute"] if item["read"] == A]
    assert on_a
    assert all(item["point"]["q:door"]["probabilities"]["yes"] == 1 for item in on_a)
    # q:door's own abstention is laundered too: d:open is Unsure, q:a's source is unasked,
    # and d:kept drops the unasked pick into a list.
    assert {item["read"] for item in report["execute"]} == {A, "q:door"}


def dropped_a(r: ReadView) -> object:
    return ["a"] if pick_with_margin(r, A, 0.8, 0.2) == "yes" else []


def owed(r: ReadView) -> object:
    got = extract(r, "q:owed", 0.5)
    return 0.0 if isinstance(got, Undetermined) else got


def act_on_amount(r: ReadView) -> Plan:
    amount = r.read("d:amount")
    return Plan("execute") if isinstance(amount, float) else Plan("escalate", reason="no")


def test_a_dropped_extract_abstention_is_reported() -> None:
    designed = Design(
        state=STATE_T,
        sources={"read": Source({"q:owed": extraction_question()})},
        rules={"amount": Rule("d:amount", t.number(), owed, version="1", reads=["q:owed"])},
        policy=Policy(act_on_amount, version="1", reads=["d:amount"]),
    )
    values = {
        "d:amount": [1.0],
        "q:owed": [12.0],
        "t:state.body": ["You owe 12."],
        "t:state.today": ["2026-03-10"],
    }
    report = unsure_to_execute(load(designed), values=values)
    (item,) = report["execute"]
    assert item["read"] == "q:owed"
    assert item["point"]["q:owed"]["value"] is None
    assert report["points"] == 1 + 1


def scored(r: ReadView) -> object:
    level = pick_with_thresholds(r, "q:level", {"high": 0.6}, 0.1)
    return ["a"] if level == "high" else []


def test_a_dropped_thresholds_pick_over_a_score_is_reported() -> None:
    designed = Design(
        state=t.record(),
        sources={
            "rate": Source({"q:level": Question("score", Text("How bad?"), labels=["low", "high"])})
        },
        rules={"kept": Rule("d:kept", t.list_(t.string), scored, version="1", reads=["q:level"])},
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    report = unsure_to_execute(load(designed), values=VALUES)
    (item,) = report["execute"]
    assert item["point"]["q:level"]["probabilities"] == {"low": 0, "high": 0}


def test_free_inputs_the_turns_need_are_named_before_any_run() -> None:
    designed = Design(
        state=t.record(note=t.string),
        sources={"ask": Source({A: Question("noul", Text("Is it so?"))}, needs=["t:state.note"])},
        rules={"kept": Rule("d:kept", t.list_(t.string), dropped_a, version="1", reads=[A])},
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    with pytest.raises(ValueError, match=r'unfilled: "t:state\.note"'):
        unsure_to_execute(load(designed), values=VALUES)


def flag_set(r: ReadView) -> bool:
    return r.read("t:state.flag") is True


def test_a_state_field_gating_the_target_s_source_through_a_when_ranges() -> None:
    # No rule reads t:state.flag; only the source's when does. At the flag's first point,
    # False, q:a is not asked, so holding the flag would hide the finding.
    designed = Design(
        state=t.record(flag=t.boolean),
        sources={
            "ask": Source(
                {A: Question("noul", Text("Is it so?"))},
                when=When(flag_set, version="1"),
            )
        },
        rules={"kept": Rule("d:kept", t.list_(t.string), dropped_a, version="1", reads=[A])},
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    report = unsure_to_execute(load(designed), values=VALUES)
    (item,) = report["execute"]
    assert item["read"] == A
    assert item["point"]["t:state"] == {"flag": True}


# ---------------------------------------------------------------------------------------------
# A candidate-valued target, a design with both kinds of target, and what else ranges

SENDERS = "t:senders"


def sender_dropped(r: ReadView) -> object:
    """Drops the abstention only when bob is among the labels, which only the set's second
    value renders; else passes it on."""
    pick = pick_with_margin(r, "q:sender", 0.8, 0.2)
    answer: Any = r.read("q:sender")
    if isinstance(pick, Undetermined) and "bob@x.org" not in getattr(answer, "labels", ()):
        return pick
    return []


def test_a_candidate_valued_question_abstains_over_the_labels_its_run_renders() -> None:
    designed = Design(
        state=t.record(),
        sources={
            "find": Source({SENDERS: {"type": t.list_(t.string), "candidates": True}}),
            "ask": Source(
                {
                    "q:sender": Question(
                        "choice",
                        Text("Who sent it?"),
                        {"fixed": ["none"], "candidates": SENDERS},
                    )
                }
            ),
        },
        rules={
            "kept": Rule(
                "d:kept", t.list_(t.string), sender_dropped, version="1", reads=["q:sender"]
            )
        },
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    # The label set ranges, so its second value, which alone makes the rule drop the
    # abstention, is tried.
    values = {**VALUES, SENDERS: [["ana@x.org"], ["bob@x.org"]]}
    report = unsure_to_execute(load(designed), values=values)
    (item,) = report["execute"]
    assert item["read"] == "q:sender"
    written = item["point"]["q:sender"]
    assert written["labels"] == ["none", "bob@x.org"]
    assert set(written["probabilities"].values()) == {0}


def both_kinds(r: ReadView) -> object:
    plain = pick_with_margin(r, A, 0.8, 0.2)
    each = pick_each_with_margin(r, "q:each", 0.8, 0.2)
    kept = [c for c, pick in each.items() if pick == "yes"] if isinstance(each, dict) else []
    return kept + (["a"] if plain == "yes" else [])


def mixed(set_values: list[object]) -> tuple[Design, dict[str, Any]]:
    designed = Design(
        state=t.record(),
        sources={
            "look": Source({"t:set": {"type": t.list_(t.string), "candidates": True}}),
            "ask": Source({A: Question("noul", Text("Is it so?"))}),
            "each": Source(
                {
                    "q:each": Question(
                        "noul",
                        Text("Is '{c}' so?", [{"name": "c", "origin": {"candidate": "t:set"}}]),
                    )
                },
                grounded="t:set",
            ),
        },
        rules={
            "kept": Rule("d:kept", t.list_(t.string), both_kinds, version="1", reads=[A, "q:each"])
        },
        policy=Policy(act, version="1", reads=["d:kept"]),
    )
    return designed, {**VALUES, "t:set": set_values}


def test_a_design_with_plain_and_grounded_targets_reports_each_once() -> None:
    designed, values = mixed([["x", "y"]])
    report = unsure_to_execute(load(designed), values=values)
    # Each target's first run, every other read and entry at its first point, finds it.
    assert report["points"] == 1 + 1 + 1 + 1
    assert {item["read"] for item in report["execute"]} == {A, 'q:each["x"]', 'q:each["y"]'}


def test_a_grounded_set_ranges_over_its_values() -> None:
    designed, values = mixed([[], ["x"]])
    report = unsure_to_execute(load(designed), values=values)
    # q:a's first run, the empty set, finds it; the entry exists only at the set's second
    # value, so the set must range for it to be a target at all.
    assert report["points"] == 1 + 1 + 1
    assert {item["read"] for item in report["execute"]} == {A, 'q:each["x"]'}


def gated_policy(r: ReadView) -> Plan:
    """Executes only when the tool says go; the first search sets d:kept to Unsure."""
    if r.read("t:go") is True and isinstance(r.read("d:kept"), list):
        return Plan("execute")
    return Plan("escalate", reason="no")


def test_a_policy_read_ranges_and_its_findings_follow_the_first_search_s() -> None:
    designed = Design(
        state=t.record(),
        sources={
            "tool": Source({"t:go": t.boolean}),
            "ask": Source({A: Question("noul", Text("Is it so?"))}),
        },
        rules={"kept": Rule("d:kept", t.list_(t.string), dropped_a, version="1", reads=[A])},
        policy=Policy(gated_policy, version="1", reads=["d:kept", "t:go"]),
    )
    report = unsure_to_execute(load(designed), values=VALUES)
    (item,) = report["execute"]
    # t:go ranges, true and false; only true executes on q:a's abstention.
    assert item["read"] == A
    assert item["point"]["t:go"] is True
    assert report["points"] == 2 + 2


def pass_through_policy(r: ReadView) -> Plan:
    return Plan("execute")


def test_the_first_search_s_items_come_before_the_abstention_search_s() -> None:
    designed = Design(
        state=t.record(),
        sources={"ask": Source({A: Question("noul", Text("Is it so?"))})},
        rules={"kept": Rule("d:kept", t.list_(t.string), dropped_a, version="1", reads=[A])},
        policy=Policy(pass_through_policy, version="1", reads=["d:kept"]),
    )
    report = unsure_to_execute(load(designed), values=VALUES)
    assert [item["read"] for item in report["execute"]] == ["d:kept", A]


@pytest.mark.parametrize(
    ("value", "abstains"),
    [
        ({"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0, "no": 0}}, True),
        ({"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0, "no": 1}}, False),
        ({"kind": "extraction", "value": None, "p_null": 1}, True),
        ({"kind": "extraction", "value": 12, "p_null": 0}, False),
        ({"kind": "Unsure", "cause": {"code": "abstained", "refs": []}}, False),
        ([0, 0], False),
    ],
)
def test_an_abstention_answer_is_every_probability_0_or_a_null_value(
    value: object, abstains: bool
) -> None:
    assert is_abstention(value) is abstains


def test_a_key_for_a_question_its_type_fills_is_refused() -> None:
    with pytest.raises(ValueError, match='"q:a", which its grid points or its type fill'):
        unsure_to_execute(load(design()), values={**VALUES, A: [{"yes": 1, "no": 0}]})
