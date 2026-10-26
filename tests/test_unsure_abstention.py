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
    pick_with_margin,
    pick_with_thresholds,
    unsure_to_execute,
)

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
    assert [item["read"] for item in report["execute"]] == [A] * 3 + [B] * 3
    for item in report["execute"]:
        assert item["point"][item["read"]] == ZEROS
    # One policy call with d:kept Unsure, then for each target the other question's three
    # points, q:spare held at its first.
    assert report["points"] == 1 + 3 + 3


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


def picks_ab(r: ReadView) -> object:
    return [q for q in (A, B) if pick_with_margin(r, q, 0.8, 0.2) == "yes"]


def picks_abc(r: ReadView) -> object:
    return [q for q in (A, B, "q:c") if pick_with_margin(r, q, 0.8, 0.2) == "yes"]


@pytest.mark.parametrize(
    ("rule", "questions", "runs"),
    [(picks_ab, (A, B, SPARE), 3 + 3), (picks_abc, (A, B, "q:c", SPARE), 9 + 9 + 9)],
    ids=["two targets", "three targets"],
)
def test_each_target_runs_the_cross_of_the_other_ranging_reads(
    rule: Any, questions: tuple[str, ...], runs: int
) -> None:
    # Each target at its abstention point, each other target over three points, q:spare held.
    reads = tuple(q for q in questions if q != SPARE)
    loaded = load(design(rule=rule, reads=reads, questions=questions))
    assert unsure_to_execute(loaded, values=VALUES)["points"] == 1 + runs


def opened(r: ReadView) -> object:
    return pick_with_margin(r, "q:door", 0.5, 0.0)


def test_a_question_gating_the_target_s_source_ranges() -> None:
    # q:door, which no lineage rule reads, decides through d:open whether q:a's source is
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
                needs=["t:state.flag"],
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
