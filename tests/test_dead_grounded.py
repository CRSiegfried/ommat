"""The dead-question lint over a per-candidate question: one grid entry on the declared ref,
each grounded entry crossed after the other reads with its own marker last, a ``d:`` set
learned by a probe run that counts as the first point, the ``not-grounded`` status, and a
report keyed by the declared ref whose points name each grounded entry."""

from typing import Any, cast

from designs import roles
from ommat import Answer, Design, Parameter, Plan, Policy, Question, Rule, Source, Text, load
from ommat import types as t
from ommat.helpers import dead_questions

SET = "t:folders"
D_SET = "d:folders"
BAR = 0.8
RUNS: list[str] = []


def wording(set_ref: str) -> Text:
    return Text("Is '{c}' new?", [{"name": "c", "origin": {"candidate": set_ref}}])


def asking(set_ref: str) -> Source:
    return Source(
        {
            "q:role": Question("noul", wording(set_ref)),
            "q:plain": Question("noul", Text("Plain?")),
        },
        grounded=set_ref,
    )


def entry_b(r: Any) -> Any:
    """Candidate ``b``'s entry as the rule reads it, or ``None`` when ``b`` is not grounded."""
    got: Any = r.read("q:role")
    return cast("dict[str, Any]", got).get("b") if isinstance(got, dict) else None


def b_is_new(r: Any) -> bool:
    b = entry_b(r)
    return isinstance(b, Answer) and b.probabilities["yes"] >= r.param("role/bar")


def b_is_missing(r: Any) -> bool:
    b = entry_b(r)
    return b is not None and not isinstance(b, Answer)


def heeds_nothing(r: Any) -> bool:
    r.read("q:role")
    return False


def by_new(r: Any) -> Plan:
    return Plan("execute") if r.read("d:new") is True else Plan("escalate", reason="r")


def folders(r: Any) -> list[str]:
    RUNS.append("folders")
    return ["a", "b"] if r.read("t:state.many") else []


def design(fn: Any = b_is_new, *, d_set: bool = False) -> Design:
    set_ref = D_SET if d_set else SET
    new = Rule(
        "d:new",
        t.boolean,
        fn,
        version="1",
        reads=["q:role"],
        params=["role/bar"],
        boundaries={"q:role": roles(bar=["role/bar"])},
    )
    rules = {"new": new}
    sources = {"ask": asking(set_ref)}
    state = t.record()
    if d_set:
        state = t.record(many=t.boolean)
        rules["folders"] = Rule(
            D_SET, t.list_(t.string), folders, version="1", reads=["t:state.many"], candidates=True
        )
    else:
        sources["look"] = Source({SET: {"type": t.list_(t.string), "candidates": True}})
    return Design(
        state=state,
        sources=sources,
        rules=rules,
        policy=Policy(by_new, version="p", reads=["d:new"]),
        parameters={"role/bar": Parameter(t.number(0, 1), BAR)},
    )


def noul(yes: float, no: float) -> dict[str, Any]:
    return {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": yes, "no": no}}


# The noul axis on a bar of 0.8: 0, 0.2, 0.5, 0.8, 1, then the marker.
ENTRY = 6
PLAIN = 3  # q:plain off the grid: yes at 1, no at 1, the marker
AB = {SET: [["a", "b"]]}


def test_a_grounded_question_live_through_one_candidate_is_live_with_its_witness() -> None:
    report = dead_questions(load(design()), values=AB)
    assert list(report["questions"]) == ["q:plain", "q:role"]  # keyed by the declared ref
    assert report["questions"]["q:plain"] == {"status": "not-on-grid", "witness": None}
    role = report["questions"]["q:role"]
    assert role["status"] == "live"
    first, later = role["witness"]
    # The first assignment of the other reads; entry "a" held at its first point, and "b",
    # varying fastest, from its first point to the bar.
    assert first == {
        "q:plain": noul(1.0, 0.0),
        'q:role["a"]': noul(0.0, 1.0),
        'q:role["b"]': noul(0.0, 1.0),
        SET: ["a", "b"],
        "t:state": {},
    }
    assert later == {**first, 'q:role["b"]': noul(BAR, 0.2)}


def test_a_grounded_question_that_changes_nothing_is_no_change_found() -> None:
    report = dead_questions(load(design(heeds_nothing)), values=AB)
    assert report["questions"]["q:role"] == {"status": "no-change-found", "witness": None}


def test_an_always_empty_set_grounds_nothing_and_the_question_is_not_grounded() -> None:
    report = dead_questions(load(design()), values={SET: [[]]})
    assert report["questions"]["q:role"] == {"status": "not-grounded", "witness": None}
    assert report["points"] == PLAIN  # one run per assignment, no entry to cross


def test_the_points_sum_each_assignments_block() -> None:
    report = dead_questions(load(design()), values={SET: [[], ["b"], ["a", "b"]]})
    assert report["points"] == PLAIN * (1 + ENTRY + ENTRY * ENTRY)


def test_an_entry_at_its_own_marker_point_is_shown_under_its_grounded_ref() -> None:
    report = dead_questions(load(design(b_is_missing)), values={SET: [["b"]]})
    first, later = report["questions"]["q:role"]["witness"]
    assert list(first) == list(later) == ["q:plain", 'q:role["b"]', SET, "t:state"]
    assert "q:role" not in first
    assert later['q:role["b"]'] == {
        "kind": "Unavailable",
        "cause": {"code": "refused", "refs": ['q:role["b"]']},
    }


def test_a_d_set_is_learned_by_a_probe_run_that_counts_as_the_first_point() -> None:
    RUNS.clear()
    report = dead_questions(load(design(d_set=True)))
    # t:state.many False grounds nothing; True grounds "a" and "b", whose block's first
    # point is the probe, run once.
    assert report["points"] == PLAIN * (1 + ENTRY * ENTRY)
    assert len(RUNS) == report["points"]
    role = report["questions"]["q:role"]
    assert role["status"] == "live"
    first, later = role["witness"]
    assert first["t:state"] == later["t:state"] == {"many": True}
    assert D_SET not in first  # a d: set is no read: the run computes it
    assert (first['q:role["b"]'], later['q:role["b"]']) == (noul(0.0, 1.0), noul(BAR, 0.2))


def test_a_d_set_that_is_always_empty_is_not_grounded() -> None:
    def none(r: Any) -> list[str]:
        r.read("t:state.many")
        return []

    d = design(d_set=True)
    rules = {
        **d.rules,
        "folders": Rule(
            D_SET, t.list_(t.string), none, version="1", reads=["t:state.many"], candidates=True
        ),
    }
    d = Design(
        state=d.state, sources=d.sources, rules=rules, policy=d.policy, parameters=d.parameters
    )
    report = dead_questions(load(d))
    assert report["questions"]["q:role"] == {"status": "not-grounded", "witness": None}
    assert report["points"] == PLAIN * 2
