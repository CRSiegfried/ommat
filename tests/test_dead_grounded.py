"""The dead-question lint over a per-candidate question: one grid entry on the declared ref,
each grounded entry crossed after the other reads with its own marker last, a ``d:`` set
learned by a probe run that counts as the first point, the ``not-grounded`` status, and a
report keyed by the declared ref whose points name each grounded entry."""

import asyncio
from typing import Any, cast

import pytest

from designs import roles
from ommat import Answer, Design, Parameter, Question, Rule, Source, Text, load
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


def by_new(r: Any) -> str:
    return "send" if r.read("d:new") is True else "hold"


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
    rules = {
        "new": new,
        "decision": Rule(
            "d:decision", t.label("send", "hold"), by_new, version="p", reads=["d:new"]
        ),
    }
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
        policy="d:decision",
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


def test_every_point_runs_on_one_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # A loop per point exhausts Windows's ephemeral ports on a large grid: each loop opens a
    # loopback socket pair whose ports wait minutes to close.
    made: list[object] = []
    real = asyncio.events.new_event_loop

    def counted() -> asyncio.AbstractEventLoop:
        loop = real()
        made.append(loop)
        return loop

    monkeypatch.setattr(asyncio.events, "new_event_loop", counted)
    report = dead_questions(load(design()), values={SET: [[], ["b"], ["a", "b"]]})
    assert report["points"] > 1
    assert len(made) == 1


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


def second_set(r: Any) -> list[str]:
    """``["x"]`` once candidate ``a`` of ``q:one`` is answered yes, else none: a set that moves
    with another per-candidate question's entry."""
    got: Any = r.read("q:one")
    a = cast("dict[str, Any]", got).get("a") if isinstance(got, dict) else None
    return ["x"] if isinstance(a, Answer) and a.probabilities["yes"] >= BAR else []


def by_two(r: Any) -> str:
    return "hold"


def test_a_d_set_that_changes_within_an_assignment_is_refused() -> None:
    design = Design(
        name="moving",
        state=t.record(),
        sources={
            "one": Source({"q:one": Question("noul", wording("t:s1"))}, grounded="t:s1"),
            "s1": Source({"t:s1": {"type": t.list_(t.string), "candidates": True}}),
            "two": Source({"q:two": Question("noul", wording("d:s2"))}, grounded="d:s2"),
        },
        rules={
            "s2": Rule(
                "d:s2",
                t.list_(t.string),
                second_set,
                version="1",
                reads=["q:one"],
                candidates=True,
            ),
            "decision": Rule("d:decision", t.string, by_two, version="1"),
        },
        policy="d:decision",
    )
    with pytest.raises(ValueError, match="'d:s2' changes within one assignment"):
        dead_questions(load(design), values={"t:s1": [["a"]]})


def test_a_question_that_decides_a_candidate_set_is_live() -> None:
    """Runs whose other reads agree compare at their grounded entries' first points, whatever
    entries each grounds, so a gate that empties the set is live."""
    from designs import slot as _slot
    from ommat import Answer, Design, Question, ReadView, Rule, Source, Text
    from ommat import load as _load
    from ommat import types as _t
    from ommat.helpers import dead_questions as _dead

    person = _t.record(name=_t.string)

    def finder(r: ReadView) -> object:
        gate = r.read("q:gate")
        yes = isinstance(gate, Answer) and gate.probabilities["yes"] >= 0.99
        return [{"name": "Ana"}] if yes else []

    def count(r: ReadView) -> object:
        found: object = r.read("d:s")
        return f"n={len(cast('list[object]', found))}" if isinstance(found, list) else "m"

    def decision(r: ReadView) -> str:
        return str(r.read("d:n"))

    design = Design(
        name="gate",
        state=_t.record(),
        sources={
            "g": Source({"q:gate": Question("noul", Text("Gate?"))}),
            "two": Source(
                {"q:two": Question("noul", Text("Is {c}?", [_slot("c", candidate="d:s")]))},
                grounded="d:s",
            ),
        },
        rules={
            "s": Rule(
                "d:s", _t.list_(person), finder, version="1", reads=["q:gate"], candidates="name"
            ),
            "n": Rule("d:n", _t.string, count, version="1", reads=["d:s"]),
            "decision": Rule("d:decision", _t.string, decision, version="1", reads=["d:n"]),
        },
        policy="d:decision",
    )
    report = _dead(_load(design))
    assert report["questions"]["q:gate"]["status"] == "live"
