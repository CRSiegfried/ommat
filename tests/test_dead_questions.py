# pyright: reportPrivateUsage=false
"""The dead-question lint, ``dead_questions``: one ``decide_sync`` run
per point through a binding answering from the point, each status with its witness, the
byte comparison of the named entries, ``refs`` checked, the ``ValueError``s before any run and
for a broken entry, and a report of plain data. The points themselves are tested in
``test_read_points.py``."""

import copy
import json
import re
from typing import Any

import pytest

import ommat._decide as _decide
from designs import roles
from ommat import (
    Cause,
    CauseCode,
    Design,
    Parameter,
    Question,
    Rule,
    Source,
    Text,
    Unavailable,
    Undetermined,
    Unsure,
    load,
)
from ommat import types as t
from ommat._canonical import encode
from ommat.helpers import dead_questions, pick_with_margin, pick_with_thresholds

CALLS: list[str] = []


def go(r: Any) -> object:
    CALLS.append("go")
    picked = pick_with_margin(r, "q:go", r.param("go/bar"), r.param("go/margin"))
    return picked if isinstance(picked, Undetermined) else picked == "yes"


def mood(r: Any) -> object:
    return pick_with_thresholds(r, "q:mood", r.param("mood/th"), 0)


def by_go(r: Any) -> str:
    CALLS.append("policy")
    return "send" if r.read("d:go") is True else "hold"


def decision(reads: list[str], fn: Any) -> Rule:
    return Rule("d:decision", t.label("send", "hold"), fn, version="p", reads=reads)


QUESTIONS = {
    "q:go": Question("noul", Text("Go?")),
    "q:tone": Question("choice", Text("Tone?"), labels=("warm", "cold")),
}
GO = Rule(
    "d:go",
    t.boolean,
    go,
    version="1",
    reads=["q:go"],
    params=["go/bar", "go/margin"],
    boundaries={"q:go": roles(bar=["go/bar"], margin=["go/margin"])},
)
PARAMETERS = {
    "go/bar": Parameter(t.number(0, 1), 0.8),
    "go/margin": Parameter(t.number(0, 1), 0.2),
}


def design(**changes: Any) -> Design:
    """The rule ``d:decision`` beside ``d:go``; rules given in ``changes`` are added beside them
    unless ``changes`` says ``alone``."""
    fields: dict[str, Any] = {
        "state": t.record(),
        "sources": {"ask": Source(QUESTIONS)},
        "rules": {"go": GO},
        "parameters": PARAMETERS,
    }
    fields["rules"]["decision"] = decision(["d:go"], by_go)
    alone = changes.pop("alone", False)
    if "rules" in changes and not alone:
        changes = {**changes, "rules": {**fields["rules"], **changes["rules"]}}
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())


def answer(kind: str, labels: list[str], **probabilities: float) -> dict[str, Any]:
    return {"kind": kind, "labels": labels, "probabilities": probabilities}


# ---------------------------------------------------------------------------------------------
# Statuses and the witness


def test_a_question_whose_answer_changes_the_decision_is_live_with_its_exact_witness() -> None:
    report = dead_questions(LOADED, ["d:decision"])
    assert report["questions"]["q:go"] == {
        "status": "live",
        "witness": [
            {
                "q:go": answer("noul", ["yes", "no"], yes=0.0, no=1.0),
                "q:tone": answer("choice", ["warm", "cold"], warm=1.0, cold=0.0),
                "t:state": {},
            },
            {
                "q:go": answer("noul", ["yes", "no"], yes=0.8, no=0.2),
                "q:tone": answer("choice", ["warm", "cold"], warm=1.0, cold=0.0),
                "t:state": {},
            },
        ],
    }


def test_a_question_off_the_grid_that_changes_nothing_is_not_on_grid() -> None:
    assert dead_questions(LOADED, ["d:decision"])["questions"]["q:tone"] == {
        "status": "not-on-grid",
        "witness": None,
    }


def test_a_question_on_the_grid_that_changes_nothing_is_no_change_found() -> None:
    mood_rule = Rule(
        "d:mood",
        t.label("calm", "wild"),
        mood,
        version="1",
        reads=["q:mood"],
        params=["mood/th"],
        boundaries={"q:mood": roles(thresholds=["mood/th"])},
    )
    loaded = load(
        design(
            sources={
                "ask": Source(
                    {**QUESTIONS, "q:mood": Question("choice", Text("?"), labels=("calm", "wild"))}
                )
            },
            rules={"go": GO, "mood": mood_rule},
            parameters={
                **PARAMETERS,
                "mood/th": Parameter(t.mapping(t.number(0, 1)), {"calm": 0.6}),
            },
        )
    )
    report = dead_questions(loaded, ["d:decision"])
    assert report["questions"]["q:mood"] == {"status": "no-change-found", "witness": None}
    assert report["questions"]["q:go"]["status"] == "live"


def test_the_points_count_every_run() -> None:
    CALLS.clear()
    report = dead_questions(LOADED, ["d:decision"])
    # q:go: nine axis points and the marker; q:tone: two labels and the marker; t:state: {}.
    assert report["points"] == 10 * 3 * 1
    assert CALLS.count("policy") == 30


def test_the_binding_answers_under_default_and_every_declared_oracle_name() -> None:
    sources = {
        "ask": Source({"q:go": QUESTIONS["q:go"]}, oracle="model"),
        "look": Source({"t:n": t.integer()}),
    }
    loaded = load(design(sources=sources))
    report = dead_questions(loaded, ["d:decision"], values={"t:n": [1, 2]})
    assert report["points"] == 10 * 2
    assert report["questions"]["q:go"]["status"] == "live"
    witness = report["questions"]["q:go"]["witness"]
    assert [w["t:n"] for w in witness] == [1, 1]


def test_a_point_with_the_marker_reaches_the_rule_as_unavailable_refused() -> None:
    seen: list[object] = []

    def watch(r: Any) -> object:
        seen.append(r.read("q:go"))
        return True

    rule = Rule("d:go", t.boolean, watch, version="1", reads=["q:go"])
    loaded = load(design(sources={"ask": Source({"q:go": QUESTIONS["q:go"]})}, rules={"go": rule}))
    dead_questions(loaded, ["d:decision"])
    last = seen[-1]
    assert type(last).__name__ == "Unavailable"
    assert last.cause.code == "refused"  # type: ignore[attr-defined]
    assert last.cause.refs == ("q:go",)  # type: ignore[attr-defined]


def test_the_state_is_the_points_t_state_live() -> None:
    seen: list[object] = []

    def watch(r: Any) -> str:
        seen.append(r.read("t:state.blob"))
        return "hold"

    loaded = load(
        design(
            state=t.record(blob=t.bytes_),
            alone=True,
            rules={"decision": decision(["t:state.blob"], watch)},
            sources={},
            parameters={},
        )
    )
    report = dead_questions(loaded, ["d:decision"], values={"t:state.blob": [b"\x01"]})
    assert seen == [b"\x01"]
    assert report == {"points": 1, "questions": {}, "narrowed": []}


# ---------------------------------------------------------------------------------------------
# ValueErrors before any run


@pytest.mark.parametrize(
    ("values", "says"),
    [
        (None, 'unfilled: "t:n"'),
        ({"t:n": [1], "t:other": [1]}, '"t:other", a read the lint does not fill'),
        ({"t:n": [1], "q:go": [{"yes": 1, "no": 0}]}, '"q:go", which its grid points'),
        ({"t:n": ["x"]}, 'a value for "t:n" does not conform'),
        ({"t:n": [1, 1]}, 'values for "t:n" lists one value twice'),
    ],
)
def test_a_value_error_is_raised_before_any_run(values: Any, says: str) -> None:
    loaded = load(design(sources={**design().sources, "look": Source({"t:n": t.integer()})}))
    CALLS.clear()
    with pytest.raises(ValueError, match=re.escape(says)):
        dead_questions(loaded, ["d:decision"], values=values)
    assert CALLS == []


@pytest.mark.parametrize("refs", ["d:decision", {"d:decision"}, None])
def test_refs_that_are_not_a_list_or_tuple_are_a_type_error(refs: Any) -> None:
    CALLS.clear()
    with pytest.raises(TypeError, match="not a list or tuple of refs"):
        dead_questions(LOADED, refs)
    assert CALLS == []


@pytest.mark.parametrize(
    ("refs", "says"),
    [
        ([], "refs is empty: name at least one d: ref a rule provides"),
        (["d:nope"], 'refs names "d:nope", which is not a whole d: ref a rule provides'),
        (["q:go"], 'refs names "q:go", which is not a whole d: ref a rule provides'),
        (["t:state"], 'refs names "t:state", which is not a whole d: ref'),
        (["d:decision.x"], 'refs names "d:decision.x", which is not a whole d: ref'),
        (["decision"], 'refs names "decision", which is not a whole d: ref'),
        (["d:go", 5, "q:go"], 'refs holds 5, which is not a ref; refs names "q:go"'),
    ],
)
def test_refs_naming_anything_but_a_whole_d_ref_a_rule_provides_is_a_value_error_before_any_run(
    refs: Any, says: str
) -> None:
    CALLS.clear()
    with pytest.raises(ValueError, match=re.escape(says)):
        dead_questions(LOADED, refs)
    assert CALLS == []


def test_loaded_not_exactly_a_loaded_design_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        dead_questions(object(), ["d:decision"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The report


def test_the_report_is_plain_data_json_and_deepcopy_take() -> None:
    report = dead_questions(LOADED, ["d:decision"])
    assert json.loads(json.dumps(report)) == report
    assert copy.deepcopy(report) == report


def test_two_runs_give_the_same_report_bytes() -> None:
    assert encode(dead_questions(LOADED, ["d:decision"])) == encode(
        dead_questions(LOADED, ["d:decision"])
    )


def test_the_lint_raises_nothing_on_a_finding_and_changes_no_loaded_design() -> None:
    before = (LOADED.document, LOADED.design_hash, repr(LOADED._grid))
    report = dead_questions(LOADED, ["d:decision"])
    assert report["questions"]["q:tone"]["status"] == "not-on-grid"
    assert (LOADED.document, LOADED.design_hash, repr(LOADED._grid)) == before


def test_a_decision_differing_only_in_its_marker_cause_is_live() -> None:
    def causes(r: Any) -> object:
        code = CauseCode.conflict if r.read("d:go") is True else CauseCode.abstained
        return Unsure(Cause(code, ["d:go"]))

    rule = Rule("d:decision", t.label("send", "hold"), causes, version="p", reads=["d:go"])
    loaded = load(design(rules={"decision": rule}))
    assert dead_questions(loaded, ["d:decision"])["questions"]["q:go"]["status"] == "live"


def test_a_decision_differing_only_in_its_marker_refs_is_live() -> None:
    def refs(r: Any) -> object:
        return Unsure(Cause(CauseCode.abstained, ["d:go"] if r.read("d:go") is True else []))

    rule = Rule("d:decision", t.label("send", "hold"), refs, version="p", reads=["d:go"])
    loaded = load(design(rules={"decision": rule}))
    assert dead_questions(loaded, ["d:decision"])["questions"]["q:go"]["status"] == "live"


def cold(r: Any) -> object:
    tone = r.read("q:tone")
    return tone if isinstance(tone, Undetermined) else tone.probabilities["cold"] >= 0.5


COLD = Rule("d:cold", t.boolean, cold, version="1", reads=["q:tone"])


def test_a_question_is_live_when_it_changes_any_entry_named() -> None:
    loaded = load(design(rules={"cold": COLD}))

    def statuses(*refs: str) -> tuple[str, str]:
        questions = dead_questions(loaded, list(refs))["questions"]
        return questions["q:go"]["status"], questions["q:tone"]["status"]

    assert statuses("d:decision") == ("live", "not-on-grid")
    assert statuses("d:cold") == ("no-change-found", "live")
    assert statuses("d:go", "d:cold") == ("live", "live")
    assert dead_questions(loaded, ("d:cold", "d:go", "d:cold")) == dead_questions(
        loaded, ["d:go", "d:cold"]
    )


def test_a_rule_changing_a_value_it_reads_reaches_no_later_point() -> None:
    seen: list[list[str]] = []

    def spoil(r: Any) -> bool:
        items = r.read("t:items")
        seen.append(list(items))
        items.append("spoiled")
        return True

    rule = Rule("d:go", t.boolean, spoil, version="1", reads=["t:items", "q:go"])
    sources = {
        "ask": Source({"q:go": QUESTIONS["q:go"]}),
        "look": Source({"t:items": t.list_(t.string)}),
    }
    loaded = load(design(sources=sources, rules={"go": rule}))
    report = dead_questions(loaded, ["d:decision"], values={"t:items": [["a"]]})
    assert report["points"] == len(seen) == 3  # q:go off the grid: two labels and the marker
    assert seen == [["a"]] * 3


def test_a_witness_found_only_under_a_later_assignment_of_the_other_reads() -> None:
    def gated(r: Any) -> str:
        return "send" if r.read("d:go") is True and r.read("t:n") == 2 else "hold"

    sources = {"ask": Source({"q:go": QUESTIONS["q:go"]}), "look": Source({"t:n": t.integer()})}
    loaded = load(design(sources=sources, rules={"decision": decision(["d:go", "t:n"], gated)}))
    witness = dead_questions(loaded, ["d:decision"], values={"t:n": [1, 2]})["questions"]["q:go"][
        "witness"
    ]
    assert [w["t:n"] for w in witness] == [2, 2]
    assert [w["q:go"]["probabilities"]["yes"] for w in witness] == [0.0, 0.8]


def heeds_nothing(r: Any) -> bool:
    r.read("q:go")
    return True


def test_a_question_whose_entry_yields_no_answer_point_is_not_on_grid() -> None:
    rule = Rule(
        "d:go",
        t.boolean,
        heeds_nothing,
        version="1",
        reads=["q:go"],
        boundaries={"q:go": roles(mass=[{"labels": ["yes", "no"], "bounds": ["go/bar"]}])},
    )
    loaded = load(design(rules={"go": rule}))
    report = dead_questions(loaded, ["d:decision"])
    assert report["questions"]["q:go"]["status"] == "not-on-grid"
    assert report["points"] == 3 * 3  # q:go and q:tone: two labels at 1 and the marker each


def test_a_one_label_question_on_the_grid_that_changes_nothing_is_no_change_found() -> None:
    def heeds_only(r: Any) -> bool:
        r.read("q:only")
        return True

    only = Question("choice", Text("Only?"), labels=("only",))
    rule = Rule(
        "d:only",
        t.boolean,
        heeds_only,
        version="1",
        reads=["q:only"],
        boundaries={"q:only": roles(bar=["go/bar"])},
    )
    loaded = load(
        design(
            sources={"ask": Source({**QUESTIONS, "q:only": only})}, rules={"go": GO, "only": rule}
        )
    )
    report = dead_questions(loaded, ["d:decision"])
    assert report["questions"]["q:only"] == {"status": "no-change-found", "witness": None}


# ---------------------------------------------------------------------------------------------
# Narrowing an off-grid question


def test_an_off_grid_question_narrowed_by_values_tries_its_listed_labels_and_is_named() -> None:
    whole = dead_questions(LOADED, ["d:decision"])
    report = dead_questions(LOADED, ["d:decision"], values={"q:tone": ["cold"]})
    assert report["narrowed"] == ["q:tone"]
    assert whole["narrowed"] == []
    # q:tone's points fall from warm, cold and the marker to cold and the marker.
    assert (whole["points"], report["points"]) == (30, 20)
    assert json.loads(json.dumps(report)) == report
    witness = report["questions"]["q:go"]["witness"]
    assert witness[0]["q:tone"] == answer("choice", ["warm", "cold"], warm=0.0, cold=1.0)
    assert report["questions"]["q:tone"] == {"status": "not-on-grid", "witness": None}


def test_a_question_on_the_grid_is_not_narrowed() -> None:
    with pytest.raises(ValueError, match='"q:go", which its grid points or its type fill'):
        dead_questions(LOADED, ["d:decision"], values={"q:go": ["yes"]})


# ---------------------------------------------------------------------------------------------
# A broken entry


def _crashing(_state: object) -> object:
    raise RuntimeError("the codec failed")


def _malformed(_state: object) -> object:
    return 5  # not a record, so t:state does not conform


@pytest.mark.parametrize(("codec", "code"), [(_crashing, "raised"), (_malformed, "invalid-return")])
def test_a_broken_entry_raises_naming_the_point(
    monkeypatch: pytest.MonkeyPatch, codec: Any, code: str
) -> None:
    real = _decide._decided_sync  # pyright: ignore[reportPrivateUsage]

    def broken(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **{**kwargs, "codec": codec})

    monkeypatch.setattr("ommat._decide._decided_sync", broken)
    with pytest.raises(ValueError, match=re.escape(f"cause {code!r}")) as caught:
        dead_questions(LOADED, ["d:decision"])
    assert "broken entry" in str(caught.value)
    assert "'t:state': {}" in str(caught.value)
    assert "'q:go'" in str(caught.value)


@pytest.mark.parametrize(
    "marker",
    [
        Unsure(Cause(CauseCode.design_changed, [])),
        Unavailable(Cause(CauseCode.invalid_return, ["t:state"])),
    ],
)
def test_a_rule_returning_an_entry_cause_is_compared_not_refused(marker: Any) -> None:
    def marked(r: Any) -> object:
        return marker if r.read("d:go") is True else "hold"

    loaded = load(design(rules={"decision": decision(["d:go"], marked)}))
    report = dead_questions(loaded, ["d:decision"])
    assert report["questions"]["q:go"]["status"] == "live"
    assert report["questions"]["q:tone"] == {"status": "not-on-grid", "witness": None}
