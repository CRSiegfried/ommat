# pyright: reportPrivateUsage=false
"""Per-candidate questions: a source grounded over a ``list<string>`` candidate set asks each
question holding a candidate slot once per candidate, under its grounded ref, beside its other
questions asked once; load's checks 3, 4 and 12 on grounded declarations; the markers a set and
a cut give the declared ref; the read view's ``{candidate: entry}``; the format-6 record's
``grounded`` and where it admits a grounded ref; the redacted view, the transcript, replay and
``from_record``."""

import copy
import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from designs import minimal, rule, slot, with_rule
from ledger_logs import before_key_order, rehashed
from ommat import (
    Answer,
    CallResult,
    Cause,
    Design,
    FindingKind,
    Group,
    InvalidDesign,
    InvalidRecord,
    LoadedDesign,
    Plan,
    PlanDeclaration,
    Policy,
    Question,
    ReadView,
    Record,
    Rule,
    RunSettings,
    Source,
    SourceCall,
    Text,
    Unavailable,
    When,
    decide_sync,
    from_record,
    load,
    read_record,
    redact,
    replay_sync,
    transcript,
)
from ommat import types as t
from ommat._canonical import decode
from ommat._refs import code_point_sorted, grounded_ref
from test_views import ROOT, digest

OUT = "q:lamps/out"
URGENT = "q:lamps/urgent"
SET = "d:streets"

STREETS = ['Quay "St"', "Zoë Row", "Bridge ] Lane", "Back\\slash", "\uff5a Lane", "😀 Lane"]
"""Candidates in an order that is not their grounded refs' code-point order: each needs an
escape, a bracket or a non-ASCII character in its ref, and the fullwidth z (U+FF5A) comes before the
emoji by code point but after it by UTF-16 code unit."""

REFS = [grounded_ref(OUT, s) for s in STREETS]

YES = {"yes": 0.9, "no": 0.1}


def test_the_candidates_refs_are_escaped_and_out_of_the_sets_order() -> None:
    assert REFS[:4] == [
        'q:lamps/out["Quay \\"St\\""]',
        'q:lamps/out["Zoë Row"]',
        'q:lamps/out["Bridge ] Lane"]',
        'q:lamps/out["Back\\\\slash"]',
    ]
    ordered = code_point_sorted(REFS)
    assert ordered != REFS
    assert ordered == [REFS[i] for i in (3, 2, 0, 1, 4, 5)]


# ---------------------------------------------------------------------------------------------
# The design


def streets(r: ReadView) -> object:
    value = r.read("t:state.streets")
    if value == ["raise"]:
        raise RuntimeError("no streets")
    return value


READS: list[tuple[object, object]] = []


def out(r: ReadView) -> object:
    """Keep what the read of the declared ref gives, and the set read beside it."""
    READS.append((r.read(OUT), r.read(SET)))
    return []


def escalate(r: ReadView) -> Plan:
    return Plan("escalate", reason="checked")


def out_question(template: str = "Is a lamp out on {s}?", set_ref: str = SET) -> Question:
    return Question("noul", Text(template, [slot("s", candidate=set_ref)]))


def lamps(*, urgent: bool = True, **changes: Any) -> Source:
    provides: dict[str, Any] = {OUT: out_question()}
    if urgent:
        provides[URGENT] = Question("noul", Text("Does the report call it urgent?"))
    fields: dict[str, Any] = {"needs": ["t:state.text"], "grounded": SET}
    fields.update(changes)
    return Source(provides, **fields)


def design(source: Source | None = None, **changes: Any) -> Design:
    fields: dict[str, Any] = {
        "name": "lamps",
        "state": t.record(text=t.string, streets=t.list_(t.string)),
        "sources": {"lamps": lamps() if source is None else source},
        "rules": {
            "streets": Rule(
                SET,
                t.list_(t.string),
                streets,
                version="1",
                reads=["t:state.streets"],
                candidates=True,
            ),
            "out": Rule("d:out", t.list_(t.string), out, version="1", reads=[OUT]),
        },
        "policy": Policy(escalate, version="1", reads=["d:out"]),
    }
    fields.update(changes)
    return Design(**fields)


LOADED = load(design())

ASKED: list[tuple[SourceCall, ...]] = []


async def answer(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    ASKED.append(batch)
    return CallResult({ref: dict(YES) for s in batch for ref in s.questions})


def decided(
    names: list[str] = STREETS,
    loaded: LoadedDesign = LOADED,
    binding: Callable[..., Any] = answer,
    settings: RunSettings | None = None,
) -> Record:
    ASKED.clear()
    READS.clear()
    state = {"text": "Lamps are out", "streets": names}
    bindings: dict[str, Any] = {"default": binding}
    if any(s.oracle == "tools" for s in loaded._design.sources.values()):
        bindings["tools"] = tools
    return decide_sync(loaded, state, bindings, settings).record


def entries(record: Record) -> dict[str, dict[str, Any]]:
    return {i["ref"]: i for i in record.log if i["item"] == "entry"}


def q_refs(record: Record) -> list[str]:
    return [i["ref"] for i in record.log if i["item"] == "entry" and i["ref"].startswith("q:")]


def calls(record: Record) -> list[dict[str, Any]]:
    return [i for i in record.log if i["item"] == "call"]


def marker(item: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    m = item["marker"]
    return m["kind"], m["cause"]["code"], m["cause"]["refs"]


# ---------------------------------------------------------------------------------------------
# Load


def failures(d: Design) -> list[tuple[int, str]]:
    with pytest.raises(InvalidDesign) as info:
        load(d)
    return [(f.check, f.message) for f in info.value.failures]


def test_the_loaded_design_maps_each_per_candidate_ref_to_its_set() -> None:
    assert dict(LOADED._grounded) == {OUT: SET}
    assert "streets" in LOADED._graph["lamps"]  # an edge from the set's provider
    assert LOADED._needs["lamps"] == frozenset({"t:state.text"})  # the set is no need


def test_a_source_declaring_its_set_a_need_has_it_among_its_needs() -> None:
    loaded = load(design(lamps(needs=["t:state.text", SET])))
    assert loaded._needs["lamps"] == frozenset({"t:state.text", SET})
    decided(loaded=loaded)
    ((source,),) = ASKED
    assert source.needs[SET] == STREETS


def test_a_source_grounded_over_a_t_set_loads_with_an_edge_from_its_provider() -> None:
    sources = {
        "lookup": Source(
            {"t:streets": {"type": t.list_(t.string), "candidates": True}}, needs=["t:state"]
        ),
        "lamps": Source({OUT: out_question(set_ref="t:streets")}, grounded="t:streets"),
    }
    loaded = load(minimal(sources=sources))
    assert dict(loaded._grounded) == {OUT: "t:streets"}
    assert loaded._graph["lamps"] == frozenset({"lookup"})
    assert loaded._needs["lamps"] == frozenset()


CANDIDATES = Source({"t:s": {"type": t.list_(t.string), "candidates": True}})


def over_t(*sources: tuple[str, Source], **changes: Any) -> Design:
    """``minimal`` with a candidate set ``t:s`` and the given sources."""
    return minimal(sources={"set": CANDIDATES, **dict(sources)}, **changes)


def per(template: str = "Is {c} it?", set_ref: str = "t:s", **changes: Any) -> Question:
    return Question("noul", Text(template, [slot("c", candidate=set_ref)]), **changes)


ONCE = Question("noul", Text("Is it?"))

GROUNDED = ("g", Source({"q:x": per(), "q:y": Question("noul", Text("Once?"))}, grounded="t:s"))


def test_a_grounded_source_and_a_reader_of_its_declared_ref_load() -> None:
    reader = rule("r", reads=("q:x", "t:s"))
    loaded = load(with_rule(over_t(GROUNDED), "r", reader))
    assert dict(loaded._grounded) == {"q:x": "t:s"}


EXTRACTION = Question(
    "extraction",
    Text("What is owed for {c}?", [slot("c", candidate="t:s")]),
    value_type="number",
    text={"ref": "t:state.body"},
)
LABEL_SET = Question(
    "choice",
    Text("Which for {c}?", [slot("c", candidate="t:s")]),
    {"fixed": ["none"], "candidates": "t:s"},
)

REFUSED: list[tuple[Design, str]] = [
    (with_rule(over_t(), "r", rule("r", grounded="t:s")), "grounded rule"),
    (over_t(plans={"ask": PlanDeclaration("clarify", grounded="t:s")}), "plan declaration"),
    (
        over_t(("g", Source({"q:x": per(), "t:y": t.string}, grounded="t:s"))),
        "grounded source of t: refs",
    ),
    (
        over_t(("g", Source({"q:x": EXTRACTION}, grounded="t:s")), state=t.record(body=t.string)),
        "per-candidate extraction question",
    ),
    (
        over_t(("g", Source({"q:x": LABEL_SET}, grounded="t:s"))),
        "per-candidate question over a candidate-valued label set",
    ),
    (
        over_t(GROUNDED, ("h", Source({"q:z": ONCE}, needs=["q:x"]))),
        "need or slot ref on a grounded declared ref",
    ),
    (
        over_t(GROUNDED, ("h", Source({"q:z": ONCE}, needs=['q:x["a"]']))),
        "need or slot ref on a grounded declared ref",
    ),
    (
        over_t(
            GROUNDED,
            ("h", Source({"q:z": Question("noul", Text("Is {a}?", [slot("a", ref="q:x")]))})),
        ),
        "need or slot ref on a grounded declared ref",
    ),
]


@pytest.mark.parametrize(("d", "says"), REFUSED)
def test_check_12_refuses_only_the_grounded_forms_this_engine_does_not_implement(
    d: Design, says: str
) -> None:
    ((check, message),) = failures(d)
    assert check == 12
    assert says in message


@pytest.mark.parametrize(
    ("d", "says"),
    [
        (
            over_t(
                ("set2", Source({"t:s2": {"type": t.list_(t.string), "candidates": True}})),
                ("g", Source({"q:x": per(set_ref="t:s2")}, grounded="t:s")),
            ),
            'has a candidate slot naming "t:s2", not its source\'s grounding set "t:s"',
        ),
        (
            over_t(
                ("o", Source({"t:o": t.list_(t.string)})),
                ("g", Source({"q:x": per(set_ref="t:o")}, grounded="t:o")),
            ),
            'grounding names "t:o", which is not a candidate set',
        ),
        (
            over_t(("g", Source({"q:x": per(set_ref="t:none")}, grounded="t:none"))),
            'grounding names "t:none", which has no provider',
        ),
        (
            over_t(("g", Source({"q:x": per(set_ref="t:s.f")}, grounded="t:s.f"))),
            'grounding names "t:s.f", which is not a candidate set',
        ),
    ],
)
def test_check_3_holds_a_candidate_slot_and_a_grounding_to_a_candidate_set(
    d: Design, says: str
) -> None:
    ((check, message),) = failures(d)
    assert (check, says in message) == (3, True), message


@pytest.mark.parametrize(
    ("d", "says"),
    [
        (over_t(("g", Source({"q:x": per()}))), "candidate only when it is grounded"),
        (
            over_t(("g", Source({"q:x": ONCE}, grounded="t:s"))),
            "is grounded, but no question of it holds a candidate slot",
        ),
        (
            over_t(
                ("o", Source({"q:o": ONCE})),
                ("g", Source({"q:x": per(set_ref="q:o")}, grounded="q:o")),
            ),
            'grounding reads "q:o", which is not at level d: or t:',
        ),
        (
            with_rule(over_t(GROUNDED), "r", rule("r", reads=('q:x["a"]',))),
            'rule "r" names "q:x[\\"a\\"]", a grounded ref',
        ),
        (
            over_t(GROUNDED, groups={"both": Group(['q:x["a"]', "q:y"])}),
            'group "both" names "q:x[\\"a\\"]", a grounded ref',
        ),
    ],
)
def test_check_4_holds_a_candidate_slot_a_grounding_and_a_bracket_to_their_places(
    d: Design, says: str
) -> None:
    ((check, message),) = failures(d)
    assert (check, says in message) == (4, True), message


def test_a_bracketed_grounding_is_no_candidate_set_and_a_grounded_ref() -> None:
    d = over_t(("g", Source({"q:x": per(set_ref='t:s["a"]')}, grounded='t:s["a"]')))
    assert sorted(check for check, _ in failures(d)) == [3, 4]


# ---------------------------------------------------------------------------------------------
# Asking


def test_each_candidate_is_asked_in_the_request_beside_the_question_asked_once() -> None:
    record = decided()
    ((source,),) = ASKED  # one call, one source
    assert list(source.questions) == code_point_sorted([*REFS, URGENT])
    for name, ref in zip(STREETS, REFS, strict=True):
        assert source.questions[ref].wording == f"Is a lamp out on {name}?"
    assert source.questions[URGENT].wording == "Does the report call it urgent?"
    assert source.needs == {"t:state.text": "Lamps are out"}
    assert q_refs(record) == code_point_sorted([*REFS, URGENT])
    for ref in REFS:
        assert entries(record)[ref]["value"]["probabilities"] == YES


def test_the_call_item_keys_a_per_candidate_question_by_its_declared_ref_once() -> None:
    (call,) = calls(decided())
    assert list(call["questions"]) == [OUT, URGENT]
    assert call["questions"][OUT]["kind"] == "noul"


def test_the_settings_item_names_each_grounded_refs_set() -> None:
    record = decided()
    assert record.log[1]["grounded"] == {OUT: SET}
    assert record.log[0]["format"] == 6


def test_a_question_identity_is_one_across_candidates_and_turns() -> None:
    (one,) = calls(decided(STREETS[:1]))
    (two,) = calls(decided(STREETS[2:]))
    assert one["questions"][OUT] == two["questions"][OUT]


def test_a_set_holding_unavailable_marks_the_declared_ref_and_asks_the_rest() -> None:
    record = decided(["raise"])
    found = entries(record)
    assert marker(found[OUT]) == ("Unavailable", "need-unavailable", [SET])
    assert found[OUT]["call"] is None
    assert not any(r.startswith(OUT + "[") for r in found)
    ((source,),) = ASKED
    assert list(source.questions) == [URGENT]
    assert [(found_marker(record), found_marker(record, SET))] == READS


def found_marker(record: Record, ref: str = OUT) -> Unavailable:
    kind, code, refs = marker(entries(record)[ref])
    assert kind == "Unavailable"
    return Unavailable(Cause(code, refs))


def never(r: ReadView) -> bool:
    return False


def unasked_set_design(*, urgent: bool) -> Design:
    gate = Source(
        {"t:gate": t.list_(t.string)}, needs=["t:state"], oracle="tools", when=When(never, "1")
    )
    rules = {
        "streets": Rule(
            SET,
            t.list_(t.string),
            lambda r: r.read("t:gate"),
            version="1",
            reads=["t:gate"],
            candidates=True,
        ),
    }
    return design(
        sources={"gate": gate, "lamps": lamps(urgent=urgent)},
        rules=rules,
        policy=Policy(escalate, version="1"),
    )


def test_a_set_holding_unasked_gives_the_declared_ref_need_undetermined() -> None:
    record = decided(loaded=load(unasked_set_design(urgent=True)))
    assert marker(entries(record)[SET])[0] == "Unasked"
    assert marker(entries(record)[OUT]) == ("Unasked", "need-undetermined", [SET])
    ((source,),) = ASKED
    assert list(source.questions) == [URGENT]


def test_a_source_left_with_nothing_to_ask_over_a_marked_set_is_not_called() -> None:
    record = decided(loaded=load(unasked_set_design(urgent=False)))
    assert marker(entries(record)[OUT]) == ("Unasked", "need-undetermined", [SET])
    assert calls(record) == []
    assert ASKED == []


def test_an_empty_set_asks_nothing_for_its_question_and_the_rest_once() -> None:
    record = decided([])
    assert q_refs(record) == [URGENT]
    ((source,),) = ASKED
    assert list(source.questions) == [URGENT]
    assert READS == [({}, [])]


def test_a_source_with_nothing_to_ask_over_an_empty_set_is_not_called_or_charged() -> None:
    loaded = load(design(lamps(urgent=False, cost=1)))
    record = decided([], loaded, settings=RunSettings(budget=0))
    assert q_refs(record) == []
    assert calls(record) == []
    assert ASKED == []
    rounds = [i for i in record.log if i["item"] == "round"]
    assert [r["cost"] for r in rounds] == [0]
    assert READS == [({}, [])]


def raises(r: ReadView) -> object:
    raise RuntimeError("bad")


def test_the_sources_own_need_marker_goes_on_each_grounded_entry() -> None:
    d = design(
        lamps(needs=["t:state.text", "d:bad"]),
        rules={**design().rules, "bad": Rule("d:bad", t.string, raises, version="1")},
    )
    record = decided(STREETS[:3], load(d))
    found = entries(record)
    assert OUT not in found
    for ref in [*REFS[:3], URGENT]:
        assert marker(found[ref]) == ("Unavailable", "need-unavailable", ["d:bad"])
    assert q_refs(record) == code_point_sorted([*REFS[:3], URGENT])
    assert ASKED == []


def test_the_sources_own_when_marker_goes_on_each_grounded_entry() -> None:
    record = decided(STREETS[:2], load(design(lamps(when=When(never, "1")))))
    for ref in [*REFS[:2], URGENT]:
        assert marker(entries(record)[ref]) == ("Unasked", "when-false", ["lamps"])
    assert ASKED == []


def t_set_design(grounded_id: str = "lamps") -> Design:
    sources = {
        "lookup": Source(
            {"t:streets": {"type": t.list_(t.string), "candidates": True}},
            needs=["t:state"],
            oracle="tools",
            cost=1,
        ),
        grounded_id: Source(
            {OUT: out_question(set_ref="t:streets")},
            needs=["t:state.text"],
            grounded="t:streets",
            cost=1,
        ),
    }
    return design(sources=sources, rules={}, policy=Policy(escalate, version="1"))


async def tools(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"t:streets": batch[0].needs["t:state"]["streets"]})


def t_set_turn(budget: float, grounded_id: str = "lamps") -> Record:
    ASKED.clear()
    state = {"text": "x", "streets": STREETS[:3]}
    bindings = {"tools": tools, "default": answer}
    loaded = load(t_set_design(grounded_id))
    return decide_sync(loaded, state, bindings, RunSettings(budget=budget)).record


# "zlamps" sorts after the set's provider, "lookup", whose cut marker is written first.
@pytest.mark.parametrize("grounded_id", ["lamps", "zlamps"])
def test_a_cut_before_the_set_is_on_the_board_marks_the_declared_ref(grounded_id: str) -> None:
    record = t_set_turn(0.5, grounded_id)
    found = entries(record)
    assert marker(found[OUT]) == ("Unasked", "over-budget", [])
    assert marker(found["t:streets"]) == ("Unasked", "over-budget", [])
    assert q_refs(record) == [OUT]


def test_a_cut_after_the_set_is_on_the_board_marks_each_grounded_entry() -> None:
    record = t_set_turn(1)
    refs = code_point_sorted([grounded_ref(OUT, s) for s in STREETS[:3]])
    assert q_refs(record) == refs
    for ref in refs:
        assert marker(entries(record)[ref]) == ("Unasked", "over-budget", [])


def test_a_t_set_grounds_its_questions_in_the_next_round() -> None:
    record = t_set_turn(2)
    refs = code_point_sorted([grounded_ref(OUT, s) for s in STREETS[:3]])
    assert q_refs(record) == refs
    assert [c["sources"] for c in calls(record)] == [["lookup"], ["lamps"]]


# ---------------------------------------------------------------------------------------------
# Reading


def test_a_read_of_the_declared_ref_is_each_candidates_entry_in_the_sets_order() -> None:
    decided()
    ((answers, members),) = READS
    assert members == STREETS
    assert isinstance(answers, dict)
    read = cast("dict[str, Any]", answers)
    assert list(read) == STREETS
    for value in read.values():
        assert isinstance(value, Answer)
        assert value.probabilities == YES


def test_a_read_of_the_declared_ref_holds_a_candidates_marker() -> None:
    async def drops(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({r: dict(YES) for r in batch[0].questions if r != REFS[1]})

    decided(binding=drops)
    ((answers, _),) = READS
    read = cast("dict[str, Any]", answers)
    assert list(read) == STREETS
    assert isinstance(read[STREETS[1]], Unavailable)
    assert isinstance(read[STREETS[0]], Answer)


def reads_bracketed(r: ReadView) -> object:
    return r.read(REFS[0])


def test_a_bracketed_read_is_an_undeclared_read() -> None:
    rules = {
        **design().rules,
        "out": Rule("d:out", t.boolean, reads_bracketed, version="1", reads=[OUT]),
    }
    record = decided(loaded=load(design(rules=rules)))
    kind, code, _ = marker(entries(record)["d:out"])
    assert (kind, code) == ("Unavailable", "undeclared-read")


# ---------------------------------------------------------------------------------------------
# The record


def document(record: Record) -> dict[str, Any]:
    return {"log": copy.deepcopy(record.log), "unhashed": copy.deepcopy(record.unhashed)}


def test_a_per_candidate_record_reads_back_to_its_own_bytes_and_hash() -> None:
    record = decided()
    back = read_record(record.document)
    assert (back.document, back.record_hash) == (record.document, record.record_hash)


def edited(record: Record, change: Callable[[list[dict[str, Any]]], None]) -> dict[str, Any]:
    doc = document(record)
    change(doc["log"])
    return rehashed(doc)


def index_of(log: list[dict[str, Any]], ref: str) -> int:
    return next(i for i, item in enumerate(log) if item.get("ref") == ref)


def refused(doc: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidRecord, match=match):
        read_record(doc)


def test_a_bracketed_call_item_questions_key_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        call = next(i for i in log if i["item"] == "call")
        call["questions"] = {REFS[0]: call["questions"][OUT], URGENT: call["questions"][URGENT]}

    refused(edited(decided(), change), "a grounded ref; it is keyed by declared ref")


def test_a_value_at_a_grounded_declared_ref_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        item = log[index_of(log, OUT)]
        del item["marker"]
        item["value"] = {"kind": "noul", "labels": ["yes", "no"], "probabilities": YES}

    refused(edited(decided(["raise"]), change), "so its declared ref holds a marker or nothing")


def test_a_grounded_entry_beside_a_marker_at_its_declared_ref_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        first = index_of(log, code_point_sorted(REFS)[0])
        log[first] = {
            "item": "entry",
            "ref": OUT,
            "round": log[first]["round"],
            "call": None,
            "marker": {"kind": "Unavailable", "cause": {"code": "need-unavailable", "refs": [SET]}},
        }

    refused(edited(decided(), change), "written beside the marker at its declared ref")


def test_a_grounded_ref_whose_declared_ref_is_not_grounded_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        log[1]["grounded"] = {}

    refused(edited(decided(), change), "whose declared ref is not grounded in the settings")


@pytest.mark.parametrize(
    ("grounded", "match"),
    [
        ({REFS[0]: SET}, "not a provided declared ref"),
        ({"q:none": SET}, "not a provided declared ref"),
        ({"q:lamps/out.f": SET}, "not a provided declared ref"),
        ({OUT: "d:none"}, "not an ungrounded provided ref"),
        ({OUT: 1}, "not an ungrounded provided ref"),
        ({OUT: OUT}, "not an ungrounded provided ref"),
        ({OUT: SET, URGENT: OUT}, "not an ungrounded provided ref"),
    ],
)
def test_a_malformed_grounded_member_is_refused(grounded: dict[str, Any], match: str) -> None:
    def change(log: list[dict[str, Any]]) -> None:
        log[1]["grounded"] = grounded

    refused(edited(decided([]), change), match)


def test_a_grounding_set_whose_class_tree_is_no_class_name_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        log[1]["classes"][SET] = ["open"]

    refused(edited(decided([]), change), "whose class tree is not a class name")


def test_a_format_6_settings_item_without_grounded_is_refused() -> None:
    def change(log: list[dict[str, Any]]) -> None:
        del log[1]["grounded"]

    refused(edited(decided([]), change), "has no 'grounded' member")


def test_before_format_6_grounded_and_a_grounded_ref_are_refused() -> None:
    doc = document(decided())
    kept = copy.deepcopy(doc)
    before_key_order(doc)
    assert "grounded" not in doc["log"][1]
    refused(rehashed(doc), "grounded ref")
    before_key_order(kept)
    kept["log"][1]["grounded"] = {OUT: SET}
    refused(rehashed(kept), "holds grounded, which a format-5 record does not")


def test_a_format_5_record_with_no_grounded_ref_reads() -> None:
    doc = before_key_order(document(decided([])))
    assert read_record(rehashed(doc)).log[0]["format"] == 5


async def skewed(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """Every answer summing to 1.2, and the first candidate's marked truncated."""
    refs = list(batch[0].questions)
    return CallResult({r: {"yes": 0.9, "no": 0.3} for r in refs}, truncated=(REFS[0],))


def test_unhashed_sums_and_truncated_hold_grounded_refs() -> None:
    record = decided(STREETS[:2], binding=skewed)
    (entry,) = record.unhashed["calls"].values()
    assert set(entry["sums"]) == {REFS[0], REFS[1], URGENT}
    assert entry["truncated"] == [REFS[0]]
    assert read_record(record.document).unhashed == record.unhashed


def test_an_unhashed_grounded_ref_whose_declared_ref_is_not_grounded_is_refused() -> None:
    record = decided(STREETS[:2], binding=skewed)
    doc = document(record)
    (entry,) = doc["unhashed"]["calls"].values()
    entry["sums"] = {grounded_ref(URGENT, "x"): 1.2}
    refused(rehashed(doc), "is a grounded ref whose declared ref is not grounded")
    entry["sums"] = {URGENT: 1.2}
    entry["truncated"] = [grounded_ref(URGENT, "x")]
    refused(rehashed(doc), "is a grounded ref whose declared ref is not grounded")


# ---------------------------------------------------------------------------------------------
# The views


DIGESTED = re.compile(r'q:lamps/out\["hmac-sha256:[0-9a-f]{64}"\]')


def redacted(record: Record) -> dict[str, Any]:
    return cast("dict[str, Any]", decode(redact(record, ROOT)))


async def unexpected(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({**{r: dict(YES) for r in batch[0].questions}, "q:other": dict(YES)})


async def drops_first(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    refs = [r for r in batch[0].questions if r != REFS[0]]
    return CallResult({r: {"yes": 0.9, "no": 0.3} for r in refs}, truncated=(REFS[1],))


def test_redact_digests_each_candidate_inside_a_grounded_ref_under_its_sets_class() -> None:
    record = decided(STREETS[:3], binding=drops_first)
    view = redacted(record)
    expected = {grounded_ref(OUT, s): grounded_ref(OUT, digest(s, record)) for s in STREETS[:3]}
    refs = [i["ref"] for i in view["log"] if i["item"] == "entry" and i["ref"].startswith(OUT)]
    assert refs == [expected[r] for r in q_refs(record) if r != URGENT]
    assert all(DIGESTED.fullmatch(r) for r in refs)
    dropped = next(i for i in view["log"] if i.get("ref") == expected[REFS[0]])
    assert dropped["marker"]["cause"]["refs"] == [expected[REFS[0]]]  # not-returned
    (entry,) = view["unhashed"]["calls"].values()
    assert set(entry["sums"]) == {expected[REFS[1]], expected[REFS[2]], URGENT}
    assert entry["truncated"] == [expected[REFS[1]]]


def test_redact_digests_a_failed_calls_cause_refs() -> None:
    record = decided(STREETS[:1], binding=unexpected)
    view = redacted(record)
    (call,) = [i for i in view["log"] if i["item"] == "call"]
    hidden = grounded_ref(OUT, digest(STREETS[0], record))
    assert call["outcome"]["failed"]["refs"] == [hidden, URGENT]
    entry = next(i for i in view["log"] if i.get("ref") == hidden)
    assert entry["marker"]["cause"]["refs"] == [hidden, URGENT]


def test_redact_passes_a_grounded_ref_whose_set_is_open() -> None:
    record = decided(STREETS[:3], load(design(classes={SET: "open"})))
    view = redacted(record)
    refs = [i["ref"] for i in view["log"] if i["item"] == "entry" and i["ref"].startswith("q:")]
    assert refs == q_refs(record)


def test_the_transcript_asks_each_grounded_ref() -> None:
    record = decided(STREETS[:3])
    shown = transcript(record)
    assert shown is not None
    ((call,),) = shown["rounds"]
    assert call["asks"] == code_point_sorted([*REFS[:3], URGENT])
    assert list(call["answers"]) == call["asks"]


# ---------------------------------------------------------------------------------------------
# Replay and from_record


def test_a_per_candidate_turn_replays_with_no_finding() -> None:
    record = decided()
    replayed, found = replay_sync(record, LOADED)
    assert found == []
    assert replayed is not None
    assert q_refs(replayed) == q_refs(record)


def test_question_changed_names_the_declared_ref() -> None:
    record = decided(STREETS[:2])
    changed = load(reworded())
    found = replay_sync(record, changed)[1]
    assert {"kind": FindingKind.question_changed, "refs": [OUT]} in found


def reworded() -> Design:
    source = lamps()
    provides = {**source.provides, OUT: out_question("Is a lamp dark on {s}?")}
    return design(dataclasses.replace(source, provides=provides))


def with_streets(record: Record, names: list[str]) -> Record:
    def change(log: list[dict[str, Any]]) -> None:
        log[2]["value"]["streets"] = names

    return read_record(edited(record, change))


def test_a_copy_adding_a_candidate_replays_to_ref_not_in_record_and_an_entry() -> None:
    record = decided(STREETS[:3])
    added = grounded_ref(OUT, "Ash")
    found = replay_sync(with_streets(record, [*STREETS[:3], "Ash"]), LOADED)[1]
    assert {"kind": FindingKind.ref_not_in_record, "refs": [added]} in found
    refs = [f["ref"] for f in found if f["kind"] is FindingKind.entry]
    assert refs == [SET, added]


def test_a_copy_removing_a_candidate_replays_to_an_entry() -> None:
    record = decided(STREETS[:3])
    found = replay_sync(with_streets(record, STREETS[:2]), LOADED)[1]
    assert [f["kind"] for f in found] == [FindingKind.entry, FindingKind.entry]
    assert [f["ref"] for f in found] == [SET, REFS[2]]
    gone = next(f for f in found if f["ref"] == REFS[2])
    assert gone["replayed"] is None


def warnings_of(record: Record) -> list[str]:
    (entry,) = record.unhashed["calls"].values()
    return cast("list[str]", entry["warnings"])


def test_from_record_answers_each_grounded_question_from_its_declared_refs_entry() -> None:
    record = decided()
    again = decided(binding=from_record(record, LOADED))
    for ref in [*REFS, URGENT]:
        assert entries(again)[ref]["value"] == entries(record)[ref]["value"]
    assert warnings_of(again) == [f"answered from record {record.record_hash}"]


def test_from_record_refuses_a_candidate_the_stored_turn_did_not_ask() -> None:
    record = decided(STREETS[:2])
    again = decided([*STREETS[:2], "Ash"], binding=from_record(record, LOADED))
    added = grounded_ref(OUT, "Ash")
    assert marker(entries(again)[added]) == ("Unavailable", "unsupported", [added])
    for ref in [*REFS[:2], URGENT]:
        assert entries(again)[ref]["value"]["probabilities"] == YES
    assert warnings_of(again)[1:] == [f"{added!r}: the recorded call wrote no entry for it"]


def test_from_record_refuses_the_whole_source_when_its_declared_set_differs() -> None:
    loaded = load(design(lamps(needs=["t:state.text", SET])))
    record = decided(STREETS[:2], loaded)
    again = decided(STREETS[:1], loaded, binding=from_record(record, loaded))
    for ref in [REFS[0], URGENT]:
        assert marker(entries(again)[ref]) == ("Unavailable", "unsupported", [ref])
    assert any(SET in line for line in warnings_of(again))


def test_a_rules_read_set_holds_the_grounding_set_with_the_declared_ref() -> None:
    # ``out`` declares only the declared ref, yet reads the set beside it.
    decided(STREETS[:1])
    ((_, members),) = READS
    assert members == STREETS[:1]


# ---------------------------------------------------------------------------------------------
# Review round 1


def test_a_group_of_grounded_and_ungrounded_refs_or_two_sets_is_refused() -> None:
    mixed = over_t(GROUNDED, groups={"g": Group(["q:x", "q:y"])})
    assert (
        3,
        'group "g" names q: refs grounded over different candidate sets, or grounded and '
        "ungrounded ones",
    ) in failures(mixed)
    other = ("g2", Source({"q:z": per(set_ref="t:s2")}, grounded="t:s2"))
    two_sets = over_t(
        GROUNDED,
        other,
        ("sets2", Source({"t:s2": {"type": t.list_(t.string), "candidates": True}})),
        groups={"g": Group(["q:x", "q:z"])},
    )
    assert any(c == 3 and "different candidate sets" in m for c, m in failures(two_sets))
    load(over_t(GROUNDED, groups={"g": Group(["q:x"])}))  # all grounded over one set


def test_a_grounded_design_round_trips_through_its_document() -> None:
    loaded = load(design())
    again = load(loaded.document, functions={"streets": streets, "out": out, "policy": escalate})
    assert again.design_hash == loaded.design_hash
    assert dict(again._grounded) == {OUT: SET}  # pyright: ignore[reportPrivateUsage]


def reads_its_own_question(r: ReadView) -> object:
    return r.read(URGENT)


def test_a_set_rule_reading_its_grounded_source_is_a_cycle() -> None:
    rules = {
        "streets": Rule(
            SET,
            t.list_(t.string),
            reads_its_own_question,
            version="1",
            reads=[URGENT],
            candidates=True,
        ),
    }
    found = failures(design(rules=rules, policy=Policy(escalate, version="1")))
    assert any(c == 5 for c, _ in found)


def test_a_need_on_a_field_of_a_per_candidate_question_is_refused() -> None:
    reader = ("h", Source({"q:h": ONCE}, needs=["q:x.value"]))
    assert any(
        c == 12 and "need or slot ref on a grounded declared ref" in m
        for c, m in failures(over_t(GROUNDED, reader))
    )


WHEN_CALLS: list[int] = []


def counting_when(r: ReadView) -> bool:
    WHEN_CALLS.append(1)
    raise RuntimeError("never asked")


def test_a_source_with_nothing_to_ask_over_an_empty_set_calls_no_when() -> None:
    WHEN_CALLS.clear()
    loaded = load(design(lamps(urgent=False, when=When(counting_when, "1"))))
    record = decided([], loaded)
    assert WHEN_CALLS == []
    assert "exceptions" not in record.unhashed
    assert q_refs(record) == []


REQUIRED: list[object] = []


def requires(r: ReadView) -> object:
    REQUIRED.append(r.require(OUT))
    return []


def test_require_returns_the_dict_holding_a_marker_and_stops_only_on_the_declared_ref() -> None:
    rules = {
        **design().rules,
        "out": Rule("d:out", t.list_(t.string), requires, version="1", reads=[OUT]),
    }
    loaded = load(design(rules=rules))

    async def one_refused(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        refs = [ref for s in batch for ref in s.questions]
        return CallResult(
            {
                ref: Unavailable(Cause("refused", [ref])) if i == 0 else dict(YES)
                for i, ref in enumerate(refs)
            }
        )

    REQUIRED.clear()
    record = decided(STREETS[:2], loaded, one_refused)
    (got,) = REQUIRED
    assert isinstance(got, dict)
    assert list(cast("dict[str, object]", got)) == STREETS[:2]
    assert entries(record)["d:out"]["value"] == []


async def truncates_two(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    refs = list(batch[0].questions)
    return CallResult({r: dict(YES) for r in refs}, truncated=tuple(r for r in refs if r != URGENT))


def test_redact_keeps_truncated_in_code_point_order_after_digesting() -> None:
    record = decided(STREETS[:4], binding=truncates_two)
    (entry,) = redacted(record)["unhashed"]["calls"].values()
    assert len(entry["truncated"]) == 4
    assert entry["truncated"] == code_point_sorted(entry["truncated"])


def test_redact_digests_a_candidate_valued_answer_at_a_grounded_entry() -> None:
    # The engine never writes this (check 12), but redact holds over every record the read gives.
    record = decided(STREETS[:2], load(design(lamps(urgent=False))))

    def candidate_valued(log: list[dict[str, Any]]) -> None:
        for item in log:
            if item["item"] == "call":
                item["questions"][OUT].update(
                    kind="choice", labels={"fixed": ["other"], "candidates": SET, "at": 1}
                )
            elif item["item"] == "entry" and item["ref"].startswith(OUT + "["):
                name = decode(item["ref"][len(OUT) + 1 : -1].encode())
                item["value"] = {
                    "kind": "choice",
                    "labels": ["other", name],
                    "probabilities": {"other": 0.5, name: 0.5},
                }

    back = read_record(edited(record, candidate_valued))
    text = redact(back, ROOT).decode()
    for name in STREETS[:2]:
        assert json.dumps(name)[1:-1] not in text


def test_a_grounded_ref_with_a_field_path_is_refused() -> None:
    record = decided(STREETS[:1], binding=unexpected)

    def with_field(log: list[dict[str, Any]]) -> None:
        call = next(i for i in log if i["item"] == "call")
        call["outcome"]["failed"]["refs"][0] += ".f"

    refused(edited(record, with_field), "is a grounded ref whose declared ref is not grounded")
