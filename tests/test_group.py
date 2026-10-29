"""The correlated-group confirmation job: per pair of a group's ``q:`` refs, the units of a
split in which both hold an answer, those in which each erred against its gold answer and those
in which both did, for a plain and a grounded group, a tie, extraction refs, markers, three refs
and a format-1 gold entry; each refusal in its order, and a held key before the work and at the
write (§group-confirmation)."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import extraction as x
import test_per_candidate as pc
from designs import minimal
from ommat import CallResult, Group, Question, Record, Source, Text, decide_sync, load
from ommat._canonical import hash_value
from ommat._refs import grounded_ref
from ommat.evaluation import (
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    confirm_group,
    write_gold_answers,
)

R = RefusalReason
GROUPS = {
    "pair": Group(["q:b", "q:a"]),
    "three": Group(["q:c", "q:a", "t:state", "q:b"]),
    "lone": Group(["q:a", "t:state"]),
    "amounts": Group(["q:n1", "q:n2"]),
    "twins": Group(["q:a", "q:twin"]),
}
SOURCE = Source(
    {
        "q:a": Question("noul", Text("Is it spam?")),
        "q:b": Question("noul", Text("Is it junk mail?")),
        "q:c": Question("noul", Text("Would you bin it?")),
        "q:twin": Question("noul", Text("Is it spam?")),  # q:a's question: one gold key
        "q:n1": x.question("number"),
        "q:n2": x.question("number", wording=Text("How much is due?")),
    },
    needs=["t:state.body", "t:state.today"],
)
LOADED = load(minimal(state=x.STATE_T, sources={"ask": SOURCE}, groups=GROUPS))


def p(yes: float) -> dict[str, float]:
    return {"yes": yes, "no": round(1 - yes, 10)}


def amount(value: object) -> dict[str, Any]:
    return {"value": value, "guess": None, "p_null": 0.0, "confidence": 1, "span": None}


def turn(body: str, answers: dict[str, object]) -> Record:
    """A finished turn whose call answers ``answers``; every other ref is a marker."""

    async def binding(batch: Any, *, seconds: float) -> CallResult:
        return CallResult(dict(answers))

    return decide_sync(LOADED, {**x.STATE, "body": body}, {"default": binding}).record


def split(records: list[Any]) -> tuple[str, list[Any]]:
    return hash_value(sorted({r.record_hash for r in records})), records


def labelled(root: Path, turns: list[tuple[Record, dict[str, object]]]) -> EvaluationStore:
    store = EvaluationStore(root)
    write_gold_answers(
        "g", [r for r, _ in turns], {r.record_hash: g for r, g in turns}, results=store
    )
    return store


def refused(store: EvaluationStore, *args: Any, loaded: Any = LOADED) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as raised:
        confirm_group(loaded, *args, results=store)
    return raised.value


# Each turn of the plain pair: a erred in T2 and T4 (a tie), b in T1 and T2; T3 answers no b.
T1 = turn("one", {"q:a": p(0.9), "q:b": p(0.8)})
T2 = turn("two", {"q:a": p(0.3), "q:b": p(0.4)})
T3 = turn("three", {"q:a": p(0.9)})
T4 = turn("four", {"q:a": p(0.5), "q:b": p(0.9)})
GOLDS: list[tuple[Record, dict[str, object]]] = [
    (T1, {"q:a": "yes", "q:b": "no"}),
    (T2, {"q:a": "yes", "q:b": "yes"}),
    (T3, {"q:a": "yes"}),
    (T4, {"q:a": "yes", "q:b": "yes"}),
]
PAIR = {"refs": ["q:a", "q:b"], "turns": 3, "erred": [2, 2], "both": 1}


def test_a_plain_pair_counts_its_turns_errors_and_joint_errors(tmp_path: Path) -> None:
    store = labelled(tmp_path, GOLDS)
    given = split([T1, T2, T3, T4])
    entry = confirm_group(LOADED, "pair", "g", given, results=store)
    key = {"design": LOADED.design_hash, "group": "pair", "split": given[0], "gold_set": "g"}
    assert entry == {"kind": "group-counts", "format": 1, "key": key, "value": {"pairs": [PAIR]}}
    assert store.lookup("group-counts", key) == entry
    assert json.loads(json.dumps(entry)) == copy.deepcopy(entry) == entry  # no Record reachable


def test_records_are_read_as_bytes_or_parsed_documents_and_counted_once(tmp_path: Path) -> None:
    store = labelled(tmp_path, GOLDS)
    records: list[Any] = [T1.document, json.loads(T2.document), T3, T4, T1]
    given = (split([T1, T2, T3, T4])[0], records)
    entry = confirm_group(LOADED, "pair", "g", given, results=store)
    assert entry["value"] == {"pairs": [PAIR]}


def test_an_answer_outside_the_group_needs_no_gold(tmp_path: Path) -> None:
    record = turn("outside", {"q:a": p(0.9), "q:b": p(0.9), "q:c": p(0.9), "q:n1": amount(1)})
    store = labelled(tmp_path, [(record, {"q:a": "yes", "q:b": "no"})])
    entry = confirm_group(LOADED, "pair", "g", split([record]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": ["q:a", "q:b"], "turns": 1, "erred": [0, 1], "both": 0}
    ]


def test_three_refs_give_three_pairs_in_code_point_order_and_no_t_ref(tmp_path: Path) -> None:
    c1 = turn("c1", {"q:a": p(0.9), "q:b": p(0.9), "q:c": p(0.2)})
    c2 = turn("c2", {"q:a": p(0.2), "q:c": p(0.2)})
    store = labelled(
        tmp_path,
        [(c1, {"q:a": "yes", "q:b": "yes", "q:c": "yes"}), (c2, {"q:a": "yes", "q:c": "no"})],
    )
    entry = confirm_group(LOADED, "three", "g", split([c1, c2]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": ["q:a", "q:b"], "turns": 1, "erred": [0, 0], "both": 0},
        {"refs": ["q:a", "q:c"], "turns": 2, "erred": [1, 1], "both": 0},
        {"refs": ["q:b", "q:c"], "turns": 1, "erred": [0, 1], "both": 0},
    ]


def test_extraction_refs_err_by_canonical_bytes_with_null_a_value(tmp_path: Path) -> None:
    e1 = turn("e1", {"q:n1": amount(120), "q:n2": amount(None)})
    e2 = turn("e2", {"q:n1": amount(120.0), "q:n2": amount(121)})
    e3 = turn("e3", {"q:n1": amount(None), "q:n2": amount(5)})
    store = labelled(
        tmp_path,
        [
            (e1, {"q:n1": {"value": 120}, "q:n2": {"value": None}}),
            (e2, {"q:n1": {"value": 120}, "q:n2": {"value": 120}}),
            (e3, {"q:n1": {"value": 5}, "q:n2": {"value": None}}),
        ],
    )
    entry = confirm_group(LOADED, "amounts", "g", split([e1, e2, e3]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": ["q:n1", "q:n2"], "turns": 3, "erred": [1, 2], "both": 1}
    ]


def gold_key(record: Record, ref: str, candidate: str = "") -> dict[str, str]:
    call = next(i for i in record.log if i["item"] == "call")
    identity = call["questions"][ref]["identity"]
    return {
        "gold_set": "g",
        "question": identity,
        "turn": record.record_hash,
        "candidate": candidate,
    }


def test_a_format_1_gold_entry_is_read_as_its_label(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    for ref, label in {"q:a": "no", "q:b": "yes"}.items():
        store.write({"kind": "gold-answer", "format": 1, "key": gold_key(T1, ref), "value": label})
    entry = confirm_group(LOADED, "pair", "g", split([T1]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": ["q:a", "q:b"], "turns": 1, "erred": [1, 0], "both": 0}
    ]


def test_two_refs_of_one_question_share_its_gold_answer(tmp_path: Path) -> None:
    record = turn("twins", {"q:a": p(0.9), "q:twin": p(0.2)})
    store = labelled(tmp_path, [(record, {"q:a": "yes"})])  # one key, both refs
    entry = confirm_group(LOADED, "twins", "g", split([record]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": ["q:a", "q:twin"], "turns": 1, "erred": [0, 1], "both": 0}
    ]


# ---------------------------------------------------------------------------------------------
# A grounded group

OUT2 = "q:lamps/dark"
LAMPS = Source(
    {pc.OUT: pc.out_question(), OUT2: pc.out_question("Is {s} dark?")},
    needs=["t:state.text"],
    grounded=pc.SET,
)
GROUNDED = load(pc.design(LAMPS, groups={"lamps": Group([pc.OUT, OUT2])}))


STREETS = ["Elm", "Oak", "Ash"]


async def lamps(batch: Any, *, seconds: float) -> CallResult:
    values: dict[str, object] = {
        grounded_ref(pc.OUT, "Elm"): p(0.9),
        grounded_ref(pc.OUT, "Oak"): p(0.2),
        grounded_ref(pc.OUT, "Ash"): p(0.9),
        grounded_ref(OUT2, "Elm"): p(0.1),
        grounded_ref(OUT2, "Oak"): p(0.9),
    }  # OUT2 for Ash a marker: that unit holds one answer
    return CallResult(values)


LAMPS_TURN = decide_sync(
    GROUNDED, {"text": "Lamps are out", "streets": STREETS}, {"default": lamps}
).record
LAMPS_GOLDS: dict[str, object] = {
    **{grounded_ref(pc.OUT, s): "yes" for s in STREETS},
    grounded_ref(OUT2, "Elm"): "yes",
    grounded_ref(OUT2, "Oak"): "no",
}


def test_a_grounded_groups_units_are_a_turn_and_a_candidate(tmp_path: Path) -> None:
    store = labelled(tmp_path, [(LAMPS_TURN, LAMPS_GOLDS)])
    entry = confirm_group(GROUNDED, "lamps", "g", split([LAMPS_TURN]), results=store)
    assert entry["value"]["pairs"] == [
        {"refs": sorted([pc.OUT, OUT2]), "turns": 2, "erred": [2, 1], "both": 1}
    ]


def test_missing_gold_in_a_grounded_group_names_the_per_candidate_ref(tmp_path: Path) -> None:
    oak = grounded_ref(OUT2, "Oak")
    store = labelled(tmp_path, [(LAMPS_TURN, {k: v for k, v in LAMPS_GOLDS.items() if k != oak})])
    exc = refused(store, "lamps", "g", split([LAMPS_TURN]), loaded=GROUNDED)
    assert (exc.reason, exc.turn, exc.ref) == (R.missing_gold, LAMPS_TURN.record_hash, oak)


# ---------------------------------------------------------------------------------------------
# Refusals


@pytest.mark.parametrize(
    ("group", "reason"), [("nope", R.unknown_group), (3, R.unknown_group), ("lone", R.too_few_refs)]
)
def test_a_group_is_refused_before_any_record_is_read(
    tmp_path: Path, group: Any, reason: R
) -> None:
    exc = refused(EvaluationStore(tmp_path), group, "g", ("id", [b"not json"]))
    assert (exc.reason, exc.group) == (reason, group if type(group) is str else None)


def test_an_invalid_record_is_refused_before_the_split_id(tmp_path: Path) -> None:
    exc = refused(EvaluationStore(tmp_path), "pair", "g", ("id", [T1, b"not json"]))
    assert exc.reason is R.invalid_record
    assert exc.error is not None


def test_a_wrong_split_id_is_refused_before_the_held_key(tmp_path: Path) -> None:
    store = labelled(tmp_path, GOLDS)
    given = split([T1])
    confirm_group(LOADED, "pair", "g", given, results=store)
    exc = refused(store, "pair", "g", ("sha256:other", [T1]))
    assert (exc.reason, exc.split) == (R.split_id, ("sha256:other", given[0]))
    assert refused(store, "pair", "g", given).reason is R.held_key


def test_records_of_another_design_are_wrong_design_naming_the_first_given(
    tmp_path: Path,
) -> None:
    first, second = pc.decided(["Elm"]), pc.decided(["Oak"])
    exc = refused(EvaluationStore(tmp_path), "pair", "g", split([T1, first, second]))
    assert (exc.reason, exc.turn) == (R.wrong_design, first.record_hash)
    exc = refused(EvaluationStore(tmp_path), "pair", "g", split([T1, second, first]))
    assert (exc.reason, exc.turn) == (R.wrong_design, second.record_hash)


@pytest.mark.parametrize(
    "given",
    ["id", ("id",), ["id", [], []], (3, []), ("id", b"x"), ("id", "x"), ("id", {}), ("id", 3)],
)
def test_a_split_not_an_id_and_records_is_a_type_error_before_anything(
    tmp_path: Path, given: object
) -> None:
    with pytest.raises(TypeError, match="the given split"):
        confirm_group(LOADED, "nope", "g", given, results=EvaluationStore(tmp_path))


def test_an_answer_with_no_gold_is_missing_and_a_marker_needs_none(tmp_path: Path) -> None:
    store = labelled(tmp_path, [(T3, {"q:a": "yes"})])  # T3's q:b is a marker
    confirm_group(LOADED, "pair", "g", split([T3]), results=store)
    exc = refused(store, "pair", "g", split([T3, T1]))
    assert (exc.reason, exc.turn, exc.ref) == (R.missing_gold, T1.record_hash, "q:a")


def test_an_unknown_gold_entry_is_refused_after_every_missing_gold(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    store.write({"kind": "gold-answer", "format": 3, "key": gold_key(T3, "q:a"), "value": "yes"})
    exc = refused(store, "pair", "g", split([T3, T1]))  # T3's entry unknown, T1's missing
    assert exc.reason is R.missing_gold
    exc = refused(store, "pair", "g", split([T3]))
    assert (exc.reason, exc.key) == (R.unknown_entry, [gold_key(T3, "q:a")])


def test_the_refusals_come_in_their_order(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    other = pc.decided()
    order: list[tuple[str, Any, R]] = [
        ("nope", ("id", [b"not json"]), R.unknown_group),
        ("lone", ("id", [b"not json"]), R.too_few_refs),
        ("pair", ("id", [T1, b"not json"]), R.invalid_record),
        ("pair", ("id", [T1, other]), R.split_id),
        ("pair", split([T1, other]), R.wrong_design),
        ("pair", split([T1]), R.missing_gold),
    ]
    for group, given, reason in order:
        assert refused(store, group, "g", given).reason is reason


class Forgetful(EvaluationStore):
    """A store whose lookup misses every ``group-counts`` entry, as a racing run's write
    between the lookup and the write would leave it."""

    def lookup(self, kind: str, key: dict[str, str]) -> dict[str, Any] | None:
        return None if kind == "group-counts" else super().lookup(kind, key)


def test_a_held_key_is_refused_before_the_counts_and_at_the_write(tmp_path: Path) -> None:
    store = labelled(tmp_path, GOLDS)
    given = split([T1, T2])
    first = confirm_group(LOADED, "pair", "g", given, results=store)
    exc = refused(store, "pair", "g", given)
    assert (exc.reason, exc.key) == (R.held_key, [first["key"]])
    exc = refused(Forgetful(tmp_path), "pair", "g", given)
    assert (exc.reason, exc.key) == (R.held_key, [first["key"]])
    assert store.lookup("group-counts", first["key"]) == first  # the first entry stands


@pytest.mark.parametrize(
    ("group", "answers", "golds"),
    [
        (
            "amounts",
            {"q:n1": amount(120), "q:n2": amount(5)},
            {"q:n1": (1, "120"), "q:n2": (2, {"value": 5})},
        ),
        ("pair", {"q:a": p(0.9)}, {"q:a": (2, "maybe")}),
    ],
)
def test_a_gold_answer_not_of_its_questions_form_is_an_unknown_entry(
    tmp_path: Path, group: str, answers: dict[str, object], golds: dict[str, Any]
) -> None:
    record = turn("form", answers)
    store = EvaluationStore(tmp_path)
    for ref, (fmt, value) in golds.items():
        store.write(
            {"kind": "gold-answer", "format": fmt, "key": gold_key(record, ref), "value": value}
        )
    ref = next(iter(golds))
    exc = refused(store, group, "g", split([record]))
    assert (exc.reason, exc.key) == (R.unknown_entry, [gold_key(record, ref)])
