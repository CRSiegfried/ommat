# pyright: reportPrivateUsage=false
"""The sweep job: the gold copy, carrying its own hash, read back and replayed, from format-1
and format-2 gold entries, an extraction's and a candidate-valued set's (§gold-copy); a passed
plan's action, the exact binomial tail against a direct sum and scipy, the order check and both
tests' choices (§sweep-scoring); and the job end to end, each count, the scope and every
refusal in its order (§sweep)."""

import asyncio
import dataclasses
import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scipy.stats  # pyright: ignore[reportMissingTypeStubs]

import extraction as x
import test_label_sets as label_sets
from ommat import (
    CallResult,
    Design,
    Helper,
    InvalidDesign,
    LoadedDesign,
    Origin,
    Parameter,
    Plan,
    Policy,
    Question,
    Record,
    Requirement,
    Source,
    SourceRequirement,
    Step,
    Table,
    Text,
    decide_sync,
    load,
    read_record,
)
from ommat import types as t
from ommat._canonical import hash_value
from ommat._replay import FindingKind, _replay_with
from ommat.evaluation import (
    DesignStore,
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    _sweep,
    split_id,
    write_gold_answers,
)
from ommat.evaluation._sweep import (
    _BONFERRONI,
    _FIXED_SEQUENCE,
    _action,
    _certifies,
    _choose,
    _gold_answers,
    _gold_copy,
    _scope,
    _tail_parts,
    _which_test,
    sweep_sync,
)
from ommat.helpers import HELPERS
from test_gold_answers import LOADED, ONE, TYPES, edited, key, unfinished
from test_use_answers import broken

R = RefusalReason
_OUTSIDE = {FindingKind.not_in_record, FindingKind.ref_not_in_record, FindingKind.question_changed}


def put(store: EvaluationStore, record: Record, golds: dict[str, object], fmt: int = 2) -> None:
    for ref, value in golds.items():
        store.write(
            {"kind": "gold-answer", "format": fmt, "key": key("g", record, ref), "value": value}
        )


LABELS: dict[str, object] = {
    "q:spam": "yes",  # and q:junk, one question at two refs
    "q:tone": "warm",
    "q:urgency": "high",
    **{f"q:{v}": {"value": None} for v in TYPES},
    "q:number": {"value": 120},
    "q:days": {"value": ["2024-02-29"]},
}


def copy_of(record: Record, store: EvaluationStore) -> Record | str:
    log = record.log
    return _gold_copy(record, log, _gold_answers(record.record_hash, log, "g", store))


def entry(record: Record, ref: str) -> dict[str, Any]:
    return next(i for i in record.log if i["item"] == "entry" and i["ref"] == ref)


def replayed(copy: Record, loaded: Any) -> tuple[Record, list[dict[str, Any]]]:
    return asyncio.run(_replay_with(copy.log, loaded, copy.log, observer=None))


# ---------------------------------------------------------------------------------------------
# The gold copy


def test_the_gold_copy_sets_every_question_and_carries_its_own_hash(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, LABELS)
    copy = copy_of(ONE, store)
    assert isinstance(copy, Record)
    assert copy.record_hash != ONE.record_hash
    assert copy.log[0]["hash"] == ONE.log[0]["hash"]  # the design item is kept
    assert read_record(copy.document).record_hash == copy.record_hash
    assert entry(copy, "q:spam")["value"]["probabilities"] == {"yes": 1, "no": 0}
    assert entry(copy, "q:junk")["value"] == entry(copy, "q:spam")["value"]
    assert entry(copy, "q:tone")["value"]["probabilities"] == {"warm": 1, "cold": 0}
    assert entry(copy, "q:urgency")["value"]["probabilities"] == {"low": 0, "high": 1}
    number = entry(copy, "q:number")["value"]
    assert (number["value"], number["p_null"], number["confidence"]) == (120, 0, 1)
    assert (number["guess"], number["span"]) == (None, None)
    null = entry(copy, "q:date")["value"]
    assert (null["value"], null["p_null"], null["confidence"]) == (None, 1, None)
    assert entry(copy, "q:days")["value"]["value"] == ["2024-02-29"]
    assert entry(copy, "t:state") == entry(ONE, "t:state")
    rerun, found = replayed(copy, LOADED)
    assert not [f for f in found if f["kind"] in _OUTSIDE]
    assert entry(rerun, "q:spam") == entry(copy, "q:spam")  # answered from the copy


def test_a_format_1_gold_entry_is_read(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {k: v for k, v in LABELS.items() if type(v) is str}, fmt=1)
    put(store, ONE, {k: v for k, v in LABELS.items() if type(v) is not str})
    copy = copy_of(ONE, store)
    assert isinstance(copy, Record)
    assert entry(copy, "q:tone")["value"]["probabilities"] == {"warm": 1, "cold": 0}


def test_an_extraction_gold_copy_is_read_back_and_replayed(tmp_path: Path) -> None:
    loaded = x.loaded()

    async def silent(batch: Any, *, seconds: float) -> CallResult:
        return CallResult({})

    record = decide_sync(loaded, x.STATE, {"default": silent}).record
    store = EvaluationStore(tmp_path)
    put(store, record, {"q:amount": {"value": 120}, "q:dates": {"value": None}})
    copy = copy_of(record, store)
    assert isinstance(copy, Record)
    back = read_record(copy.document)
    rerun, found = replayed(back, loaded)
    assert not [f for f in found if f["kind"] in _OUTSIDE]
    assert entry(rerun, "d:amount")["value"] == "big"
    assert "marker" in entry(rerun, "d:dates")  # a null gold value extracts to none
    assert entry(record, "d:amount") != entry(rerun, "d:amount")


def test_an_unknown_gold_entry_is_refused_naming_its_key(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {"q:tone": "warm"}, fmt=3)
    with pytest.raises(EvaluationRefusal) as raised:
        copy_of(ONE, store)
    assert (raised.value.reason, raised.value.key) == (R.unknown_entry, [key("g", ONE, "q:tone")])


def test_a_turn_with_no_gold_copy_says_why(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {k: v for k, v in LABELS.items() if k != "q:tone"})
    assert copy_of(ONE, store) == "q:tone has no gold answer"
    assert copy_of(unfinished(), store) == "the record has no verdict item"
    assert copy_of(broken(), store) == "the record has no state entry"


@pytest.mark.parametrize(
    ("ref", "value"), [("q:number", "yes"), ("q:tone", "kind words"), ("q:number", {"value": "x"})]
)
def test_a_gold_answer_not_of_its_questions_form_is_an_unknown_entry(
    tmp_path: Path, ref: str, value: object
) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {**LABELS, ref: value}, fmt=1 if type(value) is str else 2)
    with pytest.raises(EvaluationRefusal) as raised:
        copy_of(ONE, store)
    assert (raised.value.reason, raised.value.key) == (R.unknown_entry, [key("g", ONE, ref)])


def test_a_candidate_valued_sets_labels_are_its_fixed_part_and_candidates(tmp_path: Path) -> None:
    record = label_sets.run(["ann", "bo"])
    store = EvaluationStore(tmp_path)
    labels = {record.record_hash: {"q:sender": "bo", "q:other": "no"}}
    write_gold_answers("g", [record], labels, results=store)
    copy = copy_of(record, store)
    assert isinstance(copy, Record)
    probabilities = entry(copy, "q:sender")["value"]["probabilities"]
    assert probabilities == {"none": 0, "several": 0, "ann": 0, "bo": 1}
    assert read_record(copy.document).record_hash == copy.record_hash


def test_a_candidate_set_holding_a_marker_leaves_its_fixed_part_as_the_labels(
    tmp_path: Path,
) -> None:
    def marked(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for item in log:
            if item.get("ref") == label_sets.SET:
                del item["value"]
                cause = {"code": "raised", "refs": [label_sets.SET]}
                item["marker"] = {"kind": "Unavailable", "cause": cause}
        return log

    record = edited(label_sets.run(["ann", "bo"]), marked)
    store = EvaluationStore(tmp_path)
    labels = {record.record_hash: {"q:sender": "none", "q:other": "no"}}
    write_gold_answers("g", [record], labels, results=store)
    copy = copy_of(record, store)
    assert isinstance(copy, Record)
    assert entry(copy, "q:sender")["value"]["probabilities"] == {"none": 1, "several": 0}
    assert read_record(copy.document).record_hash == copy.record_hash


def test_a_failed_call_leaves_no_gold_copy(tmp_path: Path) -> None:
    async def boom(batch: Any, *, seconds: float) -> CallResult:
        raise RuntimeError("down")

    record = decide_sync(LOADED, x.STATE, {"default": boom}).record
    store = EvaluationStore(tmp_path)
    put(store, record, LABELS)
    assert copy_of(record, store) == "the call that wrote q:date is Failed"


# ---------------------------------------------------------------------------------------------
# The action of a passed execute plan


def plan_log(arguments: dict[str, Any], kind: str = "execute", verdict: str = "passed") -> Any:
    step = {"tool": "notify", "arguments": arguments, "requirements": [{"ref": "d:a"}]}
    return [
        {"item": "entry", "ref": "d:a", "value": {"to": {"name": "ann"}, "n": None}},
        {"item": "entry", "ref": "d:b", "value": "ann"},
        {"item": "plan", "kind": kind, "steps": [step], "derivation": [], "rows": [1]},
        {"item": "verdict", "kind": verdict, "failures": []},
    ]


def test_an_action_replaces_each_board_ref_by_what_it_reads() -> None:
    literal = {"literal": "x"}
    got = _action(plan_log({"who": {"ref": "d:a.to.name"}, "z": literal, "s": {"step": 1}}))
    assert got is not None
    assert json.loads(got) == {
        "kind": "execute",
        "steps": [
            {
                "tool": "notify",
                "arguments": {"who": {"value": "ann"}, "z": literal, "s": {"step": 1}},
                "requirements": [{"ref": "d:a"}],  # requirements stay as written
            }
        ],
    }
    # Two refs reading one value are one action; a null on the path reads null.
    assert _action(plan_log({"who": {"ref": "d:b"}})) == _action(
        plan_log({"who": {"ref": "d:a.to.name"}})
    )
    assert b'{"value":null}' in (_action(plan_log({"who": {"ref": "d:a.n"}})) or b"")


def test_only_a_passed_execute_plan_has_an_action() -> None:
    assert _action(plan_log({}, kind="clarify")) is None
    assert _action(plan_log({}, verdict="refused")) is None
    assert _action([]) is None


# ---------------------------------------------------------------------------------------------
# The binomial test


def tail(n: int, alpha: Fraction, k: int) -> Fraction:
    return Fraction(*_tail_parts(n, alpha, k))


def direct(n: int, alpha: Fraction, k: int) -> Fraction:
    """The tail as the direct sum of binomial terms."""
    return sum(
        (math.comb(n, i) * alpha**i * (1 - alpha) ** (n - i) for i in range(min(k, n) + 1)),
        Fraction(0),
    )


@pytest.mark.parametrize("alpha", [Fraction(1, 10), Fraction(1, 2), Fraction(0.05), Fraction(0.3)])
def test_the_tail_is_exact(alpha: Fraction) -> None:
    for n in [0, 1, 2, 7, 10, 50, 97]:
        for k in {0, 1, 3, n // 2, n - 1, n} - {-1}:
            if k <= n:  # escaped never exceeds the turns
                assert tail(n, alpha, k) == direct(n, alpha, k), (n, k)
    assert tail(2, Fraction(1, 2), 0) == Fraction(1, 4)


@pytest.mark.parametrize(
    ("n", "k", "alpha"), [(50, 1, 0.1), (400, 12, 0.05), (1000, 0, 0.003), (5000, 40, 0.01)]
)
def test_the_tail_agrees_with_scipy(n: int, k: int, alpha: float) -> None:
    expected = float(scipy.stats.binom.cdf(k, n, alpha))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    assert float(tail(n, Fraction(alpha), k)) == pytest.approx(expected, rel=1e-9, abs=0)


def test_a_tail_equal_to_the_level_certifies() -> None:
    assert _certifies(1, 0, 0.5, Fraction(1, 2))  # P(Bin(1, 0.5) <= 0) is exactly 0.5
    assert not _certifies(1, 0, 0.5, Fraction(1, 2) - Fraction(1, 10**30))
    assert _choose(_FIXED_SEQUENCE, 1, [0], 0.5, 0.5) == (0, 1)


def test_alpha_is_read_as_its_binary_value() -> None:
    # 0.6 as a double is just below 0.6, so (1 - alpha)**2 is just above 0.16, and above the
    # double 0.16 too: at the decimal values the tail would equal delta and certify.
    assert direct(2, Fraction(3, 5), 0) == Fraction(4, 25)
    assert not _certifies(2, 0, 0.6, Fraction(0.16))
    assert _choose(_FIXED_SEQUENCE, 2, [0], 0.6, 0.16) == (None, 1)


def test_delta_is_read_as_its_binary_value() -> None:
    # 1 - 0.6 as doubles is exactly the double 0.4, just above 2/5: at delta's binary value the
    # tail equals the level and certifies; at its decimal value it would not.
    assert 1 - Fraction(0.6) == Fraction(0.4) > Fraction(2, 5)
    assert _choose(_FIXED_SEQUENCE, 1, [0], 0.6, 0.4) == (0, 1)
    assert _choose(_BONFERRONI, 1, [0, 0], 0.6, 0.8) == (1, 2)  # 0.8 / 2 is the double 0.4


def test_a_large_tail_is_quick() -> None:
    assert _certifies(20000, 150, 0.01, Fraction(1, 2))


def test_the_order_check_reads_the_dev_split() -> None:
    assert _which_test([0, 0, 1, 4]) == _FIXED_SEQUENCE
    assert _which_test([]) == _which_test([3]) == _FIXED_SEQUENCE
    assert _which_test([0, 2, 1]) == _BONFERRONI


def test_fixed_sequence_testing_stops_at_the_first_point_not_certifying() -> None:
    # 0.9**50 is about 0.005, P(Bin(50, 0.1) <= 1) about 0.034, P(<= 5) about 0.62.
    assert _choose(_FIXED_SEQUENCE, 50, [0, 1, 5, 0], 0.1, 0.05) == (1, 3)
    assert _choose(_FIXED_SEQUENCE, 50, [5, 0], 0.1, 0.05) == (None, 1)
    assert _choose(_FIXED_SEQUENCE, 50, [0, 1], 0.1, 0.05) == (1, 2)


def test_bonferroni_tests_every_point_at_delta_over_m() -> None:
    assert _choose(_BONFERRONI, 50, [5, 0, 1], 0.1, 0.05) == (1, 3)  # 0.034 > 0.05 / 3
    assert _choose(_BONFERRONI, 50, [1, 0, 5], 0.1, 0.05) == (1, 3)
    assert _choose(_BONFERRONI, 50, [5, 6], 0.1, 0.05) == (None, 2)
    assert _choose(_BONFERRONI, 0, [0], 0.1, 0.05) == (None, 1)  # no turn certifies nothing
    assert _choose(_BONFERRONI, 50, [0, 5], 0.1, 0.05) == (0, 2)  # not the last tested


# ---------------------------------------------------------------------------------------------
# The sweep job


def spam_policy(r: Any) -> Plan:
    if r.param("spam/bar") == BREAKS:
        raise RuntimeError("the policy breaks at this bar")
    v = r.read("d:spam")
    if v == "yes":
        step = Step("archive", {"what": Origin(ref="d:spam")}, [Requirement("d:spam", "yes")])
        return Plan("execute", [step])
    if v == "no":
        return Plan("inform", reply="kept")
    return Plan("escalate", reason="unsure")


BREAKS = 0.25
FUNCTIONS = {"policy": spam_policy}
SPAM = Design(
    state=t.record(n=t.integer()),
    sources={
        "ask": Source({"q:spam": Question("noul", Text("Spam?"))}, needs=["t:state"]),
        # Asked only when d:spam is yes, so a turn whose answer moves there asks outside its
        # record.
        "check": Source({"t:checked": t.string}, requirement=SourceRequirement("d:spam", ["yes"])),
    },
    rules={
        "spam": Table(
            "d:spam",
            'label{"no", "yes"}',
            "F",
            [Helper("pick_with_margin", "q:spam", bar="spam/bar", margin="spam/m")],
            [["yes", "yes"], ["no", "no"]],
        )
    },
    parameters={
        "spam/bar": Parameter(t.number(0, 1), 0.5),
        "spam/m": Parameter(t.number(0, 1), 0.0),
    },
    policy=Policy(spam_policy, version="1", reads=["d:spam"], params=["spam/bar"]),
    tools=["archive"],
)
SPAM_DESIGN = load(SPAM)


def spam_turn(n: int, p_yes: float, design: LoadedDesign = SPAM_DESIGN) -> Record:
    async def answer(batch: Any, *, seconds: float) -> CallResult:
        if batch[0].id == "check":
            return CallResult({"t:checked": "ok"})
        return CallResult({"q:spam": {"yes": p_yes, "no": round(1 - p_yes, 10)}})

    return decide_sync(design, {"n": n}, {"default": answer}).record


def by_hand(records: list[Record]) -> str:
    return hash_value(sorted({r.record_hash for r in records}))


def test_split_id_is_the_hash_of_its_records_hashes_each_once_in_code_point_order() -> None:
    records = [spam_turn(n, p) for n, p in [(40, 0.9), (41, 0.2), (42, 0.55)]]
    expected = by_hand(records)
    assert split_id(records) == expected
    assert split_id([records[2], records[0], records[1], records[0]]) == expected
    assert split_id(r.document for r in records) == expected  # stored documents
    assert split_id([records[0].document, records[1], records[2]]) == expected
    assert split_id([]) == hash_value([])


def test_split_id_refuses_a_record_that_fails_the_read() -> None:
    with pytest.raises(EvaluationRefusal) as raised:
        split_id([DEV[0], b"{}"])
    assert raised.value.reason is R.invalid_record
    assert raised.value.error is not None


@pytest.mark.parametrize("records", ["records", b"records", {"a": 1}, 5])
def test_split_id_takes_an_iterable_of_records(records: Any) -> None:
    with pytest.raises(TypeError, match="records"):
        split_id(records)


# Each turn: its state, the model's probability of spam, and its gold label.
DEV_TURNS = [(0, 0.9, "yes"), (1, 0.7, "no"), (2, 0.2, "no"), (3, 0.55, "yes")]
CERT_TURNS = [(10, 0.9, "yes"), (11, 0.2, "no"), (12, 0.55, "no"), (13, 0.8, "yes")]
DEV = [spam_turn(n, p) for n, p, _ in DEV_TURNS]
CERT = [spam_turn(n, p) for n, p, _ in CERT_TURNS]
POINTS: list[dict[str, Any]] = [
    {"spam/bar": 0.95},
    {"spam/bar": 0.65},
    {"spam/bar": 0.0},
    {"spam/bar": 0.0, "spam/m": 0.0},
]


def sweep_doc(**changes: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "format": 1,
        "design": SPAM_DESIGN.design_hash,
        "name": "bar",
        "points": POINTS,
        "alpha": 0.5,
        "delta": 0.1,
        "dev": split_id(DEV),
        "certification": split_id(CERT),
        "gold_set": "g",
    }
    doc.update(changes)
    return doc


class Stores:
    def __init__(self, root: Path, labelled: bool = True) -> None:
        self.designs = DesignStore(root / "designs")
        self.results = EvaluationStore(root / "results")
        self.designs.keep(SPAM_DESIGN)
        if labelled:
            for records, turns in ((DEV, DEV_TURNS), (CERT, CERT_TURNS)):
                labels = {
                    r.record_hash: {"q:spam": g}
                    for r, (_, _, g) in zip(records, turns, strict=True)
                }
                write_gold_answers("g", records, labels, results=self.results)

    def run(self, document: object = None, splits: object = None, **kw: Any) -> dict[str, Any]:
        return sweep_sync(
            sweep_doc() if document is None else document,
            ((split_id(DEV), DEV), (split_id(CERT), CERT)) if splits is None else splits,
            functions=kw.get("functions", FUNCTIONS),
            designs=self.designs,
            results=self.results,
        )

    def refused(self, *args: Any, **kw: Any) -> EvaluationRefusal:
        with pytest.raises(EvaluationRefusal) as raised:
            self.run(*args, **kw)
        return raised.value


def counts(**named: int) -> dict[str, int]:
    return {
        k: named.get(k, 0)
        for k in ("matched", "escaped", "escalated", "clarify", "inform", "broken")
    }


def test_a_sweep_certifies_by_fixed_sequence_testing_and_keeps_the_chosen_design(
    tmp_path: Path,
) -> None:
    stores = Stores(tmp_path)
    entry = stores.run()
    key = {"design": SPAM_DESIGN.design_hash, "sweep": "bar"}
    assert (entry["kind"], entry["format"], entry["key"]) == ("certificate", 1, key)
    value = entry["value"]
    assert value["sweep"] == {k: v for k, v in sweep_doc().items() if k != "format"}
    assert value["helpers"] == HELPERS
    assert value["test"] == "fixed-sequence"
    assert value["counts"]["dev"] == {
        "turns": 4,
        "points": [
            counts(escalated=4),
            counts(matched=1, escaped=1, inform=1, escalated=1),
            counts(matched=2, escaped=1, inform=1),
            counts(matched=2, escaped=1, inform=1),
        ],
    }
    assert value["counts"]["certification"] == {
        "turns": 4,
        "points": [
            counts(escalated=4),
            counts(matched=2, inform=1, escalated=1),
            counts(matched=2, escaped=1, inform=1),
            None,  # not tested: the point before it did not certify
        ],
    }
    assert value["chosen"] == 1
    chosen = stores.designs.fetch(value["chosen_design"], functions=FUNCTIONS)
    assert chosen._leaf_value("spam/bar") == 0.65
    call = next(i for i in DEV[0].log if i["item"] == "call")
    assert value["scope"] == {
        "questions": [call["questions"]["q:spam"]["identity"]],
        "providers": [],
        "models": [],
        "revisions": [],
        "elicitations": [],
    }
    assert stores.results.lookup("certificate", key) == entry
    assert json.loads(json.dumps(entry)) == entry


def test_a_sweep_takes_its_document_as_non_canonical_bytes(tmp_path: Path) -> None:
    data = json.dumps(sweep_doc(), indent=2).encode()
    assert Stores(tmp_path).run(data)["value"]["chosen"] == 1


def test_a_non_monotone_dev_split_takes_bonferroni(tmp_path: Path) -> None:
    points = [{"spam/bar": 0.65}, {"spam/bar": 0.95}]  # dev escapes 1, then 0
    value = Stores(tmp_path).run(sweep_doc(points=points, delta=0.2))["value"]
    assert value["test"] == "bonferroni"
    tested = value["counts"]["certification"]["points"]
    assert [c["escaped"] for c in tested] == [0, 0]  # every point tested
    assert value["chosen"] == 1  # 0.5**4 is within 0.2 / 2


def test_no_point_chosen_keeps_no_design(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    points = [{"spam/bar": 0.65}, {"spam/bar": 0.95}]
    value = stores.run(sweep_doc(points=points))["value"]  # 0.5**4 is past 0.1 / 2
    assert (value["test"], value["chosen"], value["chosen_design"]) == ("bonferroni", None, None)
    assert [p.name for p in (tmp_path / "designs" / "sha256").iterdir()] == [
        SPAM_DESIGN.design_hash.removeprefix("sha256:")
    ]


# ---------------------------------------------------------------------------------------------
# The refusals, in their order


@pytest.mark.parametrize(
    "document",
    [
        {k: v for k, v in sweep_doc().items() if k != "format"},
        sweep_doc(format=2),
        sweep_doc(format=True),
        sweep_doc(format=1.5),
        json.dumps(sweep_doc(format="1", alpha=7)).encode(),
    ],
)
def test_an_unknown_format_is_refused_first(tmp_path: Path, document: object) -> None:
    exc = Stores(tmp_path, labelled=False).refused(document)
    assert exc.reason is R.unknown_format


@pytest.mark.parametrize(
    "document",
    [
        b'{"format": 1,',
        b'{"format": 2, "format": 1}',
        b"\\xff",
        [sweep_doc()],
        sweep_doc(alpha=math.nan),
        sweep_doc(extra=1),
        {k: v for k, v in sweep_doc().items() if k != "gold_set"},
        sweep_doc(name=1),
        sweep_doc(gold_set=None),
        sweep_doc(design=["x"]),
        sweep_doc(dev=1),
        sweep_doc(certification=1),
        sweep_doc(points={}),
        sweep_doc(points=[1]),
        sweep_doc(points=[]),
        sweep_doc(alpha=0),
        sweep_doc(alpha=1),
        sweep_doc(delta=True),
        sweep_doc(delta="0.1"),
        sweep_doc(certification=split_id(DEV)),
    ],
)
def test_an_invalid_sweep_is_refused(tmp_path: Path, document: object) -> None:
    exc = Stores(tmp_path, labelled=False).refused(document)
    assert exc.reason is R.invalid_sweep
    assert exc.error is None


def test_a_parse_error_is_chained(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(b"{")
    assert exc.__cause__ is not None


def test_a_split_not_the_sweeps_is_refused_naming_both_ids(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    swapped = ((split_id(CERT), CERT), (split_id(DEV), DEV))
    exc = stores.refused(splits=swapped)
    assert (exc.reason, exc.split) == (R.wrong_split, (split_id(CERT), split_id(DEV)))
    wrong = ((split_id(DEV), DEV), ("sha256:" + "0" * 64, CERT))
    exc = stores.refused(splits=wrong)
    assert (exc.reason, exc.split) == (R.wrong_split, ("sha256:" + "0" * 64, split_id(CERT)))


def test_a_repeated_sweep_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    entry = stores.run()
    exc = stores.refused(
        sweep_doc(points=[{"nope": 1}]), splits=((split_id(DEV), [b"x"]), (split_id(CERT), [b"x"]))
    )
    assert (exc.reason, exc.key) == (R.repeated_sweep, [entry["key"]])


def test_a_wrong_split_is_refused_before_a_repeated_sweep(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    stores.run()
    exc = stores.refused(splits=((split_id(CERT), CERT), (split_id(DEV), DEV)))
    assert exc.reason is R.wrong_split


def test_a_write_refused_at_the_store_is_held_key(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    entry = stores.run()

    class Blind(EvaluationStore):
        def lookup(self, kind: str, key: dict[str, str]) -> dict[str, Any] | None:
            return None if kind == "certificate" else super().lookup(kind, key)

    stores.results = Blind(tmp_path / "results")
    stores.designs = DesignStore(tmp_path / "again")
    stores.designs.keep(SPAM_DESIGN)
    exc = stores.refused()
    assert (exc.reason, exc.key) == (R.held_key, [entry["key"]])
    kept = stores.designs.fetch(entry["value"]["chosen_design"], functions=FUNCTIONS)
    assert kept._leaf_value("spam/bar") == 0.65  # kept before the write was refused


def test_the_design_stores_refusal_passes_through(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    stores.designs = DesignStore(tmp_path / "empty")
    exc = stores.refused()
    assert (exc.reason, exc.key) == (R.not_held, [SPAM_DESIGN.design_hash])
    exc = stores.refused(sweep_doc(design="x"))
    assert (exc.reason, exc.key) == (R.bad_key, ["x"])


def test_a_starting_design_that_does_not_load_is_invalid_design(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(functions={"policy": spam_turn})
    assert exc.reason is R.invalid_design
    assert isinstance(exc.error, InvalidDesign)
    assert exc.point is None


def test_every_points_paths_are_checked_before_any_point_loads(tmp_path: Path) -> None:
    points = [{"spam/bar": 2.0}, {"spam/bar": 0.5, "spam/nope": 1}]
    exc = Stores(tmp_path).refused(sweep_doc(points=points))
    assert (exc.reason, exc.point, exc.path) == (R.unknown_path, 1, "spam/nope")
    exc = Stores(tmp_path / "node").refused(sweep_doc(points=[{"spam": 1}]))
    assert (exc.reason, exc.point, exc.path) == (R.unknown_path, 0, "spam")  # a node


def test_an_unknown_path_is_refused_before_any_record_is_read(tmp_path: Path) -> None:
    splits = ((split_id(DEV), [*DEV, b"{}"]), (split_id(CERT), CERT))
    exc = Stores(tmp_path).refused(sweep_doc(points=[{"spam/nope": 1}]), splits=splits)
    assert (exc.reason, exc.path) == (R.unknown_path, "spam/nope")


def test_a_point_that_does_not_load_is_invalid_design_naming_it(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(sweep_doc(points=[{"spam/bar": 0.5}, {"spam/bar": 2.0}]))
    assert (exc.reason, exc.point) == (R.invalid_design, 1)
    assert isinstance(exc.error, InvalidDesign)
    assert exc.__cause__ is exc.error


def test_a_record_that_fails_the_read_is_invalid_record(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(splits=((split_id(DEV), [*DEV, b"{}"]), (split_id(CERT), CERT)))
    assert exc.reason is R.invalid_record
    assert exc.error is not None


def test_a_split_whose_id_differs_from_its_records_is_refused(tmp_path: Path) -> None:
    given = split_id(DEV[:3])
    doc = sweep_doc(dev=given)
    exc = Stores(tmp_path).refused(doc, splits=((given, DEV), (split_id(CERT), CERT)))
    assert (exc.reason, exc.split) == (R.split_id, (given, split_id(DEV)))
    unread = (split_id(CERT), [*CERT, b"{}"])
    exc = Stores(tmp_path / "both").refused(doc, splits=((given, DEV), unread))
    assert exc.reason is R.invalid_record  # every record is read before any id is checked


def test_the_dev_splits_id_is_checked_first(tmp_path: Path) -> None:
    dev_id, cert_id = split_id(DEV[:3]), split_id(CERT[:3])
    doc = sweep_doc(dev=dev_id, certification=cert_id)
    exc = Stores(tmp_path).refused(doc, splits=((dev_id, DEV), (cert_id, CERT)))
    assert (exc.reason, exc.split) == (R.split_id, (dev_id, split_id(DEV)))


@pytest.mark.parametrize("empty", [0, 1])
def test_an_empty_split_is_refused(tmp_path: Path, empty: int) -> None:
    records = [DEV, CERT]
    records[empty] = []
    ids = [split_id(r) for r in records]
    doc = sweep_doc(dev=ids[0], certification=ids[1])
    exc = Stores(tmp_path).refused(doc, splits=tuple(zip(ids, records, strict=True)))
    assert (exc.reason, exc.split) == (R.empty_split, (ids[empty], ids[empty]))


def test_an_empty_split_is_refused_after_every_record_is_read(tmp_path: Path) -> None:
    doc = sweep_doc(dev=split_id([]))
    splits: Any = ((split_id([]), []), (split_id(CERT), [*CERT, b"{}"]))
    assert Stores(tmp_path).refused(doc, splits=splits).reason is R.invalid_record


def test_an_empty_split_is_refused_after_its_id_is_checked(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(splits=((split_id(DEV), []), (split_id(CERT), CERT)))
    assert (exc.reason, exc.split) == (R.split_id, (split_id(DEV), hash_value([])))


def test_a_record_in_both_splits_is_refused_the_first_in_code_point_order(
    tmp_path: Path,
) -> None:
    low, mid, high = sorted(CERT[:3], key=lambda r: r.record_hash)
    dev = [*DEV, high, low, mid]  # the first in code-point order neither first nor last given
    cert = [CERT[3], mid, low, high]
    doc = sweep_doc(dev=split_id(dev))
    exc = Stores(tmp_path).refused(doc, splits=((split_id(dev), dev), (split_id(cert), cert)))
    assert (exc.reason, exc.turn) == (R.shared_record, low.record_hash)


def test_a_shared_record_is_refused_before_an_unknown_entry(tmp_path: Path) -> None:
    stores = Stores(tmp_path, labelled=False)
    stores.results.write(
        {"kind": "gold-answer", "format": 3, "key": key("g", DEV[0], "q:spam"), "value": "yes"}
    )
    dev = [*DEV, CERT[0]]
    exc = stores.refused(
        sweep_doc(dev=split_id(dev)), splits=((split_id(dev), dev), (split_id(CERT), CERT))
    )
    assert (exc.reason, exc.turn) == (R.shared_record, CERT[0].record_hash)


def test_a_gold_entry_of_an_unknown_format_is_refused(tmp_path: Path) -> None:
    stores = Stores(tmp_path, labelled=False)
    stores.results.write(
        {"kind": "gold-answer", "format": 3, "key": key("g", CERT[3], "q:spam"), "value": "yes"}
    )
    exc = stores.refused()
    assert (exc.reason, exc.key) == (R.unknown_entry, [key("g", CERT[3], "q:spam")])


def test_a_format_1_gold_entry_is_read_by_the_sweep(tmp_path: Path) -> None:
    stores = Stores(tmp_path, labelled=False)
    for records, turns in ((DEV, DEV_TURNS), (CERT, CERT_TURNS)):
        for record, (_, _, label) in zip(records, turns, strict=True):
            put(stores.results, record, {"q:spam": label}, fmt=1)
    assert stores.run()["value"]["chosen"] == 1


def test_a_turn_with_no_gold_answer_has_no_gold_plan_at_any_point(tmp_path: Path) -> None:
    stores = Stores(tmp_path, labelled=False)
    labels = {r.record_hash: {"q:spam": g} for r, (_, _, g) in zip(DEV, DEV_TURNS, strict=True)}
    write_gold_answers("g", DEV, labels, results=stores.results)
    exc = stores.refused()
    assert (exc.reason, exc.turn, exc.point) == (R.no_gold_plan, CERT[0].record_hash, None)
    assert "q:spam has no gold answer" in str(exc)


def test_a_gold_copy_broken_at_a_point_has_no_gold_plan_there(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(sweep_doc(points=[{"spam/bar": 0.9}, {"spam/bar": BREAKS}]))
    assert (exc.reason, exc.turn, exc.point) == (R.no_gold_plan, DEV[0].record_hash, 1)
    assert "broken" in str(exc)


def test_a_record_its_replay_does_not_determine_is_refused(tmp_path: Path) -> None:
    document = json.loads(SPAM_DESIGN.document)
    document["parameters"]["spam/bar"]["value"] = 0.95
    cautious = load(document, functions=FUNCTIONS)
    late = spam_turn(20, 0.9, cautious)  # unsure, so the check was never asked
    assert not any(i["item"] == "call" and i["sources"] == ["check"] for i in late.log)
    stores = Stores(tmp_path)
    write_gold_answers("g", [late], {late.record_hash: {"q:spam": "no"}}, results=stores.results)
    cert = [*CERT, late]
    doc = sweep_doc(certification=split_id(cert))
    exc = stores.refused(doc, splits=((split_id(DEV), DEV), (split_id(cert), cert)))
    assert (exc.reason, exc.turn, exc.point) == (R.not_determined, late.record_hash, 1)
    assert "not-in-record" in str(exc)


def test_the_chosen_designs_keep_refused_is_held_mismatch(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    document = json.loads(SPAM_DESIGN.document)
    document["parameters"]["spam/bar"]["value"] = 0.65
    chosen = load(document, functions=FUNCTIONS)
    path = tmp_path / "designs" / "sha256" / chosen.design_hash.removeprefix("sha256:")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"other")
    exc = stores.refused()
    assert (exc.reason, exc.key) == (R.held_mismatch, [chosen.design_hash])
    assert (
        stores.results.lookup("certificate", {"design": SPAM_DESIGN.design_hash, "sweep": "bar"})
        is None
    )


def test_sweep_sync_refuses_inside_a_running_loop(tmp_path: Path) -> None:
    async def inside() -> None:
        Stores(tmp_path).run()

    with pytest.raises(RuntimeError, match="await sweep"):
        asyncio.run(inside())


# ---------------------------------------------------------------------------------------------
# Each count, the splits' sizes and shapes, and the scope


def eager_policy(r: Any) -> Plan:
    v = r.read("d:spam")
    if v not in ("yes", "no"):
        raise RuntimeError("unsure")
    step = Step("archive", {"what": Origin(ref="d:spam")}, [Requirement("d:spam", v)])
    return Plan("execute", [step])


EAGER_FUNCTIONS = {"policy": eager_policy}
EAGER = load(
    dataclasses.replace(
        SPAM,
        sources={"ask": SPAM.sources["ask"]},
        policy=Policy(eager_policy, version="1", reads=["d:spam"]),
    )
)


def sweep_of(
    root: Path,
    dev: list[tuple[Record, str]],
    cert: list[tuple[Record, str]],
    points: list[dict[str, Any]],
    design: LoadedDesign = SPAM_DESIGN,
    functions: dict[str, Any] = FUNCTIONS,
    gold_set: str = "g",
) -> dict[str, Any]:
    """The sweep's entry over two splits, each record with its gold label under
    ``gold_set``."""
    designs, results = DesignStore(root / "designs"), EvaluationStore(root / "results")
    designs.keep(design)
    splits: list[tuple[str, list[Record]]] = []
    for turns in (dev, cert):
        records = [r for r, _ in turns]
        labels = {r.record_hash: {"q:spam": g} for r, g in turns}
        write_gold_answers(gold_set, records, labels, results=results)
        splits.append((split_id(records), records))
    doc = sweep_doc(
        design=design.design_hash,
        points=points,
        dev=splits[0][0],
        certification=splits[1][0],
        gold_set=gold_set,
    )
    return sweep_sync(doc, splits, functions=functions, designs=designs, results=results)


def test_an_execute_plan_differing_from_the_gold_one_escapes_and_a_broken_plan_counts(
    tmp_path: Path,
) -> None:
    dev = [(spam_turn(n, p, EAGER), g) for n, p, g in [(0, 0.9, "yes"), (1, 0.7, "no")]]
    dev.append(dev[0])  # a record repeated in its split counts once
    dev.append((spam_turn(2, 0.2, EAGER), "no"))
    cert = [(spam_turn(n, p, EAGER), g) for n, p, g in CERT_TURNS if p != 0.55]
    cert.append((spam_turn(14, 0.1, EAGER), "no"))
    value = sweep_of(
        tmp_path, dev, cert, [{"spam/bar": 0.95}, {"spam/bar": 0.0}], EAGER, EAGER_FUNCTIONS
    )["value"]
    assert value["counts"]["dev"] == {
        "turns": 3,
        "points": [counts(broken=3), counts(matched=2, escaped=1)],  # 0.7 says yes; gold no
    }
    assert value["counts"]["certification"] == {
        "turns": 4,
        "points": [counts(broken=4), counts(matched=4)],
    }
    # Certified on the certification split's 4 turns, 0.5**4 within 0.1; 3 would not be.
    assert (value["test"], value["chosen"]) == ("fixed-sequence", 1)


def test_the_entry_shares_nothing_with_the_callers_document(tmp_path: Path) -> None:
    document = sweep_doc(points=[{"spam/bar": 0.95}, {"spam/bar": 0.65}])
    kept = json.loads(json.dumps(document))
    entry = Stores(tmp_path).run(document)
    entry["value"]["sweep"]["points"][0]["spam/bar"] = 0.1
    assert document == kept
    document["points"].append({})
    assert len(entry["value"]["sweep"]["points"]) == 2


@pytest.mark.parametrize(
    "splits",
    [
        "splits",
        (),
        [(split_id(DEV), DEV)],
        ((split_id(DEV), DEV), (split_id(CERT), CERT), (split_id(CERT), CERT)),
        ((split_id(DEV), DEV), split_id(CERT)),
        ((split_id(DEV), DEV), (split_id(CERT), CERT, 1)),
        ((1, DEV), (split_id(CERT), CERT)),
        ((split_id(DEV), 5), (split_id(CERT), CERT)),
        ((split_id(DEV), DEV), (split_id(CERT), b"records")),
        ((split_id(DEV), {"a": 1}), (split_id(CERT), CERT)),
    ],
)
def test_malformed_splits_are_a_type_error_before_anything_else(
    tmp_path: Path, splits: object
) -> None:
    with pytest.raises(TypeError, match="split"):
        Stores(tmp_path, labelled=False).run(b"{", splits=splits)


def test_a_recorded_failed_call_rerun_unreturned_is_not_determined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def answer(batch: Any, *, seconds: float) -> CallResult:
        if batch[0].id == "check":
            raise RuntimeError("down")
        return CallResult({"q:spam": {"yes": 0.9, "no": 0.1}})

    failed = decide_sync(SPAM_DESIGN, {"n": 30}, {"default": answer}).record
    real = _sweep._replay_with

    async def unreturned(log: Any, design: Any, answers: Any, **kw: Any) -> Any:
        rerun, found = await real(log, design, answers, **kw)
        asked = next(i for i in log if i["item"] == "entry" and i["ref"] == "q:spam")
        if asked["value"]["probabilities"]["yes"] != 0.9:
            return rerun, found  # a gold copy
        marker = {"kind": "Unavailable", "cause": {"code": "not-returned", "refs": ["t:checked"]}}
        rewritten = [
            {"item": "entry", "ref": i["ref"], "call": i["call"], "marker": marker}
            if i["item"] == "entry" and i["ref"] == "t:checked"
            else i
            for i in rerun.log
        ]
        return SimpleNamespace(log=rewritten), found

    monkeypatch.setattr(_sweep, "_replay_with", unreturned)
    with pytest.raises(EvaluationRefusal) as raised:
        sweep_of(tmp_path, [(failed, "yes")], [(CERT[1], "no")], [{"spam/bar": 0.5}])
    exc = raised.value
    assert (exc.reason, exc.turn, exc.point) == (R.not_determined, failed.record_hash, 0)
    assert "not-returned" in str(exc)


def test_a_gold_copy_its_replay_does_not_determine_has_no_gold_plan_there(
    tmp_path: Path,
) -> None:
    document = json.loads(SPAM_DESIGN.document)
    document["parameters"]["spam/bar"]["value"] = 0.95
    late = spam_turn(21, 0.9, load(document, functions=FUNCTIONS))  # the check never asked
    with pytest.raises(EvaluationRefusal) as raised:
        sweep_of(tmp_path, [(late, "yes")], [(CERT[1], "no")], [{"spam/bar": 0.5}])
    exc = raised.value  # the gold label yes asks the check outside the record
    assert (exc.reason, exc.turn, exc.point) == (R.no_gold_plan, late.record_hash, 0)
    assert "not-in-record" in str(exc)


@pytest.mark.parametrize("value", [math.inf, 2**60])
def test_a_point_value_outside_the_domain_is_an_invalid_sweep(
    tmp_path: Path, value: object
) -> None:
    exc = Stores(tmp_path, labelled=False).refused(sweep_doc(points=[{"spam/bar": value}]))
    assert exc.reason is R.invalid_sweep


def test_the_scope_lists_each_call_items_profile() -> None:
    def call(identity: str | None, elicitation: str | None, **members: object) -> Any:
        questions = {} if identity is None else {"q:x": {"identity": identity}}
        return {"item": "call", "questions": questions, "elicitation": elicitation, **members}

    logs = [
        [call("i2", "e1", snapshot="s1"), {"item": "plan"}],  # before format 5
        [
            call(
                "i1", None, provider="p", model_requested="m1", model_reported=None, revision="r1"
            ),
            call(
                None, "e0", provider=None, model_requested="m0", model_reported="m2", revision=None
            ),
            call("i1", "e1", snapshot=None),
        ],
    ]
    assert _scope(logs) == {
        "questions": ["i1", "i2"],
        "providers": ["p"],
        "models": ["m1", "m2"],  # the reported one, else the requested
        "revisions": ["r1", "s1"],
        "elicitations": ["e0", "e1"],
    }


def test_the_scope_is_in_code_point_order_not_the_order_found() -> None:
    found = ["b", "C", "a"]  # code-point order C, a, b: neither as found nor reversed
    logs = [
        [
            {
                "item": "call",
                "questions": {"q:x": {"identity": f"q{v}"}},
                "provider": f"p{v}",
                "model_requested": f"m{v}",
                "revision": f"r{v}",
                "elicitation": f"e{v}",
            }
        ]
        for v in found
    ]
    assert _scope(logs) == {
        k: [f"{k[0]}{v}" for v in ("C", "a", "b")]
        for k in ("questions", "providers", "models", "revisions", "elicitations")
    }


def test_the_named_gold_set_is_read(tmp_path: Path) -> None:
    dev = [(r, g) for r, (_, _, g) in zip(DEV, DEV_TURNS, strict=True)]
    cert = [(r, g) for r, (_, _, g) in zip(CERT, CERT_TURNS, strict=True)]
    flipped = {"yes": "no", "no": "yes"}
    results = EvaluationStore(tmp_path / "results")
    for turns in (dev, cert):  # the other set says the opposite of every label
        labels = {r.record_hash: {"q:spam": flipped[g]} for r, g in turns}
        write_gold_answers("g", [r for r, _ in turns], labels, results=results)
    value = sweep_of(tmp_path, dev, cert, POINTS, gold_set="v2")["value"]
    assert value["sweep"]["gold_set"] == "v2"
    # As under the true labels; the flipped ones would give 2 escapes at the third point.
    assert value["counts"]["dev"]["points"][2] == counts(matched=2, escaped=1, inform=1)
    assert value["chosen"] == 1


def test_the_scope_gathers_every_turns_call_profile_through_the_sweep(tmp_path: Path) -> None:
    def profiled(record: Record, **members: object) -> Record:
        def edit(log: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for item in log:
                if item["item"] == "call":
                    item.update(members)
            return log

        return edited(record, edit)

    dev = [
        (profiled(DEV[0], provider="p1", model_requested="m1", elicitation="e1"), "yes"),
        (DEV[2], "no"),
    ]
    cert = [
        (profiled(CERT[0], provider="p2", model_requested="m0", model_reported="m2"), "yes"),
        (profiled(CERT[1], revision="r1", elicitation="e0"), "no"),
    ]
    value = sweep_of(tmp_path, dev, cert, [{"spam/bar": 0.65}])["value"]
    call = next(i for i in DEV[0].log if i["item"] == "call")
    assert value["scope"] == {
        "questions": [call["questions"]["q:spam"]["identity"]],
        "providers": ["p1", "p2"],
        "models": ["m1", "m2"],
        "revisions": ["r1"],
        "elicitations": ["e0", "e1"],
    }
