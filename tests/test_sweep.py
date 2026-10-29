# pyright: reportPrivateUsage=false
"""The sweep job: the gold copy, carrying its own hash, read back and replayed, from format-1
and format-2 gold entries and an extraction's (§gold-copy); a passed plan's action, the exact
binomial tail against scipy where it is installed, the order check and both tests'
choices (§sweep-scoring)."""

import asyncio
import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

import extraction as x
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
    write_gold_answers,
)
from ommat.evaluation._sweep import (
    BONFERRONI,
    FIXED_SEQUENCE,
    action,
    choose,
    gold_answers,
    gold_copy,
    sweep_sync,
    tail,
    which_test,
)
from ommat.helpers import HELPERS
from test_gold_answers import LOADED, ONE, TYPES, key, unfinished
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
    return gold_copy(record, log, gold_answers(record.record_hash, log, "g", store))


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


def test_a_label_for_an_extraction_question_is_no_answer(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    put(store, ONE, {**LABELS, "q:number": "yes"}, fmt=1)
    assert copy_of(ONE, store) == "the gold answer for q:number is no answer to its question"


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
    marker = {"kind": "Unavailable", "cause": {"code": "raised", "refs": ["d:m"]}}
    return [
        {"item": "entry", "ref": "d:a", "value": {"to": {"name": "ann"}, "n": None}},
        {"item": "entry", "ref": "d:b", "value": "ann"},
        {"item": "entry", "ref": "d:m", "marker": marker},
        {"item": "plan", "kind": kind, "steps": [step], "derivation": [], "rows": [1]},
        {"item": "verdict", "kind": verdict, "failures": []},
    ]


def test_an_action_replaces_each_board_ref_by_what_it_reads() -> None:
    literal = {"literal": "x"}
    got = action(plan_log({"who": {"ref": "d:a.to.name"}, "z": literal, "s": {"step": 1}}))
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
    assert action(plan_log({"who": {"ref": "d:b"}})) == action(
        plan_log({"who": {"ref": "d:a.to.name"}})
    )
    assert b'{"value":null}' in (action(plan_log({"who": {"ref": "d:a.n"}})) or b"")
    assert b'"marker":{"cause"' in (action(plan_log({"who": {"ref": "d:m"}})) or b"")


def test_only_a_passed_execute_plan_has_an_action() -> None:
    assert action(plan_log({}, kind="clarify")) is None
    assert action(plan_log({}, verdict="refused")) is None
    assert action([]) is None


# ---------------------------------------------------------------------------------------------
# The binomial test


@pytest.mark.parametrize(("n", "k"), [(0, 0), (1, 0), (1, 1), (10, 0), (10, 3), (50, 5), (7, 9)])
def test_the_tail_is_exact(n: int, k: int) -> None:
    alpha = Fraction(1, 10)
    expected = sum(
        math.comb(n, i) * alpha**i * (1 - alpha) ** (n - i) for i in range(min(k, n) + 1)
    )
    assert tail(n, alpha, k) == expected
    assert tail(2, Fraction(1, 2), 0) == Fraction(1, 4)


@pytest.mark.parametrize(("n", "k", "alpha"), [(50, 1, 0.1), (400, 12, 0.05), (1000, 0, 0.003)])
def test_the_tail_agrees_with_scipy(n: int, k: int, alpha: float) -> None:
    stats = pytest.importorskip("scipy.stats")
    assert float(tail(n, Fraction(alpha), k)) == pytest.approx(stats.binom.cdf(k, n, alpha))


def test_the_order_check_reads_the_dev_split() -> None:
    assert which_test([0, 0, 1, 4]) == FIXED_SEQUENCE
    assert which_test([]) == which_test([3]) == FIXED_SEQUENCE
    assert which_test([0, 2, 1]) == BONFERRONI


def test_fixed_sequence_testing_stops_at_the_first_point_not_certifying() -> None:
    # 0.9**50 is about 0.005, P(Bin(50, 0.1) <= 1) about 0.034, P(<= 5) about 0.62.
    assert choose(FIXED_SEQUENCE, 50, [0, 1, 5, 0], 0.1, 0.05) == (1, 3)
    assert choose(FIXED_SEQUENCE, 50, [5, 0], 0.1, 0.05) == (None, 1)
    assert choose(FIXED_SEQUENCE, 50, [0, 1], 0.1, 0.05) == (1, 2)


def test_bonferroni_tests_every_point_at_delta_over_m() -> None:
    assert choose(BONFERRONI, 50, [5, 0, 1], 0.1, 0.05) == (1, 3)  # 0.034 > 0.05 / 3
    assert choose(BONFERRONI, 50, [1, 0, 5], 0.1, 0.05) == (1, 3)
    assert choose(BONFERRONI, 50, [5, 6], 0.1, 0.05) == (None, 2)
    assert choose(BONFERRONI, 0, [0], 0.1, 0.05) == (None, 1)  # no turn certifies nothing


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


def split_id(records: list[Record]) -> str:
    return hash_value(sorted({r.record_hash for r in records}))


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
    exc = Stores(tmp_path, labelled=False).refused(document, splits=())
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
    exc = Stores(tmp_path, labelled=False).refused(document, splits=())
    assert exc.reason is R.invalid_sweep
    assert exc.error is None


def test_a_parse_error_is_chained(tmp_path: Path) -> None:
    exc = Stores(tmp_path).refused(b"{", splits=())
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


def test_a_write_refused_at_the_store_is_held_key(tmp_path: Path) -> None:
    stores = Stores(tmp_path)
    entry = stores.run()

    class Blind(EvaluationStore):
        def lookup(self, kind: str, key: dict[str, str]) -> dict[str, Any] | None:
            return None if kind == "certificate" else super().lookup(kind, key)

    stores.results = Blind(tmp_path / "results")
    exc = stores.refused()
    assert (exc.reason, exc.key) == (R.held_key, [entry["key"]])


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


def test_a_record_in_both_splits_is_refused(tmp_path: Path) -> None:
    shared = [CERT[1], CERT[0]]
    dev = [*DEV, *shared]
    doc = sweep_doc(dev=split_id(dev))
    exc = Stores(tmp_path).refused(doc, splits=((split_id(dev), dev), (split_id(CERT), CERT)))
    first = min(r.record_hash for r in shared)
    assert (exc.reason, exc.turn) == (R.shared_record, first)


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
