# pyright: reportPrivateUsage=false
"""The sweep job: a sweep document's points, each the starting design with some parameter leaves
replaced, scored against every turn's gold plan on a dev and a certification split, and a point
chosen by the exact binomial test, written as a ``certificate`` entry (§sweep).

**The gold copy** (§gold-copy). The gold answers are looked up as the gold-answers job derives
their keys, ``_gold``'s: the turn id the record's hash, the question identity from its call item
and the candidate part of its ref. An entry of a kind or format other than ``gold-answer`` 1 and
2, or whose value is not of its question's form as the gold-answers job reads it, is refused
``unknown-entry`` by ``known`` and ``conforming``, which the group job shares. A label becomes
an answer whose ``probabilities`` give 1 to the gold label and 0 to every other, over the
question's labels as asked, a candidate-valued set's from its set's entry, its fixed part alone
when that entry holds a marker; an extraction gold answer ``{"value": v}`` becomes ``value``
``v``, ``p_null`` 0 and ``confidence`` 1, or for ``v`` ``None`` ``value`` ``None``, ``p_null`` 1
and ``confidence`` ``None``, ``guess`` and ``span`` ``None`` in both. Each answer is written as
interpret writes one, on one copy whose ``hash`` is set and which is read back through
``read_record``, by ``_edit``'s machinery; a ``t:`` entry keeps its value or marker. A turn has
no gold copy when its record has no ``verdict`` item or no state entry, a ``q:`` entry a call
wrote has no gold answer, or a call that wrote one is ``Failed``. It has no gold plan at a point
when the gold copy's replay there does not determine the turn or its plan is ``broken``.

**Scoring** (§sweep-scoring). A replay determines its turn as the switching-value job reads a
probe's. A turn is counted at a point from its record's replay alone, against its gold copy's:
``escalated`` when the escalation view holds one, ``clarify`` or ``inform`` for a passed plan of
that kind, ``broken``, then ``matched`` when a passed ``execute`` plan's action is the gold
plan's, else ``escaped``. An action is the plan's ``kind`` and ``steps``, each board-ref
argument replaced by ``{"value": v}``, what the ref reads in the same replay record, compared by
canonical bytes. ``escaped`` non-decreasing along the points on the dev split selects
fixed-sequence testing, else Bonferroni; P(Bin(n, alpha) <= escaped) is computed exactly in
integers over ``alpha``'s exact fraction, by Horner's rule over a running binomial coefficient.

**The job.** ``splits`` not two ``(id, records)`` pairs, an id not a string or records not an
iterable is a ``TypeError`` before anything else. Refused in §sweep's order: the document
(``invalid-sweep`` for bytes that do not parse or a document that is not an object,
``unknown-format``, ``invalid-sweep``), then ``wrong-split``, ``repeated-sweep``, the design
store's fetch of the starting design (``invalid-design`` for its ``InvalidDesign``),
``unknown-path`` over every point's paths, then ``invalid-design`` for each point's load,
``invalid-record`` over both splits, then ``split-id`` for each, ``shared-record``,
``unknown-entry``, ``no-gold-plan`` (every turn's gold copy first, ``point`` ``None``, then
every gold copy replayed under every point), ``not-determined`` (every record replayed under
every point), then the store's ``held-mismatch`` at the chosen point's keep and ``held-key`` at
the write. A split's records are counted once each by record hash, in the order given. Every
replay runs before the dev split's counts are read; the certification split's counts are read
only by the test. An ``EntryRefusal`` from replay passes through. The entry written is returned,
plain data reaching no ``Record``.
"""

import copy
import itertools
import sys
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType
from ommat._canonical import _DecodeError, decode, encode, hash_value
from ommat._design import InvalidDesign, LoadedDesign
from ommat._interpret import answer_written
from ommat._ledger import _ITEM_DEPTH, Record
from ommat._load import load
from ommat._markers import PlanKind, Undetermined, VerdictKind
from ommat._refs import code_point_sorted, declared_ref
from ommat._replay import _first, _replay_with
from ommat._resolve import board_entries, board_value
from ommat._values import MAX_DOCUMENT_DEPTH, domain_error, guarded_repr, type_name
from ommat._views import escalation
from ommat.evaluation._diff import OUTSIDE, _read, _run_sync, _state
from ommat.evaluation._edit import _labels, _name, _record_copy, _rewritten
from ommat.evaluation._gold import _READ, KIND, _asked, _form, _keys
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import DesignStore, EvaluationStore
from ommat.evaluation._switching import _unreturned
from ommat.helpers import HELPERS

_R = RefusalReason
_Log = list[dict[str, Any]]
_Functions = Mapping[str, Callable[..., Any]] | None


def _gold_answers(
    turn: str, log: _Log, gold_set: str, results: EvaluationStore
) -> dict[str, object | None]:
    """Each ``q:`` ref a call wrote in ``log``, in log order, to its gold answer under
    ``gold_set`` as the gold-answers job writes a label, or ``None`` when none is held;
    ``unknown-entry`` for an entry of a kind or format the job does not read, or a value not of
    its question's form."""
    asked = _asked(log)
    golds: dict[str, object | None] = {}
    for _, key, refs in _keys(gold_set, turn, asked):
        entry = known(results.lookup(KIND, key), key)
        for ref in refs:
            golds[ref] = None if entry is None else conforming(entry, key, asked[ref], log)
    return {ref: golds[ref] for ref in asked}


def known(entry: dict[str, Any] | None, key: dict[str, str]) -> dict[str, Any] | None:
    """``entry``, a gold answer looked up under ``key`` or ``None``; ``unknown-entry`` for an
    entry of a kind or format the job does not read."""
    if entry is not None and (entry["kind"] != KIND or entry["format"] not in _READ):
        raise EvaluationRefusal(
            f"the entry under {key!r} is {entry['kind']} format {entry['format']}, not "
            f"{KIND} format 1 or 2",
            reason=_R.unknown_entry,
            key=[key],
        )
    return entry


def conforming(
    entry: dict[str, Any], key: dict[str, str], question: dict[str, Any], log: _Log
) -> object:
    """A known gold ``entry``'s value as the gold-answers job writes a label for ``question``,
    asked in ``log``; ``unknown-entry``, naming ``key``, for a value not of the question's form:
    a name not among its labels, a format-1 label string for an extraction question, or an
    extraction value that does not conform."""
    form = _form(entry["value"], question, log)
    if form is None:
        raise EvaluationRefusal(
            f"the entry under {key!r} holds {guarded_repr(entry['value'])}, not a gold answer "
            "of its question's form",
            reason=_R.unknown_entry,
            key=[key],
        )
    return form


def _gold_copy(read: Record, log: _Log, golds: dict[str, object | None]) -> Record | str:
    """The gold copy of ``read``, whose log is ``log``, with each ``q:`` ref of ``golds`` set
    to its gold answer, each already of its question's form; or why the turn has none."""
    if not any(item["item"] == "verdict" for item in log):
        return "the record has no verdict item"
    if _state(log) is None:
        return "the record has no state entry"
    edited = list(log)
    at = {e["ref"]: i for i, e in enumerate(log) if e["item"] == "entry"}
    for ref, gold in golds.items():
        if gold is None:
            return f"{ref} has no gold answer"
        call = log[cast(int, log[at[ref]]["call"])]
        if call["outcome"] != "returned":
            return f"the call that wrote {ref} is Failed"
    for ref, gold in golds.items():
        question = log[cast(int, log[at[ref]]["call"])]["questions"][declared_ref(ref)]
        edited[at[ref]] = _rewritten(log[at[ref]], value=_answered(ref, gold, question, log))
    return _record_copy(edited, read.unhashed)


def _answered(ref: str, gold: object, question: dict[str, Any], log: _Log) -> object:
    """The answer's document that gives ``gold``, a label as the gold-answers job writes one,
    all the probability: over the question's labels as asked, a candidate-valued set's fixed
    part alone when the set's entry holds a marker, as the gold-answers job reads them."""
    kind = AnswerKind(question["kind"])
    value_type, is_list = None, bool(question.get("is_list", False))
    if kind is AnswerKind.extraction:
        value_type = ValueType(question["value_type"])
        v = cast("dict[str, object]", gold)["value"]
        held = (1, None) if v is None else (0, 1)
        answer: object = {
            "value": v,
            "guess": None,
            "p_null": held[0],
            "confidence": held[1],
            "span": None,
        }
        labels: tuple[str, ...] = ()
    else:
        asked = _labels(question, log)
        labels = asked if asked is not None else _fixed(question)
        answer = {name: 1.0 if name == gold else 0.0 for name in labels}
    written = answer_written(ref, answer, kind, labels, value_type, is_list, sys.maxsize)
    assert not isinstance(written, Undetermined)  # conforming checked the gold answer's form
    return written[0]


def _fixed(question: dict[str, Any]) -> tuple[str, ...]:
    """A candidate-valued set's fixed part's names, the labels when the set holds a marker."""
    return tuple(_name(n) for n in cast("list[object]", question["labels"]["fixed"]))


# ---------------------------------------------------------------------------------------------
# Scoring (§sweep-scoring)

_COUNTS: tuple[str, ...] = ("matched", "escaped", "escalated", "clarify", "inform", "broken")
"""The names a turn is _counted under at a point, in the order the entry holds them."""
_FIXED_SEQUENCE, _BONFERRONI = "fixed-sequence", "bonferroni"


def _undetermined(log: _Log, rerun: Record, found: list[dict[str, Any]]) -> str | None:
    """Why the replay of ``log`` giving ``rerun`` and ``found`` does not determine its turn, as
    the switching-value job reads a probe's: a finding of a run that asked outside the record,
    or ``not-returned`` written for a ref whose recorded call is ``Failed``; ``None`` when it
    determines it."""
    outside = next((f for f in found if f["kind"] in OUTSIDE), None)
    if outside is not None:
        return f"the replay reports {outside['kind']} for {outside.get('ref')}"
    mine = {i["ref"]: i for i in log if i["item"] == "entry"}
    if _unreturned(mine, log, rerun.log):
        return "the replay writes not-returned for a ref whose recorded call is Failed"
    return None


def _action(log: _Log) -> bytes | None:
    """The canonical bytes of the action of ``log``'s plan when it is a passed ``execute``
    plan: its ``kind`` and ``steps``, each board-ref argument replaced by ``{"value": v}``,
    what the ref reads in ``log``; else ``None``. Plan rule 4 keeps a passed plan's step from
    naming a ref that holds a marker, so a ref reads a value."""
    plan, verdict = _first(log, "plan"), _first(log, "verdict")
    passed = verdict is not None and verdict["kind"] == VerdictKind.passed
    if plan is None or plan["kind"] != PlanKind.execute or not passed:
        return None
    values = board_entries(log)
    steps: list[dict[str, Any]] = []
    for step in cast("list[dict[str, Any]]", plan["steps"]):
        arguments: dict[str, Any] = {}
        for name, origin in cast("dict[str, dict[str, Any]]", step["arguments"]).items():
            ref = origin.get("ref")
            # A literal or a binding stays as written.
            arguments[name] = origin if ref is None else {"value": board_value(values, ref)}
        steps.append({**step, "arguments": arguments})
    return encode({"kind": plan["kind"], "steps": steps}, _ITEM_DEPTH)


def _counted(rerun: Record, gold: bytes | None) -> str:
    """The name a determined replay of a turn's record is _counted under at a point, against
    ``gold``, the action of its gold plan, ``None`` when that is no passed ``execute`` plan."""
    if escalation(rerun) is not None:
        return "escalated"
    log = rerun.log
    plan = cast("dict[str, Any]", _first(log, "plan"))  # a verdict passed: escalation saw it
    if plan["kind"] in (PlanKind.clarify, PlanKind.inform):
        return str(plan["kind"])
    if plan["kind"] == PlanKind.broken:
        return "broken"
    return "matched" if gold is not None and _action(log) == gold else "escaped"


def _tail_parts(n: int, alpha: Fraction, k: int) -> tuple[int, int]:
    """P(Bin(``n``, ``alpha``) <= ``k``) as an integer numerator over ``q**n``, ``q``
    ``alpha``'s denominator: with ``p`` its numerator and ``r = q - p``, the sum of
    C(n, i) p^i r^(n-i) for i up to ``k``, by Horner's rule in ``r`` over the running binomial
    coefficient and power of ``p``, so O(k) big-integer products."""
    p, q = alpha.numerator, alpha.denominator
    if k >= n:
        return q**n, q**n
    r = q - p
    total, c, pw = 0, 1, 1  # c = C(n, i), pw = p**i
    for i in range(k + 1):
        total = total * r + c * pw
        c = c * (n - i) // (i + 1)
        pw *= p
    return total * r ** (n - k), q**n


def _certifies(n: int, escaped: int, alpha: float, level: Fraction) -> bool:
    """Whether a point with ``escaped`` of ``n`` turns _certifies at ``level``: the exact
    binomial test P(Bin(n, alpha) <= escaped) <= level, compared in integers."""
    top, bottom = _tail_parts(n, Fraction(alpha), escaped)
    return top * level.denominator <= level.numerator * bottom


def _which_test(dev_escaped: list[int]) -> str:
    """Fixed-sequence testing when ``escaped`` on the dev split is non-decreasing along the
    points, else Bonferroni."""
    ordered = all(a <= b for a, b in itertools.pairwise(dev_escaped))
    return _FIXED_SEQUENCE if ordered else _BONFERRONI


def _choose(
    test: str, n: int, escaped: list[int], alpha: float, delta: float
) -> tuple[int | None, int]:
    """The chosen point's index, or ``None``, and how many points from the first were tested.
    Fixed-sequence testing tests each point at ``delta`` from the first, stops at the first
    that does not certify and chooses the last that did; Bonferroni tests every point at
    ``delta`` / m and chooses the last that _certifies."""
    chosen: int | None = None
    if test == _FIXED_SEQUENCE:
        for i, k in enumerate(escaped):
            if not _certifies(n, k, alpha, Fraction(delta)):
                return chosen, i + 1
            chosen = i
        return chosen, len(escaped)
    level = Fraction(delta) / len(escaped)
    for i, k in enumerate(escaped):
        if _certifies(n, k, alpha, level):
            chosen = i
    return chosen, len(escaped)


# ---------------------------------------------------------------------------------------------
# The job (§sweep)

_CERTIFICATE = "certificate"
_MEMBERS: frozenset[str] = frozenset(
    ("format", "design", "name", "points", "alpha", "delta", "dev", "certification", "gold_set")
)
_STRINGS = ("design", "name", "dev", "certification", "gold_set")
_SPLITS = ("dev", "certification")


async def sweep(
    document: object,
    splits: object,
    *,
    functions: _Functions,
    designs: DesignStore,
    results: EvaluationStore,
) -> dict[str, Any]:
    """Certify the sweep ``document``, bytes or parsed, on ``splits``, the dev and the
    certification split, each ``(id, records)``; the certificate entry written."""
    given = _given_splits(splits)
    doc = _sweep_document(document)
    for name, (split_id, _) in zip(_SPLITS, given, strict=True):
        if split_id != doc[name]:
            raise EvaluationRefusal(
                f"the {name} split's id is {split_id!r}, not the sweep's {doc[name]!r}",
                reason=_R.wrong_split,
                split=(split_id, doc[name]),
            )
    key = {"design": doc["design"], "sweep": doc["name"]}
    if results.lookup(_CERTIFICATE, key) is not None:
        raise EvaluationRefusal(
            f"the store holds a {_CERTIFICATE} entry under {key!r}",
            reason=_R.repeated_sweep,
            key=[key],
        )
    try:
        start = designs.fetch(doc["design"], functions=functions)
    except InvalidDesign as exc:
        raise EvaluationRefusal(
            f"the starting design does not load: {exc}", reason=_R.invalid_design, error=exc
        ) from exc
    points = _point_designs(doc["points"], start, functions)
    turns = [_split(name, given[i][1]) for i, name in enumerate(_SPLITS)]
    for i, name in enumerate(_SPLITS):
        _refuse_split_id(name, given[i][0], turns[i])
    shared = code_point_sorted(set(turns[0]) & set(turns[1]))
    if shared:
        raise EvaluationRefusal(
            f"record {shared[0]} is in both splits", reason=_R.shared_record, turn=shared[0]
        )
    every = [(turn, read) for split in turns for turn, read in split.items()]
    golds = {turn: _gold_answers(turn, read.log, doc["gold_set"], results) for turn, read in every}
    copies: dict[str, _Log] = {}
    for turn, read in every:
        made = _gold_copy(read, read.log, golds[turn])
        if isinstance(made, str):
            raise _no_gold_plan(turn, None, made)
        copies[turn] = made.log
    gold = {
        at: _action(rerun.log) for at, rerun in (await _runs(copies, points, gold=True)).items()
    }
    runs = await _runs({turn: read.log for turn, read in every}, points, gold=False)
    count = {at: _counted(rerun, gold[at]) for at, rerun in runs.items()}
    tables = [[_counts(split, p, count) for p in range(len(points))] for split in turns]
    test = _which_test([c["escaped"] for c in tables[0]])
    n = len(turns[1])
    escaped = [c["escaped"] for c in tables[1]]
    chosen, tested = _choose(test, n, escaped, doc["alpha"], doc["delta"])
    chosen_design = None
    if chosen is not None:
        designs.keep(points[chosen])
        chosen_design = points[chosen].design_hash
    value = {
        "sweep": {k: v for k, v in doc.items() if k != "format"},
        "helpers": HELPERS,
        "test": test,
        "counts": {
            "dev": {"turns": len(turns[0]), "points": tables[0]},
            "certification": {
                "turns": n,
                "points": [c if p < tested else None for p, c in enumerate(tables[1])],
            },
        },
        "chosen": chosen,
        "chosen_design": chosen_design,
        "scope": _scope([read.log for _, read in every]),
    }
    entry = {"kind": _CERTIFICATE, "format": 1, "key": key, "value": value}
    results.write(entry)
    return entry


async def _runs(
    logs: dict[str, _Log], points: list[LoadedDesign], *, gold: bool
) -> dict[tuple[str, int], Record]:
    """Each log replayed under each point's design, by turn and point; a replay that does not
    determine its turn refused, ``no-gold-plan`` for a gold copy, whose ``broken`` plan is
    refused too, else ``not-determined``."""
    out: dict[tuple[str, int], Record] = {}
    for turn, log in logs.items():
        for p, design in enumerate(points):
            rerun, found = await _replay_with(log, design, log, observer=None)
            why = _undetermined(log, rerun, found)
            if gold:
                plan = cast("dict[str, Any]", _first(rerun.log, "plan"))
                if why is None and plan["kind"] == PlanKind.broken:
                    why = "its gold copy's plan is broken"
                if why is not None:
                    raise _no_gold_plan(turn, p, why)
            elif why is not None:
                raise EvaluationRefusal(
                    f"turn {turn}'s replay under point {p} does not determine it: {why}",
                    reason=_R.not_determined,
                    turn=turn,
                    point=p,
                )
            out[turn, p] = rerun
    return out


def sweep_sync(
    document: object,
    splits: object,
    *,
    functions: _Functions,
    designs: DesignStore,
    results: EvaluationStore,
) -> dict[str, Any]:
    """``sweep`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync(
        "sweep",
        lambda: sweep(document, splits, functions=functions, designs=designs, results=results),
    )


def _given_splits(splits: object) -> list[tuple[str, Iterable[object]]]:
    """``splits`` as two ``(id, records)`` pairs; ``TypeError`` for any other shape, a split id
    that is not a string, or records that are not an iterable other than a string, bytes or a
    mapping."""
    if not isinstance(splits, list | tuple) or len(cast("tuple[object, ...]", splits)) != 2:
        raise TypeError(
            f"splits is a {type_name(cast(object, splits))}, not a pair of (id, records) splits"
        )
    given: list[tuple[str, Iterable[object]]] = []
    for name, pair in zip(_SPLITS, cast("tuple[object, ...]", splits), strict=True):
        if not isinstance(pair, list | tuple) or len(cast("tuple[object, ...]", pair)) != 2:
            raise TypeError(
                f"the {name} split is a {type_name(cast(object, pair))}, not an (id, records) pair"
            )
        split_id, records = cast("tuple[object, object]", tuple(cast("tuple[object, ...]", pair)))
        if type(split_id) is not str:
            raise TypeError(f"the {name} split's id is a {type_name(split_id)}, not a string")
        if not isinstance(records, Iterable) or isinstance(records, str | bytes | Mapping):
            raise TypeError(
                f"the {name} split's records are a {type_name(cast(object, records))}, not records"
            )
        given.append((split_id, cast("Iterable[object]", records)))
    return given


def _sweep_document(document: object) -> dict[str, Any]:
    """The sweep document parsed and checked: ``invalid-sweep`` for bytes that do not parse,
    then ``unknown-format``, then ``invalid-sweep`` for anything else outside its shape."""
    if isinstance(document, bytes | bytearray):
        try:
            document = decode(bytes(document))
        except _DecodeError as exc:
            raise _invalid(f"the sweep's bytes do not parse: {exc}") from exc
    if type(document) is not dict:
        raise _invalid(f"the sweep is a {type_name(document)}, not an object")
    doc = cast("dict[str, Any]", document)
    fmt = doc.get("format")
    if type(fmt) is not int or fmt != 1:
        raise EvaluationRefusal(
            f"the sweep's format is {guarded_repr(fmt)}, not 1", reason=_R.unknown_format
        )
    fault = domain_error(doc, MAX_DOCUMENT_DEPTH)
    if fault is not None:
        raise _invalid(f"the sweep holds a value outside the domain: {fault}")
    if frozenset(doc) != _MEMBERS:
        missing, extra = _MEMBERS - set(doc), set(doc) - _MEMBERS
        raise _invalid(f"the sweep's members miss {sorted(missing)} and add {sorted(extra)}")
    for name in _STRINGS:
        if type(doc[name]) is not str:
            raise _invalid(f"the sweep's {name} is a {type_name(doc[name])}, not a string")
    points: object = doc["points"]
    if type(points) is not list or not all(type(p) is dict for p in cast("list[object]", points)):
        raise _invalid("the sweep's points are not a list of objects")
    if not points:
        raise _invalid("the sweep has no points")
    for name in ("alpha", "delta"):
        v = doc[name]
        if type(v) not in (int, float) or not 0 < v < 1:
            raise _invalid(f"the sweep's {name} is {guarded_repr(v)}, not a number in (0, 1)")
    if doc["dev"] == doc["certification"]:
        raise _invalid("the sweep's dev and certification split ids are one id")
    return copy.deepcopy(doc)  # the entry returned shares nothing with the caller's document


def _invalid(message: str) -> EvaluationRefusal:
    return EvaluationRefusal(message, reason=_R.invalid_sweep)


def _point_designs(
    points: list[dict[str, Any]], start: LoadedDesign, functions: _Functions
) -> list[LoadedDesign]:
    """Each point's design, the starting document with its leaves replaced, loaded; first
    ``unknown-path`` for every point's paths, then ``invalid-design`` for each point's load."""
    for p, point in enumerate(points):
        for path in point:
            if path not in start._leaf_types:
                raise EvaluationRefusal(
                    f"point {p}'s path {path!r} names no leaf of the starting design",
                    reason=_R.unknown_path,
                    point=p,
                    path=path,
                )
    loaded: list[LoadedDesign] = []
    for p, point in enumerate(points):
        document = cast("dict[str, Any]", decode(start.document))
        for path, v in point.items():
            document["parameters"][path]["value"] = v
        try:
            loaded.append(load(document, functions=functions))
        except InvalidDesign as exc:
            raise EvaluationRefusal(
                f"point {p}'s design does not load: {exc}",
                reason=_R.invalid_design,
                point=p,
                error=exc,
            ) from exc
    return loaded


def _split(name: str, records: Any) -> dict[str, Record]:
    """A split's records by record hash, each once, in the order given; ``invalid-record`` for
    a record that fails the read."""
    read: dict[str, Record] = {}
    for i, document in enumerate(records):
        record = _read(document, f"the {name} split's record {i}")
        read.setdefault(record.record_hash, record)
    return read


def _refuse_split_id(name: str, split_id: str, read: dict[str, Record]) -> None:
    """``split-id`` when the id recomputed from a split's records is not the one given."""
    found = hash_value(code_point_sorted(read))
    if found != split_id:
        raise EvaluationRefusal(
            f"the {name} split's id is {split_id}, but its records' is {found}",
            reason=_R.split_id,
            split=(split_id, found),
        )


def _no_gold_plan(turn: str, point: int | None, why: str) -> EvaluationRefusal:
    where = "at every point" if point is None else f"at point {point}"
    return EvaluationRefusal(
        f"turn {turn} has no gold plan {where}: {why}",
        reason=_R.no_gold_plan,
        turn=turn,
        point=point,
    )


def _counts(split: dict[str, Record], p: int, count: dict[tuple[str, int], str]) -> dict[str, int]:
    """A split's counts at point ``p``, ``C``."""
    out = dict.fromkeys(_COUNTS, 0)
    for turn in split:
        out[count[turn, p]] += 1
    return out


def _scope(logs: list[_Log]) -> dict[str, list[str]]:
    """The distinct question identities, providers, models, revisions and elicitations of the
    call items of ``logs``, each in code-point order, null left out: a model the reported one,
    else the requested; a revision a call item's ``snapshot`` before format 5."""
    found: dict[str, set[object]] = {
        k: set() for k in ("questions", "providers", "models", "revisions", "elicitations")
    }
    for log in logs:
        for item in log:
            if item["item"] != "call":
                continue
            found["questions"].update(q["identity"] for q in item["questions"].values())
            found["providers"].add(item.get("provider"))
            reported = item.get("model_reported")
            found["models"].add(item.get("model_requested") if reported is None else reported)
            found["revisions"].add(item["revision"] if "revision" in item else item.get("snapshot"))
            found["elicitations"].add(item["elicitation"])
    return {
        k: code_point_sorted({cast(str, v) for v in vs if v is not None}) for k, vs in found.items()
    }
