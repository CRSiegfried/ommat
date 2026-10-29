# pyright: reportPrivateUsage=false
"""The sweep job's gold copy: a record with every ``q:`` entry a call wrote set to its gold
answer, from which replay under each point's design gives the turn's gold plan (§gold-copy).

The gold answers are looked up as the gold-answers job derives their keys, ``_gold``'s: the
turn id the record's hash, the question identity from its call item and the candidate part of
its ref. An entry of a kind or a format other than ``gold-answer`` 1 and 2 is refused
``unknown-entry``, naming its key. A label becomes an answer whose ``probabilities`` give 1 to
the gold label and 0 to every other, over the question's labels as asked, a candidate-valued
set's from its set's entry; an extraction gold answer ``{"value": v}`` becomes ``value`` ``v``,
``p_null`` 0 and ``confidence`` 1, or for ``v`` ``None`` ``value`` ``None``, ``p_null`` 1 and
``confidence`` ``None``, ``guess`` and ``span`` ``None`` in both. Each answer is built and
checked as ``set_entry`` builds one, on one copy whose ``hash`` is set and which is read back
through ``read_record``, by ``_edit``'s machinery; a ``t:`` entry keeps its value or marker.

A turn has no gold copy, and so no gold plan at any point, when its record has no ``verdict``
item or no state entry, a ``q:`` entry a call wrote has no gold answer, a call that wrote one
is ``Failed``, or a gold answer is no answer to its question, as a format-1 label string given
for an extraction question is; the reason is returned in place of the copy.
"""

import itertools
import math
from fractions import Fraction
from typing import Any, cast

from ommat._answer import AnswerKind
from ommat._canonical import encode
from ommat._ledger import _ITEM_DEPTH, Record
from ommat._markers import PlanKind, VerdictKind
from ommat._refs import declared_ref, parse_ref
from ommat._resolve import board_entries, board_value
from ommat._views import escalation
from ommat.evaluation._diff import OUTSIDE, _state
from ommat.evaluation._edit import _labels, _q_value, _record_copy, _rewritten
from ommat.evaluation._gold import _READ, KIND, _asked, _keys
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore
from ommat.evaluation._switching import _unreturned

_R = RefusalReason
_Log = list[dict[str, Any]]


def gold_answers(
    turn: str, log: _Log, gold_set: str, results: EvaluationStore
) -> dict[str, object | None]:
    """Each ``q:`` ref a call wrote in ``log``, in log order, to its gold answer's value under
    ``gold_set``, or ``None`` when none is held; ``unknown-entry`` for an entry of a kind or
    format the job does not read."""
    asked = _asked(log)
    golds: dict[str, object | None] = {}
    for _, key, refs in _keys(gold_set, turn, asked):
        entry = results.lookup(KIND, key)
        if entry is not None and (entry["kind"] != KIND or entry["format"] not in _READ):
            raise EvaluationRefusal(
                f"the entry under {key!r} is {entry['kind']} format {entry['format']}, not "
                f"{KIND} format 1 or 2",
                reason=_R.unknown_entry,
                key=[key],
            )
        for ref in refs:
            golds[ref] = None if entry is None else entry["value"]
    return {ref: golds[ref] for ref in asked}


def gold_copy(read: Record, log: _Log, golds: dict[str, object | None]) -> Record | str:
    """The gold copy of ``read``, whose log is ``log``, with each ``q:`` ref of ``golds`` set
    to its gold answer; or why the turn has none."""
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
        call = log[cast(int, log[at[ref]]["call"])]
        answer = _answer(ref, gold, call, log)
        written = None if answer is None else _q_value(ref, answer, call, log, None)
        if written is None:
            return f"the gold answer for {ref} is no answer to its question"
        edited[at[ref]] = _rewritten(log[at[ref]], value=written)
    return _record_copy(edited, read.unhashed)


def _answer(ref: str, gold: object, call: dict[str, Any], log: _Log) -> object:
    """The answer, in the binding's form, that gives the gold answer all the probability;
    ``None`` when the gold answer is not of the question's form."""
    question = call["questions"][declared_ref(ref)]
    if question["kind"] == AnswerKind.extraction:
        if type(gold) is not dict or set(cast("dict[str, object]", gold)) != {"value"}:
            return None
        v = cast("dict[str, object]", gold)["value"]
        held = (1, None) if v is None else (0, 1)
        return {"value": v, "guess": None, "p_null": held[0], "confidence": held[1], "span": None}
    labels = _labels(question, log) or ()
    if gold not in labels:
        return None
    return {name: 1.0 if name == gold else 0.0 for name in labels}


# ---------------------------------------------------------------------------------------------
# Scoring (§sweep-scoring)

COUNTS = ("matched", "escaped", "escalated", "clarify", "inform", "broken")
"""The names a turn is counted under at a point, in the order the entry holds them."""
FIXED_SEQUENCE, BONFERRONI = "fixed-sequence", "bonferroni"


def undetermined(log: _Log, rerun: Record, found: list[dict[str, Any]]) -> str | None:
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


def action(log: _Log) -> bytes | None:
    """The canonical bytes of the action of ``log``'s plan when it is a passed ``execute``
    plan: its ``kind`` and ``steps``, each board-ref argument replaced by what the ref reads in
    ``log``, ``{"value": v}`` or ``{"marker": m}``; else ``None``."""
    plan, verdict = _first(log, "plan"), _first(log, "verdict")
    passed = verdict is not None and verdict["kind"] == VerdictKind.passed
    if plan is None or plan["kind"] != PlanKind.execute or not passed:
        return None
    entries: dict[str, dict[str, Any]] = {}
    for item in log:
        if item["item"] == "entry":
            entries.setdefault(item["ref"], item)
    values = board_entries(log)
    steps: list[dict[str, Any]] = []
    for step in cast("list[dict[str, Any]]", plan["steps"]):
        arguments: dict[str, Any] = {}
        for name, origin in cast("dict[str, dict[str, Any]]", step["arguments"]).items():
            ref = origin.get("ref")
            if ref is None:
                arguments[name] = origin  # a literal or a binding, as written
                continue
            fields = parse_ref(ref).fields
            base = ref[: len(ref) - len(".".join(fields)) - 1] if fields else ref
            held = entries[base]
            arguments[name] = (
                {"marker": held["marker"]}
                if "marker" in held
                else {"value": board_value(values, ref)}
            )
        steps.append({**step, "arguments": arguments})
    return encode({"kind": plan["kind"], "steps": steps}, _ITEM_DEPTH)


def counted(rerun: Record, gold: bytes | None) -> str:
    """The name a determined replay of a turn's record is counted under at a point, against
    ``gold``, the action of its gold plan, ``None`` when that is no passed ``execute`` plan."""
    if escalation(rerun) is not None:
        return "escalated"
    log = rerun.log
    plan = cast("dict[str, Any]", _first(log, "plan"))  # a verdict passed: escalation saw it
    if plan["kind"] in (PlanKind.clarify, PlanKind.inform):
        return str(plan["kind"])
    if plan["kind"] == PlanKind.broken:
        return "broken"
    return "matched" if gold is not None and action(log) == gold else "escaped"


def tail(n: int, alpha: Fraction, k: int) -> Fraction:
    """P(Bin(``n``, ``alpha``) <= ``k``), exactly."""
    p, q = alpha.numerator, alpha.denominator
    total = sum(math.comb(n, i) * p**i * (q - p) ** (n - i) for i in range(min(k, n) + 1))
    return Fraction(total, q**n)


def certifies(n: int, escaped: int, alpha: float, level: Fraction) -> bool:
    """Whether a point with ``escaped`` of ``n`` turns certifies at ``level``: the exact
    binomial test P(Bin(n, alpha) <= escaped) <= level."""
    return tail(n, Fraction(alpha), escaped) <= level


def which_test(dev_escaped: list[int]) -> str:
    """Fixed-sequence testing when ``escaped`` on the dev split is non-decreasing along the
    points, else Bonferroni."""
    ordered = all(a <= b for a, b in itertools.pairwise(dev_escaped))
    return FIXED_SEQUENCE if ordered else BONFERRONI


def choose(
    test: str, n: int, escaped: list[int], alpha: float, delta: float
) -> tuple[int | None, int]:
    """The chosen point's index, or ``None``, and how many points from the first were tested.
    Fixed-sequence testing tests each point at ``delta`` from the first, stops at the first
    that does not certify and chooses the last that did; Bonferroni tests every point at
    ``delta`` / m and chooses the last that certifies."""
    chosen: int | None = None
    if test == FIXED_SEQUENCE:
        for i, k in enumerate(escaped):
            if not certifies(n, k, alpha, Fraction(delta)):
                return chosen, i + 1
            chosen = i
        return chosen, len(escaped)
    level = Fraction(delta) / len(escaped)
    for i, k in enumerate(escaped):
        if certifies(n, k, alpha, level):
            chosen = i
    return chosen, len(escaped)


def _first(log: _Log, kind: str) -> dict[str, Any] | None:
    return next((i for i in log if i["item"] == kind), None)
