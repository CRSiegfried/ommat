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

from typing import Any, cast

from ommat._answer import AnswerKind
from ommat._ledger import Record
from ommat._refs import declared_ref
from ommat.evaluation._diff import _state
from ommat.evaluation._edit import _labels, _q_value, _record_copy, _rewritten
from ommat.evaluation._gold import _READ, KIND, _asked, _keys
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore

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
