# pyright: reportPrivateUsage=false
"""The correlated-group confirmation job: for each pair of a declared group's ``q:`` refs, over
the units of a split in which both hold an answer, the units counted, the units in which each
erred against its gold answer and those in which both did, written as a ``group-counts`` entry
(§group-confirmation).

**Units and answers.** A unit is a turn and the candidate part of a ``q:`` entry's ref, ``""``
for an ungrounded one, the gold key's ``candidate``; a group's refs are declared refs, so a
grounded group's entries are its refs' per-candidate entries. A ref holds an answer at a unit
when a call wrote its entry there with a value; a marker, or no entry, holds none. A pair's
``turns`` counts units. Each gold answer is found as the gold-answers job derives its key,
``_gold``'s, and read by ``_gold``'s ``known`` and ``conforming``:
format 1 or 2, a format-1 value a label string, and of its question's form, as the gold-answers
job checks a label.

**Erring.** A probability answer errs when its most probable label is not the gold label, a
tie for most probable an error; an extraction answer when ``{"value": v}``, ``v`` its value,
differs from the gold answer by canonical bytes, null a value like any other.

**The job.** A ``split`` not an ``(id, records)`` pair, checked by ``_splits``, is a ``TypeError``
before anything else. Refused in order: ``unknown-group`` and ``too-few-refs``, naming the group,
before any record is read; ``invalid-record`` for each record in turn; ``split-id``; ``wrong-
design``, each record in the order given; ``missing-gold``, naming the turn and the entry's ref,
each answer in turn and log order; ``unknown-entry``, naming the key, after every ``missing-gold``,
for a gold entry of a kind or format the job does not read or a value not of its question's form, in
the same order; ``held-key`` when the store holds the key, before any count; and the store's ``held-
key`` at the write. A record is counted once by its record hash. The entry written is returned,
plain data reaching no ``Record``."""

import itertools
from typing import Any, NamedTuple, cast

from ommat._answer import AnswerKind
from ommat._canonical import encode
from ommat._design import LoadedDesign
from ommat._refs import declared_ref, parse_ref
from ommat._values import guarded_repr
from ommat.evaluation._gold import KIND as GOLD
from ommat.evaluation._gold import _asked, _keys, conforming, known
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._splits import given_split, read_split, refuse_split_id
from ommat.evaluation._store import EvaluationStore

_R = RefusalReason
_Unit = tuple[str, str]  # a turn and a candidate


class _Answered(NamedTuple):
    """A group ref's answer at a unit, with what its gold answer is found and checked by."""

    unit: _Unit
    declared: str  # the group's ref
    ref: str  # the entry's ref
    answer: Any  # the entry's value, an answer's document
    key: dict[str, str]  # its gold key
    gold: dict[str, Any] | None  # the entry held under it
    question: dict[str, Any]  # its call item's question
    log: list[dict[str, Any]]


KIND = "group-counts"


def confirm_group(
    loaded: LoadedDesign,
    group: str,
    gold_set: str,
    split: object,
    *,
    results: EvaluationStore,
) -> dict[str, Any]:
    """Count, for each pair of ``group``'s ``q:`` refs in ``loaded``, the units of ``split``,
    ``(id, records)``, in which both hold an answer, those in which each erred against its gold
    answer under ``gold_set`` and those in which both did; the ``group-counts`` entry written."""
    split_id, records = given_split("given", split)
    refs = loaded._groups.get(group) if type(group) is str else None
    if refs is None:
        raise EvaluationRefusal(
            f"the design declares no group {guarded_repr(group)}",
            reason=_R.unknown_group,
            group=group if type(group) is str else None,
        )
    questions = [r for r in refs if r.startswith("q:")]  # in code-point order, as declared
    if len(questions) < 2:
        raise EvaluationRefusal(
            f"group {group!r} has {len(questions)} q: refs, fewer than two",
            reason=_R.too_few_refs,
            group=group,
        )
    reads = read_split("given", records)
    refuse_split_id("given", split_id, reads)
    for turn, read in reads.items():
        if read.design_hash != loaded.design_hash:
            raise EvaluationRefusal(
                f"turn {turn}'s design hash is {read.design_hash}, not the loaded design's "
                f"{loaded.design_hash}",
                reason=_R.wrong_design,
                turn=turn,
            )
    answered: list[_Answered] = []
    for turn, read in reads.items():
        log = read.log
        asked = _asked(log)
        keyed = {ref: k for _, k, of in _keys(gold_set, turn, asked) for ref in of}
        values = {e["ref"]: e for e in log if e["item"] == "entry" and "value" in e}
        for ref in asked:
            declared = declared_ref(ref)
            if ref in values and declared in questions:
                answered.append(
                    _Answered(
                        unit=(turn, parse_ref(ref).candidate or ""),
                        declared=declared,
                        ref=ref,
                        answer=values[ref]["value"],
                        key=keyed[ref],
                        gold=results.lookup(GOLD, keyed[ref]),
                        question=asked[ref],
                        log=log,
                    )
                )
    for one in answered:
        if one.gold is None:
            raise EvaluationRefusal(
                f"turn {one.unit[0]} answers {one.ref} with no gold answer under {gold_set!r}",
                reason=_R.missing_gold,
                turn=one.unit[0],
                ref=one.ref,
            )
    erred: dict[_Unit, dict[str, bool]] = {}
    for one in answered:
        held = cast("dict[str, Any]", known(one.gold, one.key))
        value = conforming(held, one.key, one.question, one.log)
        erred.setdefault(one.unit, {})[one.declared] = _erred(one.answer, value)
    key = {"design": loaded.design_hash, "group": group, "split": split_id, "gold_set": gold_set}
    if results.lookup(KIND, key) is not None:
        raise EvaluationRefusal(
            f"the store holds a {KIND} entry under {key!r}", reason=_R.held_key, key=[key]
        )
    pairs: list[dict[str, Any]] = []
    for a, b in itertools.combinations(questions, 2):
        both = [e for e in erred.values() if a in e and b in e]
        pairs.append(
            {
                "refs": [a, b],
                "turns": len(both),
                "erred": [sum(e[a] for e in both), sum(e[b] for e in both)],
                "both": sum(e[a] and e[b] for e in both),
            }
        )
    entry = {"kind": KIND, "format": 1, "key": key, "value": {"pairs": pairs}}
    results.write(entry)
    return entry


def _erred(answer: Any, gold: object) -> bool:
    """Whether ``answer``, an answer's document, errs against ``gold``, a gold answer's value."""
    if answer["kind"] == AnswerKind.extraction:
        return encode({"value": answer["value"]}) != encode(gold)
    probabilities = cast("dict[str, float]", answer["probabilities"])
    top = max(probabilities.values())
    most = [label for label, p in probabilities.items() if p == top]
    return most != [gold]
