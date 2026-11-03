# pyright: reportPrivateUsage=false
"""The answers table: one row per question a call answered in each record, per candidate for
a per-candidate question, a pure read of records under one design.

``answers(records, loaded, *, gold_set=None, results=None)`` reads each record in list order and
each ``q:`` entry a call wrote, in log order, and gives a row of the turn, the entry's ref, its
declared ref and candidate, the question identity, kind and labels as asked, the answer or the
marker, the top label, the question's targets as declared and the groups that name it with their
reasons. With ``gold_set`` and ``results``, an ``EvaluationStore``, each row also gives the gold
answer, found as ``write_gold_answers`` keys it, and whether the answer erred against it: a
probability answer errs when its top label, a tie for the top none, is not the gold label; an
extraction answer when ``{"value": v}``, ``v`` its value, differs from the gold answer by
canonical bytes. A row with no gold answer, or holding a marker, has ``erred`` null; no row is
dropped. Counts, rates and comparisons over the rows are the caller's.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ommat._answer import AnswerKind
from ommat._canonical import decode, encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._refs import declared_ref, parse_ref
from ommat._values import type_name
from ommat.evaluation._diff import _read
from ommat.evaluation._edit import _labels, _name
from ommat.evaluation._gold import KIND, conforming, known
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore

_R = RefusalReason
_Log = list[dict[str, Any]]


def answers(
    records: Iterable[object],
    loaded: LoadedDesign,
    *,
    gold_set: str | None = None,
    results: EvaluationStore | None = None,
) -> list[dict[str, Any]]:
    """The answers table of ``records`` under ``loaded``, with each row's gold answer under
    ``gold_set`` when given."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    if not isinstance(records, Iterable) or isinstance(records, str | bytes | Mapping):
        raise TypeError(f"records are a {type_name(cast(object, records))}, not records")
    if (gold_set is None) != (results is None):
        raise TypeError("gold_set and results are given together or not at all")
    if gold_set is not None and type(gold_set) is not str:
        raise TypeError(f"gold_set is a {type_name(gold_set)}, not a str")
    if results is not None and type(results) is not EvaluationStore:
        raise TypeError(f"results is a {type_name(results)}, not an EvaluationStore")
    sources = cast("dict[str, Any]", decode(loaded.document))["sources"]
    groups = {name: list(loaded._group_reasons[name]) for name in loaded._groups}
    rows: list[dict[str, Any]] = []
    for i, document in enumerate(records):
        record = _read(document, f"records[{i}]")
        if record.design_hash != loaded.design_hash:
            raise EvaluationRefusal(
                f"records[{i}]'s design hash is {record.design_hash}, not the loaded design's "
                f"{loaded.design_hash}",
                reason=_R.wrong_design,
                turn=record.record_hash,
            )
        log = record.log  # read once: every read decodes afresh
        for entry in log:
            if entry["item"] != "entry" or entry["call"] is None:
                continue
            ref = cast(str, entry["ref"])
            if not ref.startswith("q:"):
                continue
            question = log[cast(int, entry["call"])]["questions"][declared_ref(ref)]
            row = _row(record.record_hash, entry, question, log, loaded, sources, groups)
            if gold_set is not None:
                _gold(row, question, log, gold_set, cast(EvaluationStore, results))
            rows.append(row)
    return rows


def _row(
    turn: str,
    entry: dict[str, Any],
    question: dict[str, Any],
    log: _Log,
    loaded: LoadedDesign,
    sources: Mapping[str, Any],
    groups: Mapping[str, list[str]],
) -> dict[str, Any]:
    ref = cast(str, entry["ref"])
    declared = declared_ref(ref)
    answer = entry.get("value")
    labels = None
    if question["kind"] != AnswerKind.extraction:
        names = _labels(question, log)
        if names is None:  # a candidate set holding a marker: its fixed part's names stand
            names = tuple(_name(n) for n in cast("list[object]", question["labels"]["fixed"]))
        labels = list(names)
    provider = loaded._providers.get(declared)
    asked = {} if provider is None else sources[provider]["provides"].get(declared, {})
    return {
        "turn": turn,
        "ref": ref,
        "question": declared,
        "candidate": parse_ref(ref).candidate or "",
        "identity": question["identity"],
        "kind": question["kind"],
        "labels": labels,
        "answer": answer,
        "marker": entry.get("marker"),
        "top": _top(answer),
        "target": asked.get("target"),
        "latent_targets": [dict(link) for link in asked.get("latent_targets", [])],
        "groups": [
            {"group": name, "reason": list(reasons)}
            for name, reasons in groups.items()
            if declared in loaded._groups[name]
        ],
    }


def _top(answer: dict[str, Any] | None) -> str | None:
    """A probability answer's most probable label, ``None`` on a tie for it or for any other
    answer."""
    if answer is None or answer["kind"] == AnswerKind.extraction:
        return None
    probabilities = cast("dict[str, float]", answer["probabilities"])
    best = max(probabilities.values())
    tops = [label for label, p in probabilities.items() if p == best]
    return tops[0] if len(tops) == 1 else None


def _gold(
    row: dict[str, Any],
    question: dict[str, Any],
    log: _Log,
    gold_set: str,
    results: EvaluationStore,
) -> None:
    """The row's gold answer and whether its answer erred; both null with no gold answer, and
    ``erred`` null for a marker."""
    key = {
        "gold_set": gold_set,
        "question": row["identity"],
        "turn": row["turn"],
        "candidate": row["candidate"],
    }
    entry = known(results.lookup(KIND, key), key)
    if entry is None:
        row.update(gold=None, erred=None)
        return
    gold = conforming(entry, key, question, log)
    answer = row["answer"]
    if answer is None:
        erred = None
    elif answer["kind"] == AnswerKind.extraction:
        erred = encode({"value": answer["value"]}) != encode(gold)
    else:
        erred = row["top"] != gold
    row.update(gold=gold, erred=erred)
