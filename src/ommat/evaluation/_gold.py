# pyright: reportPrivateUsage=false
"""The gold-answers job: a labeller's labels for the questions finished turns asked, written
to the evaluation store under a named gold set, with an old set's labels copied for the keys
given none (§evaluation-store).

A question is a ``q:`` entry a call wrote, a marker included: its call item's ``questions``
gives the identity, the kind and the declared labels, whose names alone a label is checked
against, a candidate-valued set's from its set's entry in the same record, as ``_edit`` reads
them; its ref gives the candidate part. No design is read. ``labels`` is
``{turn id: {ref: label}}``, the turn id the record hash of the record passed, so the original
turn's; a label is a name, a string, or for an extraction question ``{"value": v}``, ``v``
conforming to the question's value type and list flag or ``None``; a label of ``None``
withdraws. The refs of one gold key are the turn's refs asking one question for one candidate.

Refused in order, each before any write: ``invalid-record``, each record read in turn;
``not-found``, then ``invalid-label``, each label in ``labels``' order; ``conflicting-labels``,
each key in record and log order; ``unknown-entry`` for an old entry of a format other than 1
and 2, then ``empty-copy``, every key looked up under ``copy_from``; ``held-key`` naming every
key to be written that the store already holds. A key is written with its given label, or with
none given its old one, its value as the old entry holds it; the job writes nothing for a
key given ``None`` or left with no label. A held key at a write is another run's, refused with
the held key first, then those this run wrote, in write order. The entries written are
returned, in write order, plain data.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType, _typed_member
from ommat._canonical import encode
from ommat._refs import declared_ref, parse_ref
from ommat._values import guarded_repr, type_name
from ommat.evaluation._diff import _read
from ommat.evaluation._edit import _labels
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore

_R = RefusalReason
_Log = list[dict[str, Any]]
_Key = dict[str, str]

KIND = "gold-answer"
FORMAT = 2
_READ = (1, 2)  # a format-1 value is a label string


def write_gold_answers(
    gold_set: str,
    records: Iterable[object],
    labels: Mapping[str, Mapping[str, object]],
    *,
    results: EvaluationStore,
    copy_from: str | None = None,
) -> list[dict[str, Any]]:
    """Write ``labels``, ``{turn id: {ref: label}}``, for the turns of ``records`` under
    ``gold_set``, each key given no label taking its label under ``copy_from``; the entries
    written."""
    logs: dict[str, _Log] = {}
    for i, document in enumerate(records):
        read = _read(document, f"records[{i}]")
        logs.setdefault(read.record_hash, read.log)  # read once: every read decodes afresh
    asked = {turn: _asked(log) for turn, log in logs.items()}
    given = _given(labels, asked)
    formed: dict[tuple[str, str], object] = {}  # each given label as written, None withdrawing
    for turn, ref, label in given:
        question = asked[turn][ref]
        form = None if label is None else _form(label, question, logs[turn])
        if label is not None and form is None:
            fits = (
                '{"value": v}, v conforming to its value type or null'
                if question["kind"] == AnswerKind.extraction
                else "the name of one of its question's labels"
            )
            raise EvaluationRefusal(
                f"label {guarded_repr(label)} for {ref} in turn {turn} is not {fits}",
                reason=_R.invalid_label,
                turn=turn,
                ref=ref,
            )
        formed[turn, ref] = form
    keys: list[tuple[_Key, list[object]]] = []  # each key and its refs' given labels
    for turn in asked:
        for key, refs in _keys(gold_set, turn, asked[turn]):
            mine = [ref for ref in refs if (turn, ref) in formed]
            forms = [formed[turn, ref] for ref in mine]
            if len({encode(f) for f in forms}) > 1:
                shown = ", ".join(
                    f"{r} {guarded_repr(f)}" for r, f in zip(mine, forms, strict=True)
                )
                raise EvaluationRefusal(
                    f"turn {turn} gives one gold key differing labels: {shown}",
                    reason=_R.conflicting_labels,
                    turn=turn,
                    key=[key],
                )
            keys.append((key, forms))
    old = _copied(copy_from, [key for key, _ in keys], results)
    written: list[dict[str, Any]] = []
    for key, forms in keys:
        value = forms[0] if forms else old.get(_id(key))
        if value is not None:
            written.append({"kind": KIND, "format": FORMAT, "key": key, "value": value})
    held: list[_Key] = [e["key"] for e in written if results.lookup(KIND, e["key"]) is not None]
    if held:
        raise EvaluationRefusal(
            f"the store holds {len(held)} of the {KIND} keys to write", reason=_R.held_key, key=held
        )
    done: list[_Key] = []
    for entry in written:
        try:
            results.write(entry)
        except EvaluationRefusal as exc:
            if exc.reason is not _R.held_key:
                raise
            raise EvaluationRefusal(
                f"another run wrote {entry['key']!r} first, after {len(done)} of this run's",
                reason=_R.held_key,
                key=[entry["key"], *done],
            ) from exc
        done.append(entry["key"])
    return written


def _asked(log: _Log) -> dict[str, dict[str, Any]]:
    """Each ``q:`` entry a call wrote, by ref in log order, to its call item's question."""
    return {
        e["ref"]: log[e["call"]]["questions"][declared_ref(e["ref"])]
        for e in log
        if e["item"] == "entry" and e["ref"].startswith("q:") and e["call"] is not None
    }


def _given(
    labels: Mapping[str, Mapping[str, object]], asked: dict[str, dict[str, Any]]
) -> list[tuple[str, str, object]]:
    """Each label as ``(turn, ref, label)``, in ``labels``' order; ``not-found`` for a turn
    not passed or a ref no call wrote in it."""
    if not isinstance(labels, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"labels is a {type_name(labels)}, not a mapping of turn ids")
    given: list[tuple[str, str, object]] = []
    for turn, by_ref in labels.items():
        if not isinstance(by_ref, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"labels[{turn!r}] is a {type_name(by_ref)}, not a mapping of refs")
        for ref, label in by_ref.items():
            if ref not in asked.get(turn, {}):
                where = f"turn {turn}" if turn in asked else f"no turn passed, {turn!r}"
                raise EvaluationRefusal(
                    f"no call wrote {ref!r} in {where}",
                    reason=_R.not_found,
                    turn=turn if type(turn) is str else None,
                    ref=ref if type(ref) is str else None,
                )
            given.append((turn, ref, label))
    return given


def _form(label: object, question: dict[str, Any], log: _Log) -> object | None:
    """The label as written: a name of the question's labels, or an extraction label's
    ``{"value": v}`` with ``v`` as the answer holds it, a list a list; ``None`` when it is
    neither."""
    if question["kind"] == AnswerKind.extraction:
        if type(label) is not dict or set(cast("dict[object, object]", label)) != {"value"}:
            return None
        try:
            v = _typed_member(
                ValueType(question["value_type"]),
                question["is_list"],
                cast("dict[str, object]", label)["value"],
                "value",
            )
        except ValueError:
            return None
        return {"value": list(cast("tuple[object, ...]", v)) if type(v) is tuple else v}
    names = _labels(question, log)
    return label if type(label) is str and names is not None and label in names else None


def _keys(
    gold_set: str, turn: str, asked: dict[str, dict[str, Any]]
) -> list[tuple[_Key, list[str]]]:
    """Each gold key of a turn's questions, in log order of its first ref, with its refs."""
    keys: dict[tuple[str, str], tuple[_Key, list[str]]] = {}
    for ref, question in asked.items():
        candidate = parse_ref(ref).candidate or ""
        key = {
            "gold_set": gold_set,
            "question": question["identity"],
            "turn": turn,
            "candidate": candidate,
        }
        keys.setdefault((key["question"], candidate), (key, []))[1].append(ref)
    return list(keys.values())


def _id(key: _Key) -> tuple[str, str, str]:
    return key["question"], key["turn"], key["candidate"]


def _copied(
    copy_from: str | None, keys: list[_Key], results: EvaluationStore
) -> dict[tuple[str, str, str], object]:
    """Each key's value under ``copy_from``, by question, turn and candidate; ``unknown-entry``
    for an entry of a format the job does not know, then ``empty-copy`` when none is held."""
    if copy_from is None:
        return {}
    old: dict[tuple[str, str, str], object] = {}
    for key in keys:
        entry = results.lookup(KIND, {**key, "gold_set": copy_from})
        if entry is None:
            continue
        if entry["format"] not in _READ:
            raise EvaluationRefusal(
                f"the {KIND} entry under {entry['key']!r} is format {entry['format']}, not 1 or 2",
                reason=_R.unknown_entry,
                key=[entry["key"]],
            )
        old[_id(key)] = entry["value"]
    if not old:
        raise EvaluationRefusal(
            f"gold set {copy_from!r} holds no key the job looks up", reason=_R.empty_copy
        )
    return old
