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
``unfinished``, a record with no ``verdict`` item; ``not-found``, then ``invalid-label``, each
label in ``labels``' order; ``conflicting-labels``, each key in record and log order;
``empty-copy`` when ``copy_from`` holds none of the keys, whatever its entries' formats; for each
key given no label, in the same order, ``unknown-entry`` for its old entry of a format other
than 1 and 2, then ``invalid-label`` for an old value that is not a label of the question,
checked as a given one; ``held-key`` naming every key to be written that the store already
holds. A key is written with its given label, or with none given its old one; the job writes
nothing for a key given ``None`` or left with no label. A candidate-valued set whose entry
holds a marker leaves its fixed part's names as the labels.

The writes are not atomic across keys: an ``OSError`` or a racing run mid-write leaves the keys
already written, and the store being write-once, a rerun refuses ``held-key`` naming them. A
held key at a write is another run's, refused with the held key first, then those this run
wrote, in write order; any other refusal passes through. The entries written are returned, in
write order, plain data.

``known`` and ``conforming`` read a stored gold answer as the sweep job and the correlated-group
job read one: format 1 or 2, and of its question's form, else ``unknown-entry``.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType, _typed_member
from ommat._canonical import encode
from ommat._refs import declared_ref, parse_ref
from ommat._values import guarded_repr, type_name
from ommat.evaluation._diff import _read
from ommat.evaluation._edit import _labels, _name
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore

_R = RefusalReason
_Log = list[dict[str, Any]]
_Key = dict[str, str]
_Id = tuple[str, str, str]  # a key's turn, question and candidate

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
    for turn, log in logs.items():
        if not any(item["item"] == "verdict" for item in log):
            raise EvaluationRefusal(
                f"turn {turn} has no verdict item", reason=_R.unfinished, turn=turn
            )
    asked = {turn: _asked(log) for turn, log in logs.items()}
    formed: dict[tuple[str, str], object] = {}  # each given label as written, None withdrawing
    for turn, ref, label in _given(labels, asked):
        formed[turn, ref] = None if label is None else _checked(label, turn, ref, asked, logs)
    keys: list[tuple[_Id, _Key, list[str], list[object]]] = []  # its refs and their labels
    for turn in asked:
        for ident, key, refs in _keys(gold_set, turn, asked[turn]):
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
            keys.append((ident, key, refs, forms))
    old = _held(copy_from, [(ident, key) for ident, key, _, _ in keys], results)
    written: list[dict[str, Any]] = []
    for ident, key, refs, forms in keys:
        value = forms[0] if forms else _copied(old[ident], refs[0], key, asked, logs)
        if value is not None:
            written.append({"kind": KIND, "format": FORMAT, "key": key, "value": value})
    held: list[_Key] = [e["key"] for e in written if results.lookup(KIND, e["key"]) is not None]
    if held:
        raise EvaluationRefusal(
            f"the store holds {len(held)} of the {KIND} keys to write", reason=_R.held_key, key=held
        )
    done: list[_Key] = []  # not atomic across keys: a failure here leaves these written
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


def _checked(
    label: object, turn: str, ref: str, asked: dict[str, dict[str, Any]], logs: dict[str, _Log]
) -> object:
    """``label`` as written for the question at ``ref`` in ``turn``; ``invalid-label``
    otherwise."""
    question = asked[turn][ref]
    form = _form(label, question, logs[turn])
    if form is None:
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
    return form


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
    if names is None:  # a candidate set holding a marker: its fixed part's names stand
        names = tuple(_name(n) for n in cast("list[object]", question["labels"]["fixed"]))
    return label if type(label) is str and label in names else None


def _keys(
    gold_set: str, turn: str, asked: dict[str, dict[str, Any]]
) -> list[tuple[_Id, _Key, list[str]]]:
    """Each gold key of a turn's questions, in log order of its first ref, with its refs."""
    keys: dict[_Id, tuple[_Id, _Key, list[str]]] = {}
    for ref, question in asked.items():
        ident = (turn, question["identity"], parse_ref(ref).candidate or "")
        _, identity, candidate = ident
        key = {"gold_set": gold_set, "question": identity, "turn": turn, "candidate": candidate}
        keys.setdefault(ident, (ident, key, []))[2].append(ref)
    return list(keys.values())


def _held(
    copy_from: str | None, keys: list[tuple[_Id, _Key]], results: EvaluationStore
) -> dict[_Id, dict[str, Any] | None]:
    """Each key's entry under ``copy_from``, or ``None``; ``empty-copy`` when none is held,
    whatever its format."""
    if copy_from is None:
        return {ident: None for ident, _ in keys}
    old = {ident: results.lookup(KIND, {**key, "gold_set": copy_from}) for ident, key in keys}
    if all(entry is None for entry in old.values()):
        raise EvaluationRefusal(
            f"gold set {copy_from!r} holds no key the job looks up", reason=_R.empty_copy
        )
    return old


def _copied(
    entry: dict[str, Any] | None,
    ref: str,
    key: _Key,
    asked: dict[str, dict[str, Any]],
    logs: dict[str, _Log],
) -> object:
    """The old entry's value, checked as a given label is; ``unknown-entry`` for a format the
    job does not read; ``None`` with no entry."""
    if entry is None:
        return None
    known(entry, entry["key"])
    return _checked(entry["value"], key["turn"], ref, asked, logs)


def known(entry: dict[str, Any] | None, key: dict[str, str]) -> dict[str, Any] | None:
    """``entry``, a gold answer looked up under ``key`` or ``None``; ``unknown-entry`` for an
    entry of a kind or format the job does not read."""
    # The store's lookup by kind, and its envelope check, make the entry's kind KIND.
    if entry is not None and entry["format"] not in _READ:
        raise EvaluationRefusal(
            f"the {KIND} entry under {key!r} is format {entry['format']}, not 1 or 2",
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
