# pyright: reportPrivateUsage=false
"""The question-reach job: for each question, whether its answer, changed alone, changes an
entry the caller names in a recorded turn (§question-reach).

It asks the dead-question lint's question from turns the design is known to reach. ``refs`` is
checked as the lint checks it, a non-empty list or tuple of declared whole ``d:`` refs each a
rule provides, then ``values`` as the lints check it, through ``points`` over every ``q:`` ref a
source provides, both before any record is read; a key naming any other read, an unfilled read
and each other slip is one ``ValueError``. Each record is then read and refused as the
deciding inputs refuse one, ``invalid-record``, ``wrong-design`` for a record before format 8,
``unfinished``, ``nothing-to-replay``, ``wrong-design`` for another design hash and
``baseline-differs``, each but the first naming its turn, every record in list order before
any trial.

For each record in list order, and each ``q:`` entry it holds whose call is neither null nor
recorded ``Failed``, in log order, the job runs one trial, ``intervene``'s entry set through
``_edit``'s one procedure, at each of the entry's points other than its recorded value or marker
by canonical bytes. An entry's points are its declared question's read points, from ``points``,
a grounded entry's with its own marker last; a candidate-valued question's are each fixed label
and then the first candidate at 1, every other label the call asked with at 0; a returned call
always rendered a candidate, since an empty set leaves the question out with its call null. A
trial whose replay reports ``stale-question`` is undetermined, whatever else it finds; any other
reaches when, at some named ref, the entry it wrote differs from the record's by
``entry_bytes``, a marker's cause and refs included, or one of the two holds no entry there.

A question is ``live`` when some trial of an entry of it reaches; else ``not-determined`` when
some trial was undetermined; else ``no-change-found`` when some trial ran; else
``never-asked``. A ``live`` question's witness is the first reaching trial by record, then log,
then point order: ``{"record", "ref", "value"}``, the entry's ref and its value or marker as the
copy's record wrote it. The report is plain data.
"""

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from ommat._answer import AnswerKind
from ommat._canonical import encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign, Question
from ommat._fixture import _binding_form
from ommat._ledger import _ITEM_DEPTH, Record, entry_bytes
from ommat._markers import Cause, Unavailable
from ommat._refs import code_point_sorted, declared_ref
from ommat._values import described, is_scalar_string, quote, string_data, type_name
from ommat.evaluation._diff import OUTSIDE, _read, _refuse_unfit, _run_sync, baseline
from ommat.evaluation._edit import _applied, _labels, _q_value, _SetEntry
from ommat.helpers import points
from ommat.helpers._dead import named_refs

_Log = list[dict[str, Any]]


def _refuse_records(what: str, records: object) -> None:
    """``TypeError`` for ``records`` that are not an iterable other than a string, bytes or a
    mapping."""
    if not isinstance(records, Iterable) or isinstance(records, str | bytes | Mapping):
        raise TypeError(f"{what} are a {type_name(cast(object, records))}, not records")


async def question_reach(
    records: Iterable[object],
    loaded: LoadedDesign,
    *,
    refs: Sequence[str],
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The report ``{"records": [hash], "trials": N, "questions": {ref: {"status",
    "witness"}}, "narrowed": [ref]}``, keyed by every ``q:`` ref a source provides; before any
    record is read, a ``TypeError`` for ``refs`` that is not a list or tuple, and a
    ``ValueError`` for ``refs`` empty or naming anything but a declared whole ``d:`` ref a rule
    provides, then for a slip in ``values`` or a question nothing fills. ``values`` maps a
    ``q:`` ref a source provides to a list of its values, as the dead-question lint takes it;
    for a label question with no grid entry, the labels to try."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    _refuse_records("records", records)
    named = named_refs(loaded, refs)
    provided = code_point_sorted(
        ref for s in loaded._design.sources.values() for ref in s.provides if ref.startswith("q:")
    )
    fills, narrowed = _filled(loaded, provided, values)
    read: list[tuple[Record, _Log]] = []
    for i, document in enumerate(records):
        record = _read(document, f"record {i}")
        log = record.log  # read once: every read decodes afresh
        _refuse_unfit((f"record {i}",), (record,), (log,), loaded)
        await baseline(log, loaded, turn=record.record_hash)
        read.append((record, log))
    trials = 0
    ran: set[str] = set()
    undetermined: set[str] = set()
    witnesses: dict[str, dict[str, Any]] = {}
    for record, log in read:
        for entry in log:
            if not _asked(entry, log):
                continue
            ref = cast(str, entry["ref"])
            question = declared_ref(ref)
            for value in _points(ref, entry, log, loaded, fills[question]):
                done = await _applied(record, log, loaded, [_SetEntry(ref, value)], None)
                found = done.findings
                trials += 1
                ran.add(question)
                if any(f["kind"] in OUTSIDE for f in found):
                    undetermined.add(question)
                elif question not in witnesses and _reaches(named, log, done.record.log):
                    written = done.items[0]
                    shown = written["value"] if "value" in written else written["marker"]
                    witnesses[question] = {
                        "record": record.record_hash,
                        "ref": ref,
                        "value": copy.deepcopy(shown),
                    }
    questions: dict[str, Any] = {}
    for question in provided:
        if question in witnesses:
            status = "live"
        elif question in undetermined:
            status = "not-determined"
        elif question in ran:
            status = "no-change-found"
        else:
            status = "never-asked"
        questions[question] = {"status": status, "witness": witnesses.get(question)}
    return {
        "records": [record.record_hash for record, _ in read],
        "trials": trials,
        "questions": questions,
        "narrowed": narrowed,
    }


def question_reach_sync(
    records: Iterable[object],
    loaded: LoadedDesign,
    *,
    refs: Sequence[str],
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``question_reach`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync(
        "question_reach", lambda: question_reach(records, loaded, refs=refs, values=values)
    )


def _reaches(named: tuple[str, ...], log: _Log, rerun: _Log) -> bool:
    """Whether the entry at some named ref differs between ``log`` and ``rerun`` by
    ``entry_bytes``."""
    return any(entry_bytes(log, ref) != entry_bytes(rerun, ref) for ref in named)


def _asked(entry: dict[str, Any], log: _Log) -> bool:
    """Whether ``entry`` is a ``q:`` entry a returned call wrote."""
    if entry["item"] != "entry" or not cast(str, entry["ref"]).startswith("q:"):
        return False
    call = entry["call"]
    return call is not None and log[cast(int, call)]["outcome"] == "returned"


def _filled(
    loaded: LoadedDesign, provided: list[str], values: Mapping[str, Any] | None
) -> tuple[dict[str, list[Any]], list[str]]:
    """Each question's read points, in record form, and the questions ``values`` narrowed, in
    code-point order; one ``ValueError`` names, as the lints name them, each key that is not a
    string, each slip ``points`` finds, each key naming no question a source provides, then
    every question nothing fills."""
    if values is None:
        values = {}
    if not issubclass(type(values), Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"values is a {type_name(values)}, not a mapping or None")
    given: dict[str, list[object]] = {}
    problems: list[str] = []
    for key, listed in cast("Mapping[object, object]", values).items():
        if not is_scalar_string(key):
            problems.append(f"the values key {described(key)} is not a string")
            continue
        name = string_data(cast(str, key))
        if type(listed) is not list and type(listed) is not tuple:
            raise TypeError(f"values for {quote(name)} is not a list")
        given[name] = list(cast("list[object]", listed))
    fills: dict[str, list[Any]] = {}
    unfilled: list[str] = []
    for question in provided:
        try:
            fills[question] = points(loaded, question, values=given.get(question))
        except ValueError as exc:
            if question not in given and str(exc) == _UNFILLED.format(quote(question)):
                unfilled.append(question)
            else:
                problems.append(str(exc))
    problems += [
        f"values names {quote(key)}, a read the lint does not fill"
        for key in given
        if key not in provided
    ]
    if unfilled:
        problems.append(_UNFILLED.format(", ".join(quote(q) for q in unfilled)))
    if problems:
        raise ValueError("; ".join(problems))
    narrowed = [q for q in given if q in provided and _kind(loaded, q) != AnswerKind.extraction]
    return fills, code_point_sorted(narrowed)


_UNFILLED = "unfilled: {}; give each a list in values"
"""The lints' line naming the reads nothing fills."""


def _kind(loaded: LoadedDesign, ref: str) -> object:
    """The kind of the question a source provides at ``ref``."""
    return cast(Question, loaded._design.sources[loaded._providers[ref]].provides[ref]).kind


def _points(
    ref: str, entry: dict[str, Any], log: _Log, loaded: LoadedDesign, points: list[Any]
) -> list[object]:
    """The entry's points in their order, each as an entry set takes it, less the one whose record
    form is the entry's recorded value or marker by canonical bytes: an answer in the binding's
    form, a marker, a grounded entry's naming that entry, and a candidate-valued question's
    label at 1 over the labels its call asked."""
    question = declared_ref(ref)
    call = log[cast(int, entry["call"])]
    rendered: tuple[str, ...] | None = None
    if question in loaded._label_sets:
        # A returned call asked it over a set holding a value of at least one candidate: a
        # marker, an empty set or labels the engine refuses leave the question out with its call
        # null (§candidate-sets).
        rendered = _labels(call["questions"][question], log)
        assert rendered is not None
    recorded = encode(entry["value"] if "value" in entry else entry["marker"], _ITEM_DEPTH)
    out: list[object] = []
    for point in points:
        record = cast("dict[str, Any]", point)
        if "cause" in record:  # the marker; a grounded entry's names that entry
            cause = cast("dict[str, Any]", record["cause"])
            refs = [ref] if ref != question else list(cast("list[str]", cause["refs"]))
            document = {**record, "cause": {**cause, "refs": refs}}
            if encode(document, _ITEM_DEPTH) != recorded:
                out.append(Unavailable(Cause(cast(str, cause["code"]), refs)))
            continue
        if "on_top" in record:
            assert rendered is not None
            label = cast("str | None", record["on_top"])
            top = rendered[loaded._label_at[question]] if label is None else label
            live: object = {name: 1.0 if name == top else 0.0 for name in rendered}
        else:
            live = _binding_form(record)
        written = _q_value(ref, live, call, log, loaded)
        if written is not None and encode(written, _ITEM_DEPTH) == recorded:
            continue  # the recorded value: replay would find nothing
        out.append(copy.deepcopy(live))
    return out
