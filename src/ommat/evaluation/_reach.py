# pyright: reportPrivateUsage=false
"""The question-reach job: for each question, whether its answer, changed alone, changes the
plan or the verdict of a recorded turn (§question-reach).

It asks the dead-question lint's question from turns the design is known to reach. ``values``
is checked as the lints check it, through ``_points``' fill over every ``q:`` ref a source
provides, before any record is read; a key naming any other read, an unfilled read and each
other slip is one ``ValueError``. Each record is then read and refused as the deciding inputs
refuse one, ``invalid-record``, ``unfinished``, ``nothing-to-replay``, ``wrong-design`` and
``baseline-differs``, each but the first naming its turn, every record in list order before
any trial.

For each record in list order, and each ``q:`` entry it holds whose call is neither null nor
recorded ``Failed``, in log order, the job runs one form 1 trial, ``set_entry``'s edit through
``_edit``'s one procedure, at each of the entry's points other than its recorded value or marker
by canonical bytes. An entry's points are its declared question's lint points, a grounded
entry's with its own marker last; a candidate-valued question's are each fixed label and then
the first candidate at 1, every other label the call asked with at 0; a returned call always
rendered a candidate, since an empty set leaves the question out with its call null. A trial
whose replay reports ``stale-question`` is
undetermined, whatever else it finds; any other reaches when its findings hold ``plan`` or
``verdict``, a plan compared without its ``context``.

A question is ``live`` when some trial of an entry of it reaches; else ``not-determined`` when
some trial was undetermined; else ``no-change-found`` when some trial ran; else
``never-asked``. A ``live`` question's witness is the first reaching trial by record, then log,
then point order: ``{"record", "ref", "value"}``, the entry's ref and its value or marker as the
copy's record wrote it. The report is plain data.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ommat._canonical import encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._ledger import _ITEM_DEPTH, Record
from ommat._markers import Undetermined
from ommat._refs import code_point_sorted, declared_ref
from ommat._replay import FindingKind
from ommat._values import type_name
from ommat.evaluation._diff import OUTSIDE, _read, _refuse_unfit, _run_sync, baseline
from ommat.evaluation._edit import _applied, _labels, _q_value, _SetEntry
from ommat.evaluation._splits import _refuse_records
from ommat.helpers._points import OnTop, Point, entry_points, filled, fresh

_K = FindingKind
_DECIDING = frozenset({_K.plan, _K.verdict})
_Log = list[dict[str, Any]]


async def question_reach(
    records: Iterable[object], loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"records": [hash], "trials": N, "questions": {ref: {"status",
    "witness"}}, "narrowed": [ref]}``, keyed by every ``q:`` ref a source provides; a
    ``ValueError`` for a slip in ``values`` or a question nothing fills, before any record is
    read. ``values`` maps a ``q:`` ref a source provides to a list of its values, as the
    dead-question lint takes it; for a label question with no grid entry, the labels to try."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    _refuse_records("records", records)
    provided = code_point_sorted(
        ref for s in loaded._design.sources.values() for ref in s.provides if ref.startswith("q:")
    )
    fills, _, narrowed = filled(loaded, provided, values)
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
                items, _, found = await _applied(
                    record, log, loaded, [_SetEntry(ref, value)], None, None
                )
                trials += 1
                ran.add(question)
                if any(f["kind"] in OUTSIDE for f in found):
                    undetermined.add(question)
                elif question not in witnesses and any(f["kind"] in _DECIDING for f in found):
                    written = items[0]
                    shown = written["value"] if "value" in written else written["marker"]
                    witnesses[question] = {
                        "record": record.record_hash,
                        "ref": ref,
                        "value": fresh(shown),
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
    records: Iterable[object], loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """``question_reach`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync("question_reach", lambda: question_reach(records, loaded, values=values))


def _asked(entry: dict[str, Any], log: _Log) -> bool:
    """Whether ``entry`` is a ``q:`` entry a returned call wrote."""
    if entry["item"] != "entry" or not cast(str, entry["ref"]).startswith("q:"):
        return False
    call = entry["call"]
    return call is not None and log[cast(int, call)]["outcome"] == "returned"


def _points(
    ref: str, entry: dict[str, Any], log: _Log, loaded: LoadedDesign, points: list[Point]
) -> list[object]:
    """The entry's points in their order, each as form 1 takes it, less the one whose record
    form is the entry's recorded value or marker by canonical bytes."""
    question = declared_ref(ref)
    if ref != question:  # a grounded entry: its own marker last
        points = entry_points(points, ref)
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
        live = point.live
        if isinstance(live, Undetermined):  # the marker, whose record form is its document
            if encode(point.record, _ITEM_DEPTH) != recorded:
                out.append(live)
            continue
        if isinstance(live, OnTop):
            assert rendered is not None
            live = live.over(rendered, loaded._label_at[question])
        written = _q_value(ref, live, call, log, loaded)
        if written is not None and encode(written, _ITEM_DEPTH) == recorded:
            continue  # the recorded value: replay would find nothing
        out.append(fresh(live))
    return out
