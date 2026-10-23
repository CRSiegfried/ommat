# pyright: reportPrivateUsage=false
"""The dead-question lint, ``dead_questions``: whether any point the lint ran found a
question's answer changing the plan or the verdict.

It fills the exogenous values, every ref a source provides and ``t:state``, by the lint points,
and runs ``decide_sync`` once per point with the default run settings, no codec, the point's
``t:state`` as the state and one binding, under ``default`` and every declared oracle name,
that answers each asked ref from the point. So the ``d:`` values, ``Unasked`` and ``Unsure``
come from the rules and the ``when``s, which decide runs. Two runs differ when their ``plan``
or ``verdict`` items' canonical bytes differ, as replay compares them. Each point
runs once, and every question is judged from those runs. A candidate-valued question is
answered over the labels each run renders, each fixed label on top and then a candidate; it
is never on the grid, and a witness shows it as that run's record wrote its entry.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import ommat._decide as _decide
from ommat._binding import CallResult, SourceCall
from ommat._canonical import encode
from ommat._design import LoadedDesign
from ommat._ledger import _ITEM_DEPTH
from ommat._refs import code_point_sorted
from ommat._values import type_name
from ommat.helpers._points import OnTop, Point, filled, fresh, grid_order

_STATE = "t:state"
_DEFAULT = "default"


def dead_questions(
    loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"points": N, "questions": {ref: {"status", "witness"}}}``, keyed by
    every ``q:`` ref a source provides; a ``ValueError`` before any run for a read nothing
    fills or a slip in ``values``. ``values`` maps a read the lint fills, ``t:state`` or a
    field path under it, or a ``t:`` or ``q:`` ref a source provides, to a list of its values;
    a ``d:`` ref is never a key, since rules compute those values."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    sources = loaded._design.sources.values()
    reads = code_point_sorted({ref for s in sources for ref in s.provides} | {_STATE})
    fills, on_grid = filled(loaded, reads, values)
    keys = {s.oracle for s in sources if s.oracle is not None}
    lists = [fills[r] for r in reads]
    sizes = [len(points) for points in lists]
    runs: list[tuple[bytes, bytes]] = []
    written: list[dict[str, Any]] = []
    for point in grid_order(lists):
        run, entries = _run(loaded, dict(zip(reads, point, strict=True)), keys)
        runs.append(run)
        written.append(entries)
    questions: dict[str, Any] = {}
    for k, ref in enumerate(reads):
        if not ref.startswith("q:"):
            continue
        pair = _witness(runs, sizes, k)
        if pair is not None:
            witness: Any = [{**_shown(reads, lists, sizes, i), **written[i]} for i in pair]
            status = "live"
        else:
            witness = None
            status = "no-change-found" if ref in on_grid else "not-on-grid"
        questions[ref] = {"status": status, "witness": witness}
    return {"points": len(runs), "questions": questions}


def _run(
    loaded: LoadedDesign, point: dict[str, Point], keys: set[str]
) -> tuple[tuple[bytes, bytes], dict[str, Any]]:
    """One turn at ``point``: its ``plan`` and ``verdict`` items' canonical bytes, and each
    candidate-valued question's value or marker as the run's record wrote its entry."""

    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        values: dict[str, object] = {ref: fresh(point[ref].live) for c in batch for ref in c.refs}
        for call in batch:
            for ref, q in call.questions.items():
                live = point[ref].live
                if isinstance(live, OnTop):
                    values[ref] = live.over(q.labels, loaded._label_at[ref])
                else:
                    values[ref] = fresh(live)
        return CallResult(values)

    bindings = {_DEFAULT: binding, **dict.fromkeys(keys, binding)}
    decision = _decide.decide_sync(loaded, fresh(point[_STATE].live), bindings)
    from ommat._replay import _first  # imported here: replay imports this package

    log = decision.record.log
    plan, verdict = _first(log, "plan"), _first(log, "verdict")
    assert plan is not None  # decide_sync returned, so the log holds both
    assert verdict is not None
    entries = {
        i["ref"]: fresh(i["value"] if "value" in i else i["marker"])
        for i in log
        if i["item"] == "entry" and i["ref"] in loaded._label_sets
    }
    return (encode(plan, _ITEM_DEPTH), encode(verdict, _ITEM_DEPTH)), entries


def _witness(
    runs: Sequence[tuple[bytes, bytes]], sizes: Sequence[int], k: int
) -> tuple[int, int] | None:
    """Of the pairs of points that differ only at read ``k`` and whose runs differ, the one
    whose earlier point comes first in grid order, then whose later point does."""
    stride = 1
    for size in sizes[k + 1 :]:
        stride *= size
    best: tuple[int, int] | None = None
    firsts: dict[tuple[int, int], int] = {}  # each assignment of the other reads: its first
    found: set[tuple[int, int]] = set()
    for i, run in enumerate(runs):
        key = (i // (stride * sizes[k]), i % stride)
        first = firsts.setdefault(key, i)
        if key in found or run == runs[first]:
            continue
        found.add(key)  # the first that differs is this assignment's least later point
        if best is None or (first, i) < best:
            best = (first, i)
    return best


def _shown(
    reads: Sequence[str], lists: Sequence[list[Point]], sizes: Sequence[int], i: int
) -> dict[str, Any]:
    """The ``i``-th point in grid order as a report shows it, each read in record form."""
    shown: dict[str, Any] = {}
    for ref, points, size in reversed(list(zip(reads, lists, sizes, strict=True))):
        i, j = divmod(i, size)
        shown[ref] = fresh(points[j].record)
    return {ref: shown[ref] for ref in reads}
