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

A per-candidate question is no read of its own. At each assignment of the other reads its
set's value is fixed, a ``t:`` set's by the point and a ``d:`` set's by a probe run with every
grounded entry at its first point, which counts as that run. Each candidate grounds one entry,
filled as its declaration is with its own marker last, and the entries cross after the other
reads in code-point order of their grounded refs, so each assignment holds a block of runs
whose size follows its sets. An empty set, or one holding a marker, grounds nothing.
"""

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import ommat._decide as _decide
from ommat._binding import CallResult, SourceCall
from ommat._canonical import encode
from ommat._design import LoadedDesign
from ommat._ledger import _ITEM_DEPTH
from ommat._refs import code_point_sorted, declared_ref, grounded_ref
from ommat._values import type_name
from ommat.helpers._points import OnTop, Point, entry_points, filled, fresh

_STATE = "t:state"
_DEFAULT = "default"

_Run = tuple[bytes, bytes]


@dataclass(frozen=True, slots=True)
class _Ran:
    """One run: its ``plan`` and ``verdict`` bytes, the index of each other read's point, the
    index of each grounded entry's point by grounded ref, and its point as a report shows it."""

    run: _Run
    at: tuple[int, ...]
    entries: dict[str, int]
    shown: dict[str, Any]


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
    provided = {ref for s in sources for ref in s.provides} | {_STATE}
    grounded = provided & loaded._grounded.keys()
    reads = code_point_sorted(provided - grounded)
    fills, on_grid = filled(loaded, code_point_sorted(provided), values)
    keys = {s.oracle for s in sources if s.oracle is not None}
    lists = [fills[r] for r in reads]
    ran: list[_Ran] = []
    for at in itertools.product(*(range(len(points)) for points in lists)):  # grid order
        point = {r: lists[k][i] for k, (r, i) in enumerate(zip(reads, at, strict=True))}
        ran += _block(loaded, point, keys, fills, tuple(at))
    asked = {declared_ref(g) for r in ran for g in r.entries}  # grounded by some assignment
    questions: dict[str, Any] = {}
    for ref in code_point_sorted(provided):
        if not ref.startswith("q:"):
            continue
        if ref in grounded:
            pair = _witness(ran, _entry_keys(ran, ref))
        else:
            k = reads.index(ref)
            pair = _witness(ran, _read_keys(ran, k))
        if pair is not None:
            witness: Any = [fresh(ran[i].shown) for i in pair]
            status = "live"
        else:
            witness = None
            if ref in grounded and ref not in asked:
                status = "not-grounded"
            else:
                status = "no-change-found" if ref in on_grid else "not-on-grid"
        questions[ref] = {"status": status, "witness": witness}
    return {"points": len(ran), "questions": questions}


class _At(dict[str, Point]):
    """A run's point: each other read's and each grounded entry's, and, for a grounded entry
    it does not hold, as the probe asks one, its declared question's first point."""

    def __init__(self, points: Mapping[str, Point], firsts: Mapping[str, Point]) -> None:
        super().__init__(points)
        self.firsts = firsts

    def __missing__(self, ref: str) -> Point:
        return self.firsts[declared_ref(ref)]


def _block(
    loaded: LoadedDesign,
    point: dict[str, Point],
    keys: set[str],
    fills: Mapping[str, list[Point]],
    at: tuple[int, ...],
) -> list[_Ran]:
    """An assignment's runs: each grounded entry's points crossed, the last fastest."""
    firsts = {ref: fills[ref][0] for ref in loaded._grounded if ref in fills}
    sets: dict[str, object] = {}
    probe = None
    for set_ref in {loaded._grounded[ref] for ref in firsts}:
        if set_ref in point:  # a t: set, fixed by the point
            sets[set_ref] = point[set_ref].live
    if any(loaded._grounded[ref] not in sets for ref in firsts):  # a d: set: run the probe
        probe = _run(loaded, _At(point, firsts), keys)
        sets.update(probe[2])
    entries: dict[str, list[Point]] = {}
    for ref in firsts:
        for candidate in _candidates(sets.get(loaded._grounded[ref])):
            entry = grounded_ref(ref, candidate)
            entries[entry] = entry_points(fills[ref], entry)
    names = code_point_sorted(entries)
    out: list[_Ran] = []
    for combo in itertools.product(*(range(len(entries[n])) for n in names)):
        chosen = dict(zip(names, combo, strict=True))
        given = {**point, **{n: entries[n][i] for n, i in chosen.items()}}
        done = (
            probe
            if probe is not None and not any(combo)
            else _run(loaded, _At(given, firsts), keys)
        )
        shown = {ref: fresh(given[ref].record) for ref in code_point_sorted(given)}
        shown.update(done[1])  # a candidate-valued question as its run's record wrote it
        out.append(_Ran(done[0], at, chosen, shown))
    return out


def _run(
    loaded: LoadedDesign, point: dict[str, Point], keys: set[str]
) -> tuple[_Run, dict[str, Any], dict[str, Any]]:
    """One turn at ``point``: its ``plan`` and ``verdict`` items' canonical bytes, each
    candidate-valued question's value or marker as the run's record wrote its entry, and each
    grounding ``d:`` set's value or marker as the record wrote it."""

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
    written = {
        i["ref"]: i["value"] if "value" in i else i["marker"] for i in log if i["item"] == "entry"
    }
    entries = {r: fresh(v) for r, v in written.items() if r in loaded._label_sets}
    sets = {r: written.get(r) for r in loaded._grounded.values() if r.startswith("d:")}
    return (encode(plan, _ITEM_DEPTH), encode(verdict, _ITEM_DEPTH)), entries, sets


def _candidates(value: object) -> list[str]:
    """The candidates a set's value grounds: none for a marker, a missing entry or no list."""
    if type(value) is not list:
        return []
    return list(dict.fromkeys(c for c in cast("list[object]", value) if type(c) is str))


def _items(entries: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(entries.items()))


_Key = tuple[object, ...]  # what a comparison holds fixed: all of a run's point but one read


def _read_keys(ran: Sequence[_Ran], k: int) -> list[list[_Key]]:
    """Each run's key for the ``k``-th other read: every other read and every entry fixed."""
    return [[(r.at[:k] + r.at[k + 1 :], _items(r.entries))] for r in ran]


def _entry_keys(ran: Sequence[_Ran], ref: str) -> list[list[_Key]]:
    """Each run's keys for a per-candidate question, one per grounded entry of it: the
    entry, the other reads and the other entries fixed. A run grounding none has none."""
    return [
        [
            (g, r.at, _items({h: i for h, i in r.entries.items() if h != g}))
            for g in r.entries
            if declared_ref(g) == ref
        ]
        for r in ran
    ]


def _witness(ran: Sequence[_Ran], keys: Sequence[list[_Key]]) -> tuple[int, int] | None:
    """Of the pairs of runs that differ only at one varied point and whose runs differ, the
    one whose earlier point comes first in grid order, then whose later point does."""
    best: tuple[int, int] | None = None
    firsts: dict[object, int] = {}  # each assignment of the rest: its first run
    found: set[object] = set()
    for i, run_keys in enumerate(keys):
        for rest in run_keys:
            first = firsts.setdefault(rest, i)
            if rest in found or ran[i].run == ran[first].run:
                continue
            found.add(rest)  # the first that differs is this assignment's least later point
            if best is None or (first, i) < best:
                best = (first, i)
    return best
