# pyright: reportPrivateUsage=false
"""The grid form of the rescoring job: ``rescore`` over the designs a list of parameter points
gives, each record rerun once per region of points whose runs cannot differ, not once per point
(§evaluation-jobs).

A point's design is ``loaded`` with the point's leaves holding its values, built once per point
as ``with_parameters`` builds it. A run reaches a numeric leaf only by comparing values the turn
holds against it, so two points on the same side of every value a run compared give that run
again: the same questions asked, the same values read and compared, the same decision. The job
reruns one uncovered point with the switching job's light ledger, reads from its log each grid
leaf's compared values and the side a leaf value equal to one joins, and gives that run's row to
every uncovered point whose values fall in the same bucket on every grid leaf, until every point
is covered. A tie joins the side the predicate puts it on where the document fixes the predicate,
the pick helpers' and ``extract``'s at or above their bar, a ``margin`` measured against the exact
lead, and a table cell's comparison; a function rule's other ``boundaries`` roles and its bare
reads put a tie in a bucket of its own. A ``boolean`` or ``label`` leaf, and a leaf that fills
some question's wording slot, whose value reaches the wording and not a comparison, give each
distinct grid value its own bucket. A run in which some table's ``candidate`` read finds its set's
entry holding no value, so that its compared values cannot be read, covers its own point alone.

Before any run the job refuses a grid leaf a run could reach other than by a comparison it can
see: a function rule declaring it whose ``boundaries`` do not name it, a ``C+``, ``C<`` or ``C>``
table aggregating it beside other rows, a reader of an entry that holds it, a table row outputting
``Param`` of it under any hit policy, and a function rule declaring it whose output type can hold
a number. A function rule's ``boundaries`` are trusted as the switching job trusts them. Records
of one hash are rerun once. The rows are ``rescore``'s, in its order, with each point's design
hash, and equal ``rescore`` over the points' designs row for row.
"""

from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode, encode
from ommat._decide import EntryRefusal
from ommat._design import InvalidDesign, LoadedDesign, Question
from ommat._ledger import _copy
from ommat._load import with_parameters
from ommat._refs import code_point_sorted
from ommat._typed import decoded, typed
from ommat._types import MappingOf, Type
from ommat._values import type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _run_sync, _state
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._rescore import NOT_DETERMINED, _broken_state
from ommat.evaluation._switching import (
    _HIGH,
    _LOW,
    _comparisons,
    _components,
    _contains,
    _discrete,
    _finite,
    _hidden,
    _leaf_type,
    _margin_lead,
    _rerun,
)
from ommat.helpers._numbers import exact

_R = RefusalReason
_Log = list[dict[str, Any]]
_Doc = dict[str, Any]
_Point = Mapping[str, object]
_Result = tuple[dict[str, Any] | str, list[str]]  # a row's decision and stale refs
_STATE = "t:state"


async def rescore_grid(
    records: Sequence[object],
    recorded: LoadedDesign,
    loaded: LoadedDesign,
    points: Sequence[_Point],
) -> list[dict[str, Any]]:
    """``rescore(records, recorded, designs)``, ``designs`` being ``loaded`` with each point's
    leaves holding its values, ``points`` a sequence of ``{leaf path: value}`` mappings all
    naming one non-empty set of leaves: the same rows, in the same order, one per point and
    record, each with its point's design hash. Each record is rerun once per region of points
    that cannot differ, not once per point. Each record is checked as ``rescore`` checks it,
    then the grid leaves and each point's design, before any run."""
    if type(recorded) is not LoadedDesign:
        raise EntryRefusal(f"recorded is a {type_name(recorded)}, not exactly a LoadedDesign")
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    if isinstance(points, str | bytes) or not isinstance(points, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"points are a {type_name(points)}, not a sequence of leaf-value mappings")
    for i, point in enumerate(points):
        if not isinstance(point, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"points[{i}] is a {type_name(point)}, not a mapping of leaf values")
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"records are a {type_name(records)}, not a sequence of records")
    names = tuple(f"records[{i}]" for i in range(len(records)))
    read = tuple(_read(r, name) for r, name in zip(records, names, strict=True))
    logs = tuple(r.log for r in read)  # read once: every read decodes afresh
    _refuse_unfit(names, read, logs, recorded)
    if not points:
        return []
    grid = _Grid(recorded, loaded, points)
    results: list[list[_Result]] = []
    by_hash: dict[str, list[_Result]] = {}  # records of one hash are rerun once
    for r, log in zip(read, logs, strict=True):
        found = by_hash.get(r.record_hash)
        if found is None:
            found = by_hash[r.record_hash] = await grid.results(log)
        results.append(found)
    rows: list[dict[str, Any]] = []
    for i, design in enumerate(grid.designs):
        for r, found in zip(read, results, strict=True):
            decision, stale = found[i]
            rows.append(
                {
                    "design": design.design_hash,
                    "record": r.record_hash,
                    "decision": _copy(decision),
                    "stale": list(stale),
                }
            )
    return rows


def rescore_grid_sync(
    records: Sequence[object],
    recorded: LoadedDesign,
    loaded: LoadedDesign,
    points: Sequence[_Point],
) -> list[dict[str, Any]]:
    """``rescore_grid`` run to completion on one event loop; ``RuntimeError`` inside a running
    event loop."""
    return _run_sync("rescore_grid", lambda: rescore_grid(records, recorded, loaded, points))


class _Grid:
    """The grid's value-free work: its leaves, checked; each point's design; and each point's
    bucket material per leaf, a compared leaf's values exact."""

    def __init__(self, recorded: LoadedDesign, loaded: LoadedDesign, points: Sequence[_Point]):
        self.recorded = recorded
        self.loaded = loaded
        self.policy: str = loaded._design.policy
        self.paths = list(points[0])
        if not self.paths or any(set(point) != set(self.paths) for point in points):
            raise EvaluationRefusal(
                "the points do not all name one non-empty set of leaves",
                reason=_R.uneven_grid,
            )
        types: dict[str, Type] = {path: _leaf_type(path, loaded) for path in self.paths}
        self.document = cast("_Doc", decode(loaded.document))
        worded = _worded(loaded)
        self.compared = [
            path for path in self.paths if _finite(types[path]) is None and path not in worded
        ]
        for path in self.compared:
            unseen = _unseen(self.document, loaded, path)
            if unseen:
                raise EvaluationRefusal(
                    f"the leaf at {path} reaches {', '.join(unseen)} other than by a comparison "
                    "the grid can see",
                    reason=_R.hidden_leaf,
                    path=path,
                )
        self.designs: list[LoadedDesign] = []
        for i, point in enumerate(points):
            try:
                self.designs.append(with_parameters(loaded, point))
            except InvalidDesign as exc:
                raise EvaluationRefusal(
                    f"points[{i}]'s design does not load: {exc}",
                    reason=_R.invalid_design,
                    error=exc,
                ) from exc
        self.material: list[list[object]] = [
            [_material(point[path], types[path], path in self.compared) for path in self.paths]
            for point in points
        ]

    async def results(self, log: _Log) -> list[_Result]:
        """One record's decision and stale refs per point."""
        state = cast("dict[str, Any]", _state(log))  # the record checks found a value there
        t = self.loaded._types[_STATE]
        if typed(decoded(state["value"], t), t) is None:  # the same under every point
            return [(_broken_state(), [])] * len(self.designs)
        out: list[_Result | None] = [None] * len(self.designs)
        uncovered = set(range(len(self.designs)))
        while uncovered:
            at = min(uncovered)
            rerun, fixture = await _rerun(log, self.designs[at], log, design=self.recorded)
            stale = code_point_sorted(fixture.stale)
            result: _Result = (NOT_DETERMINED, stale) if stale else (self.decision(rerun.log), [])
            for q in self.region(at, uncovered, rerun.log):
                out[q] = result
                uncovered.discard(q)
        return cast("list[_Result]", out)

    def decision(self, log: _Log) -> dict[str, Any]:
        """The policy's entry as decide writes it, from the rerun's log."""
        entry = next(i for i in log if i["item"] == "entry" and i["ref"] == self.policy)
        return {k: entry[k] for k in ("value", "marker") if k in entry}

    def region(self, at: int, uncovered: set[int], log: _Log) -> list[int]:
        """The uncovered points whose runs cannot differ from the run at ``at``, read from its
        log: ``at`` alone when some table's compared values could not be read."""
        buckets: dict[str, Callable[[Fraction], int]] = {}
        for path in self.compared:
            found, blind = _comparisons(self.document, path, log, margin=_margin_lead)
            if blind:
                return [at]
            buckets[path] = _axis(found)
        mine = self.signature(at, buckets)
        return [q for q in sorted(uncovered) if self.signature(q, buckets) == mine]

    def signature(
        self, point: int, buckets: Mapping[str, Callable[[Fraction], int]]
    ) -> tuple[object, ...]:
        """Where ``point`` falls on every grid leaf: a compared leaf's bucket, or each key's, a
        by-value leaf's canonical bytes."""
        out: list[object] = []
        for path, material in zip(self.paths, self.material[point], strict=True):
            bucket = buckets.get(path)
            if bucket is None:
                out.append(material)
            elif type(material) is Fraction:
                out.append(bucket(material))
            else:
                keyed = cast("tuple[tuple[str, Fraction], ...]", material)
                out.append(tuple((k, bucket(x)) for k, x in keyed))
        return tuple(out)


def _material(value: object, t: Type, compared: bool) -> object:
    """A point's value on one leaf as the signature reads it: exact for a compared leaf, each
    key's for a numeric mapping, else its canonical bytes."""
    if not compared:
        return encode(value)
    if isinstance(t, MappingOf):
        held = cast("Mapping[str, object]", value)
        return tuple((k, exact(cast(float, held[k]))) for k in sorted(held))
    return exact(cast(float, value))


def _axis(found: list[tuple[str, str | None, Fraction]]) -> Callable[[Fraction], int]:
    """The bucket of a leaf value among the compared values ``found``: the open interval it
    falls in, or at a compared value the interval its tie side joins, ``low`` the one below,
    ``high`` the one above, else a bucket of its own; a value every comparison puts on one
    side joins that side."""
    sides: dict[Fraction, set[str | None]] = {}
    for _, side, x in found:
        sides.setdefault(x, set()).add(side)
    values = sorted(sides)
    joins = [next(iter(sides[x])) if len(sides[x]) == 1 else None for x in values]

    def bucket(v: Fraction) -> int:
        i = bisect_left(values, v)
        if i < len(values) and values[i] == v:
            return 2 * i if joins[i] == _LOW else 2 * i + 2 if joins[i] == _HIGH else 2 * i + 1
        return 2 * i

    return bucket


def _unseen(document: _Doc, loaded: LoadedDesign, path: str) -> list[str]:
    """Each component a run could reach the leaf at ``path`` through other than by a
    comparison the grid can see, in code-point order: the switching job's hidden ones, each
    table with a row outputting ``Param`` of the leaf, and each function rule declaring it whose
    output type can hold a number."""
    out = _hidden(document, loaded, path)
    for name, c in _components(document).items():
        if c["form"] == "table":
            if any(_contains(row.get("output"), {"param": path}) for row in c["rows"]):
                out.add(name)
        elif path in c["params"] and not _discrete(loaded._types[c["provides"]]):
            out.add(name)
    return code_point_sorted(out)


def _worded(loaded: LoadedDesign) -> set[str]:
    """Each leaf some question's wording slot takes its value from."""
    out: set[str] = set()
    for source in loaded._design.sources.values():
        for provided in source.provides.values():
            if isinstance(provided, Question):
                for slot in provided.wording.slots:
                    origin = cast("Mapping[str, str]", slot["origin"])
                    if "param" in origin:
                        out.add(origin["param"])
    return out
