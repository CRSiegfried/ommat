# pyright: reportPrivateUsage=false
"""The grid form of the rescoring job: ``rescore`` over the designs a list of parameter points
gives, each record rerun once per region of points whose runs cannot differ, not once per point
(§rescore-grid).

A point's design is ``loaded`` with the point's leaves holding its values, built once per point
as ``with_parameters`` builds it. A point names a leaf whole, or one key of a numeric mapping
leaf by its key path, ``leaf.key``, that key's value replacing the one ``loaded``'s leaf holds.
A run reaches a numeric leaf only by comparing values the turn holds against it, so two points
on the same side of every value a run compared give that run again: the same questions asked,
the same values read and compared, the same entries. The job reruns one uncovered point with the
switching job's light ledger, keeping only the entries the job reads: the named refs', each
table input's read and grounded set, and each function rule's ``boundaries`` reads. It reads
from that log each grid leaf's compared values, from every comparison the rerun makes whatever
refs are named, and the side a leaf value equal to one joins, and gives that run's row to every
uncovered point whose values fall in the same bucket on every grid leaf, until every point is
covered; a point whose ``boolean``, ``label`` and wording-slot values no other uncovered point
shares covers itself without a region search. A key path, and each key of a mapping leaf
given whole, is bucketed on its own axis: the comparisons naming the leaf whole or that key's
path. A tie joins the side the predicate puts it on where the document
fixes the predicate, a table's pick helper and ``extract`` bindings and a table cell's own
comparison; a function rule's ``boundaries`` roles and its bare reads, compared in code the
document does not hold, put a tie in a bucket of its own. A ``boolean`` or ``label`` leaf, and a
leaf, or a key of one, that fills some question's wording slot, whose value reaches the wording
and not a comparison, give each distinct grid value its own bucket. A run in which some table's
``candidate`` read finds its set's entry holding no value, so that its compared values cannot be
read, covers its own point alone.

Before any run the job refuses a grid leaf a run could reach other than by a comparison it can
see: a function rule declaring it whose ``boundaries`` name neither the leaf whole nor the path
of each key the grid varies, a ``C+``, ``C<`` or ``C>`` table aggregating it beside other rows,
a reader of an entry that holds it, a table row outputting ``Param`` of it under any hit policy,
and a function rule declaring it whose output type can hold a number. A function rule's
``boundaries`` are trusted as the switching job trusts them, each key path for its key. Records
of one hash are rerun once. The rows are ``rescore``'s, in its order, with each point's design
hash, and equal ``rescore`` over the points' designs, with the same ``refs``, row for row.
"""

from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode, encode
from ommat._decide import EntryRefusal
from ommat._design import InvalidDesign, LoadedDesign, Question
from ommat._interpret import Reuse
from ommat._ledger import _copy
from ommat._load import with_parameters
from ommat._refs import code_point_sorted, key_path
from ommat._typed import decoded, typed
from ommat._types import MappingOf, Type
from ommat._values import type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _run_sync, _state
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._rescore import NOT_DETERMINED, _named, _run_of
from ommat.evaluation._switching import (
    _HIGH,
    _LOW,
    _comparisons,
    _components,
    _contains,
    _declared_ref,
    _discrete,
    _Entries,
    _finite,
    _hidden,
    _leaf_type,
    _rerun,
)
from ommat.helpers._numbers import exact

_R = RefusalReason
_Log = list[dict[str, Any]]
_Doc = dict[str, Any]
_Point = Mapping[str, object]
_Result = tuple[dict[str, Any] | str, list[str]]  # a row's entries and stale refs
_STATE = "t:state"


async def rescore_grid(
    records: Sequence[object],
    recorded: LoadedDesign,
    loaded: LoadedDesign,
    points: Sequence[_Point],
    *,
    refs: Sequence[str],
) -> list[dict[str, Any]]:
    """``rescore(records, recorded, designs, refs=refs)``, ``designs`` being ``loaded`` with each
    point's leaves holding its values, ``points`` a sequence of ``{leaf path: value}`` mappings
    all naming one non-empty set of leaves: the same rows, in the same order, one per point and
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
    named = _named(refs, {"loaded": loaded})
    names = tuple(f"records[{i}]" for i in range(len(records)))
    read = tuple(_read(r, name) for r, name in zip(records, names, strict=True))
    # Read once, as every read decodes afresh: one log object per record, which the fixture's
    # construction key holds by identity, so a later run reuses the first run's calls.
    logs = tuple(r.log for r in read)
    _refuse_unfit(names, read, logs, recorded)
    for log in logs:
        _run_of(log)  # a recorded setting decide refuses is refused here, before any run
    if not points:
        return []
    grid = _Grid(recorded, loaded, points, named)
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
            entries, stale = found[i]
            rows.append(
                {
                    "design": design.design_hash,
                    "record": r.record_hash,
                    "entries": _copy(entries),
                    "stale": list(stale),
                }
            )
    return rows


def rescore_grid_sync(
    records: Sequence[object],
    recorded: LoadedDesign,
    loaded: LoadedDesign,
    points: Sequence[_Point],
    *,
    refs: Sequence[str],
) -> list[dict[str, Any]]:
    """``rescore_grid`` run to completion on one event loop; ``RuntimeError`` inside a running
    event loop."""
    return _run_sync(
        "rescore_grid", lambda: rescore_grid(records, recorded, loaded, points, refs=refs)
    )


class _Grid:
    """The grid's value-free work: its leaves, checked; each point's design; and each point's
    bucket material per leaf, a compared leaf's values exact."""

    def __init__(
        self,
        recorded: LoadedDesign,
        loaded: LoadedDesign,
        points: Sequence[_Point],
        refs: Sequence[str],
    ):
        self.recorded = recorded
        self.loaded = loaded
        self.refs = refs
        self.paths = list(points[0])
        if not self.paths or any(set(point) != set(self.paths) for point in points):
            raise EvaluationRefusal(
                "the points do not all name one non-empty set of leaves",
                reason=_R.uneven_grid,
            )
        self.paths = code_point_sorted(self.paths)
        for path in self.paths:
            keyed = key_path(path)
            if keyed is not None and keyed[0] in points[0]:
                raise EvaluationRefusal(
                    f"the points name the leaf at {keyed[0]} both whole and by its key path {path}",
                    reason=_R.overlapping_paths,
                    path=path,
                )
        types: dict[str, Type] = {path: _grid_type(path, loaded) for path in self.paths}
        self.document = cast("_Doc", decode(loaded.document))
        worded = _worded(loaded)
        self.compared = [
            path
            for path in self.paths
            if _finite(types[path]) is None and _leaf_of(path) not in worded
        ]
        # Each compared mapping leaf's keys across the points, each bucketed on its own axis.
        self.keys: dict[str, list[str]] = {
            path: code_point_sorted({k for p in points for k in _held_keys(p[path])})
            for path in self.compared
            if isinstance(types[path], MappingOf)
        }
        for path in self.compared:
            unseen = _unseen(self.document, loaded, path, points)
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
                self.designs.append(with_parameters(loaded, _leaf_values(loaded, point)))
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
        # The points sharing each one's values on every by-value path: only they can share a run.
        by_value: dict[tuple[object, ...], set[int]] = {}
        for i, material in enumerate(self.material):
            held = tuple(
                m for p, m in zip(self.paths, material, strict=True) if p not in self.compared
            )
            by_value.setdefault(held, set()).add(i)
        self.peers: list[frozenset[int]] = [frozenset()] * len(points)
        for group in by_value.values():
            for i in group:
                self.peers[i] = frozenset(group)
        self.needed = _needed(self.document, refs)
        self.kept: dict[str, bool] = {}

    async def results(self, log: _Log) -> list[_Result]:
        """One record's entries and stale refs per point."""
        state = cast("dict[str, Any]", _state(log))  # the record checks found a value there
        t = self.loaded._types[_STATE]
        if typed(decoded(state["value"], t), t) is None:  # the same under every point
            # A broken entry: decide writes t:state alone, so no named entry.
            return [({}, []) for _ in self.designs]
        out: list[_Result | None] = [None] * len(self.designs)
        uncovered = set(range(len(self.designs)))
        reuse = Reuse()  # the first region's rerun, which each later one fires unchanged rules from
        while uncovered:
            at = min(uncovered)
            rerun, fixture = await _rerun(
                log, self.designs[at], log, design=self.recorded, reuse=reuse, keep=self.keeps
            )
            stale = code_point_sorted(fixture.stale)
            result: _Result = (NOT_DETERMINED, stale) if stale else (self.entries(rerun.log), [])
            peers = uncovered & self.peers[at]
            for q in [at] if len(peers) == 1 else self.region(at, peers, rerun.log):
                out[q] = result
                uncovered.discard(q)
        return cast("list[_Result]", out)

    def keeps(self, ref: str) -> bool:
        """Whether a rerun's log keeps the entry at ``ref``: one ``region`` or ``entries``
        reads."""
        found = self.kept.get(ref)
        if found is None:
            found = self.kept[ref] = _declared_ref(ref) in self.needed
        return found

    def entries(self, log: _Log) -> dict[str, Any]:
        """The entry at each named ref the rerun wrote, in their order, as decide writes it."""
        held = {i["ref"]: i for i in log if i["item"] == "entry" and i["ref"] in self.refs}
        return {
            ref: {k: held[ref][k] for k in ("value", "marker") if k in held[ref]}
            for ref in self.refs
            if ref in held
        }

    def region(self, at: int, uncovered: set[int], log: _Log) -> list[int]:
        """The uncovered points whose runs cannot differ from the run at ``at``, read from its
        log: ``at`` alone when some table's compared values could not be read."""
        entries = _Entries(log)
        buckets: dict[str, Callable[[Fraction], int]] = {}
        for path in self.compared:
            # A mapping leaf's each key on its own axis: the comparisons naming the leaf whole
            # or that key's path.
            for at_path in (
                [f"{path}.{k}" for k in self.keys[path]] if path in self.keys else [path]
            ):
                found, blind = _comparisons(self.document, at_path, entries)
                if blind:
                    return [at]
                buckets[at_path] = _axis(found)
        mine = self.signature(at, buckets)
        return [q for q in sorted(uncovered) if self.signature(q, buckets) == mine]

    def signature(
        self, point: int, buckets: Mapping[str, Callable[[Fraction], int]]
    ) -> tuple[object, ...]:
        """Where ``point`` falls on every grid path: a compared leaf's or key's bucket, or each
        key's, a by-value path's canonical bytes."""
        out: list[object] = []
        for path, material in zip(self.paths, self.material[point], strict=True):
            if path not in self.compared:
                out.append(material)
            elif type(material) is Fraction:
                out.append(buckets[path](material))
            else:
                keyed = cast("tuple[tuple[str, Fraction], ...]", material)
                out.append(tuple((k, buckets[f"{path}.{k}"](x)) for k, x in keyed))
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


def _needed(document: _Doc, refs: Sequence[str]) -> frozenset[str]:
    """The declared refs whose entries ``region`` and ``entries`` read from a rerun: ``refs``,
    each table input's read and grounded set, and each function rule's ``boundaries``
    reads."""
    out = set(refs)
    for c in _components(document).values():
        if c["form"] == "table":
            out.update(_declared_ref(i["read"]) for i in cast("list[dict[str, Any]]", c["inputs"]))
            if c.get("grounded") is not None:
                out.add(_declared_ref(c["grounded"]))
        elif c["boundaries"] is not None:
            out.update(_declared_ref(read) for read in cast("dict[str, Any]", c["boundaries"]))
    return frozenset(out)


def _unseen(document: _Doc, loaded: LoadedDesign, path: str, points: Sequence[_Point]) -> list[str]:
    """Each component a run could reach the leaf, or key, at ``path`` through other than by a
    comparison the grid can see, in code-point order: the switching job's hidden ones, each
    table with a row outputting ``Param`` of the leaf, and each function rule declaring it whose
    output type can hold a number. A function rule sees a key path's key, and a mapping leaf's
    each key whose value differs between two points holding it, only where its ``boundaries``
    name the leaf whole or that key's path: a key no point varies gives every run the same
    value."""
    keyed = key_path(path)
    leaf = path if keyed is None else keyed[0]
    keys: set[str] | None = {keyed[1]} if keyed is not None else None
    if keyed is None and isinstance(loaded._leaf_types[leaf], MappingOf):
        held: dict[str, set[bytes]] = {}
        for point in points:
            value = point[path]
            for k in _held_keys(value):
                held.setdefault(k, set()).add(encode(cast("Mapping[str, object]", value)[k]))
        keys = {k for k, values in held.items() if len(values) > 1}
    return _unseen_keys(document, loaded, leaf, keys)


def _unseen_keys(
    document: _Doc, loaded: LoadedDesign, leaf: str, keys: set[str] | None
) -> list[str]:
    """``_unseen`` for the leaf at ``leaf``, ``keys`` the keys of a mapping leaf that vary, or
    ``None`` for any other leaf, in code-point order."""
    out = _hidden(document, loaded, leaf, keys)
    for name, c in _components(document).items():
        if c["form"] == "table":
            if any(_contains(row.get("output"), {"param": leaf}) for row in c["rows"]):
                out.add(name)
        elif leaf in c["params"] and not _discrete(loaded._types[c["provides"]]):
            out.add(name)
    return code_point_sorted(out)


def _leaf_of(path: str) -> str:
    """The leaf a grid path names: itself, or a key path's leaf."""
    keyed = key_path(path)
    return path if keyed is None else keyed[0]


def _held_keys(value: object) -> list[str]:
    """A mapping value's keys; none for any other value, which the point's design refuses."""
    if isinstance(value, Mapping):
        return [k for k in cast("Mapping[object, object]", value) if type(k) is str]
    return []


def _grid_type(path: object, loaded: LoadedDesign) -> Type:
    """A grid path's type: a leaf's as ``switching_values`` checks it; a key path's, its numeric
    mapping leaf's value type, the key one ``loaded``'s value of the leaf holds, else
    ``unknown-path``."""
    keyed = key_path(path)
    if keyed is None:
        return _leaf_type(path, loaded)
    leaf, key = keyed
    t = loaded._leaf_types.get(leaf)
    if not isinstance(t, MappingOf) or key not in cast("dict[str, Any]", loaded._leaf_value(leaf)):
        raise EvaluationRefusal(
            f"path {path!r} names no key of a mapping leaf of the parameter tree",
            reason=_R.unknown_path,
            path=cast(str, path),
        )
    try:
        _leaf_type(leaf, loaded)
    except EvaluationRefusal as exc:  # a mapping of a type the job does not vary
        raise EvaluationRefusal(str(exc), reason=exc.reason, path=cast(str, path)) from None
    return t.value


def _leaf_values(loaded: LoadedDesign, point: _Point) -> dict[str, object]:
    """A point as ``with_parameters`` takes it: each key path's value merged into its leaf's
    value as ``loaded`` holds it."""
    out: dict[str, object] = {}
    for path, value in point.items():
        keyed = key_path(path)
        if keyed is None:
            out[path] = value
            continue
        leaf, key = keyed
        if leaf not in out:
            out[leaf] = dict(cast("dict[str, object]", loaded._leaf_value(leaf)))
        cast("dict[str, object]", out[leaf])[key] = value
    return out


def _worded(loaded: LoadedDesign) -> set[str]:
    """Each leaf some question's wording slot takes its value from."""
    return set(_wording(loaded))


def _wording(loaded: LoadedDesign) -> dict[str, set[str]]:
    """Each leaf some question's wording slot takes its value from, to the sources asking such
    a question."""
    out: dict[str, set[str]] = {}
    for sid, source in loaded._design.sources.items():
        for provided in source.provides.values():
            if isinstance(provided, Question):
                for slot in provided.wording.slots:
                    origin = cast("Mapping[str, str]", slot["origin"])
                    if "param" in origin:
                        out.setdefault(origin["param"], set()).add(sid)
    return out
