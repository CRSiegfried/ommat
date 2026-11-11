# pyright: reportPrivateUsage=false
"""The switch screen: for each record and numeric leaf, whether every value of the leaf within a
window around its own value decides the record's turn as the record did, read from the record
alone, with no replay (§switch-screen).

A run reaches a numeric leaf only by comparing values the turn holds against it, so two values
of the leaf on the same side of every value a run compared give that run again (§rescore-grid).
The record is the run at the leaf's own value, so when no value the record compares the leaf
against lies in the window, past the side a tie joins, every value in the window gives the
record's own entries: the same questions asked, the same values read and compared, the same
decision. A value only a changed run would compare cannot arise inside the window, since the
run there does not change. The screen reads the record's compared values as the switching job's
round 0 reads them, and the grid's reach test: a leaf a run could reach other than by a
comparison, a table whose ``candidate`` read finds its set holding no value, and a leaf filling
a question's wording slot, are not cleared.

A value whose design does not load decides nothing. Load reads a numeric leaf's value only in
check 9's tests of a table's cells, a ``between``'s bounds and two rows' overlap, and their
result can change only where the leaf crosses another number the design holds, a literal,
another leaf's value or a type's bound, within two of it on an integer axis, or an integer.
So when no such number lies in the window, every value there loads as the leaf's own does;
else the screen loads one value at each such number and one between each two, and names each
value that fails.

The screen trusts the record as the run of ``loaded``: it reads it after the record checks and
replays nothing, so a record the switching job would refuse ``baseline-differs`` is screened as
written. A function rule's ``boundaries`` are trusted as the switching job trusts them.
"""

import itertools
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode
from ommat._design import InvalidDesign, LoadedDesign, Question
from ommat._load import with_parameters
from ommat._refs import code_point_sorted, key_path
from ommat._types import MappingOf, _numeric
from ommat._values import type_name
from ommat.evaluation._diff import _read, _refuse_unfit
from ommat.evaluation._edit import _designs
from ommat.evaluation._grid import _grid_type, _leaf_of, _unseen_keys
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._switching import _HIGH, _LOW, _comparisons, _Entries, _finite
from ommat.helpers._numbers import exact

_R = RefusalReason
_Doc = dict[str, Any]
_Log = list[dict[str, Any]]


def screen_switches(
    records: Sequence[object], loaded: LoadedDesign, within: Mapping[str, int | float]
) -> list[dict[str, Any]]:
    """For each record and each path ``within`` names, whether every value of that numeric
    leaf, or key of one, within ``within[path]`` of its own value decides the record's turn as
    the record did: one row per record and path, record-major, paths in code-point order."""
    _designs(loaded, None, ("loaded", "rerun_design"))
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"records are a {type_name(records)}, not a sequence of records")
    if not isinstance(within, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"within is a {type_name(within)}, not a mapping of leaf paths to widths")
    names = tuple(f"records[{i}]" for i in range(len(records)))
    read = tuple(_read(r, name) for r, name in zip(records, names, strict=True))
    logs = tuple(r.log for r in read)  # read once: every read decodes afresh
    _refuse_unfit(names, read, logs, loaded)
    for path in within:
        if type(path) is not str:
            raise EvaluationRefusal(
                f"path {path!r} names no leaf of the parameter tree",
                reason=_R.unknown_path,
                path=None,
            )
    paths = code_point_sorted(within)
    document = cast("_Doc", decode(loaded.document))
    leaves = [_Leaf(loaded, document, path, within[path]) for path in paths]
    rows: list[dict[str, Any]] = []
    for r, log in zip(read, logs, strict=True):
        entries = _Entries(log)
        for leaf in leaves:
            rows.append({"record": r.record_hash, **leaf.row(document, entries)})
    return rows


class _Leaf:
    """One path's screen, its value-free work done once: its type checked, each axis it varies
    with its window, the components it reaches unseen, the sources whose wording it fills, and
    the values in its windows whose design does not load."""

    def __init__(self, loaded: LoadedDesign, document: _Doc, path: str, width: object) -> None:
        self.path = path
        t = _grid_type(path, loaded)  # unknown-path, unsupported-leaf
        inner = t.value if isinstance(t, MappingOf) else t
        if _finite(inner) is not None:
            raise EvaluationRefusal(
                f"the leaf at {path} is a boolean or label, which has no window",
                reason=_R.unsupported_leaf,
                path=path,
            )
        if type(width) not in (int, float):
            raise TypeError(f"within[{path!r}] is a {type_name(width)}, not an int or float")
        if not math.isfinite(cast(float, width)) or cast(float, width) < 0:
            raise ValueError(f"within[{path!r}] is {width!r}, not a finite number of at least 0")
        self.width = cast("int | float", width)
        leaf = _leaf_of(path)
        own = loaded._leaf_value(leaf)
        keyed = key_path(path)
        # Each axis: the key, or None for a plain leaf; the path its comparisons name; own value.
        self.axes: list[tuple[str | None, str, Fraction]]
        if keyed is not None:
            held = cast("dict[str, Any]", own)
            self.axes = [(keyed[1], path, exact(held[keyed[1]]))]
            keys: set[str] | None = {keyed[1]}
        elif isinstance(t, MappingOf):
            held = cast("dict[str, Any]", own)
            self.axes = [(k, f"{path}.{k}", exact(held[k])) for k in code_point_sorted(held)]
            keys = set(held)
        else:
            self.axes = [(None, path, exact(cast(float, own)))]
            keys = None
        numeric = cast("tuple[str, tuple[int | float, int | float] | None]", _numeric(inner))
        self.integer = numeric[0] == "integer"
        self.windows = [_window(x, self.width, self.integer, numeric[1]) for _, _, x in self.axes]
        self.unseen = set(_unseen_keys(document, loaded, leaf, keys))
        self.worded = _wording_sources(loaded, leaf)
        self.not_loading = _not_loading(loaded, leaf, self.axes, self.windows, self.integer)

    def row(self, document: _Doc, entries: _Entries) -> dict[str, Any]:
        near: list[dict[str, Any]] = []
        unseen = set(self.unseen)
        for (key, at, _), (lo, hi) in zip(self.axes, self.windows, strict=True):
            found, blind = _comparisons(document, at, entries)
            unseen |= blind
            by_value: dict[Fraction, tuple[set[str], set[str | None]]] = {}
            for name, side, x in found:
                names, sides = by_value.setdefault(x, (set(), set()))
                names.add(name)
                sides.add(side)
            for x in sorted(by_value):
                names, sides = by_value[x]
                if _separates(x, sides, lo, hi):
                    near.append(
                        {
                            "key": key,
                            "value": self.form(x),
                            "components": code_point_sorted(names),
                        }
                    )
        not_loading = [{"key": key, "value": self.form(x)} for key, x in self.not_loading]
        clear = not near and not unseen and not self.worded and not not_loading
        return {
            "parameter": self.path,
            "within": self.width,
            "clear": clear,
            "near": near,
            "unseen": code_point_sorted(unseen),
            "worded": list(self.worded),
            "not_loading": not_loading,
        }

    def form(self, x: Fraction) -> int | float:
        """A value as the leaf's type holds it: an ``int`` on an integer axis, else a ``float``."""
        return int(x) if self.integer and x.denominator == 1 else float(x)


def _window(
    own: Fraction, width: int | float, integer: bool, interval: tuple[Any, Any] | None
) -> tuple[Fraction, Fraction]:
    """The values within ``width`` of ``own`` the leaf's type holds, as a closed interval: the
    integers on an integer axis, clipped to the type's interval."""
    w = exact(width)
    lo, hi = own - w, own + w
    if integer:
        lo, hi = Fraction(math.ceil(lo)), Fraction(math.floor(hi))
    if interval is not None:
        lo, hi = max(lo, exact(interval[0])), min(hi, exact(interval[1]))
    return lo, hi


def _separates(x: Fraction, sides: set[str | None], lo: Fraction, hi: Fraction) -> bool:
    """Whether a compared value can part two values of the window: one inside it, or at an end
    whose tie does not join the window's inside, ``high`` at its low end and ``low`` at its
    high end."""
    if x < lo or x > hi or lo == hi:
        return False
    return not ((x == lo and sides == {_HIGH}) or (x == hi and sides == {_LOW}))


def _wording_sources(loaded: LoadedDesign, leaf: str) -> list[str]:
    """Each source, in code-point order, one of whose questions' wording takes a slot from the
    leaf: its question differs at every other value, so the record cannot answer it."""
    out: set[str] = set()
    for sid, source in loaded._design.sources.items():
        for provided in source.provides.values():
            if isinstance(provided, Question):
                for slot in provided.wording.slots:
                    if cast("Mapping[str, str]", slot["origin"]).get("param") == leaf:
                        out.add(sid)
    return code_point_sorted(out)


def _not_loading(
    loaded: LoadedDesign,
    leaf: str,
    axes: list[tuple[str | None, str, Fraction]],
    windows: list[tuple[Fraction, Fraction]],
    integer: bool,
) -> list[tuple[str | None, Fraction]]:
    """Each value in an axis's window whose design fails load, with its key: none when no check
    reads the leaf's value; else the design is loaded at each cut in the window, a number at
    which a check's result can change, and once between each two, the result being the same
    between two cuts."""
    if leaf not in loaded._read_leaves:
        return []
    numbers: set[Fraction] | None = None
    out: list[tuple[str | None, Fraction]] = []
    for (key, _, _), (lo, hi) in zip(axes, windows, strict=True):
        if numbers is None:
            numbers = _design_numbers(loaded, leaf)
        cuts = {c + k for c in numbers for k in range(-2, 3) if lo <= c + k <= hi} | {
            Fraction(n) for n in range(math.ceil(lo), math.floor(hi) + 1)
        }
        if not cuts:
            continue  # one piece, holding the leaf's own value, which loads
        for x in _tried(sorted(cuts | {lo, hi}), integer):
            value: object = int(x) if integer else float(x)
            if key is not None:
                value = {**cast("dict[str, Any]", loaded._leaf_value(leaf)), key: value}
            try:
                with_parameters(loaded, {leaf: value})
            except InvalidDesign:
                out.append((key, x))
    return out


def _tried(cuts: list[Fraction], integer: bool) -> list[Fraction]:
    """A value the leaf can hold at each cut it can take and one in each gap between two."""
    out: list[Fraction] = []
    for c in cuts:
        if _holds(c, integer):
            out.append(c)
    for a, b in itertools.pairwise(cuts):
        mid = (a + b) / 2
        between = (Fraction(math.floor(mid)), Fraction(math.ceil(mid)))
        for m in between if integer else (exact(float(mid)),):
            if a < m < b:
                out.append(m)
                break
    return sorted(set(out))


def _holds(x: Fraction, integer: bool) -> bool:
    """Whether the leaf can hold ``x`` exactly: an integer, or a ``float``."""
    if integer:
        return x.denominator == 1
    try:
        return exact(float(x)) == x
    except OverflowError:
        return False


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _design_numbers(loaded: LoadedDesign, leaf: str) -> set[Fraction]:
    """Every number the design's document holds but the leaf's own value, as a value or
    written in a string such as a type's interval: a superset of the numbers a load check
    compares the leaf against."""
    document = cast("_Doc", decode(loaded.document))
    document["parameters"][leaf]["value"] = None  # the leaf meets its own value nowhere
    out: set[Fraction] = set()
    for x in _walk(document):
        if type(x) is int or type(x) is float:
            if math.isfinite(x):
                out.add(exact(x))
        elif type(x) is str:
            for s in _NUMBER.findall(x):
                out.add(Fraction(s))
                try:
                    f = float(s)
                except OverflowError:  # pragma: no cover - float() of a decimal gives inf
                    continue
                if math.isfinite(f):
                    out.add(exact(f))
    return out


def _walk(value: object) -> Iterator[object]:
    """Every value within ``value``, through dicts and lists, keys included."""
    stack = [value]
    while stack:
        item = stack.pop()
        yield item
        if type(item) is dict:
            for k, v in cast("dict[str, object]", item).items():
                stack += (k, v)
        elif type(item) is list:
            stack += cast("list[object]", item)
