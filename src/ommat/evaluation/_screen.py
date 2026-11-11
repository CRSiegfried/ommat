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

A value whose design does not load decides nothing. Past its type's interval, or with none past
+-(2^53 - 1) on an integer axis and the finite ``float``s on a number axis, the leaf holds no
value, so the window stops there. Within it, load reads a numeric leaf's value only in check
9's tests of a table's cells: a ``between`` holding the leaf, its low bound above its high
one, and two rows of a ``U`` or ``A`` table naming the leaf that come to overlap. Their result
changes only where the leaf crosses a number they compare it with, a number the cells hold or
name by another leaf, or for an overlap a bound of an input's type, or crosses an integer
from two below such a number's floor to two above its ceiling; between two such cuts it
changes only with whether the leaf is an integer, which a comparison over an integer input
sees. So the screen loads no value when no check reads the leaf, and otherwise the window's
ends, each cut, and between each two an integer and a value that is none, and names each
value that fails.

The screen trusts the record as the run of ``loaded``: it reads it after the record checks and
replays nothing, so a record the switching job would refuse ``baseline-differs`` is screened as
written. A function rule's ``boundaries`` are trusted as the switching job trusts them.
"""

import itertools
import math
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode
from ommat._design import InvalidDesign, LoadedDesign
from ommat._load import with_parameters
from ommat._refs import code_point_sorted, key_path
from ommat._types import (
    Interval,
    ListOf,
    MappingOf,
    Nullable,
    RecordOf,
    TypeFailure,
    _numeric,
    parse_type,
)
from ommat._values import MAX_INTEGER, type_name
from ommat.evaluation._diff import _read, _refuse_unfit
from ommat.evaluation._edit import _designs
from ommat.evaluation._grid import _grid_type, _leaf_of, _unseen_keys, _wording
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._switching import (
    _HIGH,
    _LOW,
    _comparisons,
    _components,
    _contains,
    _Entries,
    _finite,
)
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
    """One path's screen, its value-free work done once: its type and width checked, each axis
    it varies with its window, the components the design lets it reach unseen, the sources
    whose wording it fills, and the values in its window whose design does not load."""

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
        if (type(width) is float and not math.isfinite(width)) or cast(float, width) < 0:
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
        self.not_loading: list[tuple[str | None, Fraction]] = []
        if keyed is None and not isinstance(t, MappingOf):  # a cell compares a plain leaf alone
            (window,) = self.windows
            self.not_loading = [
                (None, x) for x in _not_loading(loaded, document, leaf, window, self.integer)
            ]

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
        """A value as a row gives it: an integer as an ``int`` on an integer axis, else a
        ``float``."""
        return int(x) if self.integer and x.denominator == 1 else float(x)


def _window(
    own: Fraction, width: int | float, integer: bool, interval: tuple[Any, Any] | None
) -> tuple[Fraction, Fraction]:
    """The values within ``width`` of ``own`` the leaf's type holds, as a closed interval: the
    integers on an integer axis, clipped to the type's interval, or to +-(2^53 - 1) on an
    integer axis and the finite ``float``s on a number axis."""
    w = exact(width)
    lo, hi = own - w, own + w
    if integer:
        lo, hi = Fraction(math.ceil(lo)), Fraction(math.floor(hi))
        ends = (Fraction(-MAX_INTEGER), Fraction(MAX_INTEGER))
    else:
        ends = (-_MAX_FLOAT, _MAX_FLOAT)
    if interval is not None:
        ends = (exact(interval[0]), exact(interval[1]))
    return max(lo, ends[0]), min(hi, ends[1])


_MAX_FLOAT = Fraction(sys.float_info.max)


def _separates(x: Fraction, sides: set[str | None], lo: Fraction, hi: Fraction) -> bool:
    """Whether a compared value can part two values of the window: one inside it, or one at an
    end unless its tie joins the inside, every side ``high`` at the low end or ``low`` at the
    high end; none parts a window of one value."""
    if x < lo or x > hi or lo == hi:
        return False
    return not ((x == lo and sides == {_HIGH}) or (x == hi and sides == {_LOW}))


def _wording_sources(loaded: LoadedDesign, leaf: str) -> list[str]:
    """Each source, in code-point order, one of whose questions' wording takes a slot from the
    leaf: its question differs at every other value, so the record cannot answer it."""
    return code_point_sorted(_wording(loaded).get(leaf, set()))


def _not_loading(
    loaded: LoadedDesign,
    document: _Doc,
    leaf: str,
    window: tuple[Fraction, Fraction],
    integer: bool,
) -> list[Fraction]:
    """Each value in the window whose design fails load: none when no check reads the leaf's
    value; else the design is loaded at the window's ends, at each cut, and between each two at
    an integer and at a value that is none, the result being the same for each between two.
    A cut is a number a check compares the leaf with, and the integers from two below its
    floor to two above its ceiling, where a comparison over an integer input can change."""
    numbers = _cut_numbers(loaded, document, leaf)
    if numbers is None:
        return []
    lo, hi = window
    cuts: set[Fraction] = {lo, hi}
    for c in numbers:
        near = {c} | {Fraction(b + k) for b in (math.floor(c), math.ceil(c)) for k in range(-2, 3)}
        cuts |= {x for x in near if lo <= x <= hi}
    out: list[Fraction] = []
    for x in _tried(sorted(cuts), integer):
        try:
            with_parameters(loaded, {leaf: int(x) if integer else float(x)})
        except InvalidDesign:
            out.append(x)
    return out


def _cut_numbers(loaded: LoadedDesign, document: _Doc, leaf: str) -> set[Fraction] | None:
    """The numbers a load check compares the leaf with, ``None`` when none reads its value: the
    bounds of each ``between`` holding the leaf, and, for each ``U`` or ``A`` table one of
    whose cells names it, every number its cells hold or name by another leaf, and every bound
    of a type the design declares or a helper returns."""
    out: set[Fraction] = set()
    reads = False
    for c in _components(document).values():
        if c["form"] != "table":
            continue
        rows = cast("list[dict[str, Any]]", c["rows"])
        cells: list[object] = [cell for row in rows for cell in row["cells"]]
        if not _contains(cells, {"param": leaf}):
            continue
        for cell in cells:
            pair = cast("dict[str, Any]", cell).get("between") if type(cell) is dict else None
            if pair is not None and {"param": leaf} in pair:
                reads = True
                out |= _numbers(loaded, pair, leaf)
        if c["hit"] in ("U", "A"):
            reads = True
            out |= _numbers(loaded, cells, leaf) | _type_bounds(loaded)
    return out if reads else None


def _numbers(loaded: LoadedDesign, value: object, leaf: str) -> set[Fraction]:
    """Every number within ``value``, through dicts and lists, a leaf named as ``{"param"}``
    by its value; the leaf itself by none."""
    out: set[Fraction] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is int or type(item) is float:
            out.add(exact(item))
        elif type(item) is dict:
            held = cast("dict[str, object]", item)
            path = held.get("param")
            if type(path) is str and len(held) == 1:
                if path != leaf:
                    stack.append(loaded._leaf_value(path))
            else:
                stack += held.values()
        elif type(item) is list:
            stack += cast("list[object]", item)
    return out


def _type_bounds(loaded: LoadedDesign) -> set[Fraction]:
    """Every interval bound of a type the design declares, a definition's, an entry's or a
    leaf's, and of the types the helpers return: a table input's axis is bounded by one."""
    types: list[object] = [*loaded._types.values(), *loaded._leaf_types.values()]
    for text in loaded._design.types.values():
        parsed = parse_type(text, nullable=True)
        if not isinstance(parsed, TypeFailure):
            types.append(parsed)
    out = {Fraction(0), Fraction(1), Fraction(MAX_INTEGER)}  # mass's [0, 1], size's [0, 2^53-1]
    while types:
        t = types.pop()
        if isinstance(t, Interval):
            out |= {exact(t.lo), exact(t.hi)}
        elif isinstance(t, ListOf):
            types.append(t.item)
        elif isinstance(t, MappingOf):
            types.append(t.value)
        elif isinstance(t, Nullable):
            types.append(t.inner)
        elif isinstance(t, RecordOf):
            types += [f for _, f in t.fields]
    return out


def _tried(cuts: list[Fraction], integer: bool) -> list[Fraction]:
    """A value the leaf can hold at each cut it can take and, in each gap between two, an
    integer and, on a number axis, a value that is no integer."""
    out: list[Fraction] = [c for c in cuts if _holds(c, integer)]
    for a, b in itertools.pairwise(cuts):
        whole = Fraction(math.floor(a) + 1)
        if whole < b:
            out.append(whole)
        if not integer:
            for m in (exact(float((a + b) / 2)), exact(float((3 * a + b) / 4))):
                if a < m < b and m.denominator != 1:
                    out.append(m)
                    break
    return sorted(set(out))


def _holds(x: Fraction, integer: bool) -> bool:
    """Whether the leaf can hold ``x`` exactly: an integer, or a ``float``."""
    return x.denominator == 1 if integer else exact(float(x)) == x
