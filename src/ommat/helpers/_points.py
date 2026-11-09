# pyright: reportPrivateUsage=false
"""The read points: each read's fill and the grid order the lints run them in, and ``points``,
the public generator giving one read's points in record form.

Both lints call ``fill`` with the reads they fill, and it raises one ``ValueError`` listing
every slip in ``values`` and every unfilled read before either lint runs a point. A read is
filled from the first of: its grid points, from load's grid and the parameter block through
``_leaf_value``, a key path's from its leaf's value at the key; its declared type's values; its
list in ``values``. A record's type fill is the cross product of its fields' fills, each field
filled by the same order under its field path. A label question with no grid entry, other than
a candidate-valued one, is filled from its list in ``values`` when it has one, each label
listed at 1 in declared order, and is then narrowed. An extraction read's points cross its
value points, from its grid's ``value`` axis or its list in ``values``, with its ``p_null`` and
``confidence`` points, then add the two null points. A ``q:`` read's points end with
``Unavailable`` ``refused``. A per-candidate question is filled once, on its declared ref, and
each grounded entry takes those points with a marker of its own last.

Every point is computed exactly over each boundary's ``Fraction(repr(x))`` and rounded once
to the nearest ``float``. A point is held in two forms: the record form a report shows and
the live form a board or binding is given, ``bytes`` as ``bytes`` and a ``q:`` value as the
binding's form of its kind, the probabilities or an extraction's five members.
"""

import copy
import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from ommat._answer import (
    Answer,
    AnswerKind,
    ExtractionAnswer,
    answer_document,
    binding_form,
    extraction_document,
)
from ommat._axis import _end, axis, numeric_axis
from ommat._canonical import encode
from ommat._design import LoadedDesign, Question
from ommat._interpret import _marker_document
from ommat._markers import Cause, CauseCode, Unavailable
from ommat._refs import code_point_sorted, key_path, parse_ref
from ommat._typed import typed
from ommat._types import (
    Label,
    Nullable,
    RecordOf,
    Scalar,
    Type,
    TypeFailure,
    read_type,
)
from ommat._values import (
    described,
    domain_error,
    is_scalar_string,
    quote,
    string_data,
    type_name,
)
from ommat.helpers._numbers import exact

_ZERO, _ONE = Fraction(0), Fraction(1)


@dataclass(frozen=True, slots=True)
class Point:
    """One value of a read: its record form and its live form."""

    record: object
    live: object


@dataclass(frozen=True, slots=True)
class OnTop:
    """A candidate-valued question's answer point: ``label`` at 1 and every other label the
    run renders at 0, or, for ``None``, the first candidate it renders."""

    label: str | None

    def over(self, labels: Sequence[str], at: int) -> dict[str, float]:
        """The answer over the labels a call renders, its first candidate at index ``at``."""
        top = labels[at] if self.label is None else self.label
        return {label: 1.0 if label == top else 0.0 for label in labels}


# ---------------------------------------------------------------------------------------------
# An axis


# ---------------------------------------------------------------------------------------------
# A read's type and question


def read_of(loaded: LoadedDesign, read: str) -> Type | Question:
    """A ``q:`` read's question, or a ``t:`` or ``d:`` read's type, a field path's included."""
    p = parse_ref(read)
    base = f"{p.level}:{p.name}"
    if p.level == "q":
        return question(loaded, base)
    t = read_type(loaded._types[base], p.fields, {})
    assert not isinstance(t, TypeFailure)  # load's check 9 passed every declared read
    return t


def question(loaded: LoadedDesign, ref: str) -> Question:
    """The question a source provides at ``ref``; load's checks 3 and 4 passed."""
    provided = loaded._design.sources[loaded._providers[ref]].provides[ref]
    return cast(Question, provided)


def labels_of(loaded: LoadedDesign, ref: str) -> tuple[str, ...]:
    """A question's label names in declared order: ``yes`` and ``no`` for ``noul``."""
    return loaded._labels[ref][0]


# ---------------------------------------------------------------------------------------------
# The fill


def points(loaded: LoadedDesign, ref: str, *, values: Sequence[object] | None = None) -> list[Any]:
    """The read points of ``ref`` in ``loaded``, in their order, each in record form and fresh:
    ``ref`` is ``t:state`` or a field path under it, or a declared ``t:``, ``q:`` or ``d:`` ref,
    a per-candidate question by its declared ref; ``values``, the read's own list, is taken
    where nothing else fills it, as the lints take a list in theirs. A ``TypeError`` for a
    ``loaded`` that is not exactly a ``LoadedDesign`` or a ``ref`` that is not a ``str``; a
    ``ValueError`` for a ref that names no read of ``loaded``, a ``values`` its grid points or
    its type already fill, or a read nothing fills."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    if type(ref) is not str:
        raise TypeError(f"ref is a {type_name(ref)}, not a str")
    if not _declared_read(loaded, ref):
        raise ValueError(f"{quote(ref)} names no read of the design")
    given = None if values is None else {ref: values}
    return [fresh(p.record) for p in fill(loaded, [ref], given)[ref]]


def _declared_read(loaded: LoadedDesign, ref: str) -> bool:
    """Whether ``ref`` is ``t:state``, a field path under it, or a declared ``t:``, ``q:`` or
    ``d:`` ref with no field path or bracket, as ``points`` takes one."""
    try:
        parsed = parse_ref(ref)
    except ValueError:
        return False
    base = f"{parsed.level}:{parsed.name}"
    if parsed.candidate is not None or (
        base not in loaded._types and base not in loaded._providers
    ):
        return False
    if parsed.fields and base != "t:state":
        return False
    if parsed.fields:
        return not isinstance(read_type(loaded._types[base], parsed.fields, {}), TypeFailure)
    return True


def fill(loaded: LoadedDesign, reads: Sequence[str], values: object) -> dict[str, list[Point]]:
    """Each read's points in its order, or one ``ValueError`` naming every slip in ``values``
    and every unfilled read, raised before any point runs."""
    return filled(loaded, reads, values)[0]


def filled(
    loaded: LoadedDesign, reads: Sequence[str], values: object
) -> tuple[dict[str, list[Point]], frozenset[str], list[str]]:
    """``fill``'s points, the ``q:`` reads their grid entry filled, and the questions
    ``values`` narrowed, in code-point order."""
    filler = _Filler(loaded, values)
    return filler.run(reads), frozenset(filler.on_grid), code_point_sorted(filler.narrowed)


def fill_both(
    loaded: LoadedDesign, first: Sequence[str], second: Sequence[str], values: object
) -> tuple[dict[str, list[Point]], dict[str, list[Point]], list[str]]:
    """Two fills over one ``values``, and the questions it narrowed in either: each key must
    be a read either fills, and one ``ValueError`` names every slip in either and every
    unfilled read, before any point runs."""
    fillers = [_Filler(loaded, values), _Filler(loaded, values)]
    fills = (fillers[0].collect(first), fillers[1].collect(second))
    _finish(fillers)
    return (*fills, code_point_sorted({r for f in fillers for r in f.narrowed}))


def _finish(fillers: Sequence["_Filler"]) -> None:
    """Raise one ``ValueError`` naming every slip the fillers found: a key none consumed, an
    unfilled read, and each problem a fill met, each once."""
    problems = list(dict.fromkeys(p for f in fillers for p in f.problems))
    for key in fillers[0].values:
        if any(key in f.consumed for f in fillers):
            continue
        if any(key in f.typed_paths for f in fillers):
            problems.append(f"values names {quote(key)}, which its grid points or its type fill")
        else:
            problems.append(f"values names {quote(key)}, a read the lint does not fill")
    unfilled = sorted({r for f in fillers for r in f.unfilled})
    if unfilled:
        names = ", ".join(quote(r) for r in unfilled)
        problems.append(f"unfilled: {names}; give each a list in values")
    if problems:
        raise ValueError("; ".join(problems))


class _Filler:
    def __init__(self, loaded: LoadedDesign, values: object) -> None:
        self.loaded = loaded
        self.problems: list[str] = []
        self.values = self.checked_values(values)
        self.consumed: set[str] = set()
        self.unfilled: list[str] = []
        self.typed_paths: set[str] = set()  # paths the grid or a type filled
        self.on_grid: set[str] = set()  # q: reads their grid entry filled
        self.narrowed: set[str] = set()  # label questions values filled

    def checked_values(self, values: object) -> dict[str, list[object]]:
        if values is None:
            return {}
        if not issubclass(type(values), Mapping):  # by its type alone, never __class__
            raise TypeError(f"values is a {type_name(values)}, not a mapping or None")
        out: dict[str, list[object]] = {}
        for key, listed in cast("Mapping[object, object]", values).items():
            if not is_scalar_string(key):
                self.problems.append(f"the values key {described(key)} is not a string")
                continue
            name = string_data(cast(str, key))
            if type(listed) is not list and type(listed) is not tuple:
                raise TypeError(f"values for {quote(name)} is not a list")
            out[name] = list(cast("Sequence[object]", listed))
        return out

    def run(self, reads: Sequence[str]) -> dict[str, list[Point]]:
        fills = self.collect(reads)
        _finish([self])
        return fills

    def collect(self, reads: Sequence[str]) -> dict[str, list[Point]]:
        fills: dict[str, list[Point]] = {}
        for read in reads:
            points = self.read(read)
            if points is not None:
                fills[read] = points
        return fills

    def read(self, read: str) -> list[Point] | None:
        found = read_of(self.loaded, read)
        if isinstance(found, Question) and found.kind is AnswerKind.extraction:
            points = self.extraction(read, found)
            return None if points is None else [*points, _refused(read)]
        if isinstance(found, Question):
            answers = self.answers(read, found)
            if read not in self.narrowed:
                self.typed_paths.add(read)
            return [*answers, _refused(read)]
        return self.path(read, found)

    def path(self, path: str, t: Type) -> list[Point] | None:
        """A read or field filled by the first of its grid points, its type, its values."""
        if path in self.loaded._grid:
            self.typed_paths.add(path)
            return self.numeric(path, t)
        unfilled, problems = len(self.unfilled), len(self.problems)
        consumed, typed_paths = set(self.consumed), set(self.typed_paths)
        points = self.by_type(path, t)
        if points is not None:
            self.typed_paths.add(path)
            return points
        if path in self.values:
            # A record whose fields did not all fill is filled whole: its fields' slips go.
            del self.unfilled[unfilled:]
            del self.problems[problems:]
            self.consumed, self.typed_paths = consumed, typed_paths
            return self.listed(path, t)
        if len(self.unfilled) == unfilled:
            self.unfilled.append(path)
        return None

    def by_type(self, path: str, t: Type) -> list[Point] | None:
        if isinstance(t, Nullable):
            inner = self.by_type(path, t.inner)
            return None if inner is None else [*inner, Point(None, None)]
        if isinstance(t, Label):
            return [Point(v, v) for v in t.values]
        if isinstance(t, Scalar) and t.name == "boolean":
            return [Point(False, False), Point(True, True)]
        if isinstance(t, RecordOf):
            fields: list[tuple[str, list[Point]]] = []
            whole = True
            for name, ft in t.fields:  # the order its type spells them
                points = self.path(f"{path}.{name}", ft)
                if points is None:
                    whole = False
                else:
                    fields.append((name, points))
            if not whole:
                return None
            names = [name for name, _ in fields]
            return [
                Point(
                    {n: p.record for n, p in zip(names, combo, strict=True)},
                    {n: p.live for n, p in zip(names, combo, strict=True)},
                )
                for combo in itertools.product(*(points for _, points in fields))
            ]
        return None

    def listed(self, path: str, t: Type) -> list[Point] | None:
        self.consumed.add(path)
        points: list[Point] = []
        seen: set[bytes] = set()
        for value in self.values[path]:
            forms = typed(value, t)
            if forms is None:
                self.problems.append(f"a value for {quote(path)} does not conform to its type")
                continue
            key = encode(forms[0])
            if key in seen:
                self.problems.append(f"values for {quote(path)} lists one value twice")
                continue
            seen.add(key)
            points.append(Point(*forms))
        if not self.values[path]:
            self.problems.append(f"values for {quote(path)} is empty")
        return points

    # --- grid points ---------------------------------------------------------------------

    def bound(self, boundary: str | int) -> Fraction:
        """A boundary's value: a leaf's from the block, a key path's at its key of its leaf's,
        or an integer literal's own."""
        if type(boundary) is int:
            return Fraction(boundary)
        keyed = key_path(boundary)
        if keyed is not None:
            held = cast("dict[str, float]", self.loaded._leaf_value(keyed[0]))
            return exact(held[keyed[1]])
        return exact(cast(float, self.loaded._leaf_value(cast(str, boundary))))

    def numeric(self, path: str, t: Type) -> list[Point]:
        """A numeric ``d:`` or ``t:`` read's axis, then null where a table's input admits it;
        load's check 9 made its type numeric."""
        if isinstance(t, Nullable):
            return [*self.numeric(path, t.inner), Point(None, None)]
        bounds = [self.bound(p) for p in cast("Sequence[str | int]", self.loaded._grid[path])]
        points: list[Point] = []
        for value in numeric_axis(bounds, t):
            forms = typed(value, t)
            assert forms is not None
            points.append(Point(*forms))
        return _once(points)  # a point that rounds onto another is no point

    def answers(self, ref: str, q: Question) -> list[Point]:
        """A ``q:`` read's answer points, before its marker."""
        labels = labels_of(self.loaded, ref)
        if ref in self.loaded._label_sets:  # filled as a read with no grid entry
            # Each fixed label on top, then a candidate, over the labels each run renders.
            return [Point({"on_top": label}, OnTop(label)) for label in (*labels, None)]
        entry = cast("Mapping[str, Any] | None", self.loaded._grid.get(ref))
        if entry is not None:
            if len(labels) == 1:  # one distribution, so one answer point
                points = [_answer(q, labels, {labels[0]: 1.0})]
            else:
                points = _once(self.distributions(q, labels, entry))
            if points:
                self.on_grid.add(ref)
                return points
        # Off the grid, or an entry yielding no answer point: each label at 1 in declared
        # order, or only the labels values lists.
        tried = self.narrowing(ref, labels) if ref in self.values else labels
        return [_answer(q, labels, _at(labels, {label: _ONE})) for label in tried]

    def narrowing(self, ref: str, labels: tuple[str, ...]) -> tuple[str, ...]:
        """The labels ``values`` lists for an off-grid question, in declared order."""
        self.consumed.add(ref)
        self.narrowed.add(ref)
        listed: set[str] = set()
        for value in self.values[ref]:
            if not is_scalar_string(value) or string_data(cast(str, value)) not in labels:
                self.problems.append(f"a value for {quote(ref)} is not one of its labels")
                continue
            label = string_data(cast(str, value))
            if label in listed:
                self.problems.append(f"values for {quote(ref)} lists one value twice")
                continue
            listed.add(label)
        if not self.values[ref]:
            self.problems.append(f"values for {quote(ref)} is empty")
        return tuple(label for label in labels if label in listed)

    def extraction(self, ref: str, q: Question) -> list[Point] | None:
        """An extraction read's answer points, before its marker: the cross of its value
        points, its ``p_null`` points and its ``confidence`` points, the last varying fastest,
        then ``value`` ``None`` with ``p_null`` 1 and with ``p_null`` ``None``; ``guess`` and
        ``span`` ``None`` in each. ``None`` when nothing fills its value."""
        entry = cast("Mapping[str, Any] | None", self.loaded._grid.get(ref))
        if entry is not None and entry["value"]:  # a scalar number question (check 9)
            self.typed_paths.add(ref)
            bounds = [self.bound(p) for p in entry["value"]]
            lo, hi = _end(min(bounds), -1), _end(max(bounds), 1)
            values: list[object] = [float(x) for x in axis(bounds, lo, hi, integer=False)]
        elif ref in self.values:
            values = self.extraction_values(ref, q)
        else:
            self.unfilled.append(ref)
            return None
        nulls: list[float | None] = [*self.unit_axis(entry, "null", 0.0), None]
        confidences: list[float | None] = [*self.unit_axis(entry, "confidence", 1.0), None]
        points = [_extracted(q, v, p, c) for v in values for p in nulls for c in confidences]
        points += [_extracted(q, None, 1.0, None), _extracted(q, None, None, None)]
        if entry is not None:
            self.on_grid.add(ref)
        return _once(points)

    def unit_axis(self, entry: Mapping[str, Any] | None, role: str, default: float) -> list[float]:
        """A ``number [0, 1]`` axis's points from a role's leaves, else ``default`` alone."""
        if entry is None or not entry[role]:
            return [default]
        bounds = [self.bound(p) for p in entry[role]]
        return list(dict.fromkeys(float(x) for x in axis(bounds, _ZERO, _ONE, integer=False)))

    def extraction_values(self, ref: str, q: Question) -> list[object]:
        """An extraction read's value points from ``values``: each a value of its question's
        type, list flag and grammar, as a binding returns it, none twice by canonical bytes."""
        self.consumed.add(ref)
        out: list[object] = []
        seen: set[bytes] = set()
        for value in self.values[ref]:
            if value is None:  # the null points stand apart; a value point holds a value
                self.problems.append(f"a value for {quote(ref)} does not conform to its type")
                continue
            try:
                answer = ExtractionAnswer(
                    cast(Any, q.value_type), q.is_list, value, None, None, None, None
                )
            except ValueError:
                self.problems.append(f"a value for {quote(ref)} does not conform to its type")
                continue
            if domain_error(extraction_document(answer)) is not None:
                self.problems.append(f"a value for {quote(ref)} is outside the value domain")
                continue
            held = binding_form(answer)["value"]
            key = encode(held)
            if key in seen:
                self.problems.append(f"values for {quote(ref)} lists one value twice")
                continue
            seen.add(key)
            out.append(held)
        if not self.values[ref]:
            self.problems.append(f"values for {quote(ref)} is empty")
        return out

    def distributions(
        self, q: Question, labels: tuple[str, ...], entry: Mapping[str, Any]
    ) -> list[Point]:
        bars = [self.bound(p) for p in entry["bar"]]
        margins = [self.bound(p) for p in entry["margin"]]
        masses = [
            (tuple(cast("Sequence[str]", m["labels"])), [self.bound(p) for p in m["bounds"]])
            for m in cast("Sequence[Mapping[str, Any]]", entry["mass"])
        ]
        if q.kind is AnswerKind.noul:
            return [_answer(q, labels, d) for d in _noul(bars, margins, masses)]
        thresholds = [
            {k: exact(v) for k, v in cast("dict[str, float]", self.loaded._leaf_value(p)).items()}
            for p in entry["thresholds"]
        ]
        return [_answer(q, labels, d) for d in _choice(labels, bars, thresholds, margins, masses)]


# ---------------------------------------------------------------------------------------------
# Answer points

_Distribution = dict[str, float]


def _rounded(exacts: Mapping[str, Fraction]) -> _Distribution:
    return {label: float(p) for label, p in exacts.items()}


def _at(labels: Sequence[str], placed: Mapping[str, Fraction]) -> _Distribution:
    """``placed`` on its labels and 0 on the rest, in label order."""
    return _rounded({label: placed.get(label, _ZERO) for label in labels})


def _noul(
    bars: list[Fraction],
    margins: list[Fraction],
    masses: list[tuple[tuple[str, ...], list[Fraction]]],
) -> list[_Distribution]:
    """The one p(yes) axis: bar, margin and mass boundaries on it, ascending."""
    on: list[Fraction] = []
    for b in bars:
        on += [b, 1 - b]
    for x in margins:
        on += [(1 + x) / 2, (1 - x) / 2]
    for labels, bounds in masses:
        if set(labels) == {"yes"}:
            on += bounds
        elif set(labels) == {"no"}:
            on += [1 - x for x in bounds]
    return [_rounded({"yes": p, "no": 1 - p}) for p in axis(on, _ZERO, _ONE, integer=False)]


def _choice(
    labels: tuple[str, ...],
    bars: list[Fraction],
    thresholds: list[dict[str, Fraction]],
    margins: list[Fraction],
    masses: list[tuple[tuple[str, ...], list[Fraction]]],
) -> list[_Distribution]:
    """Bar and threshold points, then margin points, label by label in declared order and
    ascending; then mass points, set by set; the caller drops a repeat."""
    n = len(labels)
    out: list[_Distribution] = []
    for label in labels:
        on = bars + [t[label] for t in thresholds if label in t]
        for p in axis(on, _ZERO, _ONE, integer=False):
            rest = (1 - p) / (n - 1)
            out.append(_rounded({k: p if k == label else rest for k in labels}))
    for i, label in enumerate(labels):
        after = labels[(i + 1) % n]
        for x in axis(margins, _ZERO, _ONE, integer=False):
            out.append(_at(labels, {label: (1 + x) / 2, after: (1 - x) / 2}))
    for members, bounds in masses:
        inside = [k for k in labels if k in members]  # declared order
        outside = [k for k in labels if k not in members]
        for x in axis(bounds, _ZERO, _ONE, integer=False):
            if (not outside and x != 1) or (not inside and x != 0):
                continue  # no distribution realises it
            placed = dict.fromkeys(inside, _ZERO)
            if inside:
                placed[inside[0]] = x
            placed.update({k: (1 - x) / len(outside) for k in outside})
            out.append(_at(labels, placed))
    return out


def _once(points: list[Point]) -> list[Point]:
    """Each point once, the first kept, by the canonical bytes of its record form."""
    seen: set[bytes] = set()
    out: list[Point] = []
    for p in points:
        key = encode(p.record)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _answer(q: Question, labels: tuple[str, ...], d: _Distribution) -> Point:
    answer = Answer(q.kind, labels, d)
    return Point(answer_document(answer), dict(answer.probabilities))


def _extracted(q: Question, value: object, p_null: float | None, confidence: float | None) -> Point:
    """An extraction answer point, ``guess`` and ``span`` ``None``."""
    answer = ExtractionAnswer(
        cast(Any, q.value_type), q.is_list, fresh(value), None, p_null, confidence, None
    )
    return Point(extraction_document(answer), binding_form(answer))


def entry_points(points: Sequence[Point], ref: str) -> list[Point]:
    """A grounded entry's points: its declaration's, a question's marker last naming ``ref``;
    a grounded ``t:`` ref's as its declaration's, which hold no marker."""
    if not ref.startswith("q:"):
        return list(points)
    return [*points[:-1], _refused(ref)]


def _refused(ref: str) -> Point:
    """The marker a binding may return, last in a ``q:`` read's points."""
    marker = Unavailable(Cause(CauseCode.refused, [ref]))
    return Point(_marker_document(marker), marker)


# ---------------------------------------------------------------------------------------------
# Grid order and the report's form


def grid_order(fills: Sequence[list[Point]]) -> Iterator[tuple[Point, ...]]:
    """The cross product of the fills, given in code-point order of their reads, the last
    varying fastest."""
    return itertools.product(*fills)


def fresh(value: object) -> object:
    """A copy no change reaches back from: a board, a binding or a report gets its own."""
    return copy.deepcopy(value)
