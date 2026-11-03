# pyright: reportPrivateUsage=false
"""A numeric axis: the points worth trying along one number, from the boundaries it is compared
against, shared by the lints' read points and the switching-value job's probes."""

import itertools
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction

from ommat._types import Type, _numeric
from ommat._values import MAX_INTEGER


def _exact(x: float) -> Fraction:
    """The shortest decimal that reads back as ``x``, exactly."""
    return Fraction(repr(x))


def axis(
    bounds: Iterable[Fraction], lo: Fraction | None, hi: Fraction | None, *, integer: bool
) -> list[Fraction]:
    """An axis's points, exact and ascending: every boundary, one inside each gap (the
    midpoint, or on an integer axis the least integer above the lower boundary, a gap
    holding none skipped) and one past each end (the type's bound, else the end boundary
    minus or plus 1, on an integer axis the nearest integer beyond it). On an integer axis a
    boundary that is not integral, or of magnitude above 2^53 - 1, is no point, but the
    integers either side of it within those bounds are, and an end past that bound is the
    bound. A boundary outside the interval is not on the axis; an axis so left with no
    boundary has the interval's two bounds, or on an unbounded integer axis the domain's. An
    axis given no boundary has no points. Rounding is the caller's."""
    given = set(bounds)
    on = {
        b
        for b in given
        if (not integer or (b.denominator == 1 and abs(b) <= MAX_INTEGER))
        and (lo is None or lo <= b)
        and (hi is None or b <= hi)
    }
    if integer:  # each side of a boundary between two integers
        for b in given:
            if b.denominator != 1 and (lo is None or lo <= b) and (hi is None or b <= hi):
                for n in (math.floor(b), math.ceil(b)):
                    side = Fraction(max(-MAX_INTEGER, min(n, MAX_INTEGER)))
                    if (lo is None or lo <= side) and (hi is None or side <= hi):
                        on.add(side)
    ordered = sorted(on)
    if not ordered:
        if given and integer and lo is None and hi is None:
            return [Fraction(-MAX_INTEGER), Fraction(MAX_INTEGER)]
        return sorted({lo, hi}) if given and lo is not None and hi is not None else []
    points = set(ordered)
    for a, b in itertools.pairwise(ordered):
        if integer:
            n = Fraction(math.floor(a) + 1)
            if n < b:
                points.add(n)
        else:
            points.add((a + b) / 2)
    first, last = ordered[0], ordered[-1]
    if lo is None:
        lo = Fraction(max(math.ceil(first) - 1, -MAX_INTEGER)) if integer else first - 1
    if hi is None:
        hi = Fraction(min(math.floor(last) + 1, MAX_INTEGER)) if integer else last + 1
    points |= {lo, hi}
    return sorted(points)


def numeric_axis(bounds: Sequence[Fraction], t: Type) -> list[int | float]:
    """A numeric type's axis over ``bounds``, each point rounded once to the type's form: an
    ``integer`` or integer interval's an ``int``, a ``number``'s a ``float``, its ends past an
    unbounded ``number``'s boundaries by ``_end``. Given no boundary, the interval's two bounds,
    or on an unbounded type ``-(2^53 - 1)`` and ``2^53 - 1``. Repeats are the caller's."""
    numeric = _numeric(t)
    assert numeric is not None  # the caller's type is numeric
    integer, interval = numeric[0] == "integer", numeric[1]
    lo: Fraction | None = None
    hi: Fraction | None = None
    if interval is not None:
        lo, hi = _exact(interval[0]), _exact(interval[1])
    if not bounds:
        ends = (lo, hi) if lo is not None and hi is not None else (-MAX_INTEGER, MAX_INTEGER)
        points: Sequence[Fraction | int] = ends
    else:
        if not integer and lo is None:
            lo, hi = _end(min(bounds), -1), _end(max(bounds), 1)
        points = axis(bounds, lo, hi, integer=integer)
    return [int(x) if integer else float(x) for x in points]


def _end(x: Fraction, step: int) -> Fraction:
    """A number axis's end past ``x``: ``x`` plus ``step``, or the nearest ``float`` beyond
    ``x`` when that rounds onto it; ``x`` itself at the largest ``float``, past which there
    is none."""
    if float(x + step) != float(x):
        return x + step
    beyond = math.nextafter(float(x), step * math.inf)
    return x if math.isinf(beyond) else Fraction(beyond)
