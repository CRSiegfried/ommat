"""The helpers subpackage: the helper library, the choice helpers and combinators that a function
rule calls with its read view; the design-check lint, ``dead_questions``, which the consumer's
tests run over a loaded design, and ``points``, the read points it ranges a read over;
``decide_many``, which decides one turn per state, many at once; and ``combine``, which unions
parts, each a design and its ports, into one design, with ``standalone``, which makes one part
loadable alone. The names ``__all__`` lists are public, under the same deprecation rule as
``ommat.__all__``.

``HELPERS`` is the helper library's version, which the record's ``design`` item carries; it
counts the library's helpers, ``determined`` among them, and combinators alone, never the lint,
``decide_many`` or ``combine``.

The choice helpers are in ``_choice``, the combinators in ``_combinator``, and the mechanics
each of them is built on in ``_call``. ``decide_many`` is in ``_many`` and ``combine`` in
``_combine``. The lint is in ``_dead``, over the read points in ``_points``, which also gives
``points``, one read's points in record form.
"""

from ommat.helpers._choice import (
    extract,
    mass,
    pick_each_with_margin,
    pick_each_with_thresholds,
    pick_with_margin,
    pick_with_thresholds,
    size,
    top,
)
from ommat.helpers._combinator import agree, collect, first, priority, unique
from ommat.helpers._combine import (
    CombineFailure,
    InvalidCombination,
    Part,
    combine,
    standalone,
)
from ommat.helpers._dead import dead_questions
from ommat.helpers._determined import determined
from ommat.helpers._many import TurnFailure, decide_many
from ommat.helpers._points import points

__all__ = [
    "CombineFailure",
    "InvalidCombination",
    "Part",
    "TurnFailure",
    "agree",
    "collect",
    "combine",
    "dead_questions",
    "decide_many",
    "determined",
    "extract",
    "first",
    "mass",
    "pick_each_with_margin",
    "pick_each_with_thresholds",
    "pick_with_margin",
    "pick_with_thresholds",
    "points",
    "priority",
    "size",
    "standalone",
    "top",
    "unique",
]

HELPERS = 4
