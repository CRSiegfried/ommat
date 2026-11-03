"""The helpers subpackage: the helper library, the choice helpers and combinators that a function
rule or the policy calls with its read view; the design-check lints, ``dead_questions`` and
``unsure_to_execute``, which the consumer's tests run over a loaded design, and ``points``, the
read points they range a read over; and the reference runner, ``run_plan``, which the caller
runs after decide, and ``decide_many``, which decides one turn per state, many at once. The
names ``__all__`` lists are public, under the same deprecation rule as ``ommat.__all__``.

``HELPERS`` is the helper library's version, which the record's ``design`` item carries; it
counts the library's helpers, ``determined`` among them, and combinators alone, never a lint, the
runner or ``decide_many``.

The choice helpers are in ``_choice``, the combinators in ``_combinator``, and the mechanics
each of them is built on in ``_call``. ``decide_many`` is in ``_many``. The lints are in
``_dead`` and ``_unsure``, over the read points both share in ``_points``, which also gives
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
)
from ommat.helpers._combinator import agree, collect, first, priority, unique
from ommat.helpers._dead import dead_questions
from ommat.helpers._determined import determined
from ommat.helpers._many import TurnFailure, decide_many
from ommat.helpers._points import points
from ommat.helpers._runner import BindingNotFound, RunRefusal, run_plan
from ommat.helpers._unsure import unsure_to_execute

__all__ = [
    "BindingNotFound",
    "RunRefusal",
    "TurnFailure",
    "agree",
    "collect",
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
    "run_plan",
    "size",
    "unique",
    "unsure_to_execute",
]

HELPERS = 3
