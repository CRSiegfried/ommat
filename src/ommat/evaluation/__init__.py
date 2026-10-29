"""The evaluation subpackage: jobs that measure a design against past turns through replay,
and the two stores they keep their inputs and results in. The names ``__all__`` lists are
public, under the same deprecation rule as ``ommat.__all__``. Do-and-diff's six jobs, the
deciding inputs, the switching-value job and the sweep job are shipped, each async with a
``_sync`` twin, and the gold-answers job and the correlated-group confirmation job, sync.

``set_entry``, ``fail_call``, ``set_state_field`` and ``constant_rule`` each change one input
of a recorded turn, an entry a call wrote, a whole call, a ``t:state`` field or a rule made
constant, and ``intervene`` several at once; each replays the change once and returns what it
edited beside replay's findings, the changes downstream. ``use_answers`` reruns one record's
turn with another record's calls answering. ``deciding_inputs`` classes each declared read of
the rule providing an entry by whether some single edit on it, as the four single-edit jobs make
one, changes that entry: ``deciding``, ``not-deciding``, ``markers-only`` or
``not-determined``. ``switching_values`` replays a record under each value of one parameter
leaf, probed at the values the turn compares it against and the gap and end values between and
beyond them, and writes, per entry and for the plan, the switching values, slacks and classes,
or a ``boolean`` or ``label`` leaf's partition, to the evaluation store.
``write_gold_answers``, sync, writes a labeller's labels for the questions finished turns asked
under a named gold set, copying an old set's for the questions given none. ``sweep`` replays
every turn of a dev and a certification split under each point of a sweep document, the
starting design with some parameter leaves replaced, scores each against the plan its gold
answers give at that point, and chooses the last point the exact binomial test certifies on the
certification split, by fixed-sequence testing when the dev split's escaped counts never fall
along the points, else Bonferroni; it keeps the chosen design in the design store and writes the
counts and the choice as a ``certificate`` entry. ``confirm_group``, sync, counts for each pair
of a declared correlated group's ``q:`` refs, over a split's turns, or turns and candidates for
a grounded group, in which both hold an answer, the units in which each erred against its gold
answer and those in which both did, and writes them as a ``group-counts`` entry, giving no
verdict. ``DesignStore`` keeps a loaded design's document under its design hash and fetches it
back through ``load``; ``EvaluationStore`` keeps write-once entries by kind and key. A job or
a store refuses with ``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

The edits are in ``_edit``, the deciding inputs, tried through them, in ``_deciding``, the
switching-value job in ``_switching``, the gold-answers job in ``_gold``, the sweep job in
``_sweep``, the correlated-group confirmation job in ``_group``, ``use_answers`` and the
record checks and replay they share in ``_diff``, the stores in ``_store`` and the refusal in
``_refusal``.
"""

from ommat.evaluation._deciding import deciding_inputs, deciding_inputs_sync
from ommat.evaluation._diff import use_answers, use_answers_sync
from ommat.evaluation._edit import (
    constant_rule,
    constant_rule_sync,
    fail_call,
    fail_call_sync,
    intervene,
    intervene_sync,
    set_entry,
    set_entry_sync,
    set_state_field,
    set_state_field_sync,
)
from ommat.evaluation._gold import write_gold_answers
from ommat.evaluation._group import confirm_group
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import DesignStore, EvaluationStore
from ommat.evaluation._sweep import sweep, sweep_sync
from ommat.evaluation._switching import switching_values, switching_values_sync

__all__ = [
    "DesignStore",
    "EvaluationRefusal",
    "EvaluationStore",
    "RefusalReason",
    "confirm_group",
    "constant_rule",
    "constant_rule_sync",
    "deciding_inputs",
    "deciding_inputs_sync",
    "fail_call",
    "fail_call_sync",
    "intervene",
    "intervene_sync",
    "set_entry",
    "set_entry_sync",
    "set_state_field",
    "set_state_field_sync",
    "sweep",
    "sweep_sync",
    "switching_values",
    "switching_values_sync",
    "use_answers",
    "use_answers_sync",
    "write_gold_answers",
]
