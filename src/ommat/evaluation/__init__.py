"""The evaluation subpackage: jobs that measure a design against past turns through replay,
and the two stores they keep their inputs and results in. The names ``__all__`` lists are
public, under the same deprecation rule as ``ommat.__all__``. Do-and-diff's six jobs, the
deciding inputs and the switching-value job are shipped, each async with a ``_sync`` twin, and
the gold-answers job, sync.

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
under a named gold set, copying an old set's for the questions given none. ``DesignStore`` keeps
a loaded design's document under its design hash and fetches it back through ``load``;
``EvaluationStore`` keeps write-once entries by kind and key. A job or a store refuses with
``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

The edits are in ``_edit``, the deciding inputs, tried through them, in ``_deciding``, the
switching-value job in ``_switching``, the gold-answers job in ``_gold``, ``use_answers`` and
the record checks and replay they share in ``_diff``, the stores in ``_store`` and the refusal
in ``_refusal``.
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
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import DesignStore, EvaluationStore
from ommat.evaluation._switching import switching_values, switching_values_sync

__all__ = [
    "DesignStore",
    "EvaluationRefusal",
    "EvaluationStore",
    "RefusalReason",
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
    "switching_values",
    "switching_values_sync",
    "use_answers",
    "use_answers_sync",
    "write_gold_answers",
]
