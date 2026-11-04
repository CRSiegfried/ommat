"""The evaluation subpackage: jobs that measure a design against past turns through replay,
and the two stores they keep their inputs and results in. The names ``__all__`` lists are
public, under the same deprecation rule as ``ommat.__all__``. Do-and-diff's three jobs, the
deciding inputs, the switching-value job, the question-reach job and the gold record are
shipped, each async with a ``_sync`` twin, and the gold-answers job, sync.

``intervene`` replays a copy of a recorded turn once under a list of interventions, each setting an
entry a call wrote, failing a whole call, setting a ``t:state`` field, making a rule constant or
asking a source afresh, and under a changed design when given; it returns the replay record,
replay's findings, the changes downstream, and the stale refs. ``scenario`` is ``intervene`` with
the stale questions asked live of the caller's bindings and every other answer pinned from the
record. ``use_answers`` reruns one record's turn with another record's calls answering.
``deciding_inputs`` classes each declared read of the rule providing an entry by whether some single
intervention on it changes that entry: ``deciding``, ``not-deciding``, ``markers-only`` or
``not-determined``. ``switching_values`` replays a record under each value of one parameter leaf,
probed at the values the turn compares it against and the gap and end values between and beyond
them, and writes, per entry, the policy's among them, the switching values, slacks and classes, or a
``boolean`` or ``label`` leaf's partition, to the evaluation store. ``write_gold_answers``, sync,
writes a labeller's labels for the questions finished turns asked under a named gold set, copying an
old set's for the questions given none. ``gold_record`` replays a turn with every question it asked
set to its gold answer, so its decision is what correct answers lead to. ``answers``, sync, gives
the answers table, one row per question a call answered in each record, with its gold answer and
whether it erred when a gold set is given. ``question_reach`` tries each question's answer, changed
alone at each of its read points by an entry set, in every ``q:`` entry a returned call wrote in the
records given, and reports a question ``live`` with its first witness when some trial changes the
decision, else ``not-determined``, ``no-change-found`` or ``never-asked``. ``DesignStore`` keeps a
loaded design's document under its design hash and fetches it back through ``load``;
``EvaluationStore`` keeps write-once entries by kind and key. A job or a store refuses with
``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

``intervene`` and ``scenario`` are in ``_edit``, the answers table in ``_answers``, the deciding
inputs, tried through it, in ``_deciding``, the switching-value job in ``_switching``, the
gold-answers job in ``_gold``, the question-reach job in ``_reach``, ``use_answers`` and the record
checks and replay they share in ``_diff``, the stores in ``_store`` and the refusal in
``_refusal``.
"""

from ommat.evaluation._answers import answers
from ommat.evaluation._deciding import deciding_inputs, deciding_inputs_sync
from ommat.evaluation._diff import use_answers, use_answers_sync
from ommat.evaluation._edit import intervene, intervene_sync, scenario, scenario_sync
from ommat.evaluation._gold import gold_record, gold_record_sync, write_gold_answers
from ommat.evaluation._reach import question_reach, question_reach_sync
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import DesignStore, EvaluationStore
from ommat.evaluation._switching import switching_values, switching_values_sync

__all__ = [
    "DesignStore",
    "EvaluationRefusal",
    "EvaluationStore",
    "RefusalReason",
    "answers",
    "deciding_inputs",
    "deciding_inputs_sync",
    "gold_record",
    "gold_record_sync",
    "intervene",
    "intervene_sync",
    "question_reach",
    "question_reach_sync",
    "scenario",
    "scenario_sync",
    "switching_values",
    "switching_values_sync",
    "use_answers",
    "use_answers_sync",
    "write_gold_answers",
]
