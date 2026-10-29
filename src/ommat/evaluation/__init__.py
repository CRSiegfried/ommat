"""The evaluation subpackage: jobs that measure a design against past turns through replay,
and the two stores they keep their inputs and results in. The names ``__all__`` lists are
public, under the same deprecation rule as ``ommat.__all__``. ``use_answers`` is the one job
shipped.

``use_answers`` answers a turn with another record's answers: it reruns one record's
turn with another record's calls answering, and returns replay's findings. ``DesignStore``
keeps a loaded design's document under its design hash and fetches it back through ``load``;
``EvaluationStore`` keeps write-once entries by kind and key. A job or a store refuses with
``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

The job is in ``_diff``, the stores in ``_store`` and the refusal in ``_refusal``.
"""

from ommat.evaluation._diff import use_answers, use_answers_sync
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import DesignStore, EvaluationStore

__all__ = [
    "DesignStore",
    "EvaluationRefusal",
    "EvaluationStore",
    "RefusalReason",
    "use_answers",
    "use_answers_sync",
]
