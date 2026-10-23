"""The evaluation subpackage: jobs that measure a design against past turns through replay.
The names ``__all__`` lists are public, under the same deprecation rule as
``ommat.__all__``. ``use_answers`` is the one job shipped.

``use_answers`` answers a turn with another record's answers: it reruns one record's
turn with another record's calls answering, and returns replay's findings. A job refuses
with ``EvaluationRefusal``, its ``reason`` a ``RefusalReason``.

The job is in ``_diff`` and the refusal in ``_refusal``.
"""

from ommat.evaluation._diff import use_answers, use_answers_sync
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

__all__ = ["EvaluationRefusal", "RefusalReason", "use_answers", "use_answers_sync"]
