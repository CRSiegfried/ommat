"""ommat: a blackboard decision engine.

Decide runs interpret, which gathers evidence from a decision model onto a blackboard and
fires the rules; the policy's estimate is the decision. It returns the decision and a
replayable record. A source's binding may call any tool, one that writes included. Replay
reruns a record under any loaded design and reports how the rerun differs. The core imports
only the standard library.
"""

from importlib.metadata import version as _version

from ommat._answer import Answer, AnswerKind, ExtractionAnswer, ValueType
from ommat._binding import CallResult, RenderedQuestion, SourceCall
from ommat._decide import Decision, EntryRefusal, RunSettings, decide, decide_sync
from ommat._design import (
    Design,
    DesignFailure,
    Group,
    Helper,
    Input,
    InvalidDesign,
    LoadedDesign,
    Param,
    Parameter,
    Question,
    Rule,
    Source,
    SourceRequirement,
    Structured,
    Table,
    Target,
    TargetRole,
    Text,
    When,
)
from ommat._fixture import from_record
from ommat._ledger import InvalidRecord, Record, read_record
from ommat._load import load
from ommat._markers import (
    Cause,
    CauseCode,
    Failed,
    Unasked,
    Unavailable,
    Undetermined,
    Unsure,
)
from ommat._read_view import ReadView
from ommat._replay import FindingKind, replay, replay_sync
from ommat._views import defects, escalation, redact, transcript

__version__ = _version("ommat")

__all__ = [
    "Answer",
    "AnswerKind",
    "CallResult",
    "Cause",
    "CauseCode",
    "Decision",
    "Design",
    "DesignFailure",
    "EntryRefusal",
    "ExtractionAnswer",
    "Failed",
    "FindingKind",
    "Group",
    "Helper",
    "Input",
    "InvalidDesign",
    "InvalidRecord",
    "LoadedDesign",
    "Param",
    "Parameter",
    "Question",
    "ReadView",
    "Record",
    "RenderedQuestion",
    "Rule",
    "RunSettings",
    "Source",
    "SourceCall",
    "SourceRequirement",
    "Structured",
    "Table",
    "Target",
    "TargetRole",
    "Text",
    "Unasked",
    "Unavailable",
    "Undetermined",
    "Unsure",
    "ValueType",
    "When",
    "__version__",
    "decide",
    "decide_sync",
    "defects",
    "escalation",
    "from_record",
    "load",
    "read_record",
    "redact",
    "replay",
    "replay_sync",
    "transcript",
]
