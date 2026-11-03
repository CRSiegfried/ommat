"""The evaluation jobs' one refusal class and its closed reasons."""

import enum
from typing import Any


class RefusalReason(enum.StrEnum):
    """Why an evaluation job refused."""

    unfinished = "unfinished"
    nothing_to_replay = "nothing-to-replay"
    wrong_design = "wrong-design"
    different_turn = "different-turn"
    null_call = "null-call"
    failed_call = "failed-call"
    marker = "marker"
    nonconforming = "nonconforming"
    fault_code = "fault-code"
    not_found = "not-found"
    candidate_rule = "candidate-rule"
    invalid_edit = "invalid-edit"
    no_edit = "no-edit"
    duplicate_target = "duplicate-target"
    conflicting_edits = "conflicting-edits"
    invalid_record = "invalid-record"
    invalid_design = "invalid-design"
    bad_key = "bad-key"
    not_held = "not-held"
    hash_mismatch = "hash-mismatch"
    held_mismatch = "held-mismatch"
    invalid_entry = "invalid-entry"
    unknown_kind = "unknown-kind"
    held_key = "held-key"
    misfiled_entry = "misfiled-entry"
    unknown_path = "unknown-path"
    unsupported_leaf = "unsupported-leaf"
    baseline_differs = "baseline-differs"
    unrepresentable_slack = "unrepresentable-slack"
    invalid_label = "invalid-label"
    conflicting_labels = "conflicting-labels"
    unknown_entry = "unknown-entry"
    empty_copy = "empty-copy"


class EvaluationRefusal(Exception):
    """An evaluation job's refusal: ``reason``, and each other attribute what the reason
    names, else ``None``. ``key`` is a list of store keys: a design hash for the design store,
    a key's members by name for the evaluation store. ``turn`` is the record hash of the record
    refused; ``error`` the ``InvalidRecord`` or ``InvalidDesign`` that caused it, also chained."""

    def __init__(
        self,
        message: str,
        *,
        reason: RefusalReason,
        key: list[Any] | None = None,
        turn: str | None = None,
        ref: str | None = None,
        source: str | None = None,
        path: str | None = None,
        group: str | None = None,
        split: tuple[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.key = key
        self.turn = turn
        self.ref = ref
        self.source = source
        self.path = path
        self.group = group
        self.split = split
        self.error = error
