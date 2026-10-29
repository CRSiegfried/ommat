# pyright: reportPrivateUsage=false
"""Another record's answers: rerun ``record``'s turn under
``loaded`` with ``other``'s call items and entries answering, ``q:`` and ``t:`` refs alike.

Both records are read first, then refused in order, each before any replay: ``unfinished``,
``nothing-to-replay`` and ``wrong-design``, each for ``record`` then ``other``, then
``different-turn`` when their state entries or ``settings`` items differ by canonical bytes.
The findings are replay's: the entries, plan and verdict against ``record``,
``question-changed`` against ``other``'s identities, and ``not-in-record`` and
``ref-not-in-record`` meaning not in ``other``. A plan is compared without its ``context``, so
a change of context text alone is no plan change. Neither record is edited, and nothing is read
from either's ``unhashed``.

The record read, its checks, the replay less context-alone plan findings and the sync runner
are shared with the edits of ``_edit``.
"""

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from ommat._canonical import encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._ledger import (
    _ITEM_DEPTH,
    InvalidRecord,
    Record,
    _with_held_order,
    read_record,
)
from ommat._replay import _STATE, FindingKind, _differ, _replay_with
from ommat._values import type_name
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

_R = RefusalReason
_NAMES = ("record", "other")
_T = TypeVar("_T")


async def use_answers(record: object, loaded: LoadedDesign, other: object) -> dict[str, Any]:
    """Rerun ``record`` under ``loaded`` answered by ``other``'s calls; ``{"other", "findings"}``,
    ``other`` its record hash."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    records = (_read(record, "record"), _read(other, "other"))
    logs = tuple(r.log for r in records)  # read once: every read decodes afresh
    _refuse_unfit(_NAMES, records, logs, loaded)
    mine, theirs = logs
    # Compared as replay compares items: an item of a record before format 6, holding no
    # order, is compared without the other's.
    states = (_state(mine), _state(theirs))
    for what, a, b in (("state entry", *states), ("settings item", mine[1], theirs[1])):
        if _differ(a, b):
            raise EvaluationRefusal(
                f"other's {what} is not record's, so it is another turn",
                reason=_R.different_turn,
                turn=records[1].record_hash,
            )
    return {"other": records[1].record_hash, "findings": await _replayed(mine, loaded, theirs)}


def _refuse_unfit(
    names: tuple[str, ...],
    records: tuple[Record, ...],
    logs: tuple[list[dict[str, Any]], ...],
    loaded: LoadedDesign,
) -> None:
    """Refuse, in order, ``unfinished``, ``nothing-to-replay`` and ``wrong-design``, each
    checked on every record in turn, its ``turn`` the record's hash (§do-and-diff)."""
    for name, r, log in zip(names, records, logs, strict=True):
        if not any(item["item"] == "verdict" for item in log):
            raise EvaluationRefusal(
                f"{name} has no verdict item", reason=_R.unfinished, turn=r.record_hash
            )
    for name, r, log in zip(names, records, logs, strict=True):
        if _state(log) is None:
            raise EvaluationRefusal(
                f"{name} has no state entry", reason=_R.nothing_to_replay, turn=r.record_hash
            )
    for name, r in zip(names, records, strict=True):
        if r.design_hash != loaded.design_hash:
            raise EvaluationRefusal(
                f"{name}'s design hash is {r.design_hash}, not the loaded design's "
                f"{loaded.design_hash}",
                reason=_R.wrong_design,
                turn=r.record_hash,
            )


async def _replayed(
    log: list[dict[str, Any]], loaded: LoadedDesign, answers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replay's findings for ``log`` rerun under ``loaded``, answered by ``answers``' calls,
    less a ``plan`` finding of context alone; the replay record is dropped (§store-integrity)."""
    _, found = await _replay_with(log, loaded, answers, observer=None)
    return [f for f in found if not _context_alone(f)]


OUTSIDE = frozenset(
    {FindingKind.not_in_record, FindingKind.ref_not_in_record, FindingKind.question_changed}
)
"""The findings of a run that asked outside the record, which therefore says nothing of it."""


async def baseline(log: list[dict[str, Any]], loaded: LoadedDesign) -> None:
    """Replay ``log`` unedited under ``loaded``; ``baseline-differs`` on any finding but
    ``helpers-mismatch``, a note on the helper version alone, the kinds in the message."""
    found = await _replayed(log, loaded, log)
    kinds = [str(f["kind"]) for f in found if f["kind"] is not FindingKind.helpers_mismatch]
    if kinds:
        raise EvaluationRefusal(
            f"the record does not replay cleanly under loaded: {', '.join(dict.fromkeys(kinds))}",
            reason=_R.baseline_differs,
        )


def changed(finding: dict[str, Any]) -> bool:
    """Whether an ``entry`` finding's two sides differ in value, or in marker kind and code."""
    recorded, replayed = finding["recorded"], finding["replayed"]
    return recorded is None or replayed is None or outcome(recorded) != outcome(replayed)


def outcome(entry: dict[str, Any]) -> bytes:
    """An entry's value, or its marker's kind and cause code, as canonical bytes: a change of
    ``rows``, ``derivation``, ``round`` or a cause's refs alone is none."""
    if "marker" in entry:
        marker = entry["marker"]
        return encode(["marker", marker["kind"], marker["cause"]["code"]])
    return encode(["value", entry["value"]], _ITEM_DEPTH)


def _context_alone(finding: dict[str, Any]) -> bool:
    """Whether a ``plan`` finding's two items differ only in their ``context``. The rest of each
    is compared with an ``order`` derived afresh over the rest alone; when either item holds no
    ``order``, as one of a record before format 6 does not, the rest is compared without one.
    The items given are not changed."""
    if finding["kind"] is not FindingKind.plan:
        return False
    mine, theirs = finding["recorded"], finding["replayed"]
    if mine is None or theirs is None:
        return False
    held = "order" in mine and "order" in theirs
    rest = [
        {k: v for k, v in item.items() if k not in ("context", "order")} for item in (mine, theirs)
    ]
    if held:
        rest = [_with_held_order(item) for item in rest]
    return encode(rest[0], _ITEM_DEPTH) == encode(rest[1], _ITEM_DEPTH)


def use_answers_sync(record: object, loaded: LoadedDesign, other: object) -> dict[str, Any]:
    """``use_answers`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    return _run_sync("use_answers", lambda: use_answers(record, loaded, other))


def _run_sync(name: str, job: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """The coroutine ``job`` builds, run to completion; ``RuntimeError`` naming the job inside a
    running event loop, raised before the coroutine is built."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(f"{name}_sync was called inside a running event loop; await {name}")
    return asyncio.run(job())


def _read(document: object, name: str) -> Record:
    """A ``Record`` as given, any other value read through ``read_record``."""
    if type(document) is Record:
        return document
    try:
        return read_record(document)
    except InvalidRecord as exc:
        raise EvaluationRefusal(
            f"{name} is not a valid record: {exc}", reason=_R.invalid_record, error=exc
        ) from exc


def _state(log: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((i for i in log if i["item"] == "entry" and i["ref"] == _STATE), None)
