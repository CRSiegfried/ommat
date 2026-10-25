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
"""

import asyncio
from typing import Any

from ommat._canonical import encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._ledger import _ITEM_DEPTH, InvalidRecord, Record, read_record
from ommat._replay import _STATE, FindingKind, _differ, _replay_with
from ommat._values import type_name
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

_R = RefusalReason
_NAMES = ("record", "other")


async def use_answers(record: object, loaded: LoadedDesign, other: object) -> dict[str, Any]:
    """Rerun ``record`` under ``loaded`` answered by ``other``'s calls; ``{"other", "findings"}``,
    ``other`` its record hash."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    records = (_read(record, "record"), _read(other, "other"))
    logs = tuple(r.log for r in records)  # read once: every read decodes afresh
    for name, r, log in zip(_NAMES, records, logs, strict=True):
        if not any(item["item"] == "verdict" for item in log):
            raise EvaluationRefusal(
                f"{name} has no verdict item", reason=_R.unfinished, turn=r.record_hash
            )
    states = tuple(_state(log) for log in logs)
    for name, r, state in zip(_NAMES, records, states, strict=True):
        if state is None:
            raise EvaluationRefusal(
                f"{name} has no state entry", reason=_R.nothing_to_replay, turn=r.record_hash
            )
    for name, r in zip(_NAMES, records, strict=True):
        if r.design_hash != loaded.design_hash:
            raise EvaluationRefusal(
                f"{name}'s design hash is {r.design_hash}, not the loaded design's "
                f"{loaded.design_hash}",
                reason=_R.wrong_design,
                turn=r.record_hash,
            )
    mine, theirs = logs
    # Compared as replay compares items: an item of a record before format 6, holding no
    # order, is compared without the other's.
    for what, a, b in (("state entry", *states), ("settings item", mine[1], theirs[1])):
        if _differ(a, b):
            raise EvaluationRefusal(
                f"other's {what} is not record's, so it is another turn",
                reason=_R.different_turn,
                turn=records[1].record_hash,
            )
    _, found = await _replay_with(mine, loaded, theirs, observer=None)
    found = [f for f in found if not _context_alone(f)]
    return {"other": records[1].record_hash, "findings": found}


def _context_alone(finding: dict[str, Any]) -> bool:
    """Whether a ``plan`` finding's two items differ only in their ``context``."""
    if finding["kind"] is not FindingKind.plan:
        return False
    mine, theirs = finding["recorded"], finding["replayed"]
    if mine is None or theirs is None:
        return False
    rest = [{k: v for k, v in item.items() if k != "context"} for item in (mine, theirs)]
    return encode(rest[0], _ITEM_DEPTH) == encode(rest[1], _ITEM_DEPTH)


def use_answers_sync(record: object, loaded: LoadedDesign, other: object) -> dict[str, Any]:
    """``use_answers`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "use_answers_sync was called inside a running event loop; await use_answers"
        )
    return asyncio.run(use_answers(record, loaded, other))


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
