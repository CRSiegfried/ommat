# pyright: reportPrivateUsage=false
"""Another record's answers: rerun ``record``'s turn under
``loaded`` with ``other``'s call items and entries answering, ``q:`` and ``t:`` refs alike.

Both records are read first, then refused in order, each before any replay: ``wrong-design``
for a record before format 8, ``unfinished``, ``nothing-to-replay``, then ``wrong-design`` for
a design hash other than the loaded design's, each for ``record`` then ``other``; then
``different-turn`` when their state entries or ``settings`` items differ by canonical bytes.
The findings are replay's: the entries against ``record``, and
``stale-question`` for each ref whose source is checked against ``other``: not answered there,
or asked with other needs or as another question. Neither record is edited, and nothing is read
from either's ``unhashed``.

The record read, its checks, the replay, the sync runner and ``own_functions``, the
``functions`` a job given ``loaded`` loads an edited design with, are shared with ``intervene``
in ``_edit``.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, TypeVar

from ommat._binding import OracleBinding
from ommat._canonical import encode
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._ledger import (
    _ITEM_DEPTH,
    InvalidRecord,
    Record,
    decision_bytes,
    finished,
    read_record,
)
from ommat._replay import _STATE, FindingKind, _differ, _replay_with
from ommat._values import POLICY_FORMAT, type_name
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
    states = (_state(mine), _state(theirs))
    for what, a, b in (("state entry", *states), ("settings item", mine[1], theirs[1])):
        if _differ(a, b):
            raise EvaluationRefusal(
                f"other's {what} is not record's, so it is another turn",
                reason=_R.different_turn,
                turn=records[1].record_hash,
            )
    _, found, _ = await _replayed(mine, loaded, theirs)
    return {"other": records[1].record_hash, "findings": found}


def _refuse_unfit(
    names: tuple[str, ...],
    records: tuple[Record, ...],
    logs: tuple[list[dict[str, Any]], ...],
    loaded: LoadedDesign,
) -> None:
    """Refuse, in order, ``wrong-design`` for a record before format 8, which holds no
    policy's entry; ``unfinished`` for one not finished, holding a ``cause`` item or no entry at
    the policy's ref; ``nothing-to-replay`` for one whose state entry is missing or holds a
    marker; and ``wrong-design`` for a design hash other than ``loaded``'s; each checked on every
    record in turn, its ``turn`` the record's hash (§do-and-diff)."""
    for name, r, log in zip(names, records, logs, strict=True):
        if log[0]["format"] < POLICY_FORMAT:
            raise EvaluationRefusal(
                f"{name} is a format-{log[0]['format']} record, before format {POLICY_FORMAT}, "
                "so it holds no policy's entry",
                reason=_R.wrong_design,
                turn=r.record_hash,
            )
    for name, r, log in zip(names, records, logs, strict=True):
        if not finished(log):
            raise EvaluationRefusal(
                f"{name} is unfinished: it holds a cause item or no entry at the policy's ref",
                reason=_R.unfinished,
                turn=r.record_hash,
            )
    for name, r, log in zip(names, records, logs, strict=True):
        state = _state(log)
        if state is None or "marker" in state:
            held = "no state entry" if state is None else "a marker at its state entry"
            raise EvaluationRefusal(
                f"{name} has {held}", reason=_R.nothing_to_replay, turn=r.record_hash
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
    log: list[dict[str, Any]],
    loaded: LoadedDesign,
    answers: list[dict[str, Any]],
    *,
    base: list[dict[str, Any]] | None = None,
    pinned: frozenset[str] = frozenset(),
    failed: frozenset[str] = frozenset(),
    design: LoadedDesign | None = None,
    asked: frozenset[str] = frozenset(),
    live: Mapping[str, OracleBinding] | None = None,
    note: str = "",
    unpinned: frozenset[str] = frozenset(),
) -> tuple[Record, list[dict[str, Any]], frozenset[str]]:
    """The replay record of ``log`` rerun under ``loaded``, answered by ``answers``' calls and
    checked against ``base`` and ``design`` as ``_replay_with`` checks, stale sources asked of
    ``live`` when given; replay's findings; and the refs answered from ``answers``."""
    rerun, found, fixture = await _replay_with(
        log,
        loaded,
        answers,
        observer=None,
        base=base,
        pinned=pinned,
        failed=failed,
        design=design,
        asked=asked,
        live=live,
        note=note,
        unpinned=unpinned,
    )
    return rerun, found, _stood(rerun.log, fixture.pinned, fixture.reproduced)


def _stood(log: list[dict[str, Any]], given: set[str], reproduced: set[str]) -> frozenset[str]:
    """The refs given an answer from the record whose call the turn wrote ``returned``, and
    those of a recorded failure given back."""
    entries = {i["ref"]: i for i in log if i["item"] == "entry"}
    stood: set[str] = set(reproduced)
    for ref in given:
        entry = entries.get(ref)
        call = None if entry is None else entry["call"]
        if call is not None and log[call]["outcome"] == "returned":
            stood.add(ref)
    return frozenset(stood)


OUTSIDE = frozenset({FindingKind.stale_question})
"""The findings of a run that asked a question the record does not answer, which therefore says
nothing of it."""


async def baseline(
    log: list[dict[str, Any]], loaded: LoadedDesign, *, turn: str | None = None
) -> None:
    """Replay ``log`` unedited under ``loaded``; ``baseline-differs`` on any finding but
    ``helpers-mismatch``, a note on the helper version alone, the kinds in the message, naming
    ``turn`` when given."""
    _, found, _ = await _replayed(log, loaded, log)
    kinds = [str(f["kind"]) for f in found if f["kind"] is not FindingKind.helpers_mismatch]
    if kinds:
        raise EvaluationRefusal(
            f"the record does not replay cleanly under loaded: {', '.join(dict.fromkeys(kinds))}",
            reason=_R.baseline_differs,
            turn=turn,
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


def use_answers_sync(record: object, loaded: LoadedDesign, other: object) -> dict[str, Any]:
    """``use_answers`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    return _run_sync("use_answers", lambda: use_answers(record, loaded, other))


def own_functions(
    loaded: LoadedDesign, functions: Mapping[str, Callable[..., Any]] | None
) -> Mapping[str, Callable[..., Any]]:
    """``functions``, or when ``None`` ``loaded``'s own callables, keyed as load's ``functions``
    are, by declared name and ``<source>.when``: an edited copy of ``loaded``'s
    document names the same code."""
    return loaded._callables if functions is None else functions


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


def decided(log: list[dict[str, Any]], rerun: list[dict[str, Any]]) -> bool:
    """Whether a rerun changed the decision: the policy's entries of the two logs differ by
    ``decision_bytes``, a marker's cause and refs included (§decide-output)."""
    return decision_bytes(log) != decision_bytes(rerun)


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
