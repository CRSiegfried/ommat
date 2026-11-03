"""Many turns: ``decide_many`` decides one turn per state, at most ``concurrency`` at once, on the
caller's running loop, and returns each turn's ``Decision`` or ``TurnFailure`` in input order.

Every turn shares ``loaded``, ``bindings``, ``settings`` and ``codec``; only its state differs,
and each has its own deadline, as ``decide`` gives one. A state given twice is decided twice,
so a repeat is a state listed N times, and identical repeats share a record hash. An exception
that ends one turn, an ``EntryRefusal``, an observer's raise or a cancellation of that turn
alone, is collected as its ``TurnFailure`` with the record so far, the others run on; a
cancellation of the call itself, or any ``BaseException`` that is not an ``Exception``,
cancels every turn and propagates. A turn not yet started when a cap is wanted is simply not
started: a caller stops the run by cancelling it, never by a binding that returns markers.
"""

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import ommat._decide as _decide
from ommat._binding import OracleBinding
from ommat._design import LoadedDesign
from ommat._ledger import Observer, Record
from ommat._values import type_name


@dataclass(frozen=True, slots=True)
class TurnFailure:
    """A turn ``decide_many`` collected in place of its ``Decision``: the exception that ended
    it and its ``record`` so far, ``None`` for an ``EntryRefusal``, which ends a turn before its
    record opens."""

    error: BaseException
    record: Record | None


async def decide_many(
    states: Iterable[object],
    loaded: LoadedDesign,
    bindings: Mapping[str, OracleBinding],
    *,
    concurrency: int,
    settings: "_decide.RunSettings | None" = None,
    codec: "_decide.Codec | None" = None,
    observer_for: Callable[[int], Observer | None] | None = None,
) -> "list[_decide.Decision | TurnFailure]":
    """Decide a turn for each of ``states``, at most ``concurrency`` at once, each observed by
    ``observer_for(index)`` when given; each turn's ``Decision`` or ``TurnFailure``, in input
    order."""
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError(f"concurrency {concurrency!r} is not an int of at least 1")
    if observer_for is not None and not callable(observer_for):
        raise TypeError(f"observer_for is a {type_name(observer_for)}, not callable or None")
    if isinstance(states, str | bytes | Mapping) or not isinstance(states, Iterable):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"states are a {type_name(cast(object, states))}, not an iterable of states"
        )
    given = list(states)
    results: list[_decide.Decision | TurnFailure | None] = [None] * len(given)
    gate = asyncio.Semaphore(concurrency)

    async def one(index: int, state: object) -> None:
        try:
            async with gate:
                results[index] = await _turn(
                    index, state, loaded, bindings, settings, codec, observer_for
                )
        except asyncio.CancelledError as exc:
            # Kept should the run end normally: something outside it cancelled this turn alone.
            results[index] = TurnFailure(exc, _record_of(exc))
            raise

    async with asyncio.TaskGroup() as group:
        for index, state in enumerate(given):
            group.create_task(one(index, state))
    # A turn cancelled from outside before it first ran set no slot.
    return [TurnFailure(asyncio.CancelledError(), None) if r is None else r for r in results]


async def _turn(
    index: int,
    state: object,
    loaded: LoadedDesign,
    bindings: Mapping[str, OracleBinding],
    settings: "_decide.RunSettings | None",
    codec: "_decide.Codec | None",
    observer_for: Callable[[int], Observer | None] | None,
) -> "_decide.Decision | TurnFailure":
    """One turn's decision, or what ended it alone; a cancellation from outside the turn, or a
    ``BaseException`` that is not an ``Exception``, propagates."""
    try:
        observer = None if observer_for is None else observer_for(index)
        return await _decide.decide(
            loaded, state, bindings, settings, codec=codec, observer=observer
        )
    except asyncio.CancelledError as exc:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise  # the run is cancelled, not this turn alone
        return TurnFailure(exc, _record_of(exc))
    except Exception as exc:
        return TurnFailure(exc, _record_of(exc))


def _record_of(exc: BaseException) -> Record | None:
    """The record decide attached to the exception, or ``None``."""
    record: Any = getattr(exc, "record", None)
    return record if type(record) is Record else None
