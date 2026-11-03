"""``decide_many``: one turn per state, at most ``concurrency`` at once, results in input order
with repeats kept; a turn ended alone is collected as a ``TurnFailure`` with its record so far,
the others run on; the run's own cancellation and a ``BaseException`` that is not an
``Exception`` propagate; each turn has its own deadline."""

import asyncio
from typing import Any

import pytest

from ommat import Decision, EntryRefusal, RunSettings
from ommat.helpers import TurnFailure, decide_many
from scripted import Scripted, returns
from test_decide import LOADED, REPLIES, STATE, bound

OTHER = {**STATE, "subject": "other"}


def binding(**changes: object) -> dict[str, Any]:
    return bound(Scripted({**REPLIES, **changes}))


def run(*args: Any, **kwargs: Any) -> list[Any]:
    return asyncio.run(decide_many(*args, **kwargs))


def test_results_come_in_input_order_and_identical_repeats_share_a_record_hash() -> None:
    out = run([STATE, OTHER, STATE], LOADED, binding(), concurrency=2)
    assert all(type(d) is Decision for d in out)
    first, other, again = (d.record for d in out)
    assert first.record_hash == again.record_hash != other.record_hash
    assert other.log[2]["value"]["subject"] == "other"


def test_no_more_than_concurrency_turns_run_at_once() -> None:
    running = [0, 0]  # now, most

    async def slow(batch: Any, *, seconds: float) -> object:
        running[0] += 1
        running[1] = max(running)
        await asyncio.sleep(0.01)
        running[0] -= 1
        return returns({"t:size": 4})

    out = run([STATE] * 6, LOADED, binding(look=slow), concurrency=2)
    assert len(out) == 6
    assert running[1] == 2


def test_an_entry_refusal_is_collected_with_no_record() -> None:
    (failure,) = run([STATE], LOADED, {"default": Scripted(REPLIES)}, concurrency=1)
    assert type(failure) is TurnFailure
    assert type(failure.error) is EntryRefusal
    assert failure.record is None


def test_an_observers_raise_ends_its_turn_alone_with_its_record() -> None:
    class Boom(Exception):
        pass

    def raising(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "plan":
            raise Boom

    out = run(
        [STATE, STATE],
        LOADED,
        binding(),
        concurrency=2,
        observer_for=lambda i: raising if i == 1 else None,
    )
    assert type(out[0]) is Decision
    failure = out[1]
    assert type(failure) is TurnFailure
    assert type(failure.error) is Boom
    assert failure.record is not None
    assert failure.record.log[-1]["item"] == "plan"


def test_a_turn_cancelled_alone_is_collected_with_its_record() -> None:
    async def cancelled(batch: Any, *, seconds: float) -> object:
        raise asyncio.CancelledError

    out = run([STATE, OTHER], LOADED, binding(look=cancelled), concurrency=1)
    for failure in out:
        assert type(failure) is TurnFailure
        assert type(failure.error) is asyncio.CancelledError
        assert failure.record is not None
        assert failure.record.log[-1] == {
            "item": "cause",
            "cause": {"code": "cancelled", "refs": []},
        }


def test_cancelling_the_run_propagates() -> None:
    async def hang(batch: Any, *, seconds: float) -> object:
        await asyncio.sleep(10)
        return returns({"t:size": 4})

    async def main() -> None:
        task = asyncio.create_task(
            decide_many([STATE] * 3, LOADED, binding(look=hang), concurrency=3)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main())


def test_a_base_exception_that_is_no_exception_aborts_the_run() -> None:
    async def interrupted(batch: Any, *, seconds: float) -> object:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run([STATE, STATE], LOADED, binding(look=interrupted), concurrency=2)


def test_each_turn_has_its_own_deadline() -> None:
    async def slow(batch: Any, *, seconds: float) -> object:
        await asyncio.sleep(0.05)
        return returns({"t:size": 4})

    out = run(
        [STATE] * 4, LOADED, binding(look=slow), concurrency=1, settings=RunSettings(deadline=1.0)
    )
    for decision in out:
        assert all(i["deadline"] is False for i in decision.record.log if i["item"] == "round")


def test_states_are_read_once_and_an_empty_list_gives_none() -> None:
    assert run([], LOADED, binding(), concurrency=1) == []
    out = run(iter([STATE, OTHER]), LOADED, binding(), concurrency=1)
    assert len(out) == 2


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"concurrency": 0}, ValueError),
        ({"concurrency": True}, ValueError),
        ({"concurrency": 1.0}, ValueError),
        ({"concurrency": 1, "observer_for": 5}, TypeError),
    ],
)
def test_arguments_of_another_form_are_refused_before_any_turn(
    kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        run([STATE], LOADED, binding(), **kwargs)


@pytest.mark.parametrize("states", ["abc", b"abc", {"a": 1}, 5])
def test_states_that_are_not_an_iterable_of_states_are_a_type_error(states: Any) -> None:
    with pytest.raises(TypeError):
        run(states, LOADED, binding(), concurrency=1)


def test_a_turn_cancelled_from_outside_the_run_keeps_its_place() -> None:
    def cancelling(index: int, item: dict[str, Any]) -> None:
        if item["item"] == "entry":
            task = asyncio.current_task()
            assert task is not None
            task.cancel()

    out = run(
        [STATE, OTHER, STATE],
        LOADED,
        binding(),
        concurrency=3,
        observer_for=lambda i: cancelling if i == 1 else None,
    )
    assert [type(r) for r in out] == [Decision, TurnFailure, Decision]
    assert type(out[1].error) is asyncio.CancelledError


def test_an_observer_for_that_raises_ends_that_turn_with_no_record() -> None:
    def broken(index: int) -> None:
        raise RuntimeError("no observer")

    (failure,) = run([STATE], LOADED, binding(), concurrency=1, observer_for=broken)
    assert type(failure) is TurnFailure
    assert (type(failure.error), failure.record) == (RuntimeError, None)


def test_settings_of_another_form_end_each_turn_with_an_entry_refusal() -> None:
    out = run([STATE, OTHER], LOADED, binding(), concurrency=1, settings="x")  # type: ignore[arg-type]
    assert [type(r.error) for r in out] == [EntryRefusal, EntryRefusal]


def test_a_turn_cancelled_before_it_starts_keeps_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    real = asyncio.TaskGroup.create_task
    made: list[asyncio.Task[Any]] = []

    def create(self: asyncio.TaskGroup, coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = real(self, coro, **kwargs)
        made.append(task)
        if len(made) == 1:
            task.cancel()  # before its first step
        return task

    monkeypatch.setattr(asyncio.TaskGroup, "create_task", create)
    out = run([STATE, OTHER], LOADED, binding(), concurrency=2)
    assert [type(r) for r in out] == [TurnFailure, Decision]
    assert (type(out[0].error), out[0].record) == (asyncio.CancelledError, None)
