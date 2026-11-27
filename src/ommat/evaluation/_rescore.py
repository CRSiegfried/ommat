# pyright: reportPrivateUsage=false
"""The rescoring job: each recorded turn decided again under each of several designs, from the
record's answers alone, with no record built (§evaluation-jobs).

Each rescore is replay's run, the fixture built from the record's calls, but the turn writes to
a ledger that checks, encodes and keeps nothing but the entries at the refs the caller names, so
no record of it exists. Each asked source is checked against the record under ``recorded``, the
design the record was decided under, where ``replay(record, design)`` checks it under ``design``
itself: so a design whose source carries another payload or ``oracle`` key than ``recorded``'s,
or another type for a ``t:`` ref it provides, gives ``not-determined`` for those refs where
replay under it decides. Apart from that, each named entry is the one replay under the design
writes. A run in which a source asks something the record does not answer, a stale question, is
``not-determined``, as in the switching-value job: its entries say nothing of the record.

Every record is read and checked as replay checks it, against ``recorded``, its recorded
settings included, before any run. A record's runs go one after another, its first under the
first design in full and kept as the reference each later one fires an unchanged rule from
(``_interpret.Reuse``), so that a design changing a few leaves reruns only what they reach. The
rows come back in design order, then record order, as plain data.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from ommat._decide import EntryRefusal, _bound, _opening, recorded_run
from ommat._design import LoadedDesign
from ommat._fixture import Base, FixtureBinding
from ommat._interpret import Reuse, interpret
from ommat._ledger import _logged
from ommat._refs import code_point_sorted
from ommat._replay import _cut
from ommat._typed import decoded, typed
from ommat._values import type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _run_sync, _state
from ommat.helpers._dead import named_refs

_STATE = "t:state"
_DEFAULT = "default"
NOT_DETERMINED = "not-determined"

_Log = list[dict[str, Any]]


async def rescore(
    records: Sequence[object],
    recorded: LoadedDesign,
    designs: Sequence[LoadedDesign],
    *,
    refs: Sequence[str],
) -> list[dict[str, Any]]:
    """Each record decided again under each of ``designs`` from its own answers, with no record
    built: one row per design and record, in that order, ``{"design", "record", "entries",
    "stale"}``. ``entries`` maps each of ``refs`` the run wrote, in code-point order, to its
    entry as decide writes it, ``{"value": v}`` or ``{"marker": {"kind", "cause"}}``, or is
    ``"not-determined"`` when the run asked a question the record does not answer; ``stale``
    lists those refs in code-point order. ``refs`` is a list or tuple of declared ``d:`` refs an
    ungrounded rule of every design provides, of ``recorded`` with no designs. Each asked source
    is checked against the record under ``recorded``, so a payload or ``oracle`` the design
    changes stales the source's refs, and a ``t:`` type it changes stales that ref. Each record
    is checked as replay checks it, against ``recorded``, its own design, before any run."""
    if type(recorded) is not LoadedDesign:
        raise EntryRefusal(f"recorded is a {type_name(recorded)}, not exactly a LoadedDesign")
    if isinstance(designs, str | bytes) or not isinstance(designs, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"designs are a {type_name(designs)}, not a sequence of LoadedDesigns")
    for i, d in enumerate(designs):
        if type(d) is not LoadedDesign:
            raise EntryRefusal(f"designs[{i}] is a {type_name(d)}, not exactly a LoadedDesign")
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"records are a {type_name(records)}, not a sequence of records")
    named = _named(
        refs, {f"designs[{i}]": d for i, d in enumerate(designs)} or {"recorded": recorded}
    )
    names = tuple(f"records[{i}]" for i in range(len(records)))
    read = tuple(_read(r, name) for r, name in zip(records, names, strict=True))
    # Read once, as every read decodes afresh: one log object per record, which the fixture's
    # construction key holds by identity, so a later run reuses the first run's calls.
    logs = tuple(r.log for r in read)
    _refuse_unfit(names, read, logs, recorded)
    for log in logs:
        _run_of(log)  # a recorded setting decide refuses is refused here, before any run
    # Record by record, so one record's reference run serves each later design (``Reuse``).
    results: list[list[tuple[dict[str, Any] | str, list[str]]]] = []
    for log in logs:
        reuse = Reuse()
        results.append([await _rescored(log, loaded, recorded, reuse, named) for loaded in designs])
    rows: list[dict[str, Any]] = []
    for i, loaded in enumerate(designs):
        for r, found in zip(read, results, strict=True):
            entries, stale = found[i]
            rows.append(
                {
                    "design": loaded.design_hash,
                    "record": r.record_hash,
                    "entries": entries,
                    "stale": stale,
                }
            )
    return rows


def rescore_sync(
    records: Sequence[object],
    recorded: LoadedDesign,
    designs: Sequence[LoadedDesign],
    *,
    refs: Sequence[str],
) -> list[dict[str, Any]]:
    """``rescore`` run to completion on one event loop; ``RuntimeError`` inside a running event
    loop."""
    return _run_sync("rescore", lambda: rescore(records, recorded, designs, refs=refs))


def _named(refs: object, designs: Mapping[str, LoadedDesign]) -> list[str]:
    """``refs`` checked as ``dead_questions`` checks them against each of ``designs``, named by
    its key in a ``ValueError``, each ref once, in code-point order."""
    named: tuple[str, ...] = ()
    for name, loaded in designs.items():
        try:
            named = named_refs(loaded, refs)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
    return code_point_sorted(named)


class _Kept:
    """The ledger surface interpret writes to, checking and keeping nothing but the entries at
    ``refs``, each as ``Record.log`` reads it back."""

    observer_error = None
    drops_unhashed = True  # its write_unhashed keeps nothing

    def __init__(self, refs: Sequence[str]) -> None:
        self.refs = frozenset(refs)
        self.entries: dict[str, dict[str, Any]] = {}
        self.count = 0

    def append(self, item: dict[str, Any]) -> int:
        if item.get("item") == "entry" and item.get("ref") in self.refs:
            logged = _logged(item)
            self.entries[item["ref"]] = {k: logged[k] for k in ("value", "marker") if k in logged}
        self.count += 1
        return self.count - 1

    def write_unhashed(self, member: str, value: object, index: int | None = None) -> None:
        pass


def _run_of(log: _Log) -> Any:
    """The record's run settings, as decide checks them; ``EntryRefusal`` for one it refuses."""
    return recorded_run(cast("dict[str, Any]", log[1]["settings"]))


async def _rescored(
    log: _Log, loaded: LoadedDesign, recorded: LoadedDesign, reuse: Reuse, refs: Sequence[str]
) -> tuple[dict[str, Any] | str, list[str]]:
    """The entries at ``refs``, in their order, or ``not-determined``, of ``log``'s turn rerun
    under ``loaded`` as replay reruns it, and the stale refs; ``reuse`` is the record's
    reference run."""
    state = cast("dict[str, Any]", _state(log))  # the record checks found a value there
    run = _run_of(log)
    base = Base(log, loaded, design=recorded)
    fixture = FixtureBinding(base.calls, loaded, base)
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    bindings = _bound(loaded, dict.fromkeys({*keys, _DEFAULT}, fixture))
    kept = _Kept(refs)
    for item in _opening(loaded, run.item()):
        kept.append(item)
    forms = typed(decoded(state["value"], loaded._types[_STATE]), loaded._types[_STATE])
    if forms is None:
        # A broken entry: decide writes t:state alone, so no named entry.
        return {}, []
    kept.append({"item": "entry", "ref": _STATE})
    await interpret(
        cast(Any, kept),
        loaded,
        forms[1],
        bindings,
        round_bound=run.round_bound,
        timeout=run.timeout,
        budget=run.budget,
        deadline=None,
        clock=None,
        cut=_cut(log),
        carried=frozenset(),
        reuse=reuse,
        replaying=True,  # the fixture's calls, over a log that keeps no unhashed write
    )
    stale = code_point_sorted(fixture.stale)
    if stale:
        return NOT_DETERMINED, stale
    # A ref the run wrote no entry at, as past an observer's cut, is absent.
    return {ref: kept.entries[ref] for ref in refs if ref in kept.entries}, []
