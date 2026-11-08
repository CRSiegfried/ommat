# pyright: reportPrivateUsage=false
"""The rescoring job: each recorded turn decided again under each of several designs, from the
record's answers alone, with no record built (§evaluation-jobs).

Each rescore is replay's run, the fixture built from the record's calls and each asked source
checked against the record under ``recorded``, the design it was decided under, but the turn
writes to a ledger that checks, encodes and keeps nothing but the policy's entry. Its decision is
therefore the one replay under that design writes, and no record of it exists. A run in which a
source asks something the record does not answer, a stale question, is ``not-determined``, as in
the switching-value job: its decision says nothing of the record.

Every record is read and checked as replay checks it, against ``recorded``, before any run. The
rows come back in design order, then record order, as plain data.
"""

from collections.abc import Sequence
from typing import Any, cast

from ommat._decide import EntryRefusal, _bound, _opening, run_settings
from ommat._design import LoadedDesign
from ommat._fixture import Base, FixtureBinding
from ommat._interpret import _marker_document, interpret
from ommat._ledger import _copy
from ommat._markers import Cause, CauseCode, Unavailable
from ommat._refs import code_point_sorted
from ommat._replay import _cut
from ommat._typed import decoded, typed
from ommat._values import type_name
from ommat.evaluation._diff import _read, _refuse_unfit, _run_sync, _state

_STATE = "t:state"
_DEFAULT = "default"
NOT_DETERMINED = "not-determined"

_Log = list[dict[str, Any]]


async def rescore(
    records: Sequence[object], recorded: LoadedDesign, designs: Sequence[LoadedDesign]
) -> list[dict[str, Any]]:
    """Each record decided again under each of ``designs`` from its own answers, with no record
    built: one row per design and record, in that order, ``{"design", "record", "decision",
    "stale"}``. ``decision`` is the policy's entry as decide writes it, ``{"value": v}`` or
    ``{"marker": {"kind", "cause"}}``, or ``"not-determined"`` when the run asked a question the
    record does not answer; ``stale`` lists those refs in code-point order. Each record is
    checked as replay checks it, against ``recorded``, its own design, before any run."""
    if type(recorded) is not LoadedDesign:
        raise EntryRefusal(f"recorded is a {type_name(recorded)}, not exactly a LoadedDesign")
    if isinstance(designs, str | bytes) or not isinstance(designs, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"designs are a {type_name(designs)}, not a sequence of LoadedDesigns")
    for i, d in enumerate(designs):
        if type(d) is not LoadedDesign:
            raise EntryRefusal(f"designs[{i}] is a {type_name(d)}, not exactly a LoadedDesign")
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"records are a {type_name(records)}, not a sequence of records")
    names = tuple(f"records[{i}]" for i in range(len(records)))
    read = tuple(_read(r, name) for r, name in zip(records, names, strict=True))
    logs = tuple(r.log for r in read)  # read once: every read decodes afresh
    _refuse_unfit(names, read, logs, recorded)
    rows: list[dict[str, Any]] = []
    for loaded in designs:
        for r, log in zip(read, logs, strict=True):
            decision, stale = await _rescored(log, loaded, recorded)
            rows.append(
                {
                    "design": loaded.design_hash,
                    "record": r.record_hash,
                    "decision": decision,
                    "stale": stale,
                }
            )
    return rows


def rescore_sync(
    records: Sequence[object], recorded: LoadedDesign, designs: Sequence[LoadedDesign]
) -> list[dict[str, Any]]:
    """``rescore`` run to completion on one event loop; ``RuntimeError`` inside a running event
    loop."""
    return _run_sync("rescore", lambda: rescore(records, recorded, designs))


class _Kept:
    """The ledger surface interpret writes to, checking and keeping nothing but the entry at
    ``policy``."""

    observer_error = None

    def __init__(self, policy: str) -> None:
        self.policy = policy
        self.decision: dict[str, Any] | None = None
        self.count = 0

    def append(self, item: dict[str, Any]) -> int:
        if item.get("item") == "entry" and item.get("ref") == self.policy:
            kept = {k: item[k] for k in ("value", "marker") if k in item}
            self.decision = cast("dict[str, Any]", _copy(kept))
        self.count += 1
        return self.count - 1

    def write_unhashed(self, member: str, value: object, index: int | None = None) -> None:
        pass


async def _rescored(
    log: _Log, loaded: LoadedDesign, recorded: LoadedDesign
) -> tuple[dict[str, Any] | str, list[str]]:
    """The policy's entry, or ``not-determined``, of ``log``'s turn rerun under ``loaded`` as
    replay reruns it, and the stale refs."""
    state = cast("dict[str, Any]", _state(log))  # the record checks found a value there
    settings = cast("dict[str, Any]", log[1]["settings"])
    run = run_settings(
        settings["round_bound"],
        settings["timeout"],
        settings["budget"],
        settings["deadline"],
        settings["certificate_cap"],
    )
    base = Base(log, loaded, design=recorded)
    fixture = FixtureBinding(base.calls, loaded, base)
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    bindings = _bound(loaded, dict.fromkeys({*keys, _DEFAULT}, fixture))
    kept = _Kept(loaded._design.policy)
    for item in _opening(loaded, run.item()):
        kept.append(item)
    forms = typed(decoded(state["value"], loaded._types[_STATE]), loaded._types[_STATE])
    if forms is None:  # decide writes the policy's entry Unavailable, as for a broken state
        marker = _marker_document(Unavailable(Cause(CauseCode.invalid_return, [_STATE])))
        return {"marker": marker}, []
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
    )
    stale = code_point_sorted(fixture.stale)
    if stale:
        return NOT_DETERMINED, stale
    if kept.decision is None:  # pragma: no cover - interpret writes the policy's entry
        raise RuntimeError("the rescored turn wrote no entry at the policy's ref")
    return kept.decision, []
