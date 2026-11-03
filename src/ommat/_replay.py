# pyright: reportPrivateUsage=false
"""Replay: rerun decide's body over a recorded turn under a given loaded design, with the
record as the only fixture, and report every way the rerun differs.

``replay`` refuses a wrong argument before it reads the record. A record
with no ``verdict`` item is ``unfinished``, and one with a ``verdict`` item and no state entry
is ``nothing-to-replay``: either returns ``None`` with that finding alone. Otherwise the record's
run settings go through decide's bounds check, and decide's body runs through its internal
entry: the recorded state entry typed by the given design in place of the codec, the fixture
binding under every declared oracle name and ``default``, the observer, and the record's cut in
place of a clock reading. A cancellation or the observer's exception
propagates as decide lets it out, with the replay record so far and no findings.

``_replay_with`` is the rerun itself, its fixture built from a given log's calls while the
state entry, the settings, the cut and the compared log are the record's; ``replay`` gives the
record's log as both, and the evaluation subpackage's ``use_answers`` another record's. Each
asked source is checked against a base log, the answering log unless ``intervene`` made it
from another, and against the base's design when ``intervene`` gives one, and a
``stale-question`` finding names each ref the fixture found stale.

The findings are plain ``dict``s in a fixed order, two items differing when their
canonical bytes differ. Replay reads nothing from ``unhashed``.
"""

import asyncio
import enum
from typing import Any, cast

from ommat._canonical import decode, encode
from ommat._decide import EntryRefusal, _bound, _turn, observer_or_none, run_settings
from ommat._design import LoadedDesign
from ommat._fixture import Base, FixtureBinding, RecordedCalls
from ommat._interpret import SORTED_TEXT
from ommat._ledger import _ITEM_DEPTH, Observer, Record
from ommat._refs import code_point_sorted
from ommat._typed import decoded
from ommat._values import CONTEXT_FORMAT, KEY_ORDER_FORMAT, type_name
from ommat.helpers import HELPERS

_STATE = "t:state"
_DEFAULT = "default"

Finding = dict[str, Any]


class FindingKind(enum.StrEnum):
    """Replay's finding kinds."""

    design_mismatch = "design-mismatch"
    helpers_mismatch = "helpers-mismatch"
    nothing_to_replay = "nothing-to-replay"
    stale_question = "stale-question"
    unfinished = "unfinished"
    entry = "entry"
    plan = "plan"
    verdict = "verdict"


_K = FindingKind


async def replay(
    record: Record, loaded: LoadedDesign, *, observer: Observer | None = None
) -> tuple[Record | None, list[Finding]]:
    """Rerun ``record`` under ``loaded`` and return the replay record, or ``None``, and the
    findings."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    if type(record) is not Record:
        raise EntryRefusal(f"record is a {type_name(record)}, not exactly a Record")
    observer_or_none(observer)
    log = record.log  # read once: every read decodes afresh
    if not any(item["item"] == "verdict" for item in log):
        return None, [{"kind": _K.unfinished}]
    if not any(item["item"] == "entry" and item["ref"] == _STATE for item in log):
        return None, [{"kind": _K.nothing_to_replay}]
    return await _replay_with(log, loaded, log, observer=observer)


async def _replay_with(
    log: list[dict[str, Any]],
    loaded: LoadedDesign,
    answers: list[dict[str, Any]],
    *,
    observer: Observer | None,
    base: list[dict[str, Any]] | None = None,
    pinned: frozenset[str] = frozenset(),
    failed: frozenset[str] = frozenset(),
    design: LoadedDesign | None = None,
    asked: frozenset[str] = frozenset(),
) -> tuple[Record, list[Finding]]:
    """Rerun the finished turn ``log`` holds, with a state entry, under ``loaded``, the
    fixture built from the call items and entries of ``answers``, each source checked against
    ``base``, ``answers`` when ``None``, and against ``design``, the base's design, the refs in
    ``pinned`` and the sources in ``failed`` never, the sources in ``asked`` stale; the state
    entry, the run settings, the cut and the log every finding compares against are
    ``log``'s."""
    # Both callers checked that the log holds a state entry.
    state = next(item for item in log if item["item"] == "entry" and item["ref"] == _STATE)
    # The settings item is second in every record from the design item on.
    recorded = cast("dict[str, Any]", log[1]["settings"])
    run = run_settings(
        recorded["round_bound"],
        recorded["timeout"],
        recorded["budget"],
        recorded["deadline"],
        recorded["certificate_cap"],
    )
    checked = Base(answers if base is None else base, loaded, design=design, asked=asked)
    fixture = FixtureBinding(RecordedCalls(answers), loaded, checked, pinned=pinned, failed=failed)
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    bindings = _bound(loaded, dict.fromkeys({*keys, _DEFAULT}, fixture))
    value = decoded(state["value"], loaded._types[_STATE])
    # A record before format 6 rendered slot text sorted; its rerun renders it so too.
    token = SORTED_TEXT.set(log[0]["format"] < KEY_ORDER_FORMAT)
    try:
        decision = await _turn(
            loaded, value, bindings, run, codec=None, clock=None, cut=_cut(log), observer=observer
        )
    finally:
        SORTED_TEXT.reset(token)
    rerun = decision.record
    return rerun, _findings(log, rerun.log, loaded, fixture)


def _cut(log: list[dict[str, Any]]) -> tuple[int, bool] | None:
    """The record's cut: the round whose ``round`` item has ``deadline`` true, and whether
    that round holds a call item; ``None`` for a record with no such round."""
    for item in log:
        if item["item"] == "round" and item["deadline"] is True:
            r = cast(int, item["round"])
            return r, any(i["item"] == "call" and i["round"] == r for i in log)
    return None


def replay_sync(
    record: Record, loaded: LoadedDesign, *, observer: Observer | None = None
) -> tuple[Record | None, list[Finding]]:
    """``replay`` run to completion; ``RuntimeError`` inside a running event loop, raised
    before the coroutine is built."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("replay_sync was called inside a running event loop; await replay")
    return asyncio.run(replay(record, loaded, observer=observer))


# ---------------------------------------------------------------------------------------------
# The findings, in their fixed order


def _findings(
    log: list[dict[str, Any]],
    rerun: list[dict[str, Any]],
    loaded: LoadedDesign,
    fixture: FixtureBinding,
) -> list[Finding]:
    found: list[Finding] = []
    design = log[0]
    if design["hash"] != loaded.design_hash:
        found.append(
            {"kind": _K.design_mismatch, "recorded": design["hash"], "replayed": loaded.design_hash}
        )
    if design["helpers"] != HELPERS:
        found.append(
            {"kind": _K.helpers_mismatch, "recorded": design["helpers"], "replayed": HELPERS}
        )
    for ref in code_point_sorted(fixture.stale):
        sid, why = fixture.stale[ref]
        found.append({"kind": _K.stale_question, "ref": ref, "source": sid, "clause": why.clause})
    found += _entry_findings(log, rerun, loaded)
    for kind in (_K.plan, _K.verdict):
        mine, theirs = _first(log, kind.value), _first(rerun, kind.value)
        compared = mine
        if kind is _K.plan and mine is not None and design["format"] < CONTEXT_FORMAT:
            compared = {**mine, "context": None}  # a record before format 5 holds no block
        if _differ(compared, theirs):
            found.append({"kind": kind, "recorded": mine, "replayed": theirs})
    return found


def _entry_findings(
    log: list[dict[str, Any]], rerun: list[dict[str, Any]], loaded: LoadedDesign
) -> list[Finding]:
    """An ``entry`` finding per ref, in code-point order, held by either record whose two
    entries differ, null for a side that holds none."""
    mine = {item["ref"]: item for item in log if item["item"] == "entry"}
    theirs = {item["ref"]: item for item in rerun if item["item"] == "entry"}
    rules = cast("dict[str, Any]", decode(loaded.document))["rules"]
    found: list[Finding] = []
    for ref in code_point_sorted({*mine, *theirs}):
        a, b = mine.get(ref), theirs.get(ref)
        if _differ(a, b):
            found.append(
                {
                    "kind": _K.entry,
                    "ref": ref,
                    "recorded": a,
                    "replayed": b,
                    "rule": _rule(ref, loaded, rules),
                }
            )
    return found


def _rule(ref: str, loaded: LoadedDesign, rules: dict[str, Any]) -> dict[str, Any] | None:
    """The given design's function rule providing a ``d:`` ref: its ``module``, ``qualname``,
    ``version`` and ``source`` from the design's document, else null."""
    if not ref.startswith("d:"):
        return None
    declared = rules.get(loaded._providers.get(ref, ""))
    if declared is None or declared["form"] != "function" or declared["provides"] != ref:
        return None
    return {k: declared[k] for k in ("module", "qualname", "version", "source")}


def _first(log: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((item for item in log if item["item"] == kind), None)


_LATER_MEMBERS = ("order", "grounded", "sets")
"""The members formats 6 and 7 add to an item, which a record before them does not hold."""


def _differ(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Two items, or a side holding none, compared by canonical bytes, ``order`` and a settings
    item's ``grounded`` and ``sets`` included; an item of a record before the format adding
    one does not hold it, and is compared without the other side's."""
    if a is None or b is None:
        return a is not b
    for member in _LATER_MEMBERS:
        if (member in a) != (member in b):
            a = {k: v for k, v in a.items() if k != member}
            b = {k: v for k, v in b.items() if k != member}
    return encode(a, _ITEM_DEPTH) != encode(b, _ITEM_DEPTH)
