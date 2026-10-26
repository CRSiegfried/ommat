# pyright: reportPrivateUsage=false
"""The ``Unsure``-to-execute lint, ``unsure_to_execute``: every ``execute`` plan the policy
returns when one of its ``d:`` reads is ``Unsure``.

It fills the policy's declared reads by the lint points: a read under another declared read's
path is filled as a field of that read, and every other read is written at its path, on a
board holding one entry per base ref. It sets each ``d:`` base ref in turn to ``Unsure``
with cause ``abstained``, naming the declared reads of the rule that writes it, while the
other reads range over their points, and calls the policy alone through ``_component.call``
on a fresh board per call: no plan rule and no guard. A call is ``execute`` only when no stop
was recorded, nothing was raised, and the return is exactly a ``Plan`` of kind ``execute``.
The report is a warning: the lint raises nothing on a finding.

A per-candidate pick's ``Unsure`` for one candidate is no read the policy makes: a rule may turn
it into a value, such as a list leaving the candidate out. So for each per-candidate question a
rule on the policy's lineage reads, the lint also runs the turn as the dead-question lint does,
each grounded entry in turn at the abstention point, every label at 0, on which every pick
abstains, and every other entry over its points. A run is reported, under the entry's grounded
ref, when it wrote the entry as that answer, an entry of a rule on the lineage lists a
returned per-candidate pick over the question, and its ``plan`` item is ``execute``. A
per-candidate question no lineage rule reads holds its first point, and a ``d:`` set that
changes within one assignment raises ``ValueError``, as the dead-question lint's does. Its
runs cross the picked questions' entries once per entry in turn, so they grow with the
candidates as the dead-question lint's cross product does.
"""

import itertools
from collections.abc import Mapping, Sequence
from typing import Any, cast

from ommat._answer import AnswerKind
from ommat._canonical import encode
from ommat._component import base, call
from ommat._design import LoadedDesign, Policy, Rule
from ommat._interpret import _marker_document
from ommat._markers import Cause, CauseCode, PlanKind, Unsure
from ommat._plan import Plan
from ommat._read_view import ComponentKind, ReadView, Role, walk
from ommat._refs import code_point_sorted, declared_ref, grounded_ref, parse_ref
from ommat._values import type_name
from ommat.helpers._dead import _STATE, _At, _candidates, _run
from ommat.helpers._points import (
    Point,
    abstention,
    entry_points,
    fill,
    fill_both,
    fresh,
    grid_order,
    question,
)

_POLICY = "policy"
_PICKS = frozenset({"pick_each_with_margin", "pick_each_with_thresholds"})


def unsure_to_execute(
    loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"points": N, "execute": [{"read", "point"}]}``, by ``read`` in code-point
    order, then by point in grid order; a ``ValueError`` before any call for a read nothing
    fills or a slip in ``values``. ``values`` maps one of the policy's declared reads, ``d:``
    or ``t:``, to a list of its values; a read declared under another is keyed by its field
    path, and a ``q:`` ref is never a key. When a rule on the policy's lineage reads a
    per-candidate question, a key may also name a read the per-candidate search needs and
    its grid and types cannot fill, such as the candidate set, and a ``read`` may be a
    grounded ref."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    policy = cast(Policy, loaded._design.policy)  # load refuses the policy table
    declared = code_point_sorted(set(policy.reads))
    top = [r for r in declared if not any(r.startswith(o + ".") for o in declared)]
    rules = _lineage(loaded, policy)
    picked = _picked(loaded, rules)
    runs_fills: dict[str, list[Point]] = {}
    if picked:
        sources = loaded._design.sources.values()
        provided = code_point_sorted({ref for src in sources for ref in src.provides} | {_STATE})
        fills, runs_fills = fill_both(loaded, top, provided, values)
    else:
        fills = fill(loaded, top, values)
    points = 0
    execute: list[dict[str, Any]] = []
    for unsure in code_point_sorted({base(r) for r in declared if r.startswith("d:")}):
        # load refuses a rule table, so a rule writes every d: ref
        writer = cast(Rule, loaded._design.rules[loaded._providers[unsure]])
        marker = Unsure(Cause(CauseCode.abstained, code_point_sorted(set(writer.reads))))
        shown_marker = _marker_document(marker)  # the record's marker form, interpret's
        others = [r for r in top if base(r) != unsure]
        for combo in grid_order([fills[r] for r in others]):
            point = dict(zip(others, combo, strict=True))
            points += 1
            if _executes(loaded, policy, _board(point, unsure, marker)):
                shown = {
                    r: fresh(shown_marker) if base(r) == unsure else _record(r, top, point)
                    for r in declared
                }
                execute.append({"read": unsure, "point": shown})
    if picked:
        runs, found = _per_candidate(loaded, rules, picked, runs_fills)
        points += runs
        execute += found
    return {"points": points, "execute": execute}


def _lineage(loaded: LoadedDesign, policy: Policy) -> list[Rule]:
    """The rules the policy's reads rest on: each writing a ``d:`` ref the policy reads, or one
    such a rule reads, and so on."""
    seen: set[str] = set()
    todo = [base(r) for r in policy.reads if r.startswith("d:")]
    while todo:
        ref = todo.pop()
        if ref in seen:
            continue
        seen.add(ref)
        rule = cast(Rule, loaded._design.rules[loaded._providers[ref]])
        todo += [base(r) for r in rule.reads if r.startswith("d:")]
    return [cast(Rule, loaded._design.rules[loaded._providers[ref]]) for ref in sorted(seen)]


def _picked(loaded: LoadedDesign, rules: Sequence[Rule]) -> list[str]:
    """The per-candidate questions of a probability kind a rule on the lineage reads."""
    read = {base(r) for rule in rules for r in rule.reads}
    return code_point_sorted(
        ref
        for ref in read & loaded._grounded.keys()
        if ref.startswith("q:") and question(loaded, ref).kind is not AnswerKind.extraction
    )


def _per_candidate(
    loaded: LoadedDesign,
    rules: Sequence[Rule],
    picked: Sequence[str],
    fills: Mapping[str, list[Point]],
) -> tuple[int, list[dict[str, Any]]]:
    """The per-candidate search: the runs it made, and each run reported, by grounded ref in
    code-point order and then in grid order."""
    grounded = loaded._grounded
    reads = code_point_sorted(set(fills) - grounded.keys())
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    firsts = {ref: fills[ref][0] for ref in grounded if ref in fills}
    lineage = {rule.provides for rule in rules}
    runs = 0
    found: dict[str, list[dict[str, Any]]] = {}
    for combo in grid_order([fills[r] for r in reads]):
        point = dict(zip(reads, combo, strict=True))
        sets: dict[str, object] = {}
        for set_ref in {grounded[ref] for ref in firsts}:
            if set_ref in point:  # a t: set, fixed by the point
                sets[set_ref] = point[set_ref].live
        if any(grounded[ref] not in sets for ref in firsts):  # a d: set: run the probe
            runs += 1
            sets.update(_run(loaded, _At(point, firsts), keys)[2])
        # A question no lineage rule reads holds its first point: only the picked
        # questions' other entries can change what a pick returns.
        entries = {
            grounded_ref(ref, c): entry_points(fills[ref], grounded_ref(ref, c))[
                slice(None) if ref in picked else slice(1)
            ]
            for ref in firsts
            for c in _candidates(sets.get(grounded[ref]))
        }
        for target in code_point_sorted(e for e in entries if declared_ref(e) in picked):
            stopped = abstention(loaded, declared_ref(target))
            names = code_point_sorted(entries)
            lists = [[stopped] if n == target else entries[n] for n in names]
            for chosen in itertools.product(*lists):
                given = {**point, **dict(zip(names, chosen, strict=True))}
                runs += 1
                done = _run(loaded, given, keys)
                for set_ref, value in done[2].items():
                    if set_ref in sets and encode(value) != encode(sets[set_ref]):
                        raise ValueError(
                            f"the candidate set {set_ref!r} changes within one assignment of the "
                            "other reads, so the lint cannot fix its grounded entries"
                        )
                if _rests(done[3], target, stopped, lineage):
                    shown = {ref: fresh(given[ref].record) for ref in code_point_sorted(given)}
                    shown.update(done[1])
                    found.setdefault(target, []).append({"read": target, "point": shown})
    return runs, [item for target in code_point_sorted(found) for item in found[target]]


def _rests(
    log: Sequence[Mapping[str, Any]], target: str, stopped: Point, lineage: set[str]
) -> bool:
    """Whether a run's ``execute`` plan rests on the abstention at ``target``: the entry was
    written as that answer, and a rule on the lineage returned a per-candidate pick over its
    question."""
    question_ref = declared_ref(target)
    written = {i["ref"]: i for i in log if i["item"] == "entry"}
    plan = next(i for i in log if i["item"] == "plan")
    entry = written.get(target)
    if plan["kind"] != "execute" or entry is None or "value" not in entry:
        return False
    if encode(entry["value"]) != encode(stopped.record):
        return False
    return any(
        item["helper"] in _PICKS and item["refs"] == [question_ref] == item["rested_on"]
        for ref in lineage
        for item in written.get(ref, {}).get("derivation", [])
    )


def _board(point: Mapping[str, Point], unsure: str, marker: Unsure) -> dict[str, object]:
    """A fresh board: each read's live value at its path, and the marker at ``unsure``."""
    board: dict[str, object] = {unsure: marker}
    for read, p in point.items():
        fields = parse_ref(read).fields
        if not fields:
            board[read] = fresh(p.live)
            continue
        node = cast("dict[str, object]", board.setdefault(base(read), {}))
        for name in fields[:-1]:
            node = cast("dict[str, object]", node.setdefault(name, {}))
        node[fields[-1]] = fresh(p.live)
    return board


def _executes(loaded: LoadedDesign, policy: Policy, board: dict[str, object]) -> bool:
    """Whether the policy alone returns an ``execute`` plan over ``board``, its stop record
    read as compile reads it."""
    leaves = {path: loaded._leaf_value(path) for path in policy.params}
    view = ReadView(Role(ComponentKind.policy, _POLICY), policy.reads, policy.params, board, leaves)
    called = call(loaded._callables[_POLICY], view)
    if called.stop is not None or called.error is not None:
        return False
    returned = called.returned
    return type(returned) is Plan and returned.kind is PlanKind.execute


def _record(read: str, top: Sequence[str], point: Mapping[str, Point]) -> object:
    """A declared read's value in record form, read through the filled read it lies under or
    is."""
    for filled in top:
        if read == filled or read.startswith(filled + "."):
            fields = parse_ref(read).fields[len(parse_ref(filled).fields) :]
            return fresh(walk(point[filled].record, fields, read))
    raise AssertionError(f"{read!r} lies under no filled read")  # pragma: no cover
