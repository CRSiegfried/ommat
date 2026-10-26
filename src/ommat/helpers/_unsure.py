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

An answer's abstention, a pick's or ``extract``'s ``Unsure``, is no read the policy makes: a
rule may turn it into a value, such as a list leaving a candidate out. So for each question a
rule on the policy's lineage reads, every rule and source its reads rest on in the dependency
graph, the lint also runs the turn as the dead-question lint does, with that question, or
each of its grounded entries in turn, at its abstention point: every label at 0, or a null
extraction value. The reads the lineage reads, needs, is gated on or is grounded over range
over their points, and every other read holds its first. A run is reported, under the
question's ref or the entry's grounded ref, when it wrote the target as its abstention
answer, an entry of a lineage rule lists a returned pick or ``extract`` over the question,
and its ``plan`` item is ``execute``. A ``d:`` set that changes within one assignment
raises ``ValueError``, as the dead-question lint's does. A target's runs are the cross of
the ranging reads' points other than its own, so they grow with those reads as the
dead-question lint's cross product does.
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
    is_abstention,
    question,
)

_POLICY = "policy"
_ONE = frozenset({"pick_with_margin", "pick_with_thresholds", "extract"})
_EACH = frozenset({"pick_each_with_margin", "pick_each_with_thresholds"})


def unsure_to_execute(
    loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"points": N, "execute": [{"read", "point"}]}``, by ``read`` in code-point
    order, then by point in grid order; a ``ValueError`` before any call for a read nothing
    fills or a slip in ``values``. ``values`` maps one of the policy's declared reads, ``d:``
    or ``t:``, to a list of its values; a read declared under another is keyed by its field
    path. When a rule on the policy's lineage reads a question, a key may also name a read
    the abstention search fills and its grid and types cannot, such as a ``t:state`` field, a
    candidate set or an extraction question, and a ``read`` may be a ``q:`` or grounded
    ref."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    policy = cast(Policy, loaded._design.policy)  # load refuses the policy table
    declared = code_point_sorted(set(policy.reads))
    top = [r for r in declared if not any(r.startswith(o + ".") for o in declared)]
    lineage = _lineage(loaded, policy)
    targets = _targets(loaded, lineage)
    runs_fills: dict[str, list[Point]] = {}
    if targets:
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
    if targets:
        runs, found = _abstentions(loaded, policy, lineage, targets, runs_fills)
        points += runs
        execute += found
    return {"points": points, "execute": execute}


def _lineage(loaded: LoadedDesign, policy: Policy) -> set[str]:
    """Every rule and source the policy's reads rest on in the dependency graph."""
    seen: set[str] = set()
    todo = [loaded._providers[base(r)] for r in policy.reads if base(r) != _STATE]
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo += loaded._graph[node]
    return seen


def _targets(loaded: LoadedDesign, lineage: set[str]) -> list[str]:
    """The questions a lineage rule reads, a per-candidate one of a probability kind only."""
    rules = loaded._design.rules
    read = {base(r) for rid in lineage if rid in rules for r in cast(Rule, rules[rid]).reads}
    return code_point_sorted(
        ref
        for ref in read
        if ref.startswith("q:")
        and (ref not in loaded._grounded or question(loaded, ref).kind is not AnswerKind.extraction)
    )


def _ranging(loaded: LoadedDesign, policy: Policy, lineage: set[str]) -> set[str]:
    """The base refs the policy or a lineage component reads, needs, is gated on or is
    grounded over; every other exogenous read can change no finding."""
    used = {base(r) for r in policy.reads}
    for node in lineage:
        if node in loaded._design.rules:
            used |= {base(r) for r in cast(Rule, loaded._design.rules[node]).reads}
            continue
        source = loaded._design.sources[node]
        used |= {base(r) for r in loaded._needs[node]}
        if source.grounded is not None:
            used.add(source.grounded)
        if source.when is not None:
            used.add(_STATE)  # a function when reads the state
        for ref in source.provides:
            if ref in loaded._label_sets:
                used.add(loaded._label_sets[ref])
    return used


def _abstentions(
    loaded: LoadedDesign,
    policy: Policy,
    lineage: set[str],
    targets: Sequence[str],
    fills: Mapping[str, list[Point]],
) -> tuple[int, list[dict[str, Any]]]:
    """The abstention search: the runs it made, and each run reported, by target in
    code-point order and then in run order."""
    grounded = loaded._grounded
    ranging = _ranging(loaded, policy, lineage)
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    rules = {
        cast(Rule, loaded._design.rules[n]).provides for n in lineage if n in loaded._design.rules
    }
    reads = code_point_sorted(set(fills) - grounded.keys())
    held = {r: fills[r] if base(r) in ranging else fills[r][:1] for r in reads}
    firsts = {ref: fills[ref][0] for ref in grounded if ref in fills}
    found: dict[str, list[dict[str, Any]]] = {}
    runs = 0

    def assignment(point: dict[str, Point]) -> dict[str, list[Point]]:
        """The grounded entries this assignment grounds, each with its points, after the
        probe that learns a ``d:`` set when one grounds entries."""
        nonlocal runs
        sets: dict[str, object] = {}
        for set_ref in {grounded[ref] for ref in firsts}:
            if set_ref in point:  # a t: set, fixed by the point
                sets[set_ref] = point[set_ref].live
        if any(grounded[ref] not in sets for ref in firsts):  # a d: set: run the probe
            runs += 1
            sets.update(_run(loaded, _At(point, firsts), keys)[2])
        entries: dict[str, list[Point]] = {}
        for ref in firsts:
            for c in _candidates(sets.get(grounded[ref])):
                entry = grounded_ref(ref, c)
                points = entry_points(fills[ref], entry)
                entries[entry] = points if ref in ranging else points[:1]
        self_sets.update(sets)
        return entries

    self_sets: dict[str, object] = {}

    def cross(point: dict[str, Point], entries: dict[str, list[Point]], target: str) -> None:
        nonlocal runs
        names = code_point_sorted(entries)
        for chosen in itertools.product(*(entries[n] for n in names)):
            given = {**point, **dict(zip(names, chosen, strict=True))}
            runs += 1
            done = _run(loaded, given, keys)
            for set_ref, value in done[2].items():
                if set_ref in self_sets and encode(value) != encode(self_sets[set_ref]):
                    raise ValueError(
                        f"the candidate set {set_ref!r} changes within one assignment of the "
                        "other reads, so the lint cannot fix its grounded entries"
                    )
            if _rests(done[3], target, rules):
                shown = {ref: fresh(given[ref].record) for ref in code_point_sorted(given)}
                shown.update(done[1])
                found.setdefault(target, []).append({"read": target, "point": shown})

    plain = [q for q in targets if q not in grounded]
    each = [q for q in targets if q in grounded]
    for target in plain:
        lists = [[abstention(loaded, r)] if r == target else held[r] for r in reads]
        for combo in grid_order(lists):
            self_sets.clear()
            point = dict(zip(reads, combo, strict=True))
            cross(point, assignment(point), target)
    if each:
        for combo in grid_order([held[r] for r in reads]):
            self_sets.clear()
            point = dict(zip(reads, combo, strict=True))
            entries = assignment(point)
            for target in code_point_sorted(e for e in entries if declared_ref(e) in each):
                stopped = {**entries, target: [abstention(loaded, declared_ref(target))]}
                cross(point, stopped, target)
    return runs, [item for target in code_point_sorted(found) for item in found[target]]


def _rests(log: Sequence[Mapping[str, Any]], target: str, rules: set[str]) -> bool:
    """Whether a run's ``execute`` plan rests on the abstention at ``target``: it was written
    as an abstention answer, and a lineage rule returned a pick or ``extract`` over its
    question, the one-ref helpers for a plain question and the per-candidate picks for a
    grounded entry."""
    question_ref = declared_ref(target)
    helpers = _ONE if target == question_ref else _EACH
    written = {i["ref"]: i for i in log if i["item"] == "entry"}
    plan = next(i for i in log if i["item"] == "plan")
    entry = written.get(target)
    if plan["kind"] != "execute" or entry is None or "value" not in entry:
        return False
    if not is_abstention(entry["value"]):
        return False
    return any(
        item["helper"] in helpers and item["refs"] == [question_ref] == item["rested_on"]
        for ref in rules
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
