# pyright: reportPrivateUsage=false
"""The ``Unsure``-to-execute lint, ``unsure_to_execute``: every ``execute`` plan the policy
returns when one of its ``d:`` reads is ``Unsure``.

It fills the policy's declared reads by the lint points: a read under another declared read's
path is filled as a field of that read, and every other read is written at its path, on a
board holding one entry per base ref. It sets each ``d:`` base ref in turn to ``Unsure``
with cause ``abstained``, naming the declared reads of the rule that writes it, but for a
table that cannot write ``Unsure``, while the
other reads range over their points, and calls the policy alone through ``_component.call``
on a fresh board per call: no plan rule and no guard. A call is ``execute`` only when no stop
was recorded, nothing was raised, and the return is exactly a ``Plan`` of kind ``execute``.
The report is a warning: the lint raises nothing on a finding.

An answer's abstention, a pick's or ``extract``'s ``Unsure``, is no read the policy makes: a
rule may turn it into a value, such as a list leaving a candidate out. So for each question a
rule on the policy's lineage reads, every rule and source its reads rest on in the dependency
graph, the lint also runs the turn as the dead-question lint does, with that question, or each
of its grounded entries in turn, at its abstention point: every label at 0, or a null
extraction value. The reads the lineage reads, needs, is gated on, is grounded over or is
labelled by range over their points, and every other read holds its first. A subject's first run
that rests on its abstention is reported, under the question's ref or the entry's grounded ref,
when it wrote the subject as its abstention answer, an entry of a lineage rule lists a returned
pick or ``extract`` over the question, and its ``plan`` item is ``execute``. A ``d:`` set a run
shows changing within one assignment raises ``ValueError``, as the dead-question lint's does. A
subject's runs vary one ranging read at a time from their first points and stop at its first
finding, so they grow with the sum of those reads' points, times the sum over the grounded
entries where both range, not their product. A finding whose run left a policy read the first
search reported holding an ``Unsure`` that names the subject names that read in ``via``: a
policy refusing the marker there would not have executed in the run.
"""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from ommat._canonical import encode
from ommat._compile import table_choice
from ommat._component import base, call
from ommat._design import LoadedDesign
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
)

_POLICY = "policy"
_ONE = frozenset({"pick_with_margin", "pick_with_thresholds", "extract"})
_EACH = frozenset({"pick_each_with_margin", "pick_each_with_thresholds"})


def unsure_to_execute(
    loaded: LoadedDesign, *, values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The report ``{"points": N, "execute": [{"read", "point"}], "narrowed": [ref]}``, by
    ``read`` in code-point order, then by point in grid order; a ``ValueError`` before any call
    for a read nothing fills or a slip in ``values``. ``values`` maps one of the policy's
    declared reads, ``d:`` or ``t:``, to a list of its values; a read declared under another is
    keyed by its field path. When a rule on the policy's lineage reads a question, a key may
    also name a read the abstention search fills and its grid and types cannot, such as a
    ``t:state`` field, a candidate set or an extraction question, or a label question with no
    grid entry, whose list is the labels to try and which ``narrowed`` names; a ``read`` may
    then be a ``q:`` or grounded ref, and its item holds ``via``, a policy read the first
    search reported, when the run left that read an ``Unsure`` naming the subject."""
    if type(loaded) is not LoadedDesign:
        raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    declared = code_point_sorted(set(loaded._reads[_POLICY]))
    top = [r for r in declared if not any(r.startswith(o + ".") for o in declared)]
    lineage = _lineage(loaded)
    subjects = _subjects(loaded, lineage)
    runs_fills: dict[str, list[Point]] = {}
    narrowed: list[str] = []
    if subjects:
        sources = loaded._design.sources.values()
        provided = code_point_sorted({ref for src in sources for ref in src.provides} | {_STATE})
        fills, runs_fills, narrowed = fill_both(loaded, top, provided, values)
    else:
        fills = fill(loaded, top, values)
    points = 0
    execute: list[dict[str, Any]] = []
    for unsure in code_point_sorted({base(r) for r in declared if r.startswith("d:")}):
        if _never_unsure(loaded, loaded._providers[unsure]):
            continue
        writer = loaded._reads[loaded._providers[unsure]]
        marker = Unsure(Cause(CauseCode.abstained, code_point_sorted(set(writer))))
        shown_marker = _marker_document(marker)  # the record's marker form, interpret's
        others = [r for r in top if base(r) != unsure]
        for combo in grid_order([fills[r] for r in others]):
            point = dict(zip(others, combo, strict=True))
            points += 1
            if _executes(loaded, _board(point, unsure, marker)):
                shown = {
                    r: fresh(shown_marker) if base(r) == unsure else _record(r, top, point)
                    for r in declared
                }
                execute.append({"read": unsure, "point": shown})
    if subjects:
        flagged = {item["read"] for item in execute}
        runs, found = _abstentions(loaded, lineage, subjects, runs_fills, flagged)
        points += runs
        execute += found
    return {"points": points, "execute": execute, "narrowed": narrowed}


def _never_unsure(loaded: LoadedDesign, rid: str) -> bool:
    """Whether a table writes the ref and cannot write ``Unsure``: it has no ``Unsure``
    output entry, and a row of ``-`` cells giving a value or a parameter always matches."""
    if rid not in loaded._table_bytes:
        return False
    rows = cast("list[dict[str, Any]]", loaded._table(rid)["rows"])
    outputs = [cast(object, row["output"]) for row in rows]
    if any(isinstance(o, dict) and "marker" in o for o in outputs):
        return False
    return any(
        all(cell == "-" for cell in row["cells"])
        and isinstance(output, dict)
        and ("literal" in output or "param" in output)
        for row, output in zip(rows, outputs, strict=True)
    )


def _lineage(loaded: LoadedDesign) -> set[str]:
    """Every rule and source the policy's reads rest on in the dependency graph."""
    seen: set[str] = set()
    todo = [loaded._providers[base(r)] for r in loaded._reads[_POLICY] if base(r) != _STATE]
    while todo:
        node = todo.pop()
        if node not in seen:
            seen.add(node)
            todo += loaded._graph[node]
    return seen


def _subjects(loaded: LoadedDesign, lineage: set[str]) -> list[str]:
    """The questions a lineage rule reads; load refuses a per-candidate extraction, so a
    per-candidate one is of a probability kind."""
    rules = loaded._design.rules
    read = {base(r) for rid in lineage if rid in rules for r in loaded._reads[rid]}
    return code_point_sorted(ref for ref in read if ref.startswith("q:"))


def _ranging(loaded: LoadedDesign, lineage: set[str]) -> set[str]:
    """The base refs the policy or a lineage component reads, needs, is gated on or is
    grounded over; every other exogenous read can change no finding."""
    used = {base(r) for r in loaded._reads[_POLICY]}
    for node in lineage:
        if node in loaded._design.rules:
            used |= {base(r) for r in loaded._reads[node]}
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
    lineage: set[str],
    subjects: Sequence[str],
    fills: Mapping[str, list[Point]],
    flagged: set[str],
) -> tuple[int, list[dict[str, Any]]]:
    """The abstention search: the runs it made, and each run reported, by subject in
    code-point order and then in run order. ``flagged`` holds the ``d:`` reads the first
    search reported."""
    grounded = loaded._grounded
    ranging = _ranging(loaded, lineage)
    keys = {s.oracle for s in loaded._design.sources.values() if s.oracle is not None}
    rules = {loaded._design.rules[n].provides for n in lineage if n in loaded._design.rules}
    # Each question a lineage table picks or extracts from: its entry holds rows, not a
    # derivation, so the input stands for the returned helper.
    picked = {
        cast(str, i["read"])
        for n in lineage
        if n in loaded._table_bytes
        for i in cast("list[dict[str, Any]]", loaded._table(n)["inputs"])
        if i["helper"] in _ONE
    }
    reads = code_point_sorted(set(fills) - grounded.keys())
    held = {r: fills[r] if base(r) in ranging else fills[r][:1] for r in reads}
    firsts = {ref: fills[ref][0] for ref in grounded if ref in fills}
    found: dict[str, dict[str, Any]] = {}
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
            set_ref = grounded[ref]
            for c in _candidates(sets.get(set_ref), loaded._candidate_sets.get(set_ref)):
                entry = grounded_ref(ref, c)
                points = entry_points(fills[ref], entry)
                entries[entry] = points if ref in ranging else points[:1]
        self_sets.update(sets)
        return entries

    self_sets: dict[str, object] = {}

    def cross(point: dict[str, Point], entries: dict[str, list[Point]], subject: str) -> bool:
        """Whether a run over these entries, varied one at a time, rests on the subject's
        abstention; the first such run is the subject's finding."""
        nonlocal runs
        names = code_point_sorted(entries)
        for chosen in _one_at_a_time([entries[n] for n in names]):
            given = {**point, **dict(zip(names, chosen, strict=True))}
            runs += 1
            done = _run(loaded, given, keys)
            for set_ref, value in done[2].items():
                if set_ref in self_sets and encode(value) != encode(self_sets[set_ref]):
                    raise ValueError(
                        f"the candidate set {set_ref!r} changes within one assignment of the "
                        "other reads, so the lint cannot fix its grounded entries"
                    )
            if _rests(done[3], subject, rules, picked):
                shown = {ref: fresh(given[ref].record) for ref in code_point_sorted(given)}
                shown.update(done[1])
                found[subject] = {"read": subject, "point": shown}
                via = _via(done[3], subject, flagged)
                if via is not None:
                    found[subject]["via"] = via
                return True
        return False

    plain = [q for q in subjects if q not in grounded]
    each = [q for q in subjects if q in grounded]
    for subject in plain:
        lists = [[abstention(loaded, r)] if r == subject else held[r] for r in reads]
        for combo in _one_at_a_time(lists):
            self_sets.clear()
            point = dict(zip(reads, combo, strict=True))
            if cross(point, assignment(point), subject):
                break
    if each:
        for combo in _one_at_a_time([held[r] for r in reads]):
            self_sets.clear()
            point = dict(zip(reads, combo, strict=True))
            entries = assignment(point)
            for subject in code_point_sorted(e for e in entries if declared_ref(e) in each):
                if subject in found:
                    continue
                stopped = {**entries, subject: [abstention(loaded, declared_ref(subject))]}
                cross(point, stopped, subject)
    return runs, [found[subject] for subject in code_point_sorted(found)]


def _one_at_a_time(lists: Sequence[list[Point]]) -> Iterator[tuple[Point, ...]]:
    """Every read at its first point, then each read in turn over its other points with the
    rest at their first: one run per point beyond the first, not their cross product."""
    first = tuple(points[0] for points in lists)
    yield first
    for i, points in enumerate(lists):
        for point in points[1:]:
            yield (*first[:i], point, *first[i + 1 :])


def _rests(
    log: Sequence[Mapping[str, Any]], subject: str, rules: set[str], picked: set[str]
) -> bool:
    """Whether a run's ``execute`` plan rests on the abstention at ``subject``: it was written
    as an abstention answer, and a lineage rule returned a pick or ``extract`` over its
    question, the one-ref helpers for a plain question and the per-candidate picks for a
    grounded entry, or a lineage table has a pick or ``extract`` input over it."""
    question_ref = declared_ref(subject)
    helpers = _ONE if subject == question_ref else _EACH
    written = {i["ref"]: i for i in log if i["item"] == "entry"}
    plan = next(i for i in log if i["item"] == "plan")
    entry = written.get(subject)
    if plan["kind"] != "execute" or entry is None or "value" not in entry:
        return False
    if not is_abstention(entry["value"]):
        return False
    if subject == question_ref and question_ref in picked:
        return True
    return any(
        item["helper"] in helpers and item["refs"] == [question_ref] == item["rested_on"]
        for ref in rules
        for item in written.get(ref, {}).get("derivation", [])
    )


def _via(log: Sequence[Mapping[str, Any]], subject: str, flagged: set[str]) -> str | None:
    """The first ``d:`` read, in code-point order, the first search reported and the run
    wrote as an ``Unsure`` naming ``subject``: a policy refusing that marker would not have
    executed in the run."""
    met = [
        item["ref"]
        for item in log
        if item["item"] == "entry"
        and item["ref"] in flagged
        and item.get("marker", {}).get("kind") == "Unsure"
        and subject in item["marker"]["cause"]["refs"]
    ]
    return code_point_sorted(met)[0] if met else None


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


def _executes(loaded: LoadedDesign, board: dict[str, object]) -> bool:
    """Whether the policy alone returns an ``execute`` plan over ``board``, its stop record
    read as compile reads it; over a policy table, compile's evaluation gives the plan."""
    if _POLICY in loaded._table_bytes:  # compile's evaluation, which catches its own raise
        return table_choice(loaded, board)[0].kind is PlanKind.execute
    reads, params = loaded._reads[_POLICY], loaded._params[_POLICY]
    leaves = {path: loaded._leaf_value(path) for path in params}
    view = ReadView(Role(ComponentKind.policy, _POLICY), reads, params, board, leaves)
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
