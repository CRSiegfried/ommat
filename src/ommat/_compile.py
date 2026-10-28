# pyright: reportPrivateUsage=false
"""Compile: call the policy once over its read view, turn what it gave into the turn's one
plan under the seven plan rules, append the ``plan`` item, call the guard and append the
``verdict`` item.

The outcome is the read view's order: the first recorded undeclared read, the first ``require``
stop, a raise, an invalid return, the plan. A policy plan is first tested for what only the
engine writes, then against the rules in their fixed order: 1, 5 and 7 over the plan, then
each step by position against 2, 3, 4, 7 and 6; the cause names the first break. The plan
returned is built from the copy the ledger stores, in exact ``dict``, ``list`` and ``str`` with
each value's live type kept, so a ``1.0`` stays a ``float`` where the record's bytes read back
as ``1``.

A policy table is evaluated, not called: its inputs read and its helpers called on a read
view over its reads, its rows tested, and the winning row's plan declaration built, its
declared text rendered from its slots; its ``plan`` item holds ``rows``, and ``slots`` when
it rendered one, where a function policy's holds ``derivation``. Compile writes no board
entry. It renders the design's context block over the board interpret left, before it calls
the policy, into every ``plan`` item it appends.
"""

from collections.abc import Callable, Mapping
from typing import Any, cast

from ommat._component import at, call, exception_record, names_read
from ommat._design import LoadedDesign, PlanDeclaration, Policy, Requirement
from ommat._guard import Verdict, guard, verdict_item
from ommat._interpret import _marker_document, _render
from ommat._ledger import Ledger
from ommat._markers import Cause, CauseCode, PlanKind, Undetermined
from ommat._plan import (
    Binding,
    Origin,
    Plan,
    Step,
    candidate_literal,
    declared_plan,
    engine_plan,
    step_document,
)
from ommat._read_view import ComponentKind, ReadView, RequireStopped, Role, UndeclaredRead, walk
from ommat._refs import candidate_values, entry_ref, grounded_ref, is_field_path, parse_ref
from ommat._table import Condition, evaluate, plan_conditions, read_inputs
from ommat._types import hold_failure
from ommat._values import domain_error, string_data

_C = CauseCode
_POLICY = "policy"
_IDEMPOTENCY_KEY = "idempotency_key"


def compile_turn(
    ledger: Ledger, loaded: LoadedDesign, board: Mapping[str, object]
) -> tuple[Plan, Verdict]:
    """Run compile over the live board interpret left, every ref on it, and
    return the plan and the verdict, both appended."""
    context = context_member(loaded, board)  # before the policy runs
    if _POLICY in loaded._table_bytes:
        plan, rows, raised = table_plan(loaded, board)
        item = plan_item(plan, None, context, rows=rows)
        index = ledger.append(item)
        if raised is not None:
            ledger.write_unhashed("exceptions", exception_record(raised), index)
        if plan.cause is None:
            plan = _plan_of(item)  # the ledger's stored copy
        verdict = guard(plan, board, loaded._requirements(), loaded._design.default_deny)
        ledger.append(verdict_item(verdict))
        return plan, verdict
    policy = cast(Policy, loaded._design.policy)
    leaves = {path: loaded._leaf_value(path) for path in policy.params}
    view = ReadView(
        Role(ComponentKind.policy, _POLICY),
        policy.reads,
        policy.params,
        board,
        leaves,
        grounded=loaded._grounded,
        sets=loaded._candidate_sets,
    )
    called = call(loaded._callables[_POLICY], view)
    stop, error = called.stop, called.error
    unnamed: str | None = None
    if isinstance(stop, UndeclaredRead):
        if names_read(loaded, stop):
            plan = engine_plan(PlanKind.broken, Cause(_C.undeclared_read, [_POLICY, stop.read]))
        else:
            plan = engine_plan(PlanKind.broken, Cause(_C.undeclared_read, [_POLICY]))
            unnamed = stop.read
    elif isinstance(stop, RequireStopped):
        plan = engine_plan(PlanKind.escalate, Cause(_C.required, [stop.ref]))
    elif error is not None:
        plan = engine_plan(PlanKind.broken, Cause(_C.raised, [_POLICY]))
    elif called.awaitable:  # tested before any other test of the return
        plan = engine_plan(PlanKind.broken, Cause(_C.invalid_return, [_POLICY]))
    else:
        refs = _Rules(loaded, board).first_break(called.returned)
        if refs is None:
            plan = cast(Plan, called.returned)
        else:
            plan = engine_plan(PlanKind.broken, Cause(_C.invalid_return, refs))
    item = plan_item(plan, called.derivation, context)
    index = ledger.append(item)
    if called.text is not None:  # whatever the outcome, a stop's included
        ledger.write_unhashed("exceptions", exception_record(called.text), index)
    if unnamed is not None:
        ledger.write_unhashed("reads", unnamed, index)
    if plan.cause is None:
        plan = _plan_of(item)  # the ledger's stored copy, not the policy's values
    verdict = guard(plan, board, loaded._requirements(), loaded._design.default_deny)
    ledger.append(verdict_item(verdict))
    return plan, verdict


# ---------------------------------------------------------------------------------------------
# The plan rules


class _Rules:
    def __init__(self, loaded: LoadedDesign, board: Mapping[str, object]) -> None:
        self.loaded = loaded
        self.board = board
        self.tools = frozenset(loaded._design.tools)
        self.reads = frozenset(string_data(r) for r in loaded._reads[_POLICY])

    def first_break(self, returned: object, *, declared: bool = False) -> list[str] | None:
        """The cause's refs for what the policy returned, or ``None`` for a valid plan. A
        ``declared`` plan, the policy table's, is the engine's own build of a declaration, so
        its ``slots`` are not a policy's slip."""
        if type(returned) is not Plan:  # a subclass may override what compile reads
            return [_POLICY]
        plan = returned
        candidates = any(_has_candidate(s) for s in plan.steps)
        engine_only = plan.cause is not None or candidates
        # A grounded declaration's expansion writes a set literal; a policy writes none.
        if not declared and any(_has_set(s) for s in plan.steps):
            engine_only = True
        if engine_only or (plan.slots and not declared):
            return [_POLICY]  # what only the engine writes
        if plan.kind is not PlanKind.execute and plan.steps:
            return ["rule:1"]
        if not _rule_5(plan):
            return ["rule:5"]
        # A text's length is tested before its code points are read.
        if not all(t is None or domain_error(t) is None for t in (plan.reason, plan.reply)):
            return ["rule:7"]
        checks: tuple[tuple[int, Callable[[Step, int], bool]], ...] = (
            (2, _rule_2),
            (3, self.rule_3),
            (4, self.rule_4),
            (7, _rule_7),  # before 6: its size bound keeps rule 6's walk of each value short
            (6, self.rule_6),
        )
        for n, step in enumerate(plan.steps, 1):
            for number, holds in checks:
                if not holds(step, n):
                    return [f"rule:{number}", f"step:{n}"]
        return None

    def rule_3(self, step: Step, n: int) -> bool:
        """A declared tool, and no argument ``idempotency_key``."""
        # Each name by its string data, so no name's hash or comparison runs.
        names = (string_data(name) for name in step.arguments)
        return step.tool in self.tools and all(name != _IDEMPOTENCY_KEY for name in names)

    def rule_4(self, step: Step, n: int) -> bool:
        """Each board-ref origin a declared read of the policy that holds a value, an expanded
        origin's declared ref a declared read and its own entry a value; the board is read
        directly, so no stop or helper item is added to the policy's view."""
        for origin in step.arguments.values():
            if type(origin) is Origin and origin.ref is not None:
                ref = origin.ref
                try:
                    p = parse_ref(ref)
                except ValueError:  # not a ref under the grammar
                    return False
                declared = f"{p.level}:{p.name}"
                if (declared if p.candidate is not None else ref) not in self.reads:
                    return False
                # A grounded declared ref's own entry holds no value: it is a dict of entries.
                if entry_ref(ref) not in self.board:
                    return False
                if issubclass(type(at(self.board, ref)), Undetermined):
                    return False
        return True

    def rule_6(self, step: Step, n: int) -> bool:
        """Each requirement on a ``d:`` ref with no field path that has a provider, and can
        hold."""
        return all(self.can_hold(r) for r in step.requirements)

    def can_hold(self, requirement: Requirement) -> bool:
        ref = cast(object, requirement.ref)
        if type(ref) is not str:  # by type alone: isinstance reads the policy's __class__
            return False
        try:
            p = parse_ref(ref)
        except ValueError:  # not a ref under the grammar
            return False
        if p.level != "d" or p.fields or p.candidate is not None:
            return False
        if ref not in self.loaded._providers or ref in self.loaded._grounded:
            return False
        return hold_failure(requirement.value, self.loaded._types[ref], {}) is None


def _has_set(step: Step) -> bool:
    return any(type(o) is Origin and o.set is not None for o in step.arguments.values())


def _has_candidate(step: Step) -> bool:
    return any(type(o) is Origin and o.candidate is not None for o in step.arguments.values())


def _text(value: str | None) -> bool:
    """Text holding a character that is not whitespace."""
    return value is not None and any(not c.isspace() for c in value)


def _rule_5(plan: Plan) -> bool:
    """``reply`` on ``clarify`` and ``inform`` only, ``reason`` required on ``escalate``, each
    carried text holding a character that is not whitespace, and the kind not ``broken``."""
    if plan.kind is PlanKind.broken:
        return False
    if (plan.reply is not None) != (plan.kind in (PlanKind.clarify, PlanKind.inform)):
        return False
    if plan.kind is PlanKind.escalate and plan.reason is None:
        return False
    return all(t is None or _text(t) for t in (plan.reason, plan.reply))


def _rule_2(step: Step, n: int) -> bool:
    """Each binding names an earlier step, by a path null or in the field-path grammar."""
    for origin in step.arguments.values():
        if type(origin) is Binding:
            if not 1 <= origin.step < n:
                return False
            if origin.path is not None and not is_field_path(origin.path):
                return False
    return True


def _rule_7(step: Step, n: int) -> bool:
    """Every literal argument, argument name and requirement value in the value domain."""
    # Each name is a value, its length tested before its code points are read; then two
    # names of one string data, which ``Step`` keeps as given, put the arguments outside the
    # domain, found by string data so no name's hash or comparison runs.
    if any(domain_error(name) is not None for name in step.arguments):
        return False
    names = [string_data(name) for name in step.arguments]
    if len(set(names)) != len(names):
        return False
    for origin in step.arguments.values():
        literal = type(origin) is Origin and origin.ref is None
        if literal and domain_error(cast(Origin, origin).literal) is not None:
            return False
    # Tested before rule 6, which walks each value against its ref's type at every place.
    return all(domain_error(r.value) is None for r in step.requirements)


# ---------------------------------------------------------------------------------------------
# The plan item and the plan built from it


def context_member(
    loaded: LoadedDesign, board: Mapping[str, object] | None
) -> dict[str, Any] | None:
    """The ``plan`` item's ``context``: null with no block; else the design's name, the
    switches as declared and each entry rendered over ``board``, or no entries at a broken
    entry, where ``board`` is ``None``."""
    block = loaded._context
    if block is None:
        return None
    entries = [] if board is None else [_entry(loaded, board, e) for e in block["entries"]]
    return {
        "design": loaded._design.name,
        "entries": entries,
        "plan": block["plan"],
        "steps": block["steps"],
        "record": block["record"],
    }


def _entry(
    loaded: LoadedDesign, board: Mapping[str, object], entry: dict[str, Any]
) -> dict[str, Any]:
    """One context entry: its text rendered by the closed rule, or, when a board-ref slot
    reads a marker, no text and that marker, the flagged ref's first."""
    ref, text = entry["ref"], entry["text"]
    values: dict[str, object] = {}
    markers: list[tuple[str, Undetermined]] = []
    for slot in text["slots"]:
        tag, origin = next(iter(slot["origin"].items()))
        if tag == "param":
            values[slot["name"]] = loaded._leaf_value(origin)
            continue
        parsed = parse_ref(origin)
        value = walk(board[f"{parsed.level}:{parsed.name}"], parsed.fields, origin)
        if isinstance(value, Undetermined):
            markers.append((origin, value))
        values[slot["name"]] = value
    if markers:
        flagged = [m for r, m in markers if r == ref]
        marker = flagged[0] if flagged else markers[0][1]
        return {"ref": ref, "text": None, "marker": _marker_document(marker)}
    return {"ref": ref, "text": _render(text["template"], values), "marker": None}


def plan_item(
    plan: Plan,
    derivation: list[dict[str, Any]] | None,
    context: dict[str, Any] | None,
    *,
    rows: list[int] | None = None,
) -> dict[str, Any]:
    """The ``plan`` item, with ``derivation`` when a function policy wrote the plan, ``rows``
    and any ``slots`` when the policy table did, neither at a broken entry, where no policy
    ran, and ``context`` as ``context_member`` gives it."""
    cause = plan.cause
    item: dict[str, Any] = {
        "item": "plan",
        "kind": plan.kind.value,
        "steps": [step_document(step) for step in plan.steps],
        "reason": plan.reason,
        "reply": plan.reply,
        "cause": None if cause is None else {"code": cause.code.value, "refs": list(cause.refs)},
        "context": context,
    }
    if derivation is not None:
        item["derivation"] = derivation
    if plan.slots:
        item["slots"] = {
            k: [{"name": s["name"], "origin": dict(s["origin"])} for s in v]
            for k, v in plan.slots.items()
        }
    if rows is not None:
        item["rows"] = rows
    return item


def _plan_of(item: dict[str, Any]) -> Plan:
    """The policy's plan rebuilt from its item, the copy the ledger stores."""
    steps = _steps(item["steps"])
    slots = cast("dict[str, Any]", item.get("slots", {}))
    if slots:
        return declared_plan(item["kind"], steps, item["reason"], item["reply"], slots)
    return Plan(item["kind"], steps, reason=item["reason"], reply=item["reply"])


def _steps(documents: list[dict[str, Any]]) -> list[Step]:
    """Steps rebuilt from their document form, the ``plan`` item's or a declaration's."""

    def origin(o: dict[str, Any]) -> Origin | Binding:
        if "step" in o:
            return Binding(o["step"], o["path"])
        if "ref" in o:
            return Origin(ref=o["ref"])
        if "set" in o:
            return candidate_literal(o["literal"], o["set"])
        return Origin(literal=o["literal"])

    return [
        Step(
            s["tool"],
            {name: origin(o) for name, o in s["arguments"].items()},
            [Requirement(r["ref"], r["value"]) for r in s["requirements"]],
        )
        for s in documents
    ]


# ---------------------------------------------------------------------------------------------
# The policy table


def table_plan(
    loaded: LoadedDesign, board: Mapping[str, object]
) -> tuple[Plan, list[int], Exception | None]:
    """The policy table's plan over ``board``, held to the plan rules: ``table_choice``'s,
    or ``broken`` naming the first rule it breaks."""
    plan, rows, raised = table_choice(loaded, board)
    if plan.cause is None:
        refs = _Rules(loaded, board).first_break(plan, declared=True)
        if refs is not None:
            return engine_plan(PlanKind.broken, Cause(_C.invalid_return, refs)), rows, None
    return plan, rows, raised


def table_choice(
    loaded: LoadedDesign, board: Mapping[str, object]
) -> tuple[Plan, list[int], Exception | None]:
    """The policy table's evaluation over ``board``: the plan its winning row's declaration
    gives, built and rendered, or rule 2's ``escalate``; the matched rows; and the exception
    a helper or a comparison raised, the plan then ``broken``. No plan rule is applied."""
    table = loaded._table(_POLICY)
    reads, params = loaded._reads[_POLICY], loaded._params[_POLICY]
    leaves = {path: loaded._leaf_value(path) for path in params}
    view = ReadView(
        Role(ComponentKind.policy, _POLICY),
        reads,
        params,
        board,
        leaves,
        grounded=loaded._grounded,
        sets=loaded._candidate_sets,
    )
    inputs = cast("list[dict[str, Any]]", table["inputs"])
    rows = cast("list[dict[str, Any]]", table["rows"])
    declarations = loaded._design.plans
    try:
        values = read_inputs(inputs, view, leaves)
        conditions = {
            k: _conditions(loaded, board, view, declarations[row["plan"]])
            for k, row in enumerate(rows)
        }
        outcome = evaluate(
            rows,
            "F",
            None,
            [i["read"] for i in inputs],
            values,
            leaves,
            conditions,
            loaded._candidate_sets,
        )
        if outcome.plan is None:
            refs = list(cast(Undetermined, outcome.value).cause.refs)
            return engine_plan(PlanKind.escalate, Cause(_C.required, refs)), outcome.rows, None
        return _declared(loaded, board, loaded._plan(outcome.plan)), outcome.rows, None
    except Exception as exc:  # a helper's raise, or an incomparable comparison
        return engine_plan(PlanKind.broken, Cause(_C.raised, [_POLICY])), [], exc
    finally:
        view.close()


def _conditions(
    loaded: LoadedDesign, board: Mapping[str, object], view: ReadView, plan: PlanDeclaration
) -> list[Condition]:
    """What a row naming ``plan`` needs held besides its cells, in rule 2's order: its
    board-ref origins and slots; for a grounded plan, first its set's ref, then each expanded
    step's origins by expansion position, an origin on a grounded declared ref as its
    candidate's grounded ref, read from the board, then its slots."""
    if plan.grounded is None:
        return [Condition(ref, view.read(ref), at) for ref, at in plan_conditions(plan)]
    set_ref = plan.grounded
    value = board[set_ref]
    out = [Condition(set_ref, value, (1, 0))]
    if isinstance(value, Undetermined):
        return out  # no expansion to walk
    origins: list[str] = []
    slots: list[str] = []
    for ref, at_ in plan_conditions(plan):
        (origins if at_[0] == 1 else slots).append(ref)
    for candidate in candidate_values(value, loaded._candidate_sets[set_ref]):
        for step in plan.steps:
            arguments = cast(Step, step).arguments
            for name in sorted(arguments):
                origin = arguments[name]
                if type(origin) is not Origin or origin.ref is None:
                    continue
                ref = _expanded(loaded, origin.ref, candidate)
                held = at(board, ref) if ref != origin.ref else view.read(ref)
                out.append(Condition(ref, held, (1, len(out))))
    out += [Condition(ref, view.read(ref), (2, len(out) + i)) for i, ref in enumerate(slots)]
    return out


def _expanded(loaded: LoadedDesign, ref: str, candidate: str) -> str:
    """An origin's ref in a candidate's copy: a grounded declared ref as that candidate's
    grounded ref, any other as it is."""
    return grounded_ref(ref, candidate) if ref in loaded._grounded else ref


def _expansion(
    loaded: LoadedDesign, board: Mapping[str, object], declaration: dict[str, Any]
) -> list[dict[str, Any]]:
    """A grounded declaration's steps, one copy per candidate in the set's order: a candidate
    origin as the candidate-value literal of the whole member, an origin on a grounded
    declared ref as the candidate's grounded ref, a binding as the same copy's step."""
    set_ref = cast(str, declaration["grounded"])
    members = cast("list[object]", board[set_ref])
    values = candidate_values(members, loaded._candidate_sets[set_ref])
    steps = cast("list[dict[str, Any]]", declaration["steps"])
    out: list[dict[str, Any]] = []
    for copy, (member, value) in enumerate(zip(members, values, strict=True)):
        for step in steps:
            arguments: dict[str, Any] = {}
            for name, o in cast("dict[str, dict[str, Any]]", step["arguments"]).items():
                if "candidate" in o:
                    arguments[name] = {"literal": member, "set": set_ref}
                elif "ref" in o:
                    arguments[name] = {"ref": _expanded(loaded, cast(str, o["ref"]), value)}
                elif "step" in o:
                    arguments[name] = {"step": copy * len(steps) + o["step"], "path": o["path"]}
                else:
                    arguments[name] = o
            out.append({**step, "arguments": arguments})
    return out


def _declared(
    loaded: LoadedDesign, board: Mapping[str, object], declaration: dict[str, Any]
) -> Plan:
    """A plan declaration's document object as the plan it gives: its steps as the document
    holds them, in canonical order, and its ``reason`` and ``reply`` rendered from their slots,
    each slot's origin in ``slots``."""
    texts: dict[str, str | None] = {}
    slots: dict[str, list[dict[str, Any]]] = {}
    for member in ("reason", "reply"):
        text = cast("dict[str, Any] | None", declaration[member])
        if text is None:
            texts[member] = None
            continue
        values: dict[str, object] = {}
        for slot in text["slots"]:
            tag, origin = next(iter(slot["origin"].items()))
            if tag == "param":
                values[slot["name"]] = loaded._leaf_value(origin)
            else:
                parsed = parse_ref(origin)
                base = board[f"{parsed.level}:{parsed.name}"]
                value: object = walk(base, parsed.fields, origin)
                field = loaded._candidate_sets.get(origin)
                if field is not None and type(value) is list:  # a set of records' candidates
                    value = candidate_values(cast("list[object]", value), field)
                values[slot["name"]] = value
        texts[member] = _render(cast(str, text["template"]), values)
        if text["slots"]:
            slots[member] = text["slots"]
    if declaration["grounded"] is not None:
        steps = _steps(_expansion(loaded, board, declaration))
    else:
        steps = _steps(declaration["steps"])
    return declared_plan(declaration["kind"], steps, texts["reason"], texts["reply"], slots)
