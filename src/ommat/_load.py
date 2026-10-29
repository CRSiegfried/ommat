# pyright: reportPrivateUsage=false
"""Load: the load checks, the design document and its hash, the graph and the
ref-to-provider map, the class trees and the answer grid.

Load gathers the providers, a refused form's included, then runs
checks 1 to 4, 7, 8, 10 and 12. Once 10 and 12 pass it builds the document and hashes it.
Once 1 to 4, 10 and 12 pass it runs check 8's leaf-type test, check 9, check 10's test of
each declared class tree, and builds the graph for check 5. It raises ``InvalidDesign`` with
every failure, or returns the loaded design.

A document passes check 11's gate first, a failure there the only one; then load builds a
``Design`` from it, resolving each function component, and runs the checks on it. Once every
component resolved and 1 to 4, 10 and 12 pass, it compares the document it computes with the
one given.
"""

import importlib
import inspect
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType
from ommat._canonical import (
    _DecodeError,
    _DuplicateKey,
    decode,
    document_depth,
    encode,
    order_of,
    ordered,
    sha256_ref,
    with_order,
)
from ommat._design import (
    _BUILT_BY_LOAD,
    Context,
    ContextEntry,
    Design,
    DesignFailure,
    Group,
    InvalidDesign,
    LoadedDesign,
    Parameter,
    PlanDeclaration,
    Policy,
    PolicyTable,
    Question,
    Requirement,
    Rule,
    Source,
    SourceRequirement,
    Structured,
    Table,
    Target,
    Text,
    When,
    document_value,
    label_parts,
    live_value,
    new_failure,
)
from ommat._form import X_ROLES, check_form
from ommat._markers import PlanKind
from ommat._plan import Binding, Origin, Step, step_document
from ommat._read_view import is_candidate_read
from ommat._refs import (
    _ParsedRef,
    code_point_sorted,
    is_declared_name,
    is_field_path,
    is_segment,
    parse_ref,
)
from ommat._table import (
    ANY,
    COMPARISONS,
    HELPERS,
    Axis,
    NotationError,
    overlap,
    plan_conditions,
    python_input,
    python_row,
)
from ommat._types import (
    AnswerType,
    DeclaredMemo,
    FailureKind,
    Interval,
    Label,
    ListOf,
    MappingOf,
    Named,
    Nullable,
    RecordOf,
    Scalar,
    Type,
    TypeFailure,
    check_declared,
    check_definitions,
    conforms,
    hold_failure,
    named,
    parse_type,
    read_type,
    render,
    resolve,
    value_conforms,
)
from ommat._values import (
    CANDIDATE_FIELD_FORMAT,
    CANDIDATE_FORMAT,
    CONTEXT_FORMAT,
    DESCRIPTION_FORMAT,
    DESIGN_FORMAT,
    DESIGN_FORMATS,
    EXTRACTION_FORMAT,
    KEY_ORDER_FORMAT,
    KNOWN_FORMAT,
    MAX_DOCUMENT_DEPTH,
    MISSING,
    NOUL_DESCRIPTION_FORMAT,
    STRUCTURED_WORDING_FORMAT,
    TARGETS_FORMAT,
    domain_error,
    in_domain,
    member_named,
    members_of,
    nesting_depth,
    power,
    quote,
    shares_a_container,
    string_data,
)

_STATE = "t:state"
_NOUL_LABELS = ("yes", "no")
_ROLES = frozenset({"policy", "codec"})
_WORDS = frozenset(
    {
        "number",
        "integer",
        "string",
        "boolean",
        "bytes",
        "label",
        "list",
        "mapping",
        "record",
        "answer",
        "or",
        "null",
    }
)
_GATING = frozenset({1, 2, 3, 4, 10, 12})
_NUMBER = Scalar("number")
_NUMBERS = MappingOf(_NUMBER)
_UNIT = Interval("number", 0, 1)
_STRING = Scalar("string")
_EXTRACTION_TREE = {
    "kind": "open",
    "value_type": "open",
    "is_list": "open",
    "value": "content",
    "guess": "content",
    "p_null": "open",
    "confidence": "open",
    "span": "content",
}
"""An extraction answer's default class tree, which is also its full tree."""
_EXTRACTION_SET = ("value", "guess", "span")
"""The members a class name declared at an extraction ``q:`` ref sets."""
_UNITS = MappingOf(_UNIT)
_COUNT = Interval("integer", 0, 2**53 - 1)
"""What ``size`` returns: an integer, at least 0, within the value domain."""
_SIGNATURES: dict[str, dict[str, Type]] = {
    "pick_with_margin": {"bar": _UNIT, "margin": _UNIT},
    "pick_with_thresholds": {"thresholds": _UNITS, "margin": _UNIT},
    "mass": {"labels": ListOf(_STRING)},
    "extract": {"null_bar": _UNIT},
}
"""Each table helper's arguments after its read, with the type a binding's leaf conforms
to."""
_TYPE_SIZE = 2**20
"""The most code points a type spells with its definitions resolved, and the types of the
refs the class trees cover, summed."""
_ANSWER_SIZE = len("answer")


def load(
    design: Design | bytes | Mapping[str, Any],
    *,
    functions: Mapping[str, Callable[..., Any]] | None = None,
) -> LoadedDesign:
    """Check a design once and return the loaded design, or raise ``InvalidDesign`` carrying
    every failure found."""
    if isinstance(design, Design):
        if functions is not None:
            raise TypeError("functions is given only with a document; a Design holds its callables")
        return _Loader(check_form(design)).run()
    if isinstance(design, bytes | Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        if functions is not None and not isinstance(functions, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("functions is a mapping from a component's key to its callable")
        return _load_document(design, {} if functions is None else functions)
    raise TypeError(f"load takes a Design or a design document, not {type(design).__name__}")


def _load_document(given: bytes | Mapping[str, Any], functions: Mapping[str, Any]) -> LoadedDesign:
    """Check 11's gate, then the design built from the document under every other check, and
    the comparison."""
    parsed, refusal = _gate(given)
    if refusal is not None:
        names, message = refusal
        raise InvalidDesign([new_failure(11, names, message)], _BUILT_BY_LOAD)
    assert parsed is not None
    resolved, unresolved, failures = _resolve(parsed, functions)
    loader = _Loader(
        check_form(_build(parsed, resolved), unresolved),
        parsed=parsed,
        given=given if isinstance(given, bytes) else None,
        fmt=cast(int, parsed["format"]),
    )
    loader.failures += failures
    return loader.run()


_q = quote
_NO_TABLE = "\0"  # no table owns the use


@dataclass(frozen=True, slots=True)
class _Use:
    """A ref a declaration reads: the declaration, its role and the parsed ref."""

    owner: str  # the declared name, or "policy" for the policy and "" for the plan
    where: str
    role: str  # need, slot, candidate, label-set, text, read, boundary, policy, requirement,
    # source-requirement, group
    ref: str
    parsed: _ParsedRef

    @property
    def base(self) -> str:
        return f"{self.parsed.level}:{self.parsed.name}"


class _Loader:
    def __init__(
        self,
        checked: Any,
        *,
        parsed: dict[str, Any] | None = None,
        given: bytes | None = None,
        fmt: int = DESIGN_FORMAT,
    ) -> None:
        # From a document: the parsed document and, given as bytes, the bytes.
        self.parsed = parsed
        self.given = given
        # The format the document is written under: a document's own, read under its rules.
        self.format = fmt
        self.d: Design = checked.design
        self.failures: list[DesignFailure] = list(checked.failures)
        # What a member check 10's form test refused still provides or declares.
        self.form_providers: tuple[tuple[str, str], ...] = checked.providers
        self.leaves: frozenset[str] = frozenset(self.d.parameters) | frozenset(checked.leaves)
        self.refused_sources: set[str] = set()
        self.refused_rules: set[str] = set()
        self.refused_plans: set[str] = set()
        self.grounded: set[str] = set()
        self.providers: dict[str, list[str | None]] = {}
        self.uses: list[_Use] = []
        self.candidates: list[tuple[str, str, str]] = []  # each candidate slot's source
        # Each candidate read, which no ref spells: its owner, where it is, the read and its
        # role, a read or a boundary's key.
        self.candidate_reads: list[tuple[str, str, str, str]] = []
        # Each plan declaration's text holding a candidate slot, which no text holds.
        self.candidate_slots: list[tuple[str, str]] = []
        self.defs: dict[str, Type] = {}
        self.tainted: set[str] = set()
        # Each definition's size, and each provided ref's and the state's, once it passes.
        self.def_sizes: dict[str, int] = {}
        # What check 10's type checks learnt of the definitions, kept across declarations.
        self.declared_memo = DeclaredMemo()
        self.ref_sizes: dict[str, int] = {}
        self.types: dict[str, Type] = {}
        self.leaf_types: dict[str, Type] = {}
        self.leaf_written: dict[str, Type] = {}  # each leaf's type as declared, for messages
        self.state_type: Type | None = None
        self.class_keys: list[tuple[str, _ParsedRef]] = []  # each classes key check 7 passed
        # Each context entry's candidate slot, by its flagged ref, refused at check 4.
        self.context_candidates: list[tuple[str, str]] = []
        # Each leaf check 8 refuses under a boundaries field, left out of check 9's boundaries
        # clauses.
        self.refused_leaves: set[str] = set()
        # Each leaf's value as Python holds it, bytes as ``bytes``; a document writes a bytes
        # value as base64url, read back here once its type passes check 10.
        self.live: dict[str, object] = {}

    def fail(self, check: int, names: tuple[str, ...], message: str) -> None:
        self.failures.append(new_failure(check, code_point_sorted(names), message))

    def failed(self, checks: frozenset[int]) -> bool:
        return any(f.check in checks for f in self.failures)

    # --- the declarations that are not refused ------------------------------------------

    def sources(self) -> Iterator[tuple[str, Source]]:
        for sid, source in self.d.sources.items():
            if sid not in self.refused_sources:
                yield sid, source

    def rules(self) -> Iterator[tuple[str, Rule]]:
        """Each function rule check 12 lets through."""
        for rid, rule in self.d.rules.items():
            if rid not in self.refused_rules and isinstance(rule, Rule):
                yield rid, rule

    def tables(self) -> Iterator[tuple[str, Table]]:
        """Each rule table check 12 lets through."""
        for rid, rule in self.d.rules.items():
            if rid not in self.refused_rules and isinstance(rule, Table):
                yield rid, rule

    def components(self) -> Iterator[tuple[str, Rule | Table]]:
        """Each rule, function or table, check 12 lets through."""
        for rid, rule in self.d.rules.items():
            if rid not in self.refused_rules:
                yield rid, rule

    def policy(self) -> Policy | None:
        return self.d.policy if isinstance(self.d.policy, Policy) else None

    def policy_table(self) -> PolicyTable | None:
        return self.d.policy if isinstance(self.d.policy, PolicyTable) else None

    def plans(self) -> Iterator[tuple[str, PlanDeclaration]]:
        """Each plan declaration check 12 lets through."""
        for name, plan in self.d.plans.items():
            if name not in self.refused_plans:
                yield name, plan

    def named_plans(self) -> list[str]:
        """The declared plans the policy table's rows name, in first-named order."""
        table = self.policy_table()
        if table is None:
            return []
        names = [cast(str, row["plan"]) for row in table.rows]
        return [
            n for n in dict.fromkeys(names) if n in self.d.plans and n not in self.refused_plans
        ]

    # --- the run ---------------------------------------------------------------------------

    def run(self) -> LoadedDesign:
        self.check_12()
        self.gather()
        self.check_2()
        self.check_3()
        self.check_4()
        self.check_6()
        self.check_1()
        self.check_10()
        self.check_8()
        document: bytes | None = None
        if not self.failed(frozenset({10, 12})):
            # Check 10 holds every value in the domain, so the encoding cannot refuse.
            document = encode(self.document(), MAX_DOCUMENT_DEPTH)
        graph: Mapping[str, frozenset[str]] = MappingProxyType({})
        if not self.failed(_GATING):
            self.check_8_types()
            self.check_9()
            self.check_10_trees()
            graph = self.graph()
            self.check_5(graph)
        if self.parsed is not None and not self.failed(_GATING | {11}):
            assert document is not None
            self.compare(document)
        if self.failures:
            raise InvalidDesign(self.failures, _BUILT_BY_LOAD)
        assert document is not None
        return self.loaded(document, graph)

    # --- check 12 --------------------------------------------------------------------------

    def check_12(self) -> None:
        def refuse(into: set[str], names: tuple[str, ...], what: str, form: str) -> None:
            into.add(names[0])
            self.fail(12, names, f"{what} is a {form}, which this engine does not implement")

        sources, rules = self.refused_sources, self.refused_rules
        per_candidate = _per_candidate_refs(self.d)
        for sid, source in self.d.sources.items():
            what = f"source {_q(sid)}"
            if source.grounded is not None:
                self.grounded.add(sid)
                for ref, provided in source.provides.items():
                    if isinstance(provided, Mapping) and cast("Mapping[str, Any]", provided).get(
                        "candidates"
                    ):
                        refuse(sources, (sid, ref), f"{what}'s {_q(ref)}", "grounded candidate set")
                for ref, question in _questions(source):
                    if ref not in per_candidate:
                        continue
                    shown = f"{what}'s {_q(ref)}"
                    if question.kind == AnswerKind.extraction:
                        refuse(sources, (sid, ref), shown, _PER_CANDIDATE_EXTRACTION)
                    elif isinstance(question.labels, Mapping):
                        refuse(sources, (sid, ref), shown, _PER_CANDIDATE_LABEL_SET)
            texts = [
                origin["ref"]
                for _, question in _questions(source)
                for _, origin in _extraction_origins(question)
                if isinstance(origin.get("ref"), str)
            ]
            for need in [*_declared_needs(source), *texts]:
                if _declared_base(need) in per_candidate:
                    refuse(
                        sources,
                        (sid, need),
                        f"{what}'s need or slot ref {_q(need)}",
                        _GROUNDED_NEED,
                    )
        for rid, rule in self.d.rules.items():
            what = f"rule {_q(rid)}"
            if rule.grounded is not None:
                self.grounded.add(rid)
                if rule.candidates:
                    refuse(rules, (rid,), what, "grounded candidate set")

    # --- gathering, and check 7 ------------------------------------------------------------

    def ref(self, ref: str, where: str, owner: str) -> _ParsedRef | None:
        """The parsed ref, or ``None`` after a check-7 failure; a refused ref is left out of
        every other check."""
        try:
            return parse_ref(ref)
        except ValueError:  # the ref grammar's refusal
            names = (owner, ref) if owner else (ref,)
            self.fail(7, names, f"{where} names {_q(ref)}, which is not a ref")
            return None

    def provide(self, ref: str, provider: str, where: str, *, checked: bool) -> None:
        if checked:
            parsed = self.ref(ref, where, provider)
            if parsed is None:
                return
            if parsed.candidate is not None or parsed.fields:
                self.fail(
                    7, (provider, ref), f"{where} provides {_q(ref)}, which is not a declared ref"
                )
                return
        else:  # a refused form: gathered as a provider only (check 12)
            try:
                parsed = parse_ref(ref)
            except ValueError:  # the ref grammar's refusal
                return
            if parsed.candidate is not None or parsed.fields:
                return
        self.providers.setdefault(ref, []).append(provider)

    def read_use(self, owner: str, where: str, role: str, ref: str) -> None:
        """A component's read: a ``candidate`` read kept apart, any other a use."""
        if isinstance(cast(object, ref), str) and is_candidate_read(ref):
            self.candidate_reads.append((owner, where, ref, "read"))
        else:
            self.use(owner, where, role, ref)

    def use(self, owner: str, where: str, role: str, ref: str) -> None:
        parsed = self.ref(ref, where, owner)
        if parsed is not None:
            self.uses.append(_Use(owner, where, role, ref, parsed))

    def gather(self) -> None:
        self.providers[_STATE] = [None]
        for sid, source in self.d.sources.items():
            checked = sid not in self.refused_sources
            for ref in source.provides:
                self.provide(ref, sid, f"source {_q(sid)}", checked=checked)
        for rid, rule in self.d.rules.items():
            checked = rid not in self.refused_rules
            self.provide(rule.provides, rid, f"rule {_q(rid)}", checked=checked)
        for ref, provider in self.form_providers:
            self.provide(ref, provider, "", checked=False)
        for sid, source in self.sources():
            where = f"source {_q(sid)}"
            if source.grounded is not None:
                self.use(sid, f"{where}'s grounding", "grounding", source.grounded)
            for ref in source.needs:
                self.use(sid, where, "need", ref)
            for ref, slot in _slots(source):
                tag, value = next(iter(slot["origin"].items()))
                if tag == "ref":
                    self.use(sid, f"{where}'s question {_q(ref)}", "slot", value)
                elif tag == "candidate":
                    self.candidates.append((sid, f"{where}'s question {_q(ref)}", value))
                    self.use(sid, f"{where}'s question {_q(ref)}", "candidate", value)
            for ref, question in _questions(source):
                if isinstance(question.labels, Mapping):
                    named_set = question.labels.get("candidates")
                    if isinstance(named_set, str):
                        shown = f"{where}'s question {_q(ref)}'s label set"
                        self.use(sid, shown, "label-set", named_set)
                for member, origin in _extraction_origins(question):
                    shown = f"{where}'s question {_q(ref)}'s {member}"
                    self.use(sid, shown, "text", origin["ref"])
        for rid, rule in self.components():
            if rule.grounded is not None:
                self.use(rid, f"rule {_q(rid)}'s grounding", "grounding", rule.grounded)
        for rid, rule in self.rules():
            for ref in rule.reads:
                self.read_use(rid, f"rule {_q(rid)}", "read", ref)
        for rid, table in self.tables():
            for n, i in enumerate(table.inputs, 1):
                role = "read" if i["helper"] is None else "helper"
                self.read_use(rid, f"rule {_q(rid)}'s input {n}", role, i["read"])
            self.table_refs(rid, f"rule {_q(rid)}", table.inputs, table.rows)
        policy = self.policy()
        if policy is not None:
            for ref in policy.reads:
                self.read_use("policy", "the policy", "policy", ref)
        policy_table = self.policy_table()
        if policy_table is not None:
            for n, i in enumerate(policy_table.inputs, 1):
                self.read_use("policy", f"the policy table's input {n}", "policy", i["read"])
            self.table_refs("policy", "the policy table", policy_table.inputs, policy_table.rows)
        for name, plan in self.plans():
            where = f"plan declaration {_q(name)}"
            if plan.grounded is not None:
                self.use(name, f"{where}'s grounding", "plan-grounding", plan.grounded)
            for n, step in enumerate(plan.steps, 1):
                for arg, origin in cast(Step, step).arguments.items():
                    shown = f"{where}'s step {n} argument {_q(arg)}"
                    if isinstance(origin, Origin) and origin.candidate is not None:
                        self.use(name, shown, "plan-candidate", origin.candidate)
                    elif isinstance(origin, Origin) and origin.ref is not None:
                        self.use(name, shown, "origin", origin.ref)
                for requirement in cast(Step, step).requirements:
                    shown = f"{where}'s step {n} requirement"
                    parsed = self.ref(requirement.ref, shown, name)
                    if parsed is not None and parsed.fields:
                        self.fail(
                            7,
                            (name, requirement.ref),
                            f"{shown} names {_q(requirement.ref)}, which has a field path",
                        )
                    elif parsed is not None:
                        self.uses.append(_Use(name, shown, "requirement", requirement.ref, parsed))
            for member, text in (("reason", plan.reason), ("reply", plan.reply)):
                if text is None:
                    continue
                for slot in text.slots:
                    tag, value = next(iter(slot["origin"].items()))
                    if tag == "ref":
                        self.use(name, f"{where}'s {member}", "plan-slot", value)
                    elif tag == "candidate":
                        self.candidate_slots.append((name, f"{where}'s {member}"))
        for owner, where, boundaries, _ in self.boundary_fields():
            for key in boundaries:
                # Check 3 finds each key's provider and check 7 its form; check 4 gives a
                # key no level test, since check 9 holds it to its component's reads.
                if isinstance(cast(object, key), str) and is_candidate_read(key):
                    self.candidate_reads.append((owner, f"{where}'s boundaries", key, "boundary"))
                else:
                    self.use(owner, f"{where}'s boundaries", "boundary", key)
        for requirement in self.d.requirements:
            # A requirement names a whole estimate: a field path fails check 7
            # and, refused, the ref is left out of every other check.
            parsed = self.ref(requirement.ref, "a plan-level requirement", "")
            if parsed is not None and parsed.fields:
                self.fail(
                    7,
                    (requirement.ref,),
                    f"a plan-level requirement names {_q(requirement.ref)}, which has a field path",
                )
            elif parsed is not None:
                self.uses.append(
                    _Use("", "a plan-level requirement", "requirement", requirement.ref, parsed)
                )
        for gname, group in self.d.groups.items():
            for ref in group.refs:
                self.use(gname, f"group {_q(gname)}", "group", ref)
        for sid, source in self.sources():
            if source.requirement is None:
                continue
            where = f"source {_q(sid)}'s requirement"
            ref = source.requirement.ref
            # A requirement names a whole estimate, on a source as on the plan.
            parsed = self.ref(ref, where, sid)
            if parsed is not None and parsed.fields:
                self.fail(7, (sid, ref), f"{where} names {_q(ref)}, which has a field path")
            elif parsed is not None:
                self.uses.append(_Use(sid, where, "source-requirement", ref, parsed))
        for key in self.d.classes:
            parsed = self.ref(key, "the classes block", "")
            if parsed is not None:
                self.class_keys.append((key, parsed))
        for entry in self.context_entries():
            where = f"the context entry for {_q(entry.ref)}"
            self.use("", where, "flagged", entry.ref)
            for slot in entry.text.slots:
                tag, value = next(iter(slot["origin"].items()))
                if tag == "ref":
                    self.use(entry.ref, f"{where}'s text", "context-slot", value)
                elif tag == "candidate":
                    self.context_candidates.append((entry.ref, where))

    def table_refs(
        self, owner: str, where: str, inputs: Sequence[Any], rows: Sequence[Any]
    ) -> None:
        """Check 7 over the refs a table's cells and pass-throughs name; check 9 holds each to
        an input of the table."""
        for k, row in enumerate(rows, 1):
            named = list(_cell_refs(row["cells"]))
            output = row.get("output")
            if isinstance(output, dict) and "ref" in output:
                named.append(cast("dict[str, str]", output)["ref"])
            for ref in named:
                if not is_candidate_read(ref):  # a candidate read is no ref (check 7)
                    self.ref(ref, f"{where}'s row {k}", owner)

    def context_entries(self) -> tuple[ContextEntry, ...]:
        """The context block's entries check 10's form test kept, none with no block."""
        context = self.d.context
        return () if context is None else tuple(context.entries)

    # --- checks 2, 3 and 4 -----------------------------------------------------------------

    def check_2(self) -> None:
        for ref, providers in self.providers.items():
            if len(providers) > 1:
                ids = [p for p in providers if p is not None]
                by = ", ".join(_q(p) for p in code_point_sorted(set(ids)))
                if None in providers:
                    by = "the state declaration, " + by
                self.fail(2, (ref, *set(ids)), f"{_q(ref)} has more than one provider: {by}")

    def check_3(self) -> None:
        for u in self.uses:
            if u.role == "candidate":
                grounding = self.d.sources[u.owner].grounded
                if grounding is not None and grounding != u.ref:
                    self.fail(
                        3,
                        (u.owner, u.ref),
                        f"{u.where} has a candidate slot naming {_q(u.ref)}, not its source's "
                        f"grounding set {_q(grounding)}",
                    )
                continue
            if u.role == "plan-candidate":
                grounding = self.d.plans[u.owner].grounded
                if grounding is not None and grounding != u.ref:
                    self.fail(
                        3,
                        (u.owner, u.ref),
                        f"{u.where} is a candidate origin naming {_q(u.ref)}, not its plan's "
                        f"grounding set {_q(grounding)}",
                    )
                continue
            found = u.base in self.providers
            if not found:
                names = (u.owner, u.ref) if u.owner else (u.ref,)
                self.fail(3, names, f"{u.where} names {_q(u.ref)}, which has no provider")
            elif u.role in ("label-set", "grounding", "plan-grounding") and u.parsed.level in (
                "t",
                "d",
            ):
                if u.parsed.candidate is not None or u.parsed.fields or not self.is_set(u.ref):
                    self.fail(
                        3,
                        (u.owner, u.ref),
                        f"{u.where} names {_q(u.ref)}, which is not a candidate set",
                    )
        per_candidate = _per_candidate_refs(self.d)
        for gname, group in self.d.groups.items():
            refs = [r for r in group.refs if _parsed(r) is not None]
            counts = Counter(refs)
            for ref in code_point_sorted({r for r, n in counts.items() if n > 1}):
                self.fail(3, (gname, ref), f"group {_q(gname)} names {_q(ref)} more than once")
            # A group's q: refs are all ungrounded, or all grounded over one candidate set.
            # A bracketed ref is check 4's to refuse.
            questions = [
                r
                for r in refs
                if r.startswith("q:") and "[" not in r and _declared_base(r) in self.providers
            ]
            if len({per_candidate.get(_declared_base(r) or r) for r in questions}) > 1:
                self.fail(
                    3,
                    (gname,),
                    f"group {_q(gname)} names q: refs grounded over different candidate sets, or "
                    "grounded and ungrounded ones",
                )
        for names, where, target in self.target_links():
            if target not in self.d.targets:
                self.fail(3, names, f"{where} names the target {_q(target)}, which is not declared")
        form_refused = {ref for ref, _ in self.form_providers}  # check 10 has failed them
        for key, parsed in self.class_keys:
            where = f"the classes block's key {_q(key)}"
            if parsed.candidate is not None:
                self.fail(3, (key,), f"{where} has a grounded bracket")
            elif parsed.fields:
                self.fail(3, (key,), f"{where} has a field path")
            elif key not in self.providers:
                self.fail(3, (key,), f"{where} has no provider")
            elif parsed.level == "q" and key not in form_refused and not self.extraction_ref(key):
                self.fail(
                    3,
                    (key,),
                    f"{where} is not t:state, a source's t: ref, a d: ref or an extraction q: ref",
                )

    def target_links(self) -> Iterator[tuple[tuple[str, ...], str, str]]:
        """Each target id a question or a rule names: the question's ref or the rule's id,
        where it is named, and the id."""
        for sid, source in self.sources():
            for ref, question in _questions(source):
                where = f"source {_q(sid)}'s question {_q(ref)}"
                if question.target is not None:
                    yield (ref,), where, question.target
                for latent in question.latent_targets:
                    yield (ref,), f"{where}'s latent_targets", latent["target"]
        for rid, rule in self.components():
            if rule.target is not None:
                yield (rid,), f"rule {_q(rid)}", rule.target

    def is_set(self, ref: str) -> bool:
        """Whether a ref's provider declares it a candidate set."""
        return bool(self.set_field(ref))

    def set_field(self, ref: str) -> bool | str:
        """A ref's provider's ``candidates`` member: a source's provision's, or that of a rule
        that provides it; false for any other ref."""
        provider = self.providers.get(ref, [None])[0]
        if provider is None:
            return False
        source = self.d.sources.get(provider)
        if source is not None:
            provided = source.provides.get(ref)
            if not isinstance(provided, Mapping):
                return False
            return cast("Mapping[str, Any]", provided).get("candidates") or False
        rule = self.d.rules.get(provider)
        if rule is None or rule.provides != ref:
            return False
        return rule.candidates

    def extraction_ref(self, ref: str) -> bool:
        """Whether a ``q:`` ref's provider declares it an extraction question."""
        provider = self.providers.get(ref, [None])[0]
        source = self.d.sources.get(provider) if provider is not None else None
        provided = None if source is None else source.provides.get(ref)
        return isinstance(provided, Question) and provided.kind is AnswerKind.extraction

    def check_4(self) -> None:
        # A source check 10's form test refused in any member is left out, its provides among them.
        form = [f.message for f in self.failures if f.check == 10]
        for sid, source in self.sources():
            where = f"source {_q(sid)}"
            if not source.provides and not any(m.startswith(where) for m in form):
                self.fail(
                    4,
                    (sid,),
                    f"source {_q(sid)} provides no ref; "
                    "a source provides one or more q: or t: refs",
                )
            provided: set[str] = set()
            for ref in source.provides:
                parsed = _declared(ref)
                if parsed is None:
                    continue
                provided.add(parsed.level)
                if parsed.level not in ("q", "t"):
                    self.fail(
                        4,
                        (sid, ref),
                        f"source {_q(sid)} provides {_q(ref)}; a source provides q: or t: refs",
                    )
            if {"q", "t"} <= provided:
                self.fail(
                    4,
                    (sid,),
                    f"source {_q(sid)} provides q: and t: refs; "
                    "a source provides refs of one level",
                )
        for rid, rule in self.components():
            parsed = _declared(rule.provides)
            if parsed is not None and rule.provides != f"d:{rid}":
                message = (
                    f"rule {_q(rid)} provides {_q(rule.provides)}; "
                    f"a rule provides its own d: ref, {_q(f'd:{rid}')}"
                )
                if parsed.level == "d":
                    message += f", and {_q(rule.provides)} would be rule {_q(parsed.name)}'s"
                self.fail(4, (rid, rule.provides), message)
        allowed = {
            "grounding": ("d", "t"),
            "plan-grounding": ("d", "t"),
            "slot": ("d", "t"),
            "label-set": ("d", "t"),
            "text": ("d", "t"),
            "policy": ("d", "t"),
            "requirement": ("d",),
            "source-requirement": ("d",),
            "flagged": ("d", "t"),
            "context-slot": ("d", "t"),
            "origin": ("d", "t"),
            "plan-slot": ("d", "t"),
        }
        for flagged, where in self.context_candidates:
            self.fail(
                4, (flagged,), f"{where} has a candidate slot; a context entry's slots hold none"
            )
        # The clause is on the slot list, so it stands whether or not the slot's ref passes
        # check 7.
        for sid, where, _ in self.candidates:
            if self.d.sources[sid].grounded is None:
                self.fail(
                    4,
                    (sid,),
                    f"{where} has a candidate slot; a source's slots hold candidate only when it "
                    "is grounded",
                )
        self.check_4_grounded_slots()
        holding = {sid for sid, _, _ in self.candidates}
        for sid, source in self.sources():
            asks = any(ref.startswith("q:") for ref in source.provides)
            if source.grounded is not None and sid not in holding and asks:
                self.fail(
                    4,
                    (sid,),
                    f"source {_q(sid)} is grounded, but no question of it holds a candidate slot",
                )
        for u in self.uses:
            if u.role in ("candidate", "plan-candidate"):
                continue
            levels = allowed.get(u.role)
            names = (u.owner, u.ref) if u.owner else (u.ref,)
            if levels is not None and u.parsed.level not in levels:
                shown = " or ".join(f"{level}:" for level in levels)
                self.fail(4, names, f"{u.where} reads {_q(u.ref)}, which is not at level {shown}")
            elif u.role == "flagged" and u.parsed.level == "t" and u.base != _STATE:
                self.fail(
                    4, names, f"{u.where} flags {_q(u.ref)}, which is not t:state or a d: ref"
                )
            elif u.parsed.candidate is not None:
                self.fail(4, names, f"{u.where} names {_q(u.ref)}, a grounded ref")
        self.check_4_helpers()
        self.check_4_grounded()

    def check_4_grounded_slots(self) -> None:
        """Only a per-candidate question holds a slot on a grounded declared ref, and only on
        one grounded over its own set."""
        per_candidate = _per_candidate_refs(self.d)
        for sid, source in self.sources():
            for ref, slot in _slots(source):
                tag, value = next(iter(slot["origin"].items()))
                if tag != "ref" or not isinstance(value, str):
                    continue
                set_ref = per_candidate.get(_declared_base(value) or "")
                if set_ref is None:
                    continue
                if ref not in per_candidate:
                    self.fail(
                        4,
                        (sid, value),
                        f"source {_q(sid)}'s question {_q(ref)} has a slot on {_q(value)}, a "
                        "grounded declared ref; only a per-candidate question holds one",
                    )
                elif set_ref != source.grounded:
                    self.fail(
                        4,
                        (sid, value),
                        f"source {_q(sid)}'s question {_q(ref)} has a slot on {_q(value)}, "
                        f"grounded over {_q(set_ref)}, not its source's set",
                    )

    def check_4_grounded(self) -> None:
        """Only a grounded rule or table reads ``candidate`` or ``candidate.f``; a table reads a
        grounded declared ref only when grounded over the same set, and the policy table none."""
        for owner, where, read, role in self.candidate_reads:
            if role == "boundary":
                continue  # check 9's
            rule = self.d.rules.get(owner)
            if rule is None or rule.grounded is None:
                self.fail(
                    4,
                    (owner,),
                    f"{where} reads {_q(read)}, which only a grounded rule or table reads",
                )
        per_candidate = _per_candidate_refs(self.d)
        for u in self.uses:
            if u.role == "plan-candidate" and self.d.plans[u.owner].grounded is None:
                self.fail(
                    4,
                    (u.owner,),
                    f"{u.where} is a candidate origin, which only a grounded plan declaration "
                    "holds",
                )
            elif u.base in per_candidate and u.role in ("requirement", "source-requirement"):
                self.fail(
                    4,
                    (u.owner, u.ref) if u.owner else (u.ref,),
                    f"{u.where} names {_q(u.ref)}, a grounded declared ref, which no requirement "
                    "names",
                )
            elif u.base in per_candidate and (
                u.role == "context-slot"
                or (u.role in ("origin", "plan-slot") and self.d.plans[u.owner].grounded is None)
            ):
                self.fail(
                    4,
                    (u.owner, u.ref) if u.owner else (u.ref,),
                    f"{u.where} names {_q(u.ref)}, a grounded declared ref, which only a grounded "
                    "plan's origin names",
                )
        for owner, where in self.candidate_slots:
            if self.d.plans[owner].grounded is None:
                self.fail(4, (owner,), f"{where} has a candidate slot, which no plan text holds")
        tables: list[tuple[str, str | None]] = [(r, t.grounded) for r, t in self.tables()]
        if self.policy_table() is not None:
            tables.append(("policy", None))
        for u in self.uses:
            grounding = next((g for owner, g in tables if owner == u.owner), _NO_TABLE)
            if grounding is _NO_TABLE or u.role in ("grounding", "boundary"):
                continue
            set_ref = per_candidate.get(u.base)
            if set_ref is not None and set_ref != grounding:
                self.fail(
                    4,
                    (u.owner, u.ref),
                    f"{u.where} reads {_q(u.ref)}, a grounded declared ref; a table reads one "
                    "only when grounded over its set",
                )

    def helper_inputs(self) -> Iterator[tuple[str, str, int, dict[str, Any]]]:
        """Each helper input of a table: its owner, where it is, its position and its input."""
        for rid, table in self.tables():
            for n, i in enumerate(table.inputs, 1):
                if i["helper"] is not None:
                    yield rid, f"rule {_q(rid)}'s input {n}", n, i
        policy_table = self.policy_table()
        if policy_table is not None:
            for n, i in enumerate(policy_table.inputs, 1):
                if i["helper"] is not None:
                    yield "policy", f"the policy table's input {n}", n, i

    def check_4_helpers(self) -> None:
        """A helper input's read conforms to its signature's ``ref``: a pick helper's or
        ``mass``'s a question of a probability kind, ``extract``'s an extraction question, and
        ``size``'s a ``t:`` or ``d:`` read, its list type check 9's."""
        for owner, where, _, i in self.helper_inputs():
            helper, read = i["helper"], i["read"]
            if helper not in HELPERS:
                continue  # check 6's
            if is_candidate_read(read):
                if helper != "size":
                    self.fail(
                        4, (owner,), f"{where}: {helper} reads {_q(read)}, which is not a question"
                    )
                continue
            parsed = _parsed(read)
            if parsed is None:
                continue  # check 7's
            names = (owner, read)
            if helper == "size":
                if parsed.level == "q":
                    self.fail(4, names, f"{where}: size reads {_q(read)}, which is no list")
                continue
            if parsed.level != "q" or parsed.fields:
                self.fail(4, names, f"{where}: {helper} reads {_q(read)}, which is not a question")
                continue
            base = f"q:{parsed.name}"
            if base not in self.providers:
                continue  # check 3's
            extraction = self.extraction_ref(base)
            if helper == "extract" and not extraction:
                self.fail(
                    4,
                    names,
                    f"{where}: extract reads {_q(read)}, which is not an extraction question",
                )
            elif helper != "extract" and extraction:
                self.fail(4, names, f"{where}: {helper} reads {_q(read)}, an extraction question")

    def check_6(self) -> None:
        for owner, where, _, i in self.helper_inputs():
            helper = i["helper"]
            if helper not in HELPERS:
                self.fail(
                    6,
                    (owner,),
                    f"{where} calls {_q(helper)}, which is not a helper a table may call: "
                    "pick_with_margin, pick_with_thresholds, size, mass or extract",
                )

    # --- check 1 ---------------------------------------------------------------------------

    def check_1(self) -> None:
        d = self.d
        kinds: dict[str, list[str]] = {}
        blocks: list[tuple[str, list[str]]] = [
            ("source", [s for s, _ in self.sources()]),
            ("rule", [r for r, _ in self.components()]),
            ("plan declaration", [n for n, _ in self.plans()]),
            ("parameter", list(self.leaves)),
            ("tool", list(d.tools)),
            ("type definition", list(d.types)),
            ("group", list(d.groups)),
        ]
        for kind, names in blocks:
            for name in names:
                shown = f"{kind} {_q(name)}"
                if not is_declared_name(name):
                    self.fail(1, (name,), f"{shown} is not in the name grammar")
                if name in _ROLES:
                    self.fail(1, (name,), f"{shown} is a role name")
                if name == "candidate":
                    self.fail(1, (name,), f"{shown} is the name a grounded component reads")
                if kind == "type definition" and name in _WORDS:
                    self.fail(1, (name,), f"{shown} is a word of the type notation")
                if kind not in ("type definition", "group"):
                    kinds.setdefault(name, []).append(kind)
        for name, used in kinds.items():
            if len(used) > 1:
                self.fail(1, (name,), f"{_q(name)} is declared more than once: {', '.join(used)}")
        flagged = Counter(entry.ref for entry in self.context_entries())
        for ref in code_point_sorted({r for r, n in flagged.items() if n > 1}):
            self.fail(1, (ref,), f"two context entries flag {_q(ref)}")
        for sid, source in self.sources():
            if source.oracle == "default":
                self.fail(1, (sid,), f'source {_q(sid)} has the oracle name "default"')
        for node, path in _nodes(self.leaves):
            self.fail(1, (node, path), f"parameter {_q(node)} is a node of {_q(path)}")
        for owner, where, what, members, refs in self.unordered():
            self.repeats(owner, where, what, members, refs=refs)
        for owner, where, rows in self.table_rows():
            for k, row in enumerate(rows, 1):
                for cell in row["cells"]:
                    kind = next(
                        (
                            tag
                            for tag in ("one_of", "markers")
                            if isinstance(cell, dict) and tag in cell
                        ),
                        None,
                    )
                    if kind is not None:
                        seen: set[bytes] = set()
                        for value in cast("list[object]", cast("dict[str, Any]", cell)[kind]):
                            key = encode(value)
                            if key in seen:
                                shown = key.decode("utf-8")
                                self.fail(
                                    1,
                                    (owner,),
                                    f"{where}'s row {k} lists {shown} twice in a {kind} cell",
                                )
                            seen.add(key)
        for sid, source in self.sources():
            if source.requirement is not None:
                self.repeated_values(sid, source.requirement.ref, source.requirement.values)
            for ref, question in _questions(source):
                where = f"source {_q(sid)}'s question {_q(ref)}"
                listed: set[tuple[str, str]] = set()
                for latent in question.latent_targets:
                    target, role = latent["target"], str(latent["role"])
                    if target == question.target:
                        self.fail(1, (ref,), f"{where} lists its own target {_q(target)} as latent")
                    if (target, role) in listed:
                        self.fail(
                            1, (ref,), f"{where} lists the target {_q(target)} twice as {role}"
                        )
                    listed.add((target, role))
        for owner, key, where, entry in self.boundary_entries():
            if isinstance(entry, list):
                self.repeats(owner, where, "leaf paths", cast("list[str]", entry), refs=False)
                continue
            if "null" in entry:
                for role in X_ROLES:
                    self.repeats(owner, where, role, entry[role], refs=False)
                continue
            for role in ("bar", "thresholds", "margin"):
                self.repeats(owner, where, role, entry[role], refs=False)
            sets: set[bytes] = set()
            seen: set[bytes] = set()
            for m in cast("list[dict[str, list[str]]]", entry["mass"]):
                self.repeats(owner, where, "mass labels", m["labels"], refs=False)
                self.repeats(owner, where, "mass bounds", m["bounds"], refs=False)
                labels = _held(_unordered(m["labels"]))
                member = _held(
                    {"labels": _unordered(m["labels"]), "bounds": _unordered(m["bounds"])}
                )
                if member in seen:
                    self.fail(1, (owner, key), f"{where} lists a mass member twice")
                if labels in sets:
                    shown = labels.decode("utf-8")
                    self.fail(
                        1, (owner, key), f"{where} names the label set {shown} twice in its mass"
                    )
                seen.add(member)
                sets.add(labels)

    def unordered(self) -> Iterator[tuple[str, str, str, Sequence[str], bool]]:
        for sid, source in self.sources():
            yield sid, f"source {_q(sid)}", "needs", source.needs, True
        for rid, rule in self.rules():
            yield rid, f"rule {_q(rid)}", "reads", rule.reads, True
            yield rid, f"rule {_q(rid)}", "params", rule.params, False
        policy = self.policy()
        if policy is not None:
            yield "policy", "the policy", "reads", policy.reads, True
            yield "policy", "the policy", "params", policy.params, False

    def table_rows(self) -> Iterator[tuple[str, str, Sequence[Any]]]:
        """Each table's owner, where it is and its rows."""
        for rid, table in self.tables():
            yield rid, f"rule {_q(rid)}", table.rows
        policy_table = self.policy_table()
        if policy_table is not None:
            yield "policy", "the policy table", policy_table.rows

    def table_inputs(self) -> Iterator[tuple[str, str, Sequence[Any]]]:
        """Each table's owner, where it is and its inputs."""
        for rid, table in self.tables():
            yield rid, f"rule {_q(rid)}", table.inputs
        policy_table = self.policy_table()
        if policy_table is not None:
            yield "policy", "the policy table", policy_table.inputs

    def boundary_fields(self) -> Iterator[tuple[str, str, Mapping[str, Any], Sequence[str]]]:
        """Each declared ``boundaries`` field of a rule check 12 lets through and of the
        function policy: its owner, where it is, the field and its owner's reads."""
        for rid, rule in self.rules():
            if rule.boundaries is not None:
                yield rid, f"rule {_q(rid)}", rule.boundaries, rule.reads
        policy = self.policy()
        if policy is not None and policy.boundaries is not None:
            yield "policy", "the policy", policy.boundaries, policy.reads

    def boundary_entries(self) -> Iterator[tuple[str, str, str, Any]]:
        """Each entry of a ``boundaries`` field whose key check 7 passed: its owner, its key,
        where it is and the entry, a list of leaf paths or an object of its roles."""
        for owner, where, boundaries, _ in self.boundary_fields():
            for key, entry in boundaries.items():
                if _parsed(key) is not None:
                    yield owner, key, f"{where}'s boundaries for {_q(key)}", entry

    def repeats(
        self, owner: str, where: str, what: str, members: Sequence[str], *, refs: bool
    ) -> None:
        seen: set[bytes] = set()
        for member in members:
            unparsed = refs and _parsed(member) is None and not _candidate_read(member)
            if unparsed or domain_error(member, MAX_DOCUMENT_DEPTH):
                continue
            key = _held(member)
            if key in seen:
                self.fail(1, (owner, member), f"{where} lists {_q(member)} twice in its {what}")
            seen.add(key)

    def repeated_values(self, sid: str, ref: str, values: Sequence[object]) -> None:
        """A source requirement's values, an unordered list, compared by canonical bytes among
        the values in the domain; check 10 reports the others."""
        seen: set[bytes] = set()
        for value in values:
            if not in_domain(value):
                continue
            key = encode(value)
            if key in seen:
                shown = key.decode("utf-8")
                self.fail(
                    1,
                    (sid, ref),
                    f"source {_q(sid)}'s requirement lists {shown} twice in its values",
                )
            seen.add(key)

    # --- check 10 --------------------------------------------------------------------------

    def parse(
        self, text: object, nullable: bool, where: str, names: tuple[str, ...]
    ) -> Type | None:
        if not isinstance(text, str):
            self.fail(10, names, f"{where} declares no type string")
            return None
        parsed = parse_type(text, nullable=nullable, sorted_fields=self.format < KEY_ORDER_FORMAT)
        if isinstance(parsed, TypeFailure):
            self.fail(10, names, f"{where}: {parsed.message}")
            return None
        return parsed

    def candidate_type(
        self, t: Type, field: bool | str, where: str, names: tuple[str, ...]
    ) -> None:
        """Check 10: a candidate set's type, its definitions resolved, is ``list<string>`` under
        ``candidates`` true, and under a field's name a list of a record type whose named field
        is exactly ``string`` and whose other fields are each a scalar, a scalar or null, or a
        list of a scalar."""
        try:
            outer = resolve(t, self.defs)
            item = resolve(outer.item, self.defs) if isinstance(outer, ListOf) else None
        except ValueError:  # a definition check 10 refused
            return
        if not isinstance(field, str):  # true
            if item != Scalar("string"):
                self.fail(
                    10, names, f"{where} is a candidate set of type {render(t)}, not list<string>"
                )
            return
        if not isinstance(item, RecordOf):
            self.fail(
                10,
                names,
                f"{where} names the candidate field {_q(field)}, but its type {render(t)} is "
                "not a list of a record type",
            )
            return
        fields = dict(item.fields)
        if field not in fields or resolve(fields[field], self.defs) != Scalar("string"):
            self.fail(
                10,
                names,
                f"{where}'s candidate field {_q(field)} is not a string field of its record type",
            )
        for name, ft in item.fields:
            if name != field and not self.flat(ft):
                self.fail(
                    10,
                    names,
                    f"{where}'s field {_q(name)} is not a scalar, a scalar or null, or a list "
                    "of a scalar; a candidate record's other fields are",
                )

    def flat(self, t: Type) -> bool:
        """Whether a candidate record's other field is a scalar, a scalar or null, or a list of
        a scalar."""
        t = resolve(t, self.defs)
        if isinstance(t, Nullable):
            return _scalar(resolve(t.inner, self.defs))
        if isinstance(t, ListOf):
            return _scalar(resolve(t.item, self.defs))
        return _scalar(t)

    def declared(
        self,
        text: object,
        nullable: bool,
        where: str,
        names: tuple[str, ...],
        ref: str | None = None,
    ) -> Type | None:
        """A type that passes check 10 with every definition it names, or ``None``; the size
        of the type ``ref`` is declared with is kept for the class trees' sum."""
        t = self.parse(text, nullable, where, names)
        if t is None or set(named(t)) & self.tainted:
            return None  # a definition it names failed, and that failure stands
        failures = check_declared(t, self.defs, nullable=nullable, memo=self.declared_memo)
        for f in failures:
            self.fail(10, (*names, *f.names), f"{where}: {f.message}")
        if failures:
            return None
        size = _type_size(cast(str, text), t, self.def_sizes)
        if size > _TYPE_SIZE:
            self.fail(10, names, f"{where}: {_too_large()}")
            return None
        if ref is not None:
            self.ref_sizes[ref] = size
        return t

    def check_10(self) -> None:
        d = self.d
        for name, body in d.types.items():
            t = self.parse(body, True, f"type definition {_q(name)}", (name,))
            if t is None:
                self.tainted.add(name)
            else:
                self.defs[name] = t
        unparsed = set(self.tainted)
        for f in check_definitions(self.defs):
            if f.kind is FailureKind.cycle:
                self.tainted.update(f.names)
            else:
                self.tainted.add(f.names[0])
                if f.names[1] in unparsed:
                    continue  # the definition it names failed to parse, and says so
            self.fail(10, f.names, f.message)
        self.taint_closure()
        # Definitions before the ones that name them, so a use of a failing one is skipped.
        order = TopologicalSorter(
            {
                n: {m for m in named(body) if m in self.defs}
                for n, body in sorted(self.defs.items())
                if n not in self.tainted
            }
        ).static_order()
        for name in order:
            # In order, so each definition it names is settled: one failing taints it.
            if name in self.tainted or set(named(self.defs[name])) & self.tainted:
                self.tainted.add(name)
                continue
            failures = check_declared(
                self.defs[name], self.defs, nullable=True, memo=self.declared_memo
            )
            for f in failures:
                self.fail(10, (name, *f.names), f"type definition {_q(name)}: {f.message}")
            if failures:
                self.tainted.add(name)
                continue
            # Every name it uses is defined and passed, so its size is taken once.
            size = _type_size(d.types[name], self.defs[name], self.def_sizes)
            if size > _TYPE_SIZE:
                self.fail(10, (name,), f"type definition {_q(name)}: {_too_large()}")
                self.tainted.add(name)
            else:
                self.def_sizes[name] = size
        self.taint_closure()

        state = self.declared(d.state, False, "the state declaration", (_STATE,), _STATE)
        if state is not None:
            if isinstance(resolve(state, self.defs), RecordOf):
                self.state_type = state
            else:
                self.fail(
                    10, (_STATE,), f"the state declaration is {render(state)}, not a record type"
                )
        for sid, source in self.sources():
            for ref, provided in source.provides.items():
                # A ref check 7 refuses is left out; its declaration is still checked.
                declared = _declared(ref) is not None
                if isinstance(provided, Question):
                    if declared:
                        self.types[ref] = AnswerType()
                        self.ref_sizes[ref] = _ANSWER_SIZE
                    continue
                text = cast("Mapping[str, Any]", provided)["type"]
                where = f"source {_q(sid)}'s {_q(ref)}"
                t = self.declared(text, True, where, (sid, ref), ref if declared else None)
                if t is not None and declared:
                    self.types[ref] = t
                field = cast("Mapping[str, Any]", provided)["candidates"]
                if t is not None and field:
                    self.candidate_type(t, field, where, (sid, ref))
            if domain_error(source.payload) is not None:
                self.fail(10, (sid,), f"source {_q(sid)}'s payload is outside the value domain")
            if source.when is not None:
                self.version(source.when.version, f"source {_q(sid)}'s when", sid)
            requirement = source.requirement
            if requirement is not None and not all(in_domain(v) for v in requirement.values):
                self.fail(
                    10,
                    (sid, requirement.ref),
                    f"source {_q(sid)}'s requirement holds a value outside the value domain",
                )
        for rid, rule in self.rules():
            provides = rule.provides if _declared(rule.provides) is not None else None
            t = self.declared(rule.outputs, True, f"rule {_q(rid)}'s outputs", (rid,), provides)
            if t is not None and provides is not None:
                self.types[rule.provides] = t
            if t is not None and rule.candidates:
                self.candidate_type(t, rule.candidates, f"rule {_q(rid)}", (rid,))
            self.version(rule.version, f"rule {_q(rid)}", rid)
        for rid, table in self.tables():
            provides = table.provides if _declared(table.provides) is not None else None
            t = self.declared(table.outputs, True, f"rule {_q(rid)}'s outputs", (rid,), provides)
            if t is not None and provides is not None:
                self.types[table.provides] = t
            if t is not None and table.candidates:
                self.candidate_type(t, table.candidates, f"rule {_q(rid)}", (rid,))
            if not in_domain(table.inputs) or not in_domain(table.rows):
                self.fail(
                    10, (rid,), f"rule {_q(rid)}'s inputs or rows are outside the value domain"
                )
        for name, plan in self.plans():
            for n, step in enumerate(plan.steps, 1):
                for arg, origin in cast(Step, step).arguments.items():
                    if (
                        isinstance(origin, Origin)
                        and origin.ref is None
                        and not in_domain(origin.literal)
                    ):
                        self.fail(
                            10,
                            (name,),
                            f"plan declaration {_q(name)}'s step {n} argument {_q(arg)} is a "
                            "literal outside the value domain",
                        )
                for r in cast(Step, step).requirements:
                    if not in_domain(r.value):
                        self.fail(
                            10,
                            (name, r.ref),
                            f"plan declaration {_q(name)}'s step {n} requirement holds a value "
                            "outside the value domain",
                        )
        policy = self.policy()
        if policy is not None:
            self.version(policy.version, "the policy", "policy")
        for requirement in d.requirements:
            if not in_domain(requirement.value):
                self.fail(
                    10,
                    (requirement.ref,),
                    f"the requirement on {_q(requirement.ref)} holds a value outside the value "
                    "domain",
                )
        if sum(self.ref_sizes.values()) > _TYPE_SIZE:
            # Every record's settings item writes each of these refs' class trees in full.
            self.fail(
                10,
                (_STATE, *self.provider_map()),
                f"the types of {_STATE} and the provided refs sum to more than "
                f"{power(_TYPE_SIZE)} code points with their definitions resolved",
            )
        for path, leaf in d.parameters.items():
            t = self.declared(leaf.type, False, f"parameter {_q(path)}", (path,))
            written = (
                leaf.value if self.parsed is not None else self.leaf_document_value(leaf.value, t)
            )
            if not in_domain(written):
                self.fail(10, (path,), f"parameter {_q(path)}'s value is outside the value domain")
            if t is None:
                continue
            full = _resolved(t, self.defs)
            if not _leaf_shape(full):
                self.fail(
                    10,
                    (path,),
                    f"parameter {_q(path)}'s type is {render(t)}, not a scalar, a list of one "
                    "or a mapping of one",
                )
                continue
            self.leaf_types[path] = full
            self.leaf_written[path] = t
            if self.parsed is not None and value_conforms(leaf.value, full, {}, form="document"):
                self.live[path] = live_value(leaf.value, full)

    def leaf_value(self, path: str) -> object:
        """A leaf's value with ``bytes`` as Python ``bytes`` where its document form was read
        back; otherwise as declared."""
        return self.live.get(path, self.d.parameters[path].value)

    def leaf_document_value(self, value: object, t: Type | None) -> object:
        """A leaf value in its document form, ``bytes`` as base64url, when its type passed
        check 10 and it conforms; otherwise as given."""
        if t is None or set(named(t)) & self.tainted:
            return value
        full = _resolved(t, self.defs)
        if _leaf_shape(full) and value_conforms(value, full, {}, form="live"):
            return document_value(value, full)
        return value

    def taint_closure(self) -> None:
        changed = True
        while changed:
            changed = False
            for name, t in self.defs.items():
                if name not in self.tainted and set(named(t)) & self.tainted:
                    self.tainted.add(name)
                    changed = True

    def version(self, version: object, where: str, owner: str) -> None:
        if not isinstance(version, str) or not in_domain(version):
            self.fail(10, (owner,), f"{where} declares no version string")

    # --- check 8 ---------------------------------------------------------------------------

    def check_8(self) -> None:
        declared = self.leaves
        named_params: list[tuple[str, str, str]] = []
        for rid, rule in self.rules():
            named_params += [(rid, f"rule {_q(rid)}", p) for p in rule.params]
        policy = self.policy()
        if policy is not None:
            named_params += [("policy", "the policy", p) for p in policy.params]
        for sid, source in self.sources():
            for ref, slot in _slots(source):
                tag, value = next(iter(slot["origin"].items()))
                if tag == "param":
                    named_params.append((sid, f"source {_q(sid)}'s question {_q(ref)}", value))
        for entry in self.context_entries():
            for slot in entry.text.slots:
                tag, value = next(iter(slot["origin"].items()))
                if tag == "param":
                    shown = f"the context entry for {_q(entry.ref)}"
                    named_params.append((entry.ref, shown, value))
        named_params += self.table_params()
        for name, plan in self.plans():
            for member, text in (("reason", plan.reason), ("reply", plan.reply)):
                for slot in () if text is None else text.slots:
                    tag, value = next(iter(slot["origin"].items()))
                    if tag == "param":
                        shown = f"plan declaration {_q(name)}'s {member}"
                        named_params.append((name, shown, value))
        for owner, where, path in named_params:
            if path not in declared:
                self.fail(8, (owner, path), f"{where} names {_q(path)}, which is not a leaf")
        for owner, _, where, entry in self.boundary_entries():
            for path in _entry_paths(entry):
                if path not in declared:
                    self.refused_leaves.add(path)
                    self.fail(8, (owner, path), f"{where} names {_q(path)}, which is not a leaf")
        for path, t in self.leaf_types.items():
            if not value_conforms(self.leaf_value(path), t, {}, form="live"):
                self.fail(8, (path,), f"parameter {_q(path)}'s value does not conform to its type")

    def table_params(self) -> list[tuple[str, str, str]]:
        """Each leaf path a table names: its helpers' bindings, its cells' operands and its
        outputs' pass-throughs, with its owner and where."""
        out: list[tuple[str, str, str]] = []
        for owner, where, inputs in self.table_inputs():
            for n, i in enumerate(inputs, 1):
                out += [(owner, f"{where}'s input {n}", path) for path in i["params"].values()]
        for owner, where, rows in self.table_rows():
            for k, row in enumerate(rows, 1):
                shown = f"{where}'s row {k}"
                out += [(owner, shown, path) for path in _cell_params(row["cells"])]
                output = row.get("output")
                if isinstance(output, dict) and "param" in output:
                    out.append((owner, shown, cast("dict[str, str]", output)["param"]))
        return out

    def check_8_types(self) -> None:
        """Check 8's test of a named leaf's type: a leaf a ``boundaries`` field names has a
        type that conforms to ``number`` or ``mapping<string, number>``; a helper binding's
        leaf conforms to its argument's type in the helper's signature, and a leaf a cell
        compares against is numeric. It runs once checks 1 to 4, 10 and 12 pass, so every
        declared leaf has its type."""
        for owner, where, inputs in self.table_inputs():
            for n, i in enumerate(inputs, 1):
                signature = _SIGNATURES.get(cast(str, i["helper"]), {})
                for arg, path in cast("dict[str, str]", i["params"]).items():
                    t, wanted = self.leaf_types.get(path), signature.get(arg)
                    if t is None or wanted is None or conforms(t, wanted, {}):
                        continue
                    self.fail(
                        8,
                        (owner, path),
                        f"{where}'s input {n} binds {arg} to {_q(path)}, whose type is "
                        f"{render(self.leaf_written[path])}, which does not conform to "
                        f"{render(wanted)}",
                    )
        for owner, where, rows in self.table_rows():
            for k, row in enumerate(rows, 1):
                for path in _cell_params(row["cells"]):
                    t = self.leaf_types.get(path)
                    if t is not None and not conforms(t, _NUMBER, {}):
                        self.fail(
                            8,
                            (owner, path),
                            f"{where}'s row {k} compares against {_q(path)}, whose type is "
                            f"{render(self.leaf_written[path])}, which is not numeric",
                        )
        for owner, _, where, entry in self.boundary_entries():
            for path in _entry_paths(entry):
                t = self.leaf_types.get(path)
                if t is None or conforms(t, _NUMBER, {}) or conforms(t, _NUMBERS, {}):
                    continue
                self.refused_leaves.add(path)
                self.fail(
                    8,
                    (owner, path),
                    f"{where} names {_q(path)}, whose type is {render(self.leaf_written[path])}, "
                    "which does not conform to number or mapping<string, number>",
                )

    # --- check 9 ---------------------------------------------------------------------------

    def check_9(self) -> None:
        for sid, source in self.sources():
            for ref, question in _questions(source):
                where = f"source {_q(sid)}'s question {_q(ref)}"
                self.text(question.wording, where, (sid, ref))
        self.check_9_context()
        for u in self.uses:
            if u.role == "boundary":
                continue  # a key's read is typed as its component's read (check 9)
            t = self.read_type(u)
            names = (u.owner, u.ref) if u.owner else (u.ref,)
            if isinstance(t, TypeFailure):
                self.fail(9, names, f"{u.where} reads {_q(u.ref)}: {t.message}")
            elif u.role == "text" and resolve(t, self.defs) != _STRING:
                self.fail(
                    9,
                    names,
                    f"{u.where} names {_q(u.ref)}, whose type is {render(t)}, not exactly "
                    "string, not nullable",
                )
        # Every requirement on the plan can hold. A step's requirements are
        # in plan declarations, which check 12 refuses, so this engine applies the clause to
        # none of them; nor the clause on a declaration's binding paths.
        for requirement in self.d.requirements:
            parsed = _parsed(requirement.ref)
            if parsed is None or parsed.fields:
                continue  # check 7 refused it, so no other check reads it
            t = self.types[f"{parsed.level}:{parsed.name}"]
            value = requirement.value
            # Check 10 refused a value outside the domain, and it gates this check.
            reason = hold_failure(value, t, self.defs)
            if reason is not None:
                self.fail(
                    9,
                    (requirement.ref,),
                    f"a plan-level requirement on {_q(requirement.ref)} can never hold: {reason}",
                )
        # A source requirement can hold when each of its values can.
        for u in self.uses:
            if u.role != "source-requirement":
                continue
            requirement = self.d.sources[u.owner].requirement
            assert requirement is not None
            t = self.types[u.base]
            for value in requirement.values:
                reason = hold_failure(value, t, self.defs)
                if reason is not None:
                    shown = encode(value).decode("utf-8")
                    self.fail(
                        9,
                        (u.owner, u.ref),
                        f"{u.where} on {_q(u.ref)} can never hold for {shown}: {reason}",
                    )
        self.check_9_boundaries()
        self.check_9_tables()
        self.check_9_grounded()

    def check_9_grounded(self) -> None:
        """A ``candidate.f`` read names a field of its set's record type, and ``candidate``
        none deeper; no ``boundaries`` field names a candidate read; only a component grounded
        over a declared ref's own set reads it by a field path."""
        for owner, where, read, role in self.candidate_reads:
            if role == "boundary":
                self.fail(9, (owner,), f"{where} names {_q(read)}, a candidate read")
                continue
            rule = self.d.rules.get(owner)
            if rule is None or rule.grounded is None:
                continue  # check 4's
            if self.candidate_read_type(owner, read) is None:
                self.fail(
                    9,
                    (owner,),
                    f"{where} reads {_q(read)}, which names no field of its set's candidate",
                )
        per_candidate = _per_candidate_refs(self.d)
        for u in self.uses:
            set_ref = per_candidate.get(u.base)
            plan_role = u.role in ("origin", "plan-slot")  # the plan check's, and check 4's
            if set_ref is None or not u.parsed.fields or u.role == "grounding" or plan_role:
                continue
            owner = self.d.rules.get(u.owner) or self.d.sources.get(u.owner)
            if owner is None or owner.grounded != set_ref:
                self.fail(
                    9,
                    (u.owner, u.ref),
                    f"{u.where} reads {_q(u.ref)} by a field path; only a component grounded over "
                    f"{_q(set_ref)} reads {_q(u.base)} so",
                )

    def candidate_read_type(self, owner: str, read: str) -> Type | None:
        """A grounded component's ``candidate`` read's type: ``string``, the candidate value;
        a ``candidate.f`` read's, the field ``f`` of its set's record type; ``None`` when the
        read names no such field, or an earlier check refused the set."""
        rule = self.d.rules[owner]
        assert rule.grounded is not None
        if read == "candidate":
            return _STRING
        field = self.set_field(rule.grounded)
        t = self.types.get(rule.grounded)
        path = read.split(".")[1:]
        if not isinstance(field, str) or t is None or len(path) != 1:
            return None
        outer = resolve(t, self.defs)
        if not isinstance(outer, ListOf):
            return None
        item = resolve(outer.item, self.defs)
        if not isinstance(item, RecordOf):
            return None
        return dict(item.fields).get(path[0])

    # --- check 9 over tables and plan declarations (§decision-tables) ----------------------

    def input_type(self, owner: str, where: str, i: Mapping[str, Any]) -> Type | None:
        """A table input's type: a bare read's, ``answer`` for a ``q:`` one; a helper's
        return; ``None`` when an earlier check refused what it rests on."""
        read, helper = cast(str, i["read"]), i["helper"]
        if is_candidate_read(read):
            rule = self.d.rules.get(owner)
            if rule is None or rule.grounded is None:
                return None
            t = self.candidate_read_type(owner, read)
            if helper is None:
                return t
            if helper != "size" or t is None:
                return None
            return _COUNT if isinstance(resolve(t, self.defs), ListOf) else None
        parsed = _parsed(read)
        if parsed is None:
            return None
        base = f"{parsed.level}:{parsed.name}"
        if helper is None:
            if parsed.level == "q":
                return AnswerType() if not parsed.fields else None
            t = self.read_type(_Use(owner, where, "read", read, parsed))
            return None if isinstance(t, TypeFailure) else t
        if helper == "size":
            return _COUNT
        if helper == "mass":
            return _UNIT
        if helper not in HELPERS or base not in self.providers or parsed.level != "q":
            return None
        question = self.question(base)
        if helper == "extract":
            if question.kind is not AnswerKind.extraction:
                return None
            item = _NUMBER if question.value_type is ValueType.number else _STRING
            return ListOf(item) if question.is_list else item
        if question.kind is AnswerKind.extraction:
            return None
        if question.kind is AnswerKind.noul:
            return Label(tuple(sorted(_NOUL_LABELS)))
        if isinstance(question.labels, Mapping):
            return _STRING  # a candidate-valued set's labels are named per turn
        return Label(tuple(sorted(label_parts(_fixed(question.labels))[0])))

    def check_9_tables(self) -> None:
        for rid, table in self.tables():
            self.check_9_table(rid, f"rule {_q(rid)}", table.inputs, table.rows, table)
        policy_table = self.policy_table()
        if policy_table is not None:
            self.check_9_table(
                "policy", "the policy table", policy_table.inputs, policy_table.rows, None
            )
            self.check_9_policy_table(policy_table)
        for name, plan in self.plans():
            self.check_9_plan(name, plan)
        self.check_9_covered()

    def check_9_table(
        self,
        owner: str,
        where: str,
        inputs: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, Any]],
        table: Table | None,
    ) -> None:
        names = (owner,)
        reads = [cast(str, i["read"]) for i in inputs]
        types = [self.input_type(owner, where, i) for i in inputs]
        fixed = [
            None if t is None else self.pick_fixed(i) for t, i in zip(types, inputs, strict=True)
        ]
        counts = Counter(reads)
        for n, i in enumerate(inputs, 1):
            self.check_9_helper(owner, f"{where}'s input {n}", i)

        def named(ref: str, shown: str, own: int | None = None) -> int | None:
            if counts[ref] == 1 and reads.index(ref) == own:
                self.fail(9, (owner, ref), f"{shown} compares {_q(ref)} with itself")
                return None
            if counts[ref] == 1:
                return reads.index(ref)
            if counts[ref] == 0:
                self.fail(9, (owner, ref), f"{shown} names {_q(ref)}, which no input of it reads")
            else:
                self.fail(9, (owner, ref), f"{shown} names {_q(ref)}, which two inputs of it read")
            return None

        outputs = None
        if table is not None and _declared(table.provides) is not None:
            outputs = self.types.get(table.provides)
        hit = "F" if table is None else table.hit
        for k, row in enumerate(rows, 1):
            shown = f"{where}'s row {k}"
            cells = row["cells"]
            if len(cells) != len(inputs):
                self.fail(9, names, f"{shown} has {len(cells)} cells for its {len(inputs)} inputs")
                continue
            for n, cell in enumerate(cells):
                self.check_9_cell(
                    owner, f"{shown}'s cell {n + 1}", cell, n, reads, types, fixed, named
                )
            if table is not None:
                self.check_9_output(owner, shown, row["output"], hit, outputs, reads, types, named)
        if table is None:
            return
        self.check_9_hit(owner, where, table, outputs)
        own = any(f.check in (6, 8, 9) and owner in f.names for f in self.failures)
        if hit in ("U", "A") and not own:
            self.check_9_overlap(owner, where, table, types)

    def pick_fixed(self, i: Mapping[str, Any]) -> tuple[str, ...] | None:
        """A pick input's fixed label names when its question has a candidate-valued label set,
        which a value cell over it may name; ``None`` for any other input."""
        if i["helper"] not in ("pick_with_margin", "pick_with_thresholds"):
            return None
        parsed = _parsed(cast(str, i["read"]))
        if parsed is None or parsed.level != "q":
            return None
        question = self.question(f"q:{parsed.name}")
        if not isinstance(question.labels, Mapping):
            return None
        return tuple(label_parts(_fixed(question.labels))[0])

    def check_9_helper(self, owner: str, where: str, i: Mapping[str, Any]) -> None:
        """A helper's ``params`` name each argument of its signature after the read, and no
        other; a ``mass`` input's ``labels`` leaf holds labels of its read, each once."""
        helper = i["helper"]
        if helper is None:
            if i["params"]:
                self.fail(
                    9, (owner,), f"{where} reads {_q(i['read'])} and binds params, with no helper"
                )
            return
        if helper not in HELPERS:
            return  # check 6's
        wanted, given = set(HELPERS[helper]), set(i["params"])
        if given != wanted:
            shown = ", ".join(HELPERS[helper]) or "no argument"
            self.fail(
                9,
                (owner,),
                f"{where} binds {', '.join(sorted(given)) or 'nothing'}; {helper} takes {shown} "
                "after its read",
            )
            return
        parsed = _parsed(i["read"])
        if parsed is None:
            return  # check 7's
        if helper == "size":
            t = self.input_type(owner, where, {**i, "helper": None})
            if t is not None and not isinstance(resolve(t, self.defs), ListOf):
                self.fail(
                    9,
                    (owner, i["read"]),
                    f"{where}: size reads {_q(i['read'])}, of type {render(t)}, "
                    "which is not a list",
                )
            return
        if helper not in ("mass", "pick_with_thresholds"):
            return
        arg = "labels" if helper == "mass" else "thresholds"
        path = i["params"][arg]
        if path not in self.leaf_types or f"q:{parsed.name}" not in self.providers:
            return
        held: object = self.leaf_value(path)
        wanted_type: type = list if helper == "mass" else dict
        if type(held) is not wanted_type:
            return  # check 8's
        question = self.question(f"q:{parsed.name}")
        if question.kind is AnswerKind.extraction:
            return  # check 4's
        if helper == "pick_with_thresholds" and question.kind is AnswerKind.noul:
            self.fail(
                9,
                (owner, i["read"]),
                f"{where}: a noul question takes pick_with_margin, not thresholds",
            )
            return
        labels = (
            _NOUL_LABELS
            if question.kind is AnswerKind.noul
            else label_parts(_fixed(question.labels))[0]
        )
        members: list[object] = (
            list(cast("dict[str, object]", held))
            if type(held) is dict
            else cast("list[object]", held)
        )
        for label in members:
            if label not in labels:
                self.fail(
                    9,
                    (owner, path),
                    f"{where}'s {arg} {_q(path)} holds {_q(str(label))}, which is not a label "
                    f"of {_q(i['read'])}",
                )
        if len(set(map(str, members))) != len(members):
            self.fail(9, (owner, path), f"{where}'s labels {_q(path)} lists a label twice")

    def check_9_cell(
        self,
        owner: str,
        where: str,
        cell: Any,
        n: int,
        reads: Sequence[str],
        types: Sequence[Type | None],
        fixed: Sequence[tuple[str, ...] | None],
        named: Callable[..., int | None],
    ) -> None:
        names = (owner,)
        if cell == ANY:
            return
        ((tag, member),) = cast("dict[str, Any]", cell).items()
        if tag in ("marker", "markers", "known"):
            return
        t = types[n]
        if t is None:
            return
        if isinstance(t, AnswerType):
            self.fail(
                9,
                names,
                f"{where} tests the answer at {_q(reads[n])}; only -, Known and marker cells apply "
                "to a q: input",
            )
            return
        inner = resolve(t, self.defs)
        if isinstance(inner, Nullable):
            inner = resolve(inner.inner, self.defs)
        if tag in ("is", "one_of"):
            names_n = fixed[n]
            for value in [member] if tag == "is" else cast("list[object]", member):
                problem = (
                    _value_cell(value, inner) if names_n is None else _fixed_cell(value, names_n)
                )
                if problem is not None:
                    self.fail(
                        9, names, f"{where}: {problem} over {_q(reads[n])}, of type {render(t)}"
                    )
            return
        if tag == "in":
            j = named(cast(str, member), where, n)
            if j is None or types[j] is None:
                return
            listed = resolve(cast(Type, types[j]), self.defs)
            if isinstance(listed, Nullable):
                listed = resolve(listed.inner, self.defs)
            if not isinstance(listed, ListOf):
                self.fail(9, names, f"{where}: {_q(member)} is not a list")
                return
            item = resolve(listed.item, self.defs)
            if isinstance(item, RecordOf):  # a candidate set of records, by its field
                if not isinstance(self.set_field(cast(str, member)), str):
                    self.fail(
                        9,
                        names,
                        f"{where}: {_q(member)} is a list of records that is no candidate set",
                    )
                elif resolve(t, self.defs) != Scalar("string"):  # exactly, not nullable
                    self.fail(
                        9,
                        names,
                        f"{where}: {_q(reads[n])} is not a string, which the candidate field "
                        f"of {_q(member)} is",
                    )
                return
            numeric = _is_numeric(inner) and _is_numeric(item)
            textual = _is_textual(inner) and _is_textual(item)
            if not (numeric or textual):
                self.fail(
                    9,
                    names,
                    f"{where}: {_q(reads[n])} and the members of {_q(member)} are not both "
                    "numeric, nor both strings or labels",
                )
            return
        if not _is_numeric(inner):
            self.fail(
                9,
                names,
                f"{where} compares {_q(reads[n])}, of type {render(t)}, which is not numeric",
            )
            return
        integer = _numeric_base(inner) == "integer"
        operands = cast("list[dict[str, Any]]", member if tag == "between" else [member])
        bounds: list[object] = []
        for operand in operands:
            if "literal" in operand:
                if not integer:
                    self.fail(
                        9,
                        names,
                        f"{where} holds an integer literal over {_q(reads[n])}, which is not "
                        "integer",
                    )
                bounds.append(operand["literal"])
            elif "param" in operand:
                path = cast(str, operand["param"])
                bounds.append(self.leaf_value(path) if path in self.leaf_types else None)
            else:
                j = named(cast(str, operand["ref"]), where, n)
                if j is not None and types[j] is not None:
                    other = resolve(cast(Type, types[j]), self.defs)
                    if isinstance(other, Nullable):
                        other = resolve(other.inner, self.defs)
                    if not _is_numeric(other):
                        self.fail(
                            9,
                            names,
                            f"{where} compares against {_q(operand['ref'])}, which is not numeric",
                        )
        if tag == "between" and all(_number_value(b) for b in bounds):
            lo, hi = cast("list[float]", bounds)
            if lo > hi:
                self.fail(9, names, f"{where}'s interval has its lower bound above its upper")

    def check_9_output(
        self,
        owner: str,
        where: str,
        output: Any,
        hit: str,
        outputs: Type | None,
        reads: Sequence[str],
        types: Sequence[Type | None],
        named: Callable[..., int | None],
    ) -> None:
        names = (owner,)
        if hit == "C#":
            return  # the form gives a count's rows no output
        ((tag, member),) = cast("dict[str, Any]", output).items()
        if tag == "marker":
            if hit == "P" or hit.startswith("C"):
                self.fail(
                    9,
                    names,
                    f"{where}'s output is Unsure, which a {hit} table cannot rank or aggregate",
                )
            return
        if hit == "P" and tag != "literal":
            self.fail(9, names, f"{where}'s output is a pass-through, which a P table cannot rank")
            return
        if outputs is None:
            return
        if tag == "literal":
            form = "document" if self.parsed is not None else "live"
            if not value_conforms(member, outputs, self.defs, form=form):
                self.fail(
                    9,
                    names,
                    f"{where}'s output {encode(member).decode('utf-8')} is not a value of "
                    f"{render(outputs)}",
                )
            elif hit.startswith("C") and not _number_value(member):
                self.fail(9, names, f"{where}'s output is not a number")
            return
        if tag == "param":
            t = self.leaf_types.get(cast(str, member))
            if t is not None and not conforms(t, outputs, self.defs):
                self.fail(
                    9,
                    names,
                    f"{where} passes {_q(member)} through, whose type does not conform to "
                    f"{render(outputs)}",
                )
            return
        j = named(cast(str, member), where)
        if j is None or types[j] is None:
            return
        t = cast(Type, types[j])
        if isinstance(t, AnswerType):
            self.fail(
                9,
                names,
                f"{where} passes the answer at {_q(member)} through; a q: input cannot be passed "
                "through",
            )
        elif not conforms(t, outputs, self.defs):
            self.fail(
                9,
                names,
                f"{where} passes {_q(member)} through, of type {render(t)}, which does not "
                f"conform to {render(outputs)}",
            )

    def check_9_hit(self, owner: str, where: str, table: Table, outputs: Type | None) -> None:
        names = (owner,)
        hit = table.hit
        if hit != "P":
            if table.order is not None:
                self.fail(9, names, f"{where} declares an order, which only a P table has")
        else:
            order = list(table.order or ())
            if table.order is None:
                self.fail(9, names, f"{where} is a P table with no order")
            keys = [encode(v) for v in order if in_domain(v)]
            if len(set(keys)) != len(keys):
                self.fail(9, names, f"{where}'s order lists a value twice")
            form = "document" if self.parsed is not None else "live"
            for v in order:
                if (
                    outputs is not None
                    and in_domain(v)
                    and not value_conforms(v, outputs, self.defs, form=form)
                ):
                    self.fail(
                        9,
                        names,
                        f"{where}'s order lists {encode(v).decode('utf-8')}, which is not a "
                        f"value of {render(outputs)}",
                    )
            for k, row in enumerate(table.rows if table.order is not None else (), 1):
                output = cast(object, row["output"])
                tagged = cast("dict[str, Any]", output) if isinstance(output, dict) else {}
                literal = tagged.get("literal")
                if "literal" in tagged and in_domain(literal) and encode(literal) not in keys:
                    self.fail(9, names, f"{where}'s row {k} gives a value its order does not list")
        if outputs is None:
            return
        if hit == "C#" and resolve(outputs, self.defs) != Scalar("integer"):
            self.fail(
                9,
                names,
                f"{where} counts its rows, but its outputs {render(outputs)} is not integer",
            )
        elif hit in ("C+", "C<", "C>") and not conforms(outputs, _NUMBER, self.defs):
            self.fail(
                9,
                names,
                f"{where} aggregates its rows, but its outputs {render(outputs)} does not "
                "conform to number",
            )

    def check_9_overlap(
        self, owner: str, where: str, table: Table, types: Sequence[Type | None]
    ) -> None:
        reads = [cast(str, i["read"]) for i in table.inputs]
        axes = [self.axis(t) for t in types]
        leaves = {p: self.leaf_value(p) for p in self.leaf_types}
        rows = list(table.rows)
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                if len(rows[a]["cells"]) != len(reads) or len(rows[b]["cells"]) != len(reads):
                    continue
                if not overlap(rows[a], rows[b], reads, axes, leaves):
                    continue
                if table.hit == "U":
                    self.fail(
                        9,
                        (owner,),
                        f"{where}'s rows {a + 1} and {b + 1} overlap, which a U table refuses",
                    )
                elif encode(rows[a]["output"]) != encode(rows[b]["output"]):
                    self.fail(
                        9,
                        (owner,),
                        f"{where}'s rows {a + 1} and {b + 1} overlap and give different outputs",
                    )

    def axis(self, t: Type | None) -> Axis:
        if t is None or isinstance(t, AnswerType):
            return Axis()
        inner = resolve(t, self.defs)
        if isinstance(inner, Nullable):
            inner = resolve(inner.inner, self.defs)
        if isinstance(inner, Interval):
            return Axis(inner.base == "integer", _exact(inner.lo), _exact(inner.hi))
        return Axis(isinstance(inner, Scalar) and inner.name == "integer")

    def check_9_policy_table(self, table: PolicyTable) -> None:
        names = ("policy",)
        for k, row in enumerate(table.rows, 1):
            plan = cast(str, row["plan"])
            if plan not in self.d.plans:
                self.fail(
                    9,
                    ("policy", plan),
                    f"the policy table's row {k} names {_q(plan)}, which is not a declared plan",
                )
        if not table.rows:
            self.fail(9, names, "the policy table has no row")
            return
        last = table.rows[-1]
        if any(cell != ANY for cell in last["cells"]):
            self.fail(9, names, "the policy table's last row holds a cell that is not -")
        plan = self.d.plans.get(cast(str, last["plan"]))
        if plan is not None and plan.grounded is not None:
            self.fail(
                9, names, f"the policy table's last row names {_q(last['plan'])}, a grounded plan"
            )
        elif plan is not None and _board_refs(plan):
            self.fail(
                9,
                names,
                f"the policy table's last row names {_q(last['plan'])}, whose plan reads the board",
            )

    def check_9_plan(self, name: str, plan: PlanDeclaration) -> None:
        where, names = f"plan declaration {_q(name)}", (name,)
        try:
            kind = PlanKind(plan.kind)
        except ValueError:
            self.fail(
                9,
                names,
                f"{where}'s kind {_q(plan.kind)} is not execute, clarify, inform or escalate",
            )
            return
        if kind is PlanKind.broken:
            self.fail(9, names, f"{where}'s kind is broken, which only the engine writes")
        if plan.grounded is not None:
            self.check_9_grounded_plan(name, plan, kind)
        if kind is not PlanKind.execute and plan.steps:
            self.fail(9, names, f"{where} is {kind.value}, which holds no steps")
        if kind in (PlanKind.clarify, PlanKind.inform) and plan.reply is None:
            self.fail(9, names, f"{where} is {kind.value}, which declares a reply")
        if kind in (PlanKind.execute, PlanKind.escalate) and plan.reply is not None:
            self.fail(9, names, f"{where} is {kind.value}, which declares no reply")
        if kind is PlanKind.escalate and plan.reason is None:
            self.fail(9, names, f"{where} is escalate, which declares a reason")
        for member, text in (("reason", plan.reason), ("reply", plan.reply)):
            if text is not None:
                self.text(text, f"{where}'s {member}", names)
        for n, step in enumerate(cast("Sequence[Step]", plan.steps), 1):
            shown = f"{where}'s step {n}"
            if step.tool not in self.d.tools:
                self.fail(
                    9, names, f"{shown} names the tool {_q(step.tool)}, which is not declared"
                )
            for arg, origin in step.arguments.items():
                if arg == "idempotency_key":
                    self.fail(9, names, f"{shown} has an argument named idempotency_key")
                if isinstance(origin, Binding) and not (
                    1 <= origin.step < n and (origin.path is None or is_field_path(origin.path))
                ):
                    self.fail(
                        9,
                        names,
                        f"{shown}'s argument {_q(arg)} binds no earlier step by a path in the "
                        "grammar",
                    )
            for r in step.requirements:
                parsed = _parsed(r.ref)
                if parsed is None or parsed.fields or r.ref not in self.types:
                    continue
                reason = hold_failure(r.value, self.types[r.ref], self.defs)
                if reason is not None:
                    self.fail(
                        9,
                        (name, r.ref),
                        f"{shown}'s requirement on {_q(r.ref)} can never hold: {reason}",
                    )

    def check_9_grounded_plan(self, name: str, plan: PlanDeclaration, kind: PlanKind) -> None:
        """A grounded plan declaration is ``execute``; its board-ref origin naming a grounded
        declared ref names one grounded over its own set, with no field path; no slot names
        the candidate value, a grounded entry or a grounded declared ref."""
        where, names = f"plan declaration {_q(name)}", (name,)
        if kind is not PlanKind.execute:
            self.fail(9, names, f"{where} is grounded, so of kind execute, not {kind.value}")
        per_candidate = _per_candidate_refs(self.d)
        for n, step in enumerate(cast("Sequence[Step]", plan.steps), 1):
            for arg, origin in step.arguments.items():
                if not isinstance(origin, Origin) or origin.ref is None:
                    continue
                parsed = _parsed(origin.ref)
                if parsed is None:
                    continue
                set_ref = per_candidate.get(f"{parsed.level}:{parsed.name}")
                if set_ref is None:
                    continue
                shown = f"{where}'s step {n} argument {_q(arg)}"
                if set_ref != plan.grounded:
                    self.fail(
                        3,
                        (name, origin.ref),
                        f"{shown} names {_q(origin.ref)}, grounded over {_q(set_ref)}, not its "
                        "plan's set",
                    )
                elif parsed.fields:
                    self.fail(
                        9,
                        (name, origin.ref),
                        f"{shown} names {_q(origin.ref)}, a grounded declared ref by a field path",
                    )
        for member, text in (("reason", plan.reason), ("reply", plan.reply)):
            for slot in () if text is None else text.slots:
                tag, value = next(iter(slot["origin"].items()))
                shown = f"{where}'s {member}"
                if tag == "candidate":
                    self.fail(9, names, f"{shown} has a candidate slot, which no plan text holds")
                elif tag == "ref" and (_declared_base(cast(str, value)) or "") in per_candidate:
                    self.fail(
                        9,
                        (name, cast(str, value)),
                        f"{shown} has a slot on {_q(value)}, a grounded declared ref",
                    )

    def table_compares(self) -> set[tuple[str, str]]:
        """Each read a table compares against a leaf, with the leaf: a comparison cell's read,
        and a helper binding's read."""
        out: set[tuple[str, str]] = set()
        for _, _, inputs, rows in self.all_tables():
            for i in inputs:
                out |= {(cast(str, i["read"]), p) for p in i["params"].values()}
            for row in rows:
                for n, cell in enumerate(row["cells"]):
                    if n < len(inputs):
                        out |= {(cast(str, inputs[n]["read"]), p) for p in _cell_params([cell])}
        return out

    def all_tables(self) -> Iterator[tuple[str, str, Sequence[Any], Sequence[Any]]]:
        for rid, table in self.tables():
            yield rid, f"rule {_q(rid)}", table.inputs, table.rows
        policy_table = self.policy_table()
        if policy_table is not None:
            yield "policy", "the policy table", policy_table.inputs, policy_table.rows

    def check_9_covered(self) -> None:
        """No ``boundaries`` field pairs a read with a leaf a table already compares that read
        against."""
        covered = self.table_compares()
        for owner, key, where, entry in self.boundary_entries():
            for path in _entry_paths(entry):
                if (key, path) in covered:
                    self.fail(
                        9,
                        (owner, key, path),
                        f"{where} names {_q(path)}, which a table already compares {_q(key)} "
                        "against",
                    )

    def check_9_context(self) -> None:
        """Each context entry's text: its template and slots, a slot naming the flagged ref
        as spelled, and no slot or flagged ref whose class tree holds ``identifier``."""
        for entry in self.context_entries():
            where = f"the context entry for {_q(entry.ref)}"
            self.text(entry.text, where, (entry.ref,))
            refs = [
                cast(str, slot["origin"]["ref"])
                for slot in entry.text.slots
                if "ref" in slot["origin"]
            ]
            if entry.ref not in refs:
                self.fail(9, (entry.ref,), f"{where} has no slot naming {_q(entry.ref)}")
        for u in self.uses:
            if u.role not in ("flagged", "context-slot"):
                continue
            tree = self.tree_at(u)
            if tree is not None and _holds_identifier(tree):
                names = (u.owner, u.ref) if u.owner else (u.ref,)
                self.fail(
                    9, names, f"{u.where} names {_q(u.ref)}, whose class tree holds identifier"
                )

    def tree_at(self, u: _Use) -> Any:
        """The full class tree at a read's field path, or ``None`` when its path names no field,
        which the read's typing reports."""
        if u.parsed.level == "q":
            return None
        t = self.state_type if u.base == _STATE else self.types.get(u.base)
        if t is None:
            return None
        tree: Any = _tree(t, self.defs, self.d.classes.get(u.base))
        for field_name in u.parsed.fields:
            if type(tree) is not dict or field_name not in tree:
                return None
            tree = cast("dict[str, Any]", tree)[field_name]
        return tree

    def check_9_boundaries(self) -> None:
        """Check 9's ``boundaries`` clauses. A leaf check 8 refused is left out. No table compares a
        read, since check 12 refuses tables, so the clause on a boundary a table already
        compares applies to none."""
        reads = {owner: set(r) for owner, _, _, r in self.boundary_fields()}
        for owner, key, where, entry in self.boundary_entries():
            names = (owner, key)
            if key not in reads[owner]:
                self.fail(9, names, f"{where}: {_q(key)} is not one of its component's reads")
            parsed = _parsed(key)
            assert parsed is not None
            if parsed.level != "q":
                t = self.read_type(_Use(owner, where, "boundary", key, parsed))
                if not isinstance(t, TypeFailure) and not conforms(t, _NUMBER, self.defs):
                    self.fail(
                        9,
                        names,
                        f"{where}: {_q(key)} is neither a q: read nor of a type that conforms "
                        f"to number: its type is {render(t)}",
                    )
                for path in self.typed(cast("list[str]", entry)):
                    self.bound(owner, where, path, self.leaf_types[path], _NUMBER, "number")
                continue
            question = self.question(f"q:{parsed.name}")
            extraction = question.kind is AnswerKind.extraction
            if extraction != ("null" in entry):
                wanted = (
                    "null, confidence and value"
                    if extraction
                    else ("bar, thresholds, margin and mass")
                )
                self.fail(
                    9,
                    names,
                    f"{where}: {_q(key)} is a {question.kind} read, whose roles are {wanted}",
                )
                continue
            if extraction:
                self.extraction_bounds(owner, key, where, entry, question)
                continue
            if question.kind is AnswerKind.noul:
                labels: tuple[str, ...] = _NOUL_LABELS
            else:  # the names alone: a description is no label; a label set's fixed part
                labels = label_parts(_fixed(question.labels))[0]
            for role in ("bar", "margin"):
                for path in self.typed(entry[role]):
                    self.bound(owner, where, path, self.leaf_types[path], _UNIT, "number [0, 1]")
            for path in self.typed(entry["thresholds"]):
                t = self.leaf_types[path]
                self.bound(owner, where, path, t, _UNITS, "mapping<string, number [0, 1]>")
                value = self.leaf_value(path)
                if type(value) is dict:
                    keys = cast("dict[str, object]", value)
                    for label in code_point_sorted(set(keys) - set(labels)):
                        self.fail(
                            9,
                            (owner, key, path),
                            f"{where}: {_q(path)} has the key {_q(label)}, which is not a label of "
                            f"{_q(key)}",
                        )
            if question.kind is AnswerKind.noul and entry["thresholds"]:
                self.fail(9, names, f"{where}: a noul question's thresholds are not empty")
            for m in cast("list[dict[str, list[str]]]", entry["mass"]):
                for path in self.typed(m["bounds"]):
                    self.bound(owner, where, path, self.leaf_types[path], _UNIT, "number [0, 1]")
                for label in code_point_sorted(set(m["labels"]) - set(labels)):
                    self.fail(
                        9,
                        names,
                        f"{where}: its mass names {_q(label)}, which is not a label of {_q(key)}",
                    )

    def extraction_bounds(
        self, owner: str, key: str, where: str, entry: Mapping[str, Any], question: Question
    ) -> None:
        """An extraction read's roles: ``null`` and ``confidence`` leaves of ``number [0, 1]``,
        ``value`` leaves of ``number``, only over a scalar ``number`` question."""
        for role in ("null", "confidence"):
            for path in self.typed(entry[role]):
                self.bound(owner, where, path, self.leaf_types[path], _UNIT, "number [0, 1]")
        if entry["value"] and (question.value_type is not ValueType.number or question.is_list):
            self.fail(
                9,
                (owner, key),
                f"{where}: its value role names leaves, but {_q(key)} is not a scalar number "
                "question",
            )
            return
        for path in self.typed(entry["value"]):
            self.bound(owner, where, path, self.leaf_types[path], _NUMBER, "number")

    def typed(self, paths: Sequence[str]) -> list[str]:
        """The leaf paths check 8 let through, each with its type."""
        return [p for p in paths if p not in self.refused_leaves and p in self.leaf_types]

    def bound(self, owner: str, where: str, path: str, t: Type, to: Type, shown: str) -> None:
        if not conforms(t, to, {}):
            self.fail(
                9,
                (owner, path),
                f"{where}: {_q(path)}'s type is {render(self.leaf_written[path])}, which does not "
                f"conform to {shown}",
            )

    def question(self, ref: str) -> Question:
        """The question that provides a ``q:`` ref; checks 3 and 4 have passed."""
        provider = self.providers[ref][0]
        assert provider is not None
        provided = self.d.sources[provider].provides[ref]
        assert isinstance(provided, Question)
        return provided

    def check_10_trees(self) -> None:
        """Check 10's test of each declared class tree against its ref's type, run once checks
        1 to 4, 10 and 12 pass."""
        for key, _ in self.class_keys:  # a key check 7 refused is left out
            if key.startswith("q:"):
                messages = _extraction_conformance(self.d.classes[key])
            else:
                t = self.state_type if key == _STATE else self.types[key]
                assert t is not None
                messages = _conformance(self.d.classes[key], t, self.defs, "")
            for message in messages:
                self.fail(10, (key,), f"the class tree of {_q(key)} {message}")

    def read_type(self, u: _Use) -> Type | TypeFailure:
        if u.base == _STATE:
            assert self.state_type is not None
            base = self.state_type
        else:
            base = self.types[u.base]
        return read_type(base, u.parsed.fields, self.defs)

    def text(self, text: Text | Structured, where: str, names: tuple[str, ...]) -> None:
        if isinstance(text, Structured):
            found = _structured_template(text.template)
        else:
            found = _template(text.template)
        if isinstance(found, str):
            self.fail(9, names, f"{where}'s template {found}")
            return
        placeholders, has_text = found
        slot_names = [cast(str, s["name"]) for s in text.slots]
        counts = Counter(slot_names)
        for name in code_point_sorted({n for n, k in counts.items() if k > 1}):
            self.fail(9, (*names, name), f"{where} names the slot {_q(name)} twice")
        for name in code_point_sorted(set(placeholders) - set(slot_names)):
            self.fail(9, (*names, name), f"{where}'s placeholder {_q(name)} names no slot")
        for name in code_point_sorted(set(slot_names) - set(placeholders)):
            self.fail(9, (*names, name), f"{where}'s slot {_q(name)} is not in its template")
        # A question's wording holds a non-whitespace character outside its placeholders.
        if not has_text:
            self.fail(
                9,
                names,
                f"{where}'s template holds no character that is not whitespace outside its "
                "placeholders",
            )

    # --- the graph and check 5 -------------------------------------------------------------

    def provider_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for ref in code_point_sorted(self.providers):
            provider = self.providers[ref][0]
            if provider is not None:
                out[ref] = provider
        return out

    def needs(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {sid: set(s.needs) for sid, s in self.sources()}
        per_candidate = _per_candidate_refs(self.d)
        for u in self.uses:
            # A slot on a grounded declared ref is a need of each candidate's question alone,
            # given under its grounded ref.
            if u.role in ("slot", "text") and u.base not in per_candidate:
                out[u.owner].add(u.ref)
        return {sid: frozenset(refs) for sid, refs in out.items()}

    def graph(self) -> Mapping[str, frozenset[str]]:
        providers = self.provider_map()
        preds: dict[str, set[str]] = {n: set() for n in (*self.d.sources, *self.d.rules)}
        for u in self.uses:
            if u.role in (
                "need",
                "slot",
                "label-set",
                "grounding",
                "text",
                "read",
                "helper",
                "source-requirement",
            ) and (u.base != _STATE):
                preds[u.owner].add(providers[u.base])
        graph = {n: frozenset(preds[n]) for n in code_point_sorted(preds)}
        # Every predecessor is a key, or graphlib would add it as a root.
        assert all(p in graph for ps in graph.values() for p in ps)
        return MappingProxyType(graph)

    def check_5(self, graph: Mapping[str, frozenset[str]]) -> None:
        # Keys and predecessors in code-point order, so the cycle found is alike under every
        # hash seed; a frozenset's iteration order is not.
        ordered = {n: tuple(code_point_sorted(graph[n])) for n in code_point_sorted(graph)}
        sorter = TopologicalSorter(ordered)
        try:
            sorter.prepare()
        except CycleError as exc:
            found = cast("list[str]", exc.args[1])[:-1]  # graphlib repeats the first node
            start = found.index(min(found))
            cycle = found[start:] + found[:start]  # it starts at its least id
            shown = " -> ".join(_q(n) for n in (*cycle, cycle[0]))
            self.fail(5, tuple(cycle), f"the dependency graph has a cycle: {shown}")

    # --- the document and the loaded design ------------------------------------------------

    def document(self) -> dict[str, Any]:
        d = self.d
        policy = d.policy
        parameters: dict[str, Any] = {}
        for path, leaf in d.parameters.items():
            t = self.leaf_types.get(path)
            value = self.leaf_value(path)
            if t is not None and value_conforms(value, t, {}, form="live"):
                value = document_value(value, t)
            else:
                value = leaf.value
            parameters[path] = {"type": leaf.type, "value": value}
            if self.format >= KEY_ORDER_FORMAT:
                parameters[path]["order"] = order_of(value)
        return {
            "format": self.format,
            "name": d.name,
            "state": d.state,
            "sources": {sid: _source(s, self.format) for sid, s in d.sources.items()},
            "rules": {rid: _rule(r, self.format) for rid, r in d.rules.items()},
            "policy": (
                {
                    "form": "function",
                    "reads": _unordered(policy.reads),
                    "params": _unordered(policy.params),
                    "boundaries": _boundaries(policy.boundaries),
                    **_function(policy.function, policy.version),
                }
                if isinstance(policy, Policy)
                else {
                    "form": "table",
                    "inputs": [_input(i) for i in policy.inputs],
                    "rows": [_row(r) for r in policy.rows],
                }
            ),
            "requirements": [{"ref": r.ref, "value": r.value} for r in d.requirements],
            "default_deny": d.default_deny,
            "tools": {name: {} for name in d.tools},
            "parameters": parameters,
            "types": dict(d.types),
            "plans": {name: _plan(plan) for name, plan in d.plans.items()},
            "groups": {g: code_point_sorted(group.refs) for g, group in d.groups.items()},
            "classes": dict(d.classes),  # each tree plain data, from check 10's form test
            **({"context": _context(d.context)} if self.format >= CONTEXT_FORMAT else {}),
            **(
                {
                    "targets": {
                        tid: {"statement": target.statement, "values": target.values}
                        for tid, target in d.targets.items()
                    }
                }
                if self.format >= TARGETS_FORMAT
                else {}
            ),
        }

    # --- check 11's comparison -------------------------------------------------------------

    def compare(self, document: bytes) -> None:
        """Check 11's comparison: the document load computes from what it built and resolved
        equals the parsed document's canonical encoding, member by member, naming the
        component and member that differ; then that encoding equals the bytes given."""
        assert self.parsed is not None
        given = encode(self.parsed, MAX_DOCUMENT_DEPTH)
        if document != given:
            computed = cast("dict[str, Any]", decode(document))
            self.differences(computed, cast("dict[str, Any]", decode(given)))
            # The gate gave both documents one member set, so unequal bytes differ in a member.
            assert self.failed(frozenset({11}))
        elif self.given is not None and self.given != given:
            self.fail(11, (), "the document's bytes are not its canonical encoding")

    def differences(self, computed: dict[str, Any], given: dict[str, Any]) -> None:
        for member in code_point_sorted(computed):
            a, b = computed[member], given[member]
            if _same(a, b):
                continue
            kind = _KINDS.get(member)
            if member == "policy" and isinstance(b, dict):
                self.component("the policy", ("policy",), a, cast("dict[str, Any]", b))
            elif kind is not None and isinstance(b, dict):
                block = cast("dict[str, Any]", b)
                for name in code_point_sorted(a):
                    x, y = a[name], block[name]
                    if _same(x, y):
                        continue
                    where = f"{kind} {_q(name)}"
                    if _object(x) is not None and _object(y) is not None:
                        self.component(where, (name,), cast("dict[str, Any]", x), y)
                    else:
                        self.fail(11, (name,), f"{where} is not what load computes from it")
            else:
                self.fail(11, (), f"the document's {_q(member)} is not what load computes from it")

    def component(
        self, where: str, names: tuple[str, ...], computed: dict[str, Any], given: dict[str, Any]
    ) -> None:
        for member in code_point_sorted(computed):
            a, b = computed[member], given[member]
            if _same(a, b):
                continue
            x, y = _object(a), _object(b)
            if member == "when" and x is not None and y is not None:
                self.component(f"{where}'s when", names, x, y)
            elif member == "source" and b is None:
                self.fail(11, names, f"{where} holds a null source, where load reads its source")
            elif member == "source" and a is None:
                self.fail(
                    11, names, f"{where} holds a source hash, where load cannot read its source"
                )
            elif member in ("module", "qualname", "source"):
                self.fail(11, names, f"{where}'s {member} is not its function's")
            elif type(b) is list and _same(a, _unordered(cast("list[Any]", b))):
                self.fail(11, names, f"{where}'s {member} are not in canonical order")
            else:
                self.fail(11, names, f"{where}'s {_q(member)} is not what load computes from it")

    def label_parts(self) -> dict[str, tuple[tuple[str, ...], tuple[str | None, ...]]]:
        """Each question's label names and descriptions, aligned: ``yes`` and ``no`` for
        ``noul``, none for ``extraction``."""
        parts: dict[str, tuple[tuple[str, ...], tuple[str | None, ...]]] = {}
        for _, source in self.sources():
            for ref, provided in source.provides.items():
                if not isinstance(provided, Question):
                    continue
                if provided.kind is AnswerKind.noul and provided.labels is None:
                    parts[ref] = (_NOUL_LABELS, (None, None))
                elif provided.kind is AnswerKind.noul:  # yes then no, described
                    parts[ref] = label_parts(_fixed(provided.labels))
                elif provided.kind is AnswerKind.extraction:
                    parts[ref] = ((), ())
                else:  # a candidate-valued label set's fixed part
                    parts[ref] = label_parts(_fixed(provided.labels))
        return parts

    def label_sets(self) -> dict[str, str]:
        """Each candidate-valued question's set ref, by ``q:`` ref in code-point order."""
        sets: dict[str, str] = {}
        for _, source in self.sources():
            for ref, provided in source.provides.items():
                if isinstance(provided, Question) and isinstance(provided.labels, Mapping):
                    sets[ref] = cast("Mapping[str, str]", provided.labels)["candidates"]
        return {ref: sets[ref] for ref in code_point_sorted(sets)}

    def label_at(self) -> dict[str, int]:
        """Each candidate-valued question's position for its candidates: how many fixed labels
        come before them."""
        at: dict[str, int] = {}
        for _, source in self.sources():
            for ref, provided in source.provides.items():
                if isinstance(provided, Question) and isinstance(provided.labels, Mapping):
                    declared = provided.labels
                    at[ref] = cast(int, declared.get("at", len(declared["fixed"])))
        return {ref: at[ref] for ref in code_point_sorted(at)}

    def loaded(self, document: bytes, graph: Mapping[str, frozenset[str]]) -> LoadedDesign:
        decoded = cast("dict[str, Any]", decode(document))
        providers = self.provider_map()
        classes: dict[str, Any] = {}
        assert self.state_type is not None
        typed = {_STATE: self.state_type, **self.types}
        resolutions: dict[str, Type] = {}
        for ref in (_STATE, *providers):
            declared = self.d.classes.get(ref)
            if ref.startswith("q:") and self.extraction_ref(ref):
                classes[ref] = _extraction_tree(declared)
            else:
                classes[ref] = _tree(typed[ref], self.defs, declared)
        callables: dict[str, Callable[..., Any]] = {
            rid: rule.function for rid, rule in self.rules()
        }
        for sid, source in self.sources():
            if source.when is not None:
                # The key a field function takes: the id, a dot, the field.
                callables[f"{sid}.when"] = source.when.function
        policy = self.policy()
        if policy is not None:
            callables["policy"] = policy.function
        return LoadedDesign(
            _BUILT_BY_LOAD,
            design=self.d,
            callables=MappingProxyType(callables),
            # Each `t:` and `d:` ref's declared type, its definitions resolved, for the
            # undeclared-read naming test.
            types=MappingProxyType(
                {
                    ref: _resolved(typed[ref], self.defs, resolutions)
                    for ref in code_point_sorted(typed)
                    if not ref.startswith("q:")
                }
            ),
            leaf_types=MappingProxyType(dict(self.leaf_types)),
            needs=MappingProxyType(self.needs()),
            groups=MappingProxyType(
                {g: tuple(code_point_sorted(group.refs)) for g, group in self.d.groups.items()}
            ),
            graph=graph,
            providers=MappingProxyType(providers),
            grid=_grid(decoded),
            labels=MappingProxyType(self.label_parts()),
            label_sets=MappingProxyType(self.label_sets()),
            grounded=MappingProxyType(dict(sorted(_per_candidate_refs(self.d).items()))),
            label_at=MappingProxyType(self.label_at()),
            # Each candidate set's candidate field, None for a list<string> set.
            candidate_sets=MappingProxyType(
                {
                    ref: None if field is True else field
                    for ref in code_point_sorted(providers)
                    if ref[:2] in ("t:", "d:") and (field := self.set_field(ref))
                }
            ),
            classes_bytes=encode(classes, MAX_DOCUMENT_DEPTH),
            # Each in the order the document holds, which the turn hands on.
            payload_bytes=MappingProxyType(
                {
                    s: _held_bytes(v["payload"], v.get("payload_order"))
                    for s, v in decoded["sources"].items()
                }
            ),
            leaf_bytes=MappingProxyType(
                {
                    p: _held_bytes(v["value"], v.get("order"))
                    for p, v in decoded["parameters"].items()
                }
            ),
            # Load's own copies, from the document, never the caller's objects.
            requirement_bytes=MappingProxyType(
                {
                    sid: frozenset(encode(value) for value in v["requirement"]["values"])
                    for sid, v in decoded["sources"].items()
                    if v["requirement"] is not None
                }
            ),
            # A holder standing in the design document; each value it holds was tested.
            requirements_bytes=encode(decoded["requirements"], MAX_DOCUMENT_DEPTH - 1),
            document=document,
            hash=sha256_ref(document),
            # Load's own copy of the block, from the document; none before format 5.
            context_bytes=encode(decoded.get("context"), MAX_DOCUMENT_DEPTH - 1),
            reads=MappingProxyType(self.component_reads()),
            params=MappingProxyType(self.component_params()),
            table_bytes=MappingProxyType(
                {
                    **{
                        rid: encode(decoded["rules"][rid], MAX_DOCUMENT_DEPTH)
                        for rid, _ in self.tables()
                    },
                    **(
                        {"policy": encode(decoded["policy"], MAX_DOCUMENT_DEPTH)}
                        if self.policy_table() is not None
                        else {}
                    ),
                }
            ),
            plan_bytes=MappingProxyType(
                {name: encode(plan, MAX_DOCUMENT_DEPTH) for name, plan in decoded["plans"].items()}
            ),
        )

    def component_reads(self) -> dict[str, tuple[str, ...]]:
        """Each rule's and the policy's declared reads: a table's its inputs' reads, each once
        in input order, the policy table's then each named plan's board refs."""
        out: dict[str, tuple[str, ...]] = {rid: tuple(r.reads) for rid, r in self.rules()}
        for rid, table in self.tables():
            out[rid] = tuple(dict.fromkeys(i["read"] for i in table.inputs))
        policy, policy_table = self.policy(), self.policy_table()
        if policy is not None:
            out["policy"] = tuple(policy.reads)
        elif policy_table is not None:
            refs = [i["read"] for i in policy_table.inputs]
            for name in self.named_plans():
                plan = self.d.plans[name]
                if plan.grounded is not None:  # the set, whose value the expansion walks
                    refs.append(plan.grounded)
                refs += [ref for ref, _ in plan_conditions(plan)]
            out["policy"] = tuple(dict.fromkeys(refs))
        return out

    def component_params(self) -> dict[str, tuple[str, ...]]:
        """Each rule's and the policy's leaf paths, a table's those it names, the policy
        table's with its named plans' slot leaves."""
        out: dict[str, tuple[str, ...]] = {rid: tuple(r.params) for rid, r in self.rules()}
        named: dict[str, list[str]] = {}
        for owner, _, path in self.table_params():
            named.setdefault(owner, []).append(path)
        for rid, _ in self.tables():
            out[rid] = tuple(dict.fromkeys(named.get(rid, [])))
        policy = self.policy()
        if policy is not None:
            out["policy"] = tuple(policy.params)
        elif self.policy_table() is not None:
            paths = named.get("policy", [])
            for name in self.named_plans():
                plan = self.d.plans[name]
                for text in (plan.reason, plan.reply):
                    for slot in () if text is None else text.slots:
                        tag, value = next(iter(slot["origin"].items()))
                        if tag == "param":
                            paths.append(cast(str, value))
            out["policy"] = tuple(dict.fromkeys(paths))
        return out


# ---------------------------------------------------------------------------------------------
# Helpers


def _value_cell(value: object, t: Type) -> str | None:
    """Why a value cell's value is not one the input's type enumerates, or ``None``."""
    if isinstance(t, Label):
        return (
            None
            if isinstance(value, str) and value in t.values
            else f"{encode(value).decode('utf-8')} is not a label of the input"
        )
    if isinstance(t, Scalar) and t.name == "boolean":
        return None if type(value) is bool else f"{encode(value).decode('utf-8')} is not a boolean"
    if _numeric_base(t) == "integer":
        if type(value) is not int:
            return f"{encode(value).decode('utf-8')} is not an integer"
        if isinstance(t, Interval) and not t.lo <= value <= t.hi:
            return f"{value} is outside the input's interval"
        return None
    return "a value cell stands only over a label, boolean or integer input"


def _fixed_cell(value: object, names: tuple[str, ...]) -> str | None:
    """Why a value cell over a candidate-valued pick does not name one of its fixed labels,
    or ``None``: the candidates are named per turn, so a cell names only the fixed part."""
    if isinstance(value, str) and value in names:
        return None
    allowed = ", ".join(encode(v).decode("utf-8") for v in names) or "none"
    return (
        f"{encode(value).decode('utf-8')} is not a fixed label of the input's candidate-valued "
        f"label set (fixed labels: {allowed})"
    )


def _numeric_base(t: Type) -> str | None:
    if isinstance(t, Scalar) and t.name in ("number", "integer"):
        return t.name
    if isinstance(t, Interval):
        return t.base
    return None


def _is_numeric(t: Type) -> bool:
    return _numeric_base(t) is not None


def _is_textual(t: Type) -> bool:
    return isinstance(t, Label) or (isinstance(t, Scalar) and t.name == "string")


def _number_value(value: object) -> bool:
    return type(value) is int or type(value) is float


def _exact(x: int | float) -> Fraction:
    return Fraction(x) if type(x) is int else Fraction(repr(x))


def _board_refs(plan: PlanDeclaration) -> list[str]:
    """A plan declaration's board-ref origins and slots, in rule 2's naming order."""
    return [ref for ref, _ in plan_conditions(plan)]


def _cell_refs(cells: Sequence[Any]) -> Iterator[str]:
    """Each ref a row's cells name: an ``in`` cell's list and a two-input comparison's other
    input."""
    for cell in cells:
        if cell == ANY:
            continue
        ((tag, member),) = cast("dict[str, Any]", cell).items()
        if tag == "in":
            yield cast(str, member)
        elif tag in COMPARISONS and "ref" in member:
            yield cast("dict[str, str]", member)["ref"]


def _cell_params(cells: Sequence[Any]) -> Iterator[str]:
    """Each leaf path a row's cells compare against."""
    for cell in cells:
        if cell == ANY:
            continue
        ((tag, member),) = cast("dict[str, Any]", cell).items()
        operands = member if tag == "between" else [member] if tag in COMPARISONS else []
        for operand in cast("list[dict[str, Any]]", operands):
            if "param" in operand:
                yield cast(str, operand["param"])


def _parsed(ref: str) -> _ParsedRef | None:
    try:
        return parse_ref(ref)
    except ValueError:  # the ref grammar's refusal
        return None


def _declared(ref: str) -> _ParsedRef | None:
    parsed = _parsed(ref)
    if parsed is None:
        return None
    return parsed if parsed.candidate is None and not parsed.fields else None


_PER_CANDIDATE_EXTRACTION = "per-candidate extraction question"
_PER_CANDIDATE_LABEL_SET = "per-candidate question over a candidate-valued label set"
_GROUNDED_NEED = "need or slot ref on a grounded declared ref"


def _per_candidate_refs(design: Design) -> dict[str, str]:
    """Each grounded declared ref and its grounding set: a question of a grounded source that
    holds a candidate slot, and a grounded rule's or table's own ref (§candidate-sets)."""
    out: dict[str, str] = {}
    for rule in design.rules.values():
        grounded = getattr(rule, "grounded", None)
        provides = getattr(rule, "provides", None)
        if (
            isinstance(cast(object, rule), Rule | Table)
            and isinstance(grounded, str)
            and isinstance(provides, str)
        ):
            out[provides] = grounded
    for source in design.sources.values():
        if not isinstance(cast(object, source), Source) or not isinstance(source.grounded, str):
            continue
        for ref, question in _questions(source):
            if any(tag == "candidate" for tag, _ in _origins(question)):
                out[ref] = source.grounded
        for ref in source.provides:
            if ref.startswith("t:"):
                out[ref] = source.grounded
    return out


def _origins(question: Question) -> Iterator[tuple[object, object]]:
    """Each slot origin of a question's wording as a tag and its value, skipping any of a form
    check 10 refuses, since check 12 runs before it."""
    slots = getattr(question.wording, "slots", ())
    if not isinstance(slots, Sequence):
        return
    for slot in cast("Sequence[object]", slots):
        origin = (
            cast("Mapping[str, object]", slot).get("origin") if isinstance(slot, Mapping) else None
        )
        if isinstance(origin, Mapping) and len(cast("Mapping[str, object]", origin)) == 1:
            yield next(iter(cast("Mapping[object, object]", origin).items()))


def _candidate_read(member: object) -> bool:
    """Whether a read set's member is a ``candidate`` read, which no ref spells."""
    return isinstance(member, str) and is_candidate_read(member)


def _declared_needs(source: Source) -> Iterator[str]:
    """A source's declared needs, as strings."""
    needs = cast(object, source.needs)
    if isinstance(needs, Sequence) and not isinstance(needs, str):
        for need in cast("Sequence[object]", needs):
            if isinstance(need, str):
                yield need


def _declared_base(ref: str) -> str | None:
    """A ref's declared ref, its bracket and field path removed, or ``None`` for no ref."""
    try:
        parsed = parse_ref(ref)
    except ValueError:
        return None
    return f"{parsed.level}:{parsed.name}"


def _questions(source: Source) -> Iterator[tuple[str, Question]]:
    for ref, provided in source.provides.items():
        if isinstance(provided, Question):
            yield ref, provided


def _slots(source: Source) -> Iterator[tuple[str, Mapping[str, Any]]]:
    for ref, question in _questions(source):
        for slot in question.wording.slots:
            yield ref, slot


def _unordered(members: Sequence[Any]) -> list[Any]:
    """An unordered list in ascending order of its members' canonical bytes."""
    return sorted(members, key=_held)


def _held(member: object) -> bytes:
    """A design member's canonical bytes, encoded as a holder within the document's depth:
    only the values check 10 names are held to the size bounds."""
    return encode(member, MAX_DOCUMENT_DEPTH)


def _boundaries(boundaries: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A ``boundaries`` field as the document writes it: every list unordered, each mass
    member's lists sorted before the mass list is."""
    if boundaries is None:
        return None
    out: dict[str, Any] = {}
    for key, entry in boundaries.items():
        if _parsed(key) is None:  # check 7's key: its entry as given
            out[key] = entry
            continue
        if not isinstance(entry, dict):  # a list of leaf paths
            out[key] = _unordered(cast("list[str]", entry))
            continue
        roles = cast("dict[str, Any]", entry)
        if "null" in roles:  # an extraction read's roles
            out[key] = {role: _unordered(roles[role]) for role in X_ROLES}
            continue
        mass = [
            {"labels": _unordered(m["labels"]), "bounds": _unordered(m["bounds"])}
            for m in cast("list[dict[str, list[str]]]", roles["mass"])
        ]
        out[key] = {
            "bar": _unordered(roles["bar"]),
            "thresholds": _unordered(roles["thresholds"]),
            "margin": _unordered(roles["margin"]),
            "mass": _unordered(mass),
        }
    return out


def _entry_paths(entry: Any) -> list[str]:
    """Every leaf path a ``boundaries`` entry names, in the order it gives them."""
    if isinstance(entry, list):
        return cast("list[str]", entry)
    roles = cast("dict[str, Any]", entry)
    if "null" in roles:
        return [*roles["null"], *roles["confidence"], *roles["value"]]
    paths = [*roles["bar"], *roles["thresholds"], *roles["margin"]]
    for m in cast("list[dict[str, list[str]]]", roles["mass"]):
        paths += m["bounds"]
    return paths


def _grid(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """The answer grid, from every ``boundaries`` field in the decoded document: keyed by read in
    code-point order, field paths kept. A ``d:`` or ``t:`` read's entry is a tuple of leaf paths; a
    ``q:`` read's maps ``bar``, ``thresholds`` and ``margin`` to tuples of leaf paths and ``mass``
    to a tuple of ``{labels, bounds}``, one per label set, its bounds sharing the set's one axis.
    Each role is the union over every component that names the read, without repeats and in
    ascending order of canonical bytes; the mass sets are in ascending order of their sorted labels'
    canonical bytes, and a set's labels are held in that order too, so a lint takes "the set's first
    label in declared order" from the question. A read whose boundaries name no leaf over
    every component has no entry. Read-only at every depth. A table gives its fields as
    ``_table_boundaries`` derives them."""
    rules = list(cast("dict[str, Any]", document["rules"]).values())
    policy = cast("dict[str, Any]", document["policy"])
    fields = [r.get("boundaries") for r in rules]
    fields.append(policy.get("boundaries"))
    leaves = cast("dict[str, Any]", document["parameters"])
    for table in [r for r in rules if r["form"] == "table"] + (
        [policy] if policy["form"] == "table" else []
    ):
        fields.append(_table_boundaries(table, leaves))
    entries: dict[str, Any] = {}
    for field in fields:
        for key, entry in cast("dict[str, Any]", field or {}).items():
            if isinstance(entry, list):
                into = cast("dict[bytes, str]", entries.setdefault(key, {}))
                into.update((_held(p), p) for p in cast("list[str]", entry))
                continue
            roles = cast("dict[str, Any]", entry)
            if "null" in roles:  # an extraction read's; check 9 gave every component one form
                into_roles = entries.setdefault(key, {role: {} for role in X_ROLES})
                for role in X_ROLES:
                    into_roles[role].update((_held(p), p) for p in roles[role])
                continue
            acc = entries.setdefault(key, {"bar": {}, "thresholds": {}, "margin": {}, "mass": {}})
            for role in ("bar", "thresholds", "margin"):
                acc[role].update((_held(p), p) for p in roles[role])
            for m in cast("list[dict[str, list[str]]]", roles["mass"]):
                labels = _unordered(m["labels"])
                _, bounds = acc["mass"].setdefault(_held(labels), (labels, {}))
                bounds.update((_held(p), p) for p in m["bounds"])
    grid: dict[str, Any] = {}
    for key in code_point_sorted(entries):
        acc = entries[key]
        if not key.startswith("q:"):
            if acc:
                grid[key] = _ascending(acc)
            continue
        if "null" in acc:
            if any(acc[role] for role in X_ROLES):
                grid[key] = MappingProxyType({role: _ascending(acc[role]) for role in X_ROLES})
            continue
        if not any(acc[role] for role in ("bar", "thresholds", "margin")) and not any(
            bounds for _, bounds in acc["mass"].values()
        ):
            continue  # no leaf, so no points
        mass = tuple(
            MappingProxyType({"labels": tuple(labels), "bounds": _ascending(bounds)})
            for _, (labels, bounds) in sorted(acc["mass"].items())
        )
        grid[key] = MappingProxyType(
            {
                "bar": _ascending(acc["bar"]),
                "thresholds": _ascending(acc["thresholds"]),
                "margin": _ascending(acc["margin"]),
                "mass": mass,
            }
        )
    return MappingProxyType(grid)


def _table_boundaries(table: Mapping[str, Any], leaves: Mapping[str, Any]) -> dict[str, Any]:
    """A table's boundaries, as a ``boundaries`` field writes them (§answer-grid): a pick
    helper's bindings under its read's roles, ``extract``'s ``null_bar`` under ``null``; a
    parameter comparison over a bare ``d:`` or ``t:`` read under that read, an integer literal
    in a comparison or a value cell as its value; over a ``mass`` input under ``mass`` with
    the set its ``labels`` leaf holds; over an ``extract`` input under ``value``. A cell over
    a ``size`` input, a ``candidate`` read or another input adds nothing."""
    inputs = cast("list[dict[str, Any]]", table["inputs"])
    out: dict[str, Any] = {}

    def roles(read: str, extraction: bool) -> dict[str, Any]:
        names = X_ROLES if extraction else ("bar", "thresholds", "margin", "mass")
        empty: dict[str, list[Any]] = {r: [] for r in names}
        return cast("dict[str, Any]", out.setdefault(read, empty))

    for i in inputs:
        helper, read, params = i["helper"], cast(str, i["read"]), i["params"]
        if helper in ("pick_with_margin", "pick_with_thresholds"):
            entry = roles(read, False)
            for role in ("bar", "thresholds", "margin"):
                if role in params:
                    entry[role].append(params[role])
        elif helper == "extract":
            roles(read, True)["null"].append(params["null_bar"])
    for row in cast("list[dict[str, Any]]", table["rows"]):
        for n, cell in enumerate(cast("list[Any]", row["cells"])):
            if cell == ANY or n >= len(inputs):
                continue
            ((tag, member),) = cast("dict[str, Any]", cell).items()
            i = inputs[n]
            helper, read = i["helper"], cast(str, i["read"])
            if is_candidate_read(read):
                continue  # a cell over the candidate adds nothing (§candidate-sets)
            if tag in ("is", "one_of"):
                # An integer value cell is a literal boundary of a bare numeric read.
                values = [member] if tag == "is" else cast("list[object]", member)
                ints = [v for v in values if type(v) is int]
                if ints and helper is None and not read.startswith("q:"):
                    out.setdefault(read, []).extend(ints)
                continue
            if tag == "between":
                operands = cast("list[dict[str, Any]]", member)
            elif tag in COMPARISONS and "ref" not in member:
                operands = [cast("dict[str, Any]", member)]
            else:
                continue
            bounds = [o.get("param", o.get("literal")) for o in operands]
            if helper is None and not read.startswith("q:"):
                out.setdefault(read, []).extend(bounds)
            elif helper == "mass":
                labels = leaves[i["params"]["labels"]]["value"]
                mass = roles(read, False)["mass"]
                found = next((m for m in mass if m["labels"] == labels), None)
                if found is None:
                    mass.append({"labels": list(labels), "bounds": list(bounds)})
                else:
                    found["bounds"].extend(bounds)
            elif helper == "extract":
                roles(read, True)["value"].extend(bounds)
    return out


def _ascending(members: Mapping[bytes, str]) -> tuple[str, ...]:
    return tuple(members[b] for b in sorted(members))


def _function(fn: Callable[..., Any], version: str | None) -> dict[str, Any]:
    try:
        source: str | None = sha256_ref(inspect.getsource(fn).encode("utf-8"))
    except (OSError, TypeError, ValueError):  # ValueError: a __wrapped__ loop or overlong chain
        source = None
    return {
        "module": fn.__module__,
        "qualname": fn.__qualname__,
        "version": version,
        "source": source,
    }


def _label(label: object) -> object:
    """A label as the document writes it: a name, or an object of name and description."""
    return label if isinstance(label, str) else dict(cast("Mapping[str, str]", label))


def _structured_template(template: object, at: str = "") -> tuple[list[str], bool] | str:
    """A structured template's placeholders over its string values, and whether a key or a
    value holds a character that is not whitespace outside them, or why a string value is
    malformed, naming its path. Keys and other values are literal."""
    names: list[str] = []
    has_text = False
    members: list[tuple[str, object]]
    if type(template) is dict:
        members = list(cast("dict[str, object]", template).items())
    elif type(template) is list:
        members = [(f"[{i}]", v) for i, v in enumerate(cast("list[object]", template))]
    else:
        return [], False
    is_object = type(cast(object, template)) is dict
    for key, value in members:
        path = f"{at}[{_q(key)}]" if is_object else f"{at}{key}"
        if is_object:
            has_text = has_text or any(not c.isspace() for c in key)
        if isinstance(value, str):
            parsed = _template(value)
            if isinstance(parsed, str):
                return f"at {path} {parsed}"
            leaf_names, leaf_text = parsed
        elif type(value) in (dict, list):
            nested = _structured_template(value, path)
            if isinstance(nested, str):
                return nested
            leaf_names, leaf_text = nested
        else:
            leaf_names, leaf_text = [], value is not None
        names += leaf_names
        has_text = has_text or leaf_text
    return names, has_text


def _context(context: Context | None) -> dict[str, Any] | None:
    """The context block's document object, or ``None`` for none."""
    if context is None:
        return None
    return {
        "entries": [{"ref": e.ref, "text": _text(e.text)} for e in context.entries],
        "plan": context.plan,
        "steps": context.steps,
        "record": context.record,
    }


def _holds_identifier(tree: object) -> bool:
    """Whether a full class tree holds ``identifier`` at its root or beneath."""
    if type(tree) is str:
        return tree == "identifier"
    if type(tree) is dict:
        return any(_holds_identifier(v) for v in cast("dict[str, object]", tree).values())
    if type(tree) is list:
        return any(_holds_identifier(v) for v in cast("list[object]", tree))
    return False


def _text(text: Text | Structured, fmt: int = 0) -> dict[str, Any]:
    written: dict[str, Any] = {
        "template": text.template,
        "slots": [{"name": s["name"], "origin": dict(s["origin"])} for s in text.slots],
    }
    order = order_of(text.template) if fmt >= KEY_ORDER_FORMAT else {}
    if order:  # written only when the template is out of canonical order, so its identity holds
        written["order"] = order
    return written


def _fixed(labels: object) -> tuple[Any, ...]:
    """A question's declared labels, or a candidate-valued label set's fixed part."""
    if isinstance(labels, Mapping):
        return cast("tuple[Any, ...]", cast("Mapping[str, Any]", labels)["fixed"])
    return cast("tuple[Any, ...]", labels)


def _declared_labels(labels: object) -> Any:
    """A question's labels as the document writes them: a list of labels, or a candidate-valued
    label set's fixed part and set ref."""
    if isinstance(labels, Mapping):
        declared = cast("Mapping[str, Any]", labels)
        fixed = cast("tuple[object, ...]", declared["fixed"])
        return {
            "fixed": [_label(v) for v in fixed],
            "candidates": declared["candidates"],
            # Always written: where the candidates go among the fixed labels, by default last.
            "at": declared.get("at", len(fixed)),
        }
    return [_label(v) for v in cast("tuple[object, ...]", labels)]


def _source(source: Source, fmt: int) -> dict[str, Any]:
    provides: dict[str, Any] = {}
    for ref, provided in source.provides.items():
        if isinstance(provided, Question):
            labels = provided.labels
            provides[ref] = {
                "kind": str(provided.kind),
                "wording": _text(provided.wording, fmt),
                # As declared: a described label is its object of name and description.
                "labels": None if labels is None else _declared_labels(labels),
            }
            if provided.kind is AnswerKind.extraction:
                date = provided.reference_date
                provides[ref].update(
                    value_type=str(provided.value_type),
                    is_list=provided.is_list,
                    text=dict(cast("Mapping[str, str]", provided.text)),
                    reference_date=None if date is None else dict(date),
                )
            if fmt >= TARGETS_FORMAT:
                provides[ref].update(
                    target=provided.target,
                    latent_targets=_unordered(
                        [
                            {"target": latent["target"], "role": str(latent["role"])}
                            for latent in provided.latent_targets
                        ]
                    ),
                )
        else:
            provides[ref] = dict(cast("Mapping[str, Any]", provided))
    w, r = source.when, source.requirement
    return {
        "provides": provides,
        "needs": _unordered(source.needs),
        "when": None if w is None else _function(w.function, w.version),
        "requirement": None if r is None else {"ref": r.ref, "values": _unordered(r.values)},
        "oracle": source.oracle,
        "timeout": source.timeout,
        "payload": source.payload,
        "cost": source.cost,
        "feedback": source.feedback,
        "grounded": source.grounded,
        **({"payload_order": order_of(source.payload)} if fmt >= KEY_ORDER_FORMAT else {}),
    }


def _rule(rule: Rule | Table, fmt: int) -> dict[str, Any]:
    if isinstance(rule, Table):
        return {
            "form": "table",
            "provides": rule.provides,
            "outputs": rule.outputs,
            "hit": rule.hit,
            "order": None if rule.order is None else list(rule.order),
            "inputs": [_input(i) for i in rule.inputs],
            "rows": [_row(r) for r in rule.rows],
            "candidates": rule.candidates,
            "grounded": rule.grounded,
            **({"target": rule.target} if fmt >= TARGETS_FORMAT else {}),
        }
    return {
        "form": "function",
        "provides": rule.provides,
        "outputs": rule.outputs,
        "reads": _unordered(rule.reads),
        "params": _unordered(rule.params),
        "boundaries": _boundaries(rule.boundaries),
        "candidates": rule.candidates,
        "grounded": rule.grounded,
        **({"target": rule.target} if fmt >= TARGETS_FORMAT else {}),
        **_function(rule.function, rule.version),
    }


def _input(i: Mapping[str, Any]) -> dict[str, Any]:
    return {"read": i["read"], "helper": i["helper"], "params": dict(i["params"])}


def _row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row in the document, a ``one_of`` cell's values in canonical order."""
    cells: list[Any] = []
    for c in cast("list[Any]", row["cells"]):
        tagged = _object(c)
        if tagged is not None and "one_of" in tagged:
            cells.append({"one_of": _unordered(cast("list[Any]", tagged["one_of"]))})
        elif tagged is not None and "markers" in tagged:
            cells.append({"markers": _unordered(cast("list[Any]", tagged["markers"]))})
        else:
            cells.append(c)
    return {**row, "cells": cells}


def _plan(plan: PlanDeclaration) -> dict[str, Any]:
    return {
        "kind": plan.kind,
        "steps": [s if isinstance(s, Mapping) else step_document(s) for s in plan.steps],
        "reason": None if plan.reason is None else _text(plan.reason),
        "reply": None if plan.reply is None else _text(plan.reply),
        "grounded": plan.grounded,
    }


def _template(template: str) -> tuple[list[str], bool] | str:
    """A template's placeholders and whether it holds a character that is not whitespace
    outside them, or why it is malformed."""
    names: list[str] = []
    has_text = False
    i, n = 0, len(template)
    while i < n:
        c = template[i]
        if template.startswith("{{", i) or template.startswith("}}", i):
            has_text = True
            i += 2
        elif c == "{":
            end = template.find("}", i + 1)
            name = template[i + 1 : end] if end != -1 else ""
            if end == -1 or not is_segment(name):
                return f"has a {{ at position {i} that begins no placeholder or escape"
            names.append(name)
            i = end + 1
        elif c == "}":
            return f"has a lone }} at position {i}"
        else:
            has_text = has_text or not c.isspace()
            i += 1
    return names, has_text


def _resolved(t: Type, defs: Mapping[str, Type], memo: dict[str, Type] | None = None) -> Type:
    """A type with every definition's name in it resolved, at every depth; each definition is
    resolved once, kept in ``memo`` by name, and its resolution stands at each use."""
    memo = {} if memo is None else memo
    if isinstance(t, Named):
        found = memo.get(t.name)
        if found is None:
            # A chain of names is followed in a loop, so its length costs no recursion.
            found = memo[t.name] = _resolved(resolve(t, defs), defs, memo)
        return found
    if isinstance(t, ListOf):
        return ListOf(_resolved(t.item, defs, memo))
    if isinstance(t, MappingOf):
        return MappingOf(_resolved(t.value, defs, memo))
    if isinstance(t, RecordOf):
        return RecordOf(tuple((n, _resolved(ft, defs, memo)) for n, ft in t.fields))
    if isinstance(t, Nullable):
        return Nullable(_resolved(t.inner, defs, memo))
    return t


def _too_large() -> str:
    return (
        f"the type spells more than {power(_TYPE_SIZE)} code points with its definitions resolved"
    )


def _type_size(text: str, t: Type, sizes: Mapping[str, int]) -> int:
    """A type's size: the code points of its spelling, ``t`` parsed from it, with each
    definition's name replaced by the definition's size, from ``sizes``; at most one past the
    bound, so no count grows with the design's sharing."""
    size = len(string_data(text))
    stack = [t]
    while stack:
        node = stack.pop()
        if isinstance(node, Named):
            size += sizes[node.name] - len(node.name)
        elif isinstance(node, ListOf):
            stack.append(node.item)
        elif isinstance(node, MappingOf):
            stack.append(node.value)
        elif isinstance(node, RecordOf):
            stack.extend(ft for _, ft in node.fields)
        elif isinstance(node, Nullable):
            stack.append(node.inner)
    return min(size, _TYPE_SIZE + 1)


def _extraction_origins(question: Question) -> Iterator[tuple[str, Mapping[str, str]]]:
    """An extraction question's ``text`` and ``reference_date`` origins, each declared;
    check 10's form test has passed them."""
    if question.kind is not AnswerKind.extraction:
        return
    for member, origin in (("text", question.text), ("reference_date", question.reference_date)):
        if origin is not None:
            yield member, origin


def _extraction_tree(declared: Any) -> dict[str, Any]:
    """An extraction answer's class tree in full: a declared class name sets ``value``,
    ``guess`` and ``span``, a declared object the members it names; the rest keep the
    default."""
    tree = dict(_EXTRACTION_TREE)
    if isinstance(declared, str):
        tree.update(dict.fromkeys(_EXTRACTION_SET, declared))
    elif isinstance(declared, dict):
        tree.update(cast("dict[str, Any]", declared))
    return tree


def _extraction_conformance(tree: Any) -> list[str]:
    """Why a class tree does not conform to an extraction answer, a record of its eight
    members, each a scalar or a list of scalars, so each takes a class name."""
    if isinstance(tree, str):
        return []
    out: list[str] = []
    for key, member in cast("dict[str, Any]", tree).items():
        if key not in _EXTRACTION_TREE:
            out.append(
                f"names {_q(key)} at its root, which is not a member of an extraction answer"
            )
        elif not isinstance(member, str):
            out.append(f"holds an object at {_q(key)}, where its type is not a record")
    return out


def _scalar(t: Type) -> bool:
    return isinstance(t, Scalar | Interval | Label)


def _leaf_shape(t: Type) -> bool:
    """A leaf's type is a scalar, a list of a scalar, or a mapping to a scalar."""
    if isinstance(t, ListOf):
        return _scalar(t.item)
    if isinstance(t, MappingOf):
        return _scalar(t.value)
    return _scalar(t)


def _tree(t: Type, defs: Mapping[str, Type], declared: Any = None) -> Any:
    """A type's class tree in full, ``declared`` applied over the default:
    an object at every record, a one-member list at every mapping layer, a class name at every
    other node. A declared class name covers every value beneath it; a field a declared object
    leaves out keeps its default. ``declared`` conforms to ``t`` (check 10)."""
    t = resolve(t, defs)
    if isinstance(t, Nullable):
        return _tree(t.inner, defs, declared)
    if isinstance(t, ListOf):
        return _tree(t.item, defs, declared)
    if isinstance(t, MappingOf):
        return [_tree(t.value, defs, declared)]
    if isinstance(t, RecordOf):
        fields = cast("dict[str, Any]", declared) if isinstance(declared, dict) else {}
        return {
            n: _tree(ft, defs, declared if isinstance(declared, str) else fields.get(n))
            for n, ft in t.fields
        }
    if isinstance(declared, str):
        return declared
    if isinstance(t, Scalar) and t.name in ("string", "bytes"):
        return "content"
    assert not isinstance(t, Named)
    return "open"


def _conformance(tree: Any, t: Type, defs: Mapping[str, Type], path: str) -> list[str]:
    """Why a class tree does not conform to ``t``: a class name conforms anywhere, and an
    object where ``t``, through list, mapping and ``or null`` layers and definitions' names,
    is a record, each key naming a field whose tree conforms."""
    if isinstance(tree, str):
        return []
    t = resolve(t, defs)
    while isinstance(t, Nullable | ListOf | MappingOf):
        inner = t.inner if isinstance(t, Nullable) else t.item if isinstance(t, ListOf) else t.value
        t = resolve(inner, defs)
    where = f"at {_q(path)}" if path else "at its root"
    if not isinstance(t, RecordOf):
        return [f"holds an object {where}, where its type is not a record"]
    fields = dict(t.fields)
    out: list[str] = []
    for key, member in cast("dict[str, Any]", tree).items():
        if key not in fields:
            out.append(f"names {_q(key)} {where}, which is not a field of the record")
        else:
            out += _conformance(member, fields[key], defs, f"{path}.{key}" if path else key)
    return out


# ---------------------------------------------------------------------------------------------
# A document: check 11's gate, resolving its functions, and the Design it declares


_DOCUMENT = frozenset(
    {
        "format",
        "name",
        "state",
        "sources",
        "rules",
        "policy",
        "requirements",
        "default_deny",
        "tools",
        "parameters",
        "types",
        "plans",
        "groups",
        "classes",
    }
)
_CONTEXT = frozenset({"entries", "plan", "steps", "record"})
_TARGET = frozenset({"statement", "values"})
_LATENT = frozenset({"target", "role"})
_FUNCTION = frozenset({"module", "qualname", "version", "source"})
_SOURCE = frozenset(
    {
        "provides",
        "needs",
        "when",
        "requirement",
        "oracle",
        "timeout",
        "payload",
        "cost",
        "feedback",
        "grounded",
    }
)
_RULE = (
    frozenset(
        {"form", "provides", "outputs", "reads", "params", "boundaries", "candidates", "grounded"}
    )
    | _FUNCTION
)
_TABLE = frozenset(
    {"form", "provides", "outputs", "candidates", "grounded", "hit", "order", "inputs", "rows"}
)
_POLICY = frozenset({"form", "reads", "params", "boundaries"}) | _FUNCTION
_POLICY_TABLE = frozenset({"form", "inputs", "rows"})
_PLAN = frozenset({"kind", "steps", "reason", "reply", "grounded"})
_STEP = frozenset({"tool", "arguments", "requirements"})
_ORIGINS = frozenset(
    {
        frozenset({"literal"}),
        frozenset({"ref"}),
        frozenset({"step", "path"}),
        frozenset({"candidate"}),
    }
)
_KINDS = {
    "sources": "source",
    "rules": "rule",
    "parameters": "parameter",
    "plans": "plan declaration",
    "groups": "group",
    "tools": "tool",
    "types": "type definition",
    "classes": "the classes block's key",
}


def _gate(given: bytes | Mapping[str, Any]) -> tuple[dict[str, Any] | None, Any]:
    """The parsed document, or check 11's one failure as its names and message: the depth,
    counted before any parse; UTF-8 JSON without a duplicate key; ``format``, read first; the
    value domain; then every object's members by its form. No message holds text from the parser or
    the document's values."""
    deep = f"the document nests more than {MAX_DOCUMENT_DEPTH} deep"
    if isinstance(given, bytes):
        if document_depth(given) > MAX_DOCUMENT_DEPTH:
            return None, ((), deep)
        try:
            given.decode("utf-8")
            doc: object = decode(given)
        except UnicodeDecodeError:
            return None, ((), "the document is not UTF-8")
        except _DuplicateKey:
            return None, ((), "the document holds an object with a key twice")
        except _DecodeError:
            return None, ((), "the document is not JSON whose numbers are in the value domain")
    else:
        # Before any other walk: a container at two places would expand at each.
        if shares_a_container(given):
            return None, ((), "the document holds a list or object at two places")
        if nesting_depth(given, MAX_DOCUMENT_DEPTH) > MAX_DOCUMENT_DEPTH:
            return None, ((), deep)
        doc = given
    if type(doc) is not dict:
        return None, ((), "the document is not a JSON object")
    members = cast("dict[str, Any]", doc)
    found = member_named(members, "format")  # no code of a caller's key runs
    if found is MISSING:
        return None, ((), "the document has no format")
    if type(found) is not int or found not in DESIGN_FORMATS:
        known = ", ".join(str(f) for f in sorted(DESIGN_FORMATS))
        return None, ((), f"the document's format is not one this engine reads: {known}")
    # The document as a holder, then each value it holds to the size bounds too; it shares
    # no container, so the holder's walk is linear in it.
    if domain_error(members, MAX_DOCUMENT_DEPTH) is not None or any(
        domain_error(value) is not None for value in _value_places(members)
    ):
        return None, ((), "the document holds a value outside the value domain")
    return members, next(_member_failures(members, found), None)


def _value_places(doc: dict[str, Any]) -> Iterator[object]:
    """Each value the document holds where the size bounds apply too: each payload, leaf
    value, source requirement value and declared class tree, wherever the member checks will
    find it; the document itself is held to its depth alone. Members are found by their
    keys' string data, so no code of a caller's key runs."""
    for source in _objects_in(member_named(doc, "sources")):
        payload = member_named(source, "payload")
        if payload is not MISSING:
            yield payload
        values = member_named(member_named(source, "requirement"), "values")
        if type(values) is list or type(values) is tuple:
            yield from cast("list[object]", values)
    for leaf in _objects_in(member_named(doc, "parameters")):
        value = member_named(leaf, "value")
        if value is not MISSING:
            yield value
    yield from members_of(member_named(doc, "classes"))


def _objects_in(block: object) -> Iterator[object]:
    """A block's members that are objects."""
    for found in members_of(block):
        if type(found) is dict:
            yield found


def _missing_or_extra(
    obj: Mapping[str, Any], members: frozenset[str], where: str, names: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], str]]:
    for member in code_point_sorted(members - set(obj)):
        yield names, f"{where} has no member {_q(member)}"
    for member in code_point_sorted(set(obj) - members):
        yield names, f"{where} has the member {_q(member)}, which its form does not have"


def _objects(block: object) -> list[tuple[str, dict[str, Any]]]:
    """A block's members that are objects, in code-point order of their names; a block or
    member of another form is check 10's."""
    if not isinstance(block, dict):
        return []
    members = cast("dict[str, Any]", block)
    return [(n, members[n]) for n in code_point_sorted(members) if isinstance(members[n], dict)]


def _items(value: object) -> list[dict[str, Any]]:
    """A list's members that are objects."""
    if not isinstance(value, list | tuple):
        return []
    return [m for m in cast("list[Any]", value) if isinstance(m, dict)]


def _member_failures(doc: dict[str, Any], fmt: int) -> Iterator[tuple[tuple[str, ...], str]]:
    """Every object's missing or extra member, in one traversal order: the document's members,
    then each block's in code-point order of its name, each declaration's in code-point order
    of its own. A step is the ``plan`` item's, each
    argument one origin as a declaration writes it. A table's inputs and rows are objects of
    §table-notation's members; their values' forms are check 10's."""
    members = _DOCUMENT | {"context"} if fmt >= CONTEXT_FORMAT else _DOCUMENT
    if fmt >= TARGETS_FORMAT:
        members |= {"targets"}
    yield from _missing_or_extra(doc, members, "the document", ())
    for tid, target in _objects(doc.get("targets")):
        yield from _missing_or_extra(target, _TARGET, f"target {_q(tid)}", (tid,))
    context = doc.get("context")
    if isinstance(context, dict):
        block = cast("dict[str, Any]", context)
        where = "the context block"
        yield from _missing_or_extra(block, _CONTEXT, where, ())
        for entry in _items(block.get("entries")):
            yield from _missing_or_extra(entry, frozenset({"ref", "text"}), f"{where}'s entry", ())
            if isinstance(entry.get("text"), dict):
                yield from _text_failures(entry["text"], f"{where}'s entry text", ())
    leaf_members = frozenset({"type", "value"})
    if fmt >= KEY_ORDER_FORMAT:
        leaf_members |= {"order"}
    for path, leaf in _objects(doc["parameters"]):
        yield from _missing_or_extra(leaf, leaf_members, f"parameter {_q(path)}", (path,))
        if "order" in leaf_members and "order" in leaf:
            yield from _order_failures(
                leaf.get("value"), leaf["order"], f"parameter {_q(path)}", (path,)
            )
    for name, plan in _objects(doc["plans"]):
        where = f"plan declaration {_q(name)}"
        yield from _missing_or_extra(plan, _PLAN, where, (name,))
        for member in ("reason", "reply"):
            if isinstance(plan.get(member), dict):
                yield from _text_failures(plan[member], f"{where}'s {member}", (name,))
        for step in _items(plan.get("steps")):
            yield from _missing_or_extra(step, _STEP, f"{where}'s step", (name,))
            for _, origin in _objects(step.get("arguments")):
                if frozenset(origin) not in _ORIGINS:
                    yield (name,), f"{where} has a step argument that is not one origin"
            for r in _items(step.get("requirements")):
                required = frozenset({"ref", "value"})
                yield from _missing_or_extra(r, required, f"{where}'s step requirement", (name,))
    yield from _function_failures(
        doc["policy"], "the policy", ("policy",), _POLICY, _POLICY_TABLE, fmt
    )
    for r in _items(doc["requirements"]):
        yield from _missing_or_extra(r, frozenset({"ref", "value"}), "a plan-level requirement", ())
    policy_doc = _object(doc["policy"])
    if policy_doc is not None and policy_doc.get("form") == "table":
        yield from _table_member_failures(policy_doc, "the policy table", ("policy",), _POLICY_ROW)
    targeted: frozenset[str] = frozenset({"target"} if fmt >= TARGETS_FORMAT else ())
    for rid, rule in _objects(doc["rules"]):
        if rule.get("form") == "table":
            yield from _table_member_failures(rule, f"rule {_q(rid)}", (rid,), _RULE_ROW)
        yield from _function_failures(
            rule, f"rule {_q(rid)}", (rid,), _RULE | targeted, _TABLE | targeted, fmt
        )
        if fmt < CANDIDATE_FORMAT and rule.get("candidates") is True:
            yield (rid,), f"rule {_q(rid)} is a candidate set, which format {fmt} does not have"
        if fmt < CANDIDATE_FIELD_FORMAT and type(rule.get("candidates")) is str:
            yield (
                (rid,),
                f"rule {_q(rid)} names a candidate field, which format {fmt} does not have",
            )
    if fmt < KNOWN_FORMAT:
        tables = [(f"rule {_q(rid)}", (rid,), r) for rid, r in _objects(doc["rules"])]
        if policy_doc is not None:
            tables.append(("the policy table", ("policy",), policy_doc))
        for where, names, table in tables:
            if any(
                "known" in cell
                for row in _items(table.get("rows"))
                for cell in _items(row.get("cells"))
            ):
                yield names, f"{where} holds a Known cell, which format {fmt} does not have"
    if fmt < CANDIDATE_FIELD_FORMAT:
        components = [(f"rule {_q(rid)}", (rid,), r) for rid, r in _objects(doc["rules"])]
        policy = _object(doc["policy"])
        if policy is not None:
            components.append(("the policy", ("policy",), policy))
        for where, names, component in components:
            if any(is_candidate_read(read) for read in _component_reads(component)):
                yield names, f"{where} reads candidate, which format {fmt} does not have"
    for sid, source in _objects(doc["sources"]):
        yield from _source_failures(sid, source, fmt)
    for name, tool in _objects(doc["tools"]):
        yield from _missing_or_extra(tool, frozenset(), f"tool {_q(name)}", (name,))


def _component_reads(component: Mapping[str, Any]) -> Iterator[str]:
    """Each read a document's function rule, table or policy spells: its ``reads``, or each
    input's ``read``; anything of another form skipped, as check 11's member test reports."""
    reads: object = component.get("reads")
    if isinstance(reads, list):
        yield from (r for r in cast("list[object]", reads) if isinstance(r, str))
    inputs: object = component.get("inputs")
    if isinstance(inputs, list):
        for i in cast("list[object]", inputs):
            if isinstance(i, dict) and isinstance(cast("dict[str, Any]", i).get("read"), str):
                yield cast("dict[str, str]", i)["read"]


_INPUT = frozenset({"read", "helper", "params"})
_RULE_ROW = frozenset({"cells", "output"})
_POLICY_ROW = frozenset({"cells", "plan"})


def _table_member_failures(
    table: dict[str, Any], where: str, names: tuple[str, ...], row: frozenset[str]
) -> Iterator[tuple[tuple[str, ...], str]]:
    """Each table input's and row's missing or extra member, by position from 1."""
    for n, i in enumerate(_items(table.get("inputs")), 1):
        yield from _missing_or_extra(i, _INPUT, f"{where}'s input {n}", names)
    for k, r in enumerate(_items(table.get("rows")), 1):
        yield from _missing_or_extra(r, row, f"{where}'s row {k}", names)


def _function_failures(
    obj: object,
    where: str,
    names: tuple[str, ...],
    function: frozenset[str],
    table: frozenset[str],
    fmt: int,
) -> Iterator[tuple[tuple[str, ...], str]]:
    """A rule's or the policy's members by its ``form``, and a function's boundaries; the
    extraction roles are a form from format ``EXTRACTION_FORMAT`` on."""
    if not isinstance(obj, dict):
        return
    members = cast("dict[str, Any]", obj)
    form = members.get("form")
    if "form" not in members:
        yield names, f"{where} has no member {_q('form')}"
    elif form not in ("function", "table"):
        yield names, f"{where}'s form is neither function nor table"
    else:
        yield from _missing_or_extra(
            members, function if form == "function" else table, where, names
        )
    if form == "function":
        for key, entry in _objects(members.get("boundaries")):
            parsed = _parsed(key)
            if parsed is None or parsed.level != "q":
                continue
            shown = f"{where}'s boundaries for {_q(key)}"
            if fmt >= EXTRACTION_FORMAT and "null" in entry:  # an extraction read's roles
                yield from _missing_or_extra(entry, frozenset(X_ROLES), shown, (*names, key))
                continue
            roles = frozenset({"bar", "thresholds", "margin", "mass"})
            yield from _missing_or_extra(entry, roles, shown, (*names, key))
            for m in _items(entry.get("mass")):
                bounds = frozenset({"labels", "bounds"})
                yield from _missing_or_extra(m, bounds, f"{shown}'s mass member", (*names, key))


_QUESTION = frozenset({"kind", "wording", "labels"})
_LABEL = frozenset({"name", "description"})
_EXTRACTION = _QUESTION | {"value_type", "is_list", "text", "reference_date"}


def _source_failures(
    sid: str, source: dict[str, Any], fmt: int
) -> Iterator[tuple[tuple[str, ...], str]]:
    where, names = f"source {_q(sid)}", (sid,)
    members = _SOURCE | {"payload_order"} if fmt >= KEY_ORDER_FORMAT else _SOURCE
    yield from _missing_or_extra(source, members, where, names)
    if "payload_order" in members and "payload_order" in source:
        yield from _order_failures(source.get("payload"), source["payload_order"], where, names)
    for ref, provided in _objects(source.get("provides")):
        at, shown = (sid, ref), f"{where}'s {_q(ref)}"
        if not ref.startswith("q:"):
            yield from _missing_or_extra(provided, frozenset({"type", "candidates"}), shown, at)
            if fmt < CANDIDATE_FORMAT and provided.get("candidates") is True:
                yield at, f"{shown} is a candidate set, which format {fmt} does not have"
            if fmt < CANDIDATE_FIELD_FORMAT and type(provided.get("candidates")) is str:
                yield at, f"{shown} names a candidate field, which format {fmt} does not have"
            continue
        extraction = provided.get("kind") == AnswerKind.extraction
        if extraction and fmt < EXTRACTION_FORMAT:
            yield at, f"{shown} is an extraction question, which format {fmt} does not have"
        else:
            members = _EXTRACTION if extraction else _QUESTION
            if fmt >= TARGETS_FORMAT:
                members |= {"target", "latent_targets"}
            yield from _missing_or_extra(provided, members, shown, at)
        if fmt >= TARGETS_FORMAT:
            for latent in _items(provided.get("latent_targets")):
                yield from _missing_or_extra(latent, _LATENT, f"{shown}'s latent target", at)
        if isinstance(provided.get("wording"), dict):
            yield from _text_failures(
                provided["wording"], f"{shown}'s wording", at, held=fmt >= KEY_ORDER_FORMAT
            )
        declared: object = provided.get("labels")
        listed = declared
        template = _object(provided.get("wording"))
        structured = template is not None and type(template.get("template")) in (dict, list)
        if structured and fmt < STRUCTURED_WORDING_FORMAT:
            yield at, f"{shown} has a structured wording, which format {fmt} does not have"
            continue
        noul = provided.get("kind") == AnswerKind.noul
        if noul and declared is not None and fmt < NOUL_DESCRIPTION_FORMAT:
            yield at, f"{shown} is a noul question with labels, which format {fmt} does not have"
            continue
        if isinstance(declared, dict) and fmt < CANDIDATE_FORMAT:
            yield at, f"{shown} is a candidate-valued label set, which format {fmt} does not have"
        if isinstance(declared, dict):
            label_set = cast("dict[str, Any]", declared)
            labels = frozenset({"fixed", "candidates", "at"})
            yield from _missing_or_extra(label_set, labels, f"{shown}'s labels", at)
            listed = label_set.get("fixed")
        described = _items(cast(object, listed))
        if described and fmt < DESCRIPTION_FORMAT:
            yield at, f"{shown} has a described label, which format {fmt} does not have"
        else:
            for label in described:
                yield from _missing_or_extra(label, _LABEL, f"{shown}'s described label", at)
    if isinstance(source.get("when"), dict):
        yield from _missing_or_extra(source["when"], _FUNCTION, f"{where}'s when", names)
    if isinstance(source.get("requirement"), dict):
        required = frozenset({"ref", "values"})
        yield from _missing_or_extra(
            source["requirement"], required, f"{where}'s requirement", names
        )


def _text_failures(
    text: dict[str, Any], where: str, names: tuple[str, ...], *, held: bool = False
) -> Iterator[tuple[tuple[str, ...], str]]:
    members = frozenset({"template", "slots"})
    structured = type(text.get("template")) in (dict, list)
    if held and structured and "order" in text:
        members |= {"order"}
        if text["order"] == {}:
            yield names, f"{where} writes an empty order, which it leaves out"
        else:
            yield from _order_failures(text.get("template"), text["order"], where, names)
    yield from _missing_or_extra(text, members, where, names)
    for slot in _items(text.get("slots")):
        yield from _missing_or_extra(slot, frozenset({"name", "origin"}), f"{where}'s slot", names)
        origin = _object(slot.get("origin"))
        if origin is not None and not (
            len(origin) == 1 and set(origin) <= {"ref", "param", "candidate"}
        ):
            yield names, f"{where} has a slot whose origin is not one of ref, param or candidate"


def _order_failures(
    value: object, order: object, where: str, names: tuple[str, ...]
) -> Iterator[tuple[tuple[str, ...], str]]:
    """Check 11: an order object fits the value it orders (§key-order)."""
    try:
        with_order(value, cast("dict[str, list[int]]", order))
    except ValueError as unfit:
        yield names, f"{where}'s order does not fit its value: {unfit}"


def _ordered_value(value: object, order: object) -> Any:
    """A document value in the order its order object gives; one that does not fit is left as
    it is, check 11 having refused its document."""
    if order is None:
        return value
    try:
        return with_order(value, cast("dict[str, list[int]]", order))
    except ValueError:
        return value


def _held_bytes(value: object, order: object) -> bytes:
    """A document value in the order its order object gives, as ordered bytes."""
    if order is not None:
        value = with_order(value, cast("dict[str, list[int]]", order))
    return ordered(value).encode("utf-8")


def _unresolved(*args: Any, **kwargs: Any) -> Any:
    """What a function component load could not resolve holds; such a design fails load."""


def _components(doc: dict[str, Any]) -> list[tuple[str, str, tuple[str, ...], dict[str, Any]]]:
    """Each function component: its ``functions`` key, where it is, its names and its
    object."""
    out: list[tuple[str, str, tuple[str, ...], dict[str, Any]]] = []
    for rid, rule in _objects(doc["rules"]):
        if rule["form"] == "function":
            out.append((rid, f"rule {_q(rid)}", (rid,), rule))
    for sid, source in _objects(doc["sources"]):
        when = _object(source["when"])
        if when is not None:
            out.append((f"{sid}.when", f"source {_q(sid)}'s when", (sid,), when))
    policy = _object(doc["policy"])
    if policy is not None and policy["form"] == "function":
        out.append(("policy", "the policy", ("policy",), policy))
    return out


def _nodes(paths: frozenset[str]) -> list[tuple[str, str]]:
    """Each parameter path that is a node of another, with that path, in the other's
    code-point order: read from a trie of the paths' segments, so each path is walked once."""
    trie: list[dict[str, int]] = [{}]
    ends: set[int] = set()
    for path in paths:
        at = 0
        for part in path.split("/"):
            child = trie[at].get(part)
            if child is None:
                child = len(trie)
                trie[at][part] = child
                trie.append({})
            at = child
        ends.add(at)
    out: list[tuple[str, str]] = []
    for path in code_point_sorted(paths):
        parts = path.split("/")
        at = 0
        for i, part in enumerate(parts[:-1], 1):
            at = trie[at][part]
            if at in ends:
                out.append(("/".join(parts[:i]), path))
    return out


def _resolve(
    doc: dict[str, Any], functions: Mapping[str, Any]
) -> tuple[dict[str, Any], frozenset[str], list[DesignFailure]]:
    """Each function component's callable, by its key: through ``functions`` first, else
    through ``importlib`` by its module and qualified name. A key naming two components
    resolves neither, and check 1 reports the clash; any other component unresolved is a check
    11 failure."""
    components = _components(doc)
    keys = [key for key, _, _, _ in components]
    resolved: dict[str, Any] = {}
    unresolved: set[str] = set()
    failures: list[DesignFailure] = []
    counts = Counter(keys)
    for key, where, names, obj in components:
        if counts[key] > 1:
            unresolved.add(key)
            continue
        if key in functions:
            resolved[key] = functions[key]
            continue
        module, qualname = obj["module"], obj["qualname"]
        if module == "__main__" or (
            isinstance(qualname, str) and ("<locals>" in qualname or "<lambda>" in qualname)
        ):
            message = (
                f"{where}'s function is in __main__ or has a <locals> or <lambda> qualname, so "
                "it must be in functions"
            )
        else:
            found = _import(module, qualname)
            if found is not None:
                resolved[key] = found
                continue
            message = (
                f"{where}'s function is not in functions and its module and qualname import none"
            )
        unresolved.add(key)
        failures.append(new_failure(11, names, message))
    for key in unresolved:
        resolved[key] = _unresolved
    return resolved, frozenset(unresolved), failures


def _import(module: object, qualname: object) -> Any:
    """The object a module and qualified name name, or ``None``. Importing runs the module's
    top-level code and the qualname walk may run a module's ``__getattr__``: an ``Exception``
    either raises leaves the component unresolved, its text not kept; any other
    ``BaseException``, such as ``SystemExit``, propagates out of ``load``."""
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    try:
        found: Any = importlib.import_module(module)
        for part in qualname.split("."):
            found = getattr(found, part)
    except Exception:  # the import's or the walk's; a BaseException propagates
        return None
    return found


def _text_of(value: object) -> Any:
    if not isinstance(value, dict):
        return value
    text = cast("dict[str, Any]", value)
    if type(text.get("template")) in (dict, list):  # a structured wording
        return Structured(_ordered_value(text["template"], text.get("order")), text["slots"])
    return Text(text["template"], text["slots"])


def _build(doc: dict[str, Any], resolved: Mapping[str, Any]) -> Design:
    """The ``Design`` a document declares, each function component holding its resolved
    callable. A member of another form passes through as it is, for check 10 to refuse."""

    def block(value: object, build: Callable[[str, dict[str, Any]], Any]) -> Any:
        members = _object(value)
        if members is None:
            return value
        return {k: v if (o := _object(v)) is None else build(k, o) for k, v in members.items()}

    def provided(ref: str, value: dict[str, Any]) -> Any:
        if not ref.startswith("q:"):
            return value
        return Question(
            value["kind"],
            _text_of(value["wording"]),
            value["labels"],
            value_type=value.get("value_type"),
            is_list=value.get("is_list", False),
            text=value.get("text"),
            reference_date=value.get("reference_date"),
            target=value.get("target"),
            latent_targets=value.get("latent_targets", ()),
        )

    def source(sid: str, s: dict[str, Any]) -> Source:
        w, r = _object(s["when"]), _object(s["requirement"])
        return Source(
            provides=block(s["provides"], provided),
            needs=s["needs"],
            when=s["when"] if w is None else When(resolved[f"{sid}.when"], w["version"]),
            requirement=s["requirement"] if r is None else SourceRequirement(r["ref"], r["values"]),
            oracle=s["oracle"],
            timeout=s["timeout"],
            payload=_ordered_value(s["payload"], s.get("payload_order")),
            cost=s["cost"],
            feedback=s["feedback"],
            grounded=s["grounded"],
        )

    def rule(rid: str, r: dict[str, Any]) -> Rule | Table:
        if r["form"] == "table":
            return Table(
                r["provides"],
                r["outputs"],
                r["hit"],
                _listed(r["inputs"], python_input),
                _rows(r["rows"], policy=False, hit=r["hit"]),
                order=r["order"],
                candidates=r["candidates"],
                grounded=r["grounded"],
                target=r.get("target"),
            )
        return Rule(
            r["provides"],
            r["outputs"],
            resolved[rid],
            version=r["version"],
            reads=r["reads"],
            params=r["params"],
            boundaries=r["boundaries"],
            candidates=r["candidates"],
            grounded=r["grounded"],
            target=r.get("target"),
        )

    policy: Any = doc["policy"]
    if isinstance(policy, dict):
        p = cast("dict[str, Any]", policy)
        if p["form"] == "table":
            policy = PolicyTable(_listed(p["inputs"], python_input), _rows(p["rows"], policy=True))
        else:
            policy = Policy(
                resolved["policy"],
                version=p["version"],
                reads=p["reads"],
                params=p["params"],
                boundaries=p["boundaries"],
            )
    requirements: Any = doc["requirements"]
    if isinstance(requirements, list | tuple):
        requirements = [
            m if (o := _object(m)) is None else Requirement(o["ref"], o["value"])
            for m in cast("list[Any]", requirements)
        ]
    tools: Any = doc["tools"]
    return Design(
        state=doc["state"],
        policy=policy,
        sources=block(doc["sources"], source),
        rules=block(doc["rules"], rule),
        requirements=requirements,
        default_deny=doc["default_deny"],
        tools=list(cast("dict[str, Any]", tools)) if isinstance(tools, dict) else tools,
        parameters=block(
            doc["parameters"],
            lambda _, v: Parameter(v["type"], _ordered_value(v["value"], v.get("order"))),
        ),
        types=doc["types"],
        plans=block(
            doc["plans"],
            lambda _, v: PlanDeclaration(
                v["kind"],
                _listed(v["steps"], _step_of),
                _text_of(v["reason"]),
                _text_of(v["reply"]),
                v["grounded"],
            ),
        ),
        groups=_each(doc["groups"], Group),
        classes=doc["classes"],
        name=doc["name"],
        context=_context_of(doc.get("context")),
        targets=block(doc.get("targets", {}), lambda _, v: Target(v["statement"], v["values"])),
    )


def _rows(value: object, *, policy: bool, hit: object = None) -> Any:
    """A document's rows as a design written in Python holds them, one not of the tagged form
    as the ``NotationError`` check 10 reports."""
    return _listed(
        value,
        lambda r: (
            NotationError("a row is not an object of the tagged form")
            if (o := _object(r)) is None
            else python_row(o, policy=policy, hit=hit)
        ),
    )


def _listed(value: object, build: Callable[[Any], Any]) -> Any:
    if not isinstance(value, list):
        return value
    return [build(v) for v in cast("list[Any]", value)]


def _step_of(value: object) -> Any:
    """A document step as a ``Step``; one of another form kept whole, for check 10."""
    step = _object(value)
    if step is None or set(step) != {"tool", "arguments", "requirements"}:
        return value
    arguments, requirements = _object(step["arguments"]), step["requirements"]
    if arguments is None or not isinstance(requirements, list):
        return value
    origins: dict[str, Origin | Binding] = {}
    for name, o in arguments.items():
        origin = _object(o)
        if origin is None:
            return value
        try:
            if set(origin) == {"step", "path"}:
                origins[name] = Binding(origin["step"], origin["path"])
            elif set(origin) == {"ref"}:
                origins[name] = Origin(ref=origin["ref"])
            elif set(origin) == {"literal"}:
                origins[name] = Origin(literal=origin["literal"])
            elif set(origin) == {"candidate"}:
                origins[name] = Origin(candidate=origin["candidate"])
            else:
                return value
        except TypeError:  # a member of another form: check 10's
            return value
    built: list[Requirement] = []
    for r in cast("list[Any]", requirements):
        member = _object(r)
        if member is None or set(member) != {"ref", "value"}:
            return value
        built.append(Requirement(member["ref"], member["value"]))
    try:
        return Step(step["tool"], origins, built)
    except TypeError:  # a member of another form: check 10's
        return value


def _context_of(value: object) -> Any:
    """The ``Context`` a document's block declares; a member of another form passes through
    for check 10 to refuse."""
    block = _object(value)
    if block is None:
        return value
    entries: Any = block["entries"]
    if isinstance(entries, list):
        entries = [
            m if (o := _object(m)) is None else ContextEntry(o["ref"], _text_of(o["text"]))
            for m in cast("list[Any]", entries)
        ]
    return Context(entries, plan=block["plan"], steps=block["steps"], record=block["record"])


def _each(value: object, build: Callable[[Any], Any]) -> Any:
    if not isinstance(value, dict):
        return value
    return {k: build(v) for k, v in cast("dict[str, Any]", value).items()}


def _object(value: object) -> dict[str, Any] | None:
    """A document's object, or ``None`` for any other value."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _same(a: object, b: object) -> bool:
    return encode(a, MAX_DOCUMENT_DEPTH) == encode(b, MAX_DOCUMENT_DEPTH)
