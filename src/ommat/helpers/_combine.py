# pyright: reportPrivateUsage=false
"""Combining designs: ``combine`` unions parts into one design under a top policy, and
``standalone`` makes one part loadable alone.

A ``Part`` is a design and its ports, the ``d:`` refs it reads and no declaration of it
provides. ``combine`` adds a rule for each port, the caller's or a one-input pass-through
table it builds, and the caller's top policy, and renames nothing. It refuses what load cannot
see, the parts' structure and their ports, listing every failure in one
``InvalidCombination``; load checks the combined design as it checks any other. What combine
adds is a table where it builds one, so a combined design's document reloads by import.

``combine`` is not a helper a rule calls, so ``HELPERS`` does not count it.
"""

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from ommat._design import Design, Input, Rule, Table
from ommat._refs import _RefError, parse_ref
from ommat._types import TypeFailure, check_definitions, conforms, parse_type
from ommat._values import quote, type_name

_q = quote


def _empty() -> Mapping[str, str]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Part:
    """A design to combine and its ports: each ``d:`` ref, with no field path, that it reads
    and does not provide, mapped to the ref's type. ``ports`` is held as a read-only copy."""

    design: Design
    ports: Mapping[str, str] = field(default_factory=_empty)

    def __post_init__(self) -> None:
        if isinstance(self.ports, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            object.__setattr__(self, "ports", MappingProxyType(dict(self.ports)))


@dataclass(frozen=True, slots=True)
class CombineFailure:
    """One failure ``combine`` or ``standalone`` found: the names, ids or refs it concerns, in
    code-point order, and a one-line message."""

    names: tuple[str, ...]
    message: str


class InvalidCombination(ValueError):
    """``combine`` or ``standalone`` refused its parts. ``failures`` holds at least one
    ``CombineFailure``, alike ones once, ordered by names, then message; the string is their
    messages, one per line, in that order."""

    failures: tuple[CombineFailure, ...]

    def __init__(self, failures: Sequence[CombineFailure]) -> None:
        ordered = tuple(sorted(dict.fromkeys(failures), key=lambda f: (list(f.names), f.message)))
        if not ordered:
            raise ValueError("an InvalidCombination carries at least one failure")
        self.failures = ordered
        super().__init__("\n".join(f.message for f in ordered))


class _Failures(list[CombineFailure]):
    def add(self, names: Sequence[object], message: str) -> None:
        """Add a failure naming each of ``names`` that is a string."""
        named = (n for n in names if isinstance(n, str))
        self.append(CombineFailure(tuple(sorted(named)), message))


def _shown(value: object) -> str:
    """A string quoted, as a load failure quotes it, or another value's type in brackets."""
    return _q(value) if isinstance(value, str) else f"<{type_name(value)}>"


def _label(n: int, part: Part) -> str:
    name = part.design.name
    return f"part {n}" if name is None else f"part {n} ({_q(name)})"


def _listed(labels: Sequence[str]) -> str:
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def _whole_d_ref(ref: object) -> str | None:
    """The name of a ``d:`` ref with no field path or bracket, else ``None``."""
    if not isinstance(ref, str):
        return None
    try:
        parsed = parse_ref(ref)
    except _RefError:
        return None
    if parsed.level != "d" or parsed.fields or parsed.candidate is not None:
        return None
    return parsed.name


def _is_d_ref(ref: str) -> bool:
    """Whether ``ref`` is a ``d:`` ref with no bracket, a field path allowed."""
    try:
        parsed = parse_ref(ref)
    except _RefError:
        return False
    return parsed.level == "d" and parsed.candidate is None


def _provided(design: Design) -> set[str]:
    refs = {ref for source in design.sources.values() for ref in source.provides}
    return refs | {rule.provides for rule in design.rules.values()}


def _check_part(part: object, what: str) -> Part:
    if type(part) is not Part:
        raise TypeError(f"{what} is a {type_name(part)}, not a Part")
    if type(part.design) is not Design:
        raise TypeError(f"{what}'s design is a {type_name(part.design)}, not a Design")
    if not isinstance(part.ports, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{what}'s ports are a {type_name(part.ports)}, not a mapping")
    return part


def _port_failures(
    labelled: Sequence[tuple[str, Part]], failures: _Failures
) -> tuple[dict[str, str], set[object]]:
    """Each port of the parts that passes, by ref, with its type, and each port refused: one
    that is not a whole ``d:`` ref, whose type does not parse, that two parts declare with two
    types, or that a part provides, each with its failure. A refused port is neither bound nor
    unbound, so it fails once."""
    providers: dict[str, list[str]] = {}
    for label, part in labelled:
        for ref in _provided(part.design):
            providers.setdefault(ref, []).append(label)
    ports: dict[str, str] = {}
    refused: set[object] = set()
    declared: dict[str, list[str]] = {}
    for label, part in labelled:
        for ref, written in part.ports.items():
            if _whole_d_ref(ref) is None:
                refused.add(ref)
                failures.add(
                    [ref], f"{label}'s port {_shown(ref)} is not a d: ref with no field path"
                )
                continue
            if isinstance(parse_type(written, nullable=True), TypeFailure):
                refused.add(ref)
                failures.add(
                    [ref], f"{label}'s port {_q(ref)} has type {_shown(written)}, which is no type"
                )
                continue
            declared.setdefault(ref, []).append(label)
            if ref in ports and ports[ref] != written:
                refused.add(ref)
                failures.add(
                    [ref],
                    f"{_listed(declared[ref])} declare the port {_q(ref)} with types "
                    f"{_q(ports[ref])} and {_q(written)}",
                )
                continue
            ports.setdefault(ref, written)
    for ref in ports:
        if ref in providers:
            refused.add(ref)
            failures.add(
                [ref], f"the port {_q(ref)} is provided by {_listed(providers[ref])}, not a port"
            )
    return {ref: ports[ref] for ref in ports if ref not in refused}, refused


def _repeats(
    labelled: Sequence[tuple[str, Part]],
    what: str,
    keys: Callable[[Design], Iterable[str]],
    failures: _Failures,
) -> None:
    """A failure for each key two or more parts declare."""
    held: dict[str, list[str]] = {}
    for label, part in labelled:
        for key in dict.fromkeys(keys(part.design)):
            held.setdefault(key, []).append(label)
    for key, labels in held.items():
        if len(labels) > 1:
            failures.add([key], f"{what} {_q(key)} is declared by {_listed(labels)}")


def _conforms(outputs: str, port: str, types: Mapping[str, str]) -> bool:
    """Whether ``outputs`` conforms to the port's type; ``True`` when either type or a
    definition does not parse or resolve, which load reports."""
    defs: dict[str, Any] = {}
    for name, body in types.items():
        parsed = parse_type(body, nullable=True)
        if isinstance(parsed, TypeFailure):
            return True
        defs[name] = parsed
    if check_definitions(defs):
        return True
    s, t = parse_type(outputs, nullable=True), parse_type(port, nullable=True)
    if isinstance(s, TypeFailure) or isinstance(t, TypeFailure):
        return True
    try:
        return conforms(s, t, defs)
    except ValueError:  # a name no definition holds
        return True


def _added_ids(
    added: Mapping[str, str], taken: Mapping[str, Sequence[str]], failures: _Failures
) -> None:
    """A failure for each id combine gives a rule it adds, keyed by the ref the rule provides,
    that a part declares."""
    for ref, rid in added.items():
        if rid in taken:
            failures.add(
                [rid],
                f"the rule for {_q(ref)} takes the id {_q(rid)}, which "
                f"{_listed(taken[rid])} declares",
            )


def _ids(labelled: Sequence[tuple[str, Part]]) -> dict[str, list[str]]:
    taken: dict[str, list[str]] = {}
    for label, part in labelled:
        for rid in dict.fromkeys([*part.design.sources, *part.design.rules]):
            taken.setdefault(rid, []).append(label)
    return taken


def _merged(designs: Sequence[Design], member: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for design in designs:
        out.update(cast("Mapping[str, Any]", getattr(design, member)))
    return out


def _pass_through(port: str, type_: str, ref: str) -> Table:
    """The one-input table at ``port`` whose one row passes ``ref``'s value through; a marker
    at ``ref`` matches no row, so the port holds it with ``required`` naming ``ref``."""
    return Table(port, type_, "F", inputs=[ref], rows=[["-", Input(ref)]])


def _constant(port: str, type_: str, value: object) -> Table:
    """The table with no input at ``port`` whose one row outputs ``value``."""
    return Table(port, type_, "F", inputs=[], rows=[[value]])


def combine(
    *parts: Part,
    bind: Mapping[str, str | Rule | Table],
    policy: Rule | Table,
    name: str | None = None,
) -> Design:
    """One design holding every part's declarations, a rule for each port from ``bind`` and
    ``policy``, the top policy, as its policy; or ``InvalidCombination`` listing every failure.

    ``bind`` maps each port to its provider: a ``Rule`` or ``Table`` providing the port, or a
    ``d:`` ref, a field path allowed, whose value the port takes through a pass-through table.
    Each rule combine adds is keyed by its ref's name, the port's or the policy's."""
    if not parts:
        raise TypeError("combine takes at least one part")
    labelled = [(_label(n, _check_part(p, f"part {n}")), p) for n, p in enumerate(parts, 1)]
    if not isinstance(bind, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"bind is a {type_name(bind)}, not a mapping")
    if type(policy) not in (Rule, Table):
        raise TypeError(f"policy is a {type_name(policy)}, not a Rule or a Table")
    for ref, provider in bind.items():
        if type(provider) not in (str, Rule, Table):
            raise TypeError(
                f"bind's provider for {ref!r} is a {type_name(provider)}, not a d: ref, a Rule "
                "or a Table"
            )
    failures = _Failures()
    designs = [p.design for _, p in labelled]

    states = {d.state for d in designs}
    if len(states) > 1:
        shown = "; ".join(f"{label} declares {_q(p.design.state)}" for label, p in labelled)
        failures.add([], f"the parts' states differ: {shown}")
    _repeats(labelled, "the id", lambda d: [*d.sources, *d.rules], failures)
    _repeats(labelled, "the parameter path", lambda d: d.parameters, failures)
    _repeats(labelled, "the type name", lambda d: d.types, failures)
    _repeats(labelled, "the group id", lambda d: d.groups, failures)
    _repeats(labelled, "the target id", lambda d: d.targets, failures)
    classes: dict[str, tuple[str, Any]] = {}
    for label, part in labelled:
        for key, tree in part.design.classes.items():
            if key in classes and classes[key][1] != tree:
                failures.add(
                    [key], f"{classes[key][0]} and {label} declare different classes at {_q(key)}"
                )
            classes.setdefault(key, (label, tree))

    ports, refused = _port_failures(labelled, failures)
    types = _merged(designs, "types")
    added: dict[str, Rule | Table] = {}
    for ref, provider in bind.items():
        if ref in refused:
            continue
        if ref not in ports:
            failures.add([ref], f"bind names {_shown(ref)}, which is no port")
            continue
        if isinstance(provider, str):
            if not _is_d_ref(provider):
                failures.add(
                    [ref], f"bind's provider for {_q(ref)} is {_q(provider)}, which is no d: ref"
                )
                continue
            added[ref] = _pass_through(ref, ports[ref], provider)
            continue
        if provider.provides != ref:
            failures.add(
                [ref],
                f"bind's provider for {_q(ref)} is a rule providing {_shown(provider.provides)}",
            )
            continue
        if not _conforms(provider.outputs, ports[ref], types):
            failures.add(
                [ref],
                f"bind's provider for {_q(ref)} outputs {_shown(provider.outputs)}, which does not "
                f"conform to the port's {_q(ports[ref])}",
            )
            continue
        added[ref] = provider
    for ref in ports:
        if ref not in bind:
            failures.add([ref], f"the port {_q(ref)} is not bound")

    top = _whole_d_ref(policy.provides)
    if top is None:
        failures.add(
            [policy.provides],
            f"the policy provides {_shown(policy.provides)}, which is not a d: ref with no "
            "field path",
        )
    ids = {ref: cast(str, _whole_d_ref(ref)) for ref in added}
    if top is not None:
        if top in ids.values():
            failures.add([top], f"the policy and a port's rule both take the id {_q(top)}")
        ids[policy.provides] = top
    _added_ids(ids, _ids(labelled), failures)
    if failures:
        raise InvalidCombination(failures)

    rules = _merged(designs, "rules")
    for ref, rule in added.items():
        rules[ids[ref]] = rule
    rules[cast(str, top)] = policy
    return Design(
        state=designs[0].state,
        policy=policy.provides,
        sources=_merged(designs, "sources"),
        rules=rules,
        parameters=_merged(designs, "parameters"),
        types=types,
        groups=_merged(designs, "groups"),
        classes={key: tree for key, (_, tree) in classes.items()},
        name=name,
        targets=_merged(designs, "targets"),
    )


def standalone(part: Part, ports: Mapping[str, object]) -> Design:
    """The part's design with each port provided by a table with no input that outputs the
    value ``ports`` gives it, a value of the port's type or ``Unsure``; or
    ``InvalidCombination`` listing every failure. Each table is keyed by its port's name."""
    _check_part(part, "part")
    if not isinstance(ports, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"ports are a {type_name(ports)}, not a mapping")
    failures = _Failures()
    labelled = [(_label(1, part), part)]
    declared, refused = _port_failures(labelled, failures)
    for ref in ports:
        if ref not in part.ports and ref not in refused:
            failures.add([ref], f"ports names {_shown(ref)}, which is no port")
    for ref in declared:
        if ref not in ports:
            failures.add([ref], f"the port {_q(ref)} is given no value")
    added = {ref: _constant(ref, declared[ref], ports[ref]) for ref in declared if ref in ports}
    ids = {ref: cast(str, _whole_d_ref(ref)) for ref in added}
    _added_ids(ids, _ids(labelled), failures)
    if failures:
        raise InvalidCombination(failures)
    rules = dict(part.design.rules)
    for ref, table in added.items():
        rules[ids[ref]] = table
    return dataclasses.replace(part.design, rules=rules)
