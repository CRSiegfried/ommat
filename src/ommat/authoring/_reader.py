"""The reader: a parsed authoring document (a plain mapping, from TOML, JSON or YAML) to the
`Design`, or the `Slice` of one, that an author would write in Python.

A document is trusted as source code: the reader imports each rule's `fn`, which runs that
module's top-level code. Read only documents you would import as modules.

The reader takes a parsed mapping, never a file, and needs no parser: a TOML document is read
with `tomllib` by the caller. `import ommat` does not import this package, and the reader
takes nothing beyond the standard library. It adds conventions that expand away before `load`:

- a question is one key naming its kind, holding its wording: {noul = "Is it spam?"};
- a question's `slot = "name"` (or a list of names) fills `{name}` in its wording per candidate
  of its source's `grounded` rule; `slot = {name = "ref"}` names another rule;
- `labels` may name an entry of the top-level [labels] table;
- a rule's `provides` defaults to "d:<rule id>";
- a function is "module:qualname", or just "qualname" under a top-level `module`;
- a rule's `type` is the type its function returns, Rule's `outputs`;
- a source's `only_when = {ref = [values]}` is its SourceRequirement: the source is asked only
  when the rule `ref` holds one of the values;
- a top-level `[types]` entry names a type string, or `{ required = true }` for one the caller
  supplies; `$name` inside any type string, a rule's `type`, a source's `fetch` or a
  parameter's `type`, is replaced by it;
- a top-level `version` is every rule's version unless the rule gives its own;
- a top-level `[targets]` entry is `{statement, values}`, `values` optional;
- a question's `target` and a rule's `target` name the one target each asks about or estimates;
- a question's `latent = {target = role}` (a role, or a list of roles) is its `latent_targets`,
  and an explicit `latent_targets` list of `{target, role}` passes through beside it;
- a rule with `rows` is a decision table whose rows name only the inputs they test, and a rule
  with `like` copies another table's rows.

Every table is checked against the keys it may hold, and each value against the shape it takes
(a mapping, a string, a list of strings) for the keys that take one, so a slip is an
`AuthoringError` naming its path, never a default. It also refuses a missing key, a `hit` other
than `F`, an unknown row input, a bad `like`, and a value with two owners. `load` refuses what
the design gets wrong.
"""

from __future__ import annotations

import fnmatch
import importlib
import inspect
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from ommat import (
    Design,
    Helper,
    Input,
    Parameter,
    Question,
    Rule,
    Source,
    SourceRequirement,
    Table,
    Target,
    Text,
)
from ommat._table import HELPERS  # the engine's helper registry; only the private module holds it

KINDS = ("noul", "choice", "score", "extraction")

SLICE_KEYS = {"module", "version", "types", "labels", "parameters", "sources", "rules"}
DOC_KEYS = SLICE_KEYS | {"name", "state", "policy", "targets"}
SOURCE_KEYS = {"oracle", "needs", "only_when", "ask", "fetch", "timeout", "cost", "grounded"}
RULE_KEYS = {
    "fn",
    "type",
    "provides",
    "version",
    "reads",
    "params",
    "candidates",
    "grounded",
    "target",
}
QUESTION_KEYS = {
    *KINDS,
    "labels",
    "text",
    "slot",
    "latent",
    "latent_targets",
    "value_type",
    "is_list",
    "reference_date",
    "target",
}
PARAMETER_KEYS = {"type", "value"}
TABLE_KEYS = {"type", "hit", "target", "provides", "inputs", "rows", "like"}
LIKE_KEYS = {"like", "provides", "type", "target", "inputs"}
HITS = ("F",)
TABLE_ONLY = {"rows", "like", "inputs", "hit"}
GLOB = re.compile(r"[*?\[]")
TARGET_KEYS = {"statement", "values"}


class AuthoringError(ValueError):
    """A slip in an authoring document, with the path to it."""


@dataclass(frozen=True)
class Slice:
    """The sources, rules and parameters that a larger design merges into itself."""

    sources: dict[str, Source]
    rules: dict[str, Rule | Table]
    parameters: dict[str, Parameter]


def _is_map(value: object) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(value, Mapping)


def _is_seq(value: object) -> TypeGuard[list[Any] | tuple[Any, ...]]:
    return isinstance(value, list | tuple)


def _keys(path: str, table: Mapping[str, Any], allowed: Collection[str]) -> None:
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise AuthoringError(
            f"{path}: unknown key {unknown[0]!r}; it may hold {', '.join(sorted(allowed))}"
        )


def _mapping(path: str, value: Any) -> Mapping[str, Any]:
    if not _is_map(value):
        raise AuthoringError(f"{path}: a table, found {type(value).__name__}")
    return value


def _string(path: str, value: Any) -> str:
    if not isinstance(value, str):
        raise AuthoringError(f"{path}: a string, found {type(value).__name__}")
    return value


def _strings(path: str, value: Any) -> list[str]:
    """A list of strings; a bare string is refused, not split into characters."""
    if not _is_seq(value) or not all(isinstance(v, str) for v in value):
        raise AuthoringError(f'{path}: a list of strings, such as ["a", "b"]')
    return list(value)


def _list(path: str, value: Any) -> list[Any]:
    if not _is_seq(value):
        raise AuthoringError(f'{path}: a list, such as ["a", "b"]')
    return list(value)


def _fn(path: str, spec: str, default_module: str | None) -> Any:
    _string(path, spec)
    if default_module is not None:
        _string("module", default_module)
    module, colon, qualname = spec.partition(":")
    if not colon:
        if default_module is None:
            raise AuthoringError(
                f'{path}: {spec!r} has no module; write "module:qualname" '
                "or set a top-level `module`"
            )
        module, qualname = default_module, spec
    try:
        obj: Any = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError, ValueError, TypeError) as error:
        raise AuthoringError(f"{path}: cannot import {spec!r}: {error}") from error
    if not callable(obj):
        raise AuthoringError(f"{path}: {spec!r} is not a function")
    if inspect.iscoroutinefunction(obj):
        raise AuthoringError(f"{path}: {spec!r} is an async function; a rule is a plain function")
    return obj


def _type(path: str, spec: str, types: Mapping[str, str], _seen: tuple[str, ...] = ()) -> str:
    """`spec` with each `$name` replaced by the [types] entry of that name."""
    _string(path, spec)

    def alias(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in _seen:
            raise AuthoringError(
                f"{path}: type alias ${name} refers to itself ({' -> '.join((*_seen, name))})"
            )
        if name not in types:
            raise AuthoringError(
                f"{path}: unknown type alias ${name}; "
                f"[types] holds {', '.join(sorted(types)) or 'nothing'}"
            )
        return _type(path, types[name], types, (*_seen, name))

    return re.sub(r"\$(\w+)", alias, spec)


def _resolve_types(
    declared: Mapping[str, Any], override: Mapping[str, str] | None
) -> dict[str, str]:
    """The [types] table with its required entries filled from the caller's `types`.

    An entry is a type string or `{ required = true }`, which the caller must supply; a
    written entry may not be supplied, so a type has one owner. This runs before any alias is
    expanded, so a required entry that only another alias uses is still named."""
    override = dict(override or {})
    for name in override:
        if name not in declared:
            raise AuthoringError(
                f"types.{name}: the document declares no such type; "
                f"[types] holds {', '.join(sorted(declared)) or 'nothing'}"
            )
    table: dict[str, str] = {}
    for name, spec in declared.items():
        if isinstance(spec, str):
            if name in override:
                raise AuthoringError(
                    f"types.{name}: the document already gives a type; "
                    "write `{ required = true }` there to supply it from the caller"
                )
            table[name] = spec
        elif _is_map(spec) and dict(spec) == {"required": True}:
            if name not in override:
                raise AuthoringError(
                    f"types.{name}: the document leaves its type to the caller, "
                    "who did not pass it in `types`"
                )
            table[name] = override[name]
        else:
            raise AuthoringError(
                f"types.{name}: a type string or {{ required = true }}, "
                f"found {dict(spec) if _is_map(spec) else spec!r}"
            )
    return table


def _question(
    path: str, spec: Mapping[str, Any], labels: Mapping[str, Any], grounded: str | None
) -> Question:
    _keys(path, _mapping(path, spec), QUESTION_KEYS)
    kinds = [k for k in KINDS if k in spec]
    if len(kinds) != 1:
        raise AuthoringError(
            f"{path}: a question names exactly one kind "
            f"({', '.join(KINDS)}), found {kinds or 'none'}"
        )
    (kind,) = kinds
    rest = {k: v for k, v in spec.items() if k != kind}
    slot = rest.pop("slot", None)  # per-candidate slots over a candidates rule
    if slot is not None and not isinstance(slot, str | list | Mapping):
        raise AuthoringError(f'{path}.slot: a name, a list of names, or {{ name = "ref" }}')
    if isinstance(slot, list) and not slot:
        raise AuthoringError(f"{path}.slot: an empty list names no slot")
    if isinstance(slot, str | list):
        if grounded is None:
            raise AuthoringError(
                f"{path}.slot: names a slot but its source has no `grounded` rule; "
                'write slot = {name = "ref"}'
            )
        names = [slot] if isinstance(slot, str) else _strings(f"{path}.slot", slot)
        slot = dict.fromkeys(names, grounded)
    pairs: dict[str, str] = dict(cast("Mapping[str, str]", slot)) if slot else {}
    wording = (
        Text(spec[kind], [{"name": n, "origin": {"candidate": ref}} for n, ref in pairs.items()])
        if slot
        else Text(spec[kind])
    )
    named = rest.pop("labels", None)
    if isinstance(named, str) and named not in labels:
        raise AuthoringError(f"{path}.labels: no [labels] entry {named!r}")
    resolved = labels[named] if isinstance(named, str) else named
    if "text" in rest:
        rest["text"] = {"ref": rest["text"]}
    latent = _list(f"{path}.latent_targets", rest.pop("latent_targets", []))
    for target, roles in _mapping(f"{path}.latent", rest.pop("latent", {})).items():
        if not isinstance(roles, str | list):
            raise AuthoringError(f"{path}.latent.{target}: a role, or a list of roles")
        for role in [roles] if isinstance(roles, str) else cast("list[Any]", roles):
            latent.append({"target": target, "role": role})
    if latent:
        rest["latent_targets"] = latent
    return Question(kind, wording, resolved, **rest)


def _source(
    path: str, spec: Mapping[str, Any], labels: Mapping[str, Any], types: Mapping[str, str]
) -> Source:
    _keys(path, _mapping(path, spec), SOURCE_KEYS)
    provides: dict[str, Any] = {}
    for ref, item in _mapping(f"{path}.ask", spec.get("ask", {})).items():
        provides[ref] = _question(f"{path}.ask.{ref}", item, labels, spec.get("grounded"))
    for ref, ty in _mapping(f"{path}.fetch", spec.get("fetch", {})).items():
        if ref in provides:
            raise AuthoringError(
                f"{path}.fetch.{ref}: the source already asks {ref!r}; a ref has one provider"
            )
        provides[ref] = _type(f"{path}.fetch.{ref}", ty, types)
    only_when = spec.get("only_when")
    requirement = None
    if only_when is not None:
        if not _is_map(only_when) or len(only_when) != 1:
            found = sorted(only_when) if _is_map(only_when) else type(only_when).__name__
            raise AuthoringError(
                f"{path}.only_when: one ref and its values, found {found or 'none'}"
            )
        ((ref, values),) = only_when.items()
        requirement = SourceRequirement(ref, values)
    return Source(
        provides,
        needs=spec.get("needs", []),
        oracle=spec.get("oracle"),
        requirement=requirement,
        timeout=spec.get("timeout"),
        cost=spec.get("cost", 0),
        grounded=spec.get("grounded"),
    )


def _rule(
    path: str,
    rid: str,
    spec: Mapping[str, Any],
    version: str | None,
    types: Mapping[str, str],
    module: str | None,
    declared: Collection[str],
) -> Rule:
    _keys(path, _mapping(path, spec), RULE_KEYS)
    for key in ("fn", "type"):
        if key not in spec:
            raise AuthoringError(f"{path}: missing key {key!r}")
    return Rule(
        spec.get("provides", f"d:{rid}"),
        _type(f"{path}.type", spec["type"], types),
        _fn(f"{path}.fn", spec["fn"], module),
        version=spec.get("version", version),
        reads=spec.get("reads", []),
        params=_params(
            f"{path}.params", _strings(f"{path}.params", spec.get("params", [])), declared
        ),
        candidates=spec.get("candidates", False),
        grounded=spec.get("grounded"),
        target=spec.get("target"),
    )


def _input(path: str, item: Any) -> Any:
    """A table input: a ref, or a helper call `{ mass = "q:x", labels = "path" }`. The one key
    that names an engine helper holds the ref it reads, wherever it sits; the other keys are the
    helper's parameters."""
    if isinstance(item, str):
        return item
    if _is_map(item) and item:
        named = [key for key in item if key in HELPERS]
        if len(named) == 1 and isinstance(item[named[0]], str):
            helper = named[0]
            return Helper(helper, item[helper], **{k: v for k, v in item.items() if k != helper})
        if named:
            raise AuthoringError(
                f"{path}: names {len(named)} helpers ({', '.join(named)}) or reads no "
                f"ref; one key names the helper and holds the ref it reads"
            )
        raise AuthoringError(f"{path}: no key names a helper; the helpers are {', '.join(HELPERS)}")
    raise AuthoringError(
        f'{path}: a ref, or a helper call such as {{ mass = "q:x", labels = "p" }}'
    )


def _like(path: str, spec: Mapping[str, Any], siblings: Mapping[str, Any]) -> dict[str, Any]:
    """A table that copies another's rows, with its own id, type, target or inputs."""
    _keys(path, spec, LIKE_KEYS)
    base_id = spec["like"]
    base = siblings.get(base_id) if isinstance(base_id, str) else None
    if _is_map(base) and "like" in base:
        raise AuthoringError(f"{path}.like: {base_id!r} is itself a copy; name the table it copies")
    if not _is_map(base) or "rows" not in base:
        raise AuthoringError(f"{path}.like: {base_id!r} is not a table of this document")
    if not _is_map(base.get("inputs")):
        raise AuthoringError(f"{path}.like: {base_id!r} has no `inputs` to copy")
    inputs = dict(base["inputs"])
    for name, item in _mapping(f"{path}.inputs", spec.get("inputs", {})).items():
        if name not in inputs:
            raise AuthoringError(
                f"{path}.inputs.{name}: {base_id!r} has no such input; it has {', '.join(inputs)}"
            )
        inputs[name] = item  # replaces in place, so the input keeps its position
    merged = {k: v for k, v in base.items() if k != "provides"}
    merged.update({k: spec[k] for k in ("provides", "type", "target") if k in spec})
    merged["inputs"] = inputs
    if not any(k in spec for k in ("provides", "type", "target", "inputs")) or (
        all(spec.get(k) == base.get(k) for k in ("type", "target") if k in spec)
        and inputs == base["inputs"]
        and "provides" not in spec
    ):
        raise AuthoringError(
            f"{path}.like: changes nothing from {base_id!r}; a copy gives its own "
            f"provides, or a type, target or input that differs"
        )
    return merged


def _table(path: str, rid: str, spec: Mapping[str, Any], types: Mapping[str, str]) -> Table:
    """A decision table: named inputs, then sparse rows. A row names only the inputs it tests
    (an omitted input is `-`, anything) and holds its result in `then`."""
    _keys(path, _mapping(path, spec), TABLE_KEYS)
    for key in ("type", "hit", "inputs", "rows"):
        if key not in spec:
            raise AuthoringError(f"{path}: missing key {key!r}")
    if spec["hit"] not in HITS:
        raise AuthoringError(
            f"{path}.hit: {spec['hit']!r}; only {', '.join(map(repr, HITS))} is spelt so far"
        )
    inputs = spec["inputs"]
    if not _is_map(inputs) or not inputs:
        raise AuthoringError(f"{path}.inputs: one entry per input, `inputs.name = ...`")
    names = list(inputs)
    if "then" in inputs:
        raise AuthoringError(
            f"{path}.inputs.then: `then` is a row's result, so no input may be named it"
        )
    if not isinstance(spec["rows"], list):
        raise AuthoringError(f"{path}.rows: a list of rows")
    rows: list[list[Any]] = []
    for n, row in enumerate(cast("list[Any]", spec["rows"])):
        where = f"{path}.rows[{n}]"
        if not _is_map(row) or "then" not in row:
            raise AuthoringError(f"{where}: a row is a table with `then`, the value it returns")
        for key in row:
            if key != "then" and key not in inputs:
                raise AuthoringError(
                    f"{where}: unknown input {key!r}; the inputs are {', '.join(names)}"
                )
        then = row["then"]
        if _is_map(then):
            if set(then) != {"ref"}:
                raise AuthoringError(f'{where}.then: a value, or {{ ref = "q:x" }} to return a ref')
            then = Input(_string(f"{where}.then.ref", then["ref"]))
        rows.append([*(row.get(name, "-") for name in names), then])
    return Table(
        spec.get("provides", f"d:{rid}"),
        _type(f"{path}.type", spec["type"], types),
        spec["hit"],
        [_input(f"{path}.inputs.{name}", inputs[name]) for name in names],
        rows,
        target=spec.get("target"),
    )


def _rule_or_table(
    path: str,
    rid: str,
    spec: Mapping[str, Any],
    siblings: Mapping[str, Any],
    version: str | None,
    types: Mapping[str, str],
    module: str | None,
    declared: Collection[str],
) -> Rule | Table:
    """A rule is a function (`fn`); a table has `rows`, or copies one with `like`."""
    _mapping(path, spec)
    is_table = bool(TABLE_ONLY & set(spec))
    if "fn" in spec and is_table:
        raise AuthoringError(f"{path}: a rule is a function (`fn`) or a table (`rows`), not both")
    if "like" in spec:
        return _table(path, rid, _like(path, spec, siblings), types)
    if is_table:
        return _table(path, rid, spec, types)
    return _rule(path, rid, spec, version, types, module, declared)


def _parameter(path: str, spec: Mapping[str, Any], types: Mapping[str, str]) -> Parameter:
    _keys(path, _mapping(path, spec), PARAMETER_KEYS)
    if "type" not in spec:
        raise AuthoringError(f"{path}: missing key 'type'")
    if "value" not in spec:  # the caller supplies it, in `values`
        return Parameter(_type(f"{path}.type", spec["type"], types), _REQUIRED)
    return Parameter(_type(f"{path}.type", spec["type"], types), spec["value"])


class _Required:
    """The value of a parameter the document leaves to its caller."""

    def __repr__(self) -> str:
        return "<required>"


_REQUIRED: Any = _Required()


def _params(path: str, names: list[str], declared: Collection[str]) -> list[str]:
    """A rule's `params`, each name literal or a pattern over this document's own parameters.

    A pattern is expanded here, per document, so it never sees another slice's parameters, and
    it takes in a parameter the document declares later. A name given twice, or matched by
    two entries, is refused."""
    out: list[str] = []
    for name in names:
        if GLOB.search(name):
            found = [p for p in declared if fnmatch.fnmatchcase(p, name)]
            if not found:
                raise AuthoringError(
                    f"{path}: pattern {name!r} matches no parameter of this document"
                )
        else:
            found = [name]
        for p in found:
            if p in out:
                raise AuthoringError(f"{path}: {p!r} is named twice (pattern {name!r})")
            out.append(p)
    return out


def _target(path: str, spec: Mapping[str, Any]) -> Target:
    _keys(path, _mapping(path, spec), TARGET_KEYS)
    if "statement" not in spec:
        raise AuthoringError(f"{path}: missing key 'statement'")
    return Target(spec["statement"], values=spec.get("values"))


def slice_from(
    doc: Mapping[str, Any],
    version: str | None = None,
    values: Mapping[str, Any] | None = None,
    types: Mapping[str, str] | None = None,
) -> Slice:
    """The `Slice` a document gives: the sources, rules and parameters a larger design merges.

    `version` is every rule's version, from the caller or from the document, not both. A
    parameter that gives no `value` takes it from `values`, and one that gives a value may not
    be in `values`, so a value has one owner. A rule's `params` may hold patterns such as
    "route/bar/*", matched against this document's own parameters.
    A [types] entry written `{ required = true }` takes its type string from `types`, for a
    type the caller computes, such as a label set of the tools it was given; an entry that
    writes a string may not be in `types`, so a type has one owner."""
    _keys("document", _mapping("document", doc), SLICE_KEYS)
    labels = _mapping("labels", doc.get("labels", {}))
    types = _resolve_types(_mapping("types", doc.get("types", {})), types)
    if version is not None and "version" in doc:
        raise AuthoringError("version: the document gives one and so does the caller; keep one")
    version = version if version is not None else doc.get("version")
    parameters = {
        p: _parameter(f"parameters.{p}", v, types)
        for p, v in _mapping("parameters", doc.get("parameters", {})).items()
    }
    for path, value in (values or {}).items():
        if path not in parameters:
            raise AuthoringError(f"values.{path}: the document declares no such parameter")
        if parameters[path].value is not _REQUIRED:
            raise AuthoringError(
                f"values.{path}: the document already gives a value; "
                "omit it there to supply it from the caller"
            )
        parameters[path] = Parameter(parameters[path].type, value)
    for path, parameter in parameters.items():
        if parameter.value is _REQUIRED:
            raise AuthoringError(
                f"parameters.{path}: the document leaves its value to the "
                "caller, who did not pass it in `values`"
            )
    return Slice(
        {
            sid: _source(f"sources.{sid}", s, labels, types)
            for sid, s in _mapping("sources", doc.get("sources", {})).items()
        },
        {
            rid: _rule_or_table(
                f"rules.{rid}",
                rid,
                r,
                _mapping("rules", doc.get("rules", {})),
                version,
                types,
                doc.get("module"),
                parameters,
            )
            for rid, r in _mapping("rules", doc.get("rules", {})).items()
        },
        parameters,
    )


def design_from(
    doc: Mapping[str, Any],
    version: str | None = None,
    values: Mapping[str, Any] | None = None,
    types: Mapping[str, str] | None = None,
) -> Design:
    """The whole design: the slice, with its state, its policy rule and its targets.
    `version`, `values` and `types` are as for `slice_from`; `$name` also works in `state`."""
    _keys("document", _mapping("document", doc), DOC_KEYS)
    for key in ("state", "policy"):
        if key not in doc:
            raise AuthoringError(f"document: missing key {key!r}")
    own = {k: v for k, v in doc.items() if k in SLICE_KEYS}
    inner = slice_from(own, version, values, types)
    merged = _resolve_types(doc.get("types", {}), types)
    return Design(
        name=doc.get("name"),
        state=_type("state", doc["state"], merged),
        policy=doc["policy"],
        sources=inner.sources,
        rules=inner.rules,
        parameters=inner.parameters,
        targets={
            tid: _target(f"targets.{tid}", t)
            for tid, t in _mapping("targets", doc.get("targets", {})).items()
        },
    )
