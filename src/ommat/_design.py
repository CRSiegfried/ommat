"""The design types, the loaded design and ``InvalidDesign``.

A design type's fields are named as its object's members in the design document; a function
component takes its callable as ``function``. A design type validates
nothing when built: load's check 10 tests every member's form, so every failure is reported
in one pass. A constructor copies the containers that are its
own fields one level deep: a mapping field becomes a read-only mapping, keyed by each key's
string data when its keys are strings that stay distinct, a sequence field a tuple, and a
string its string data. So adding to or removing from a dict or list the caller passed as a
field changes no design. Values inside, a payload or a leaf's value among them, are kept as
given, so a caller who changes those before load changes the design load sees.
"""

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

from ommat._answer import AnswerKind, ValueType
from ommat._canonical import decode, encode
from ommat._read_view import ReadView
from ommat._types import ListOf, MappingOf, Scalar, Type, b64url_decode, b64url_encode
from ommat._values import MAX_DOCUMENT_DEPTH, string_data

# ---------------------------------------------------------------------------------------------
# Snapshots, which never raise


def _empty() -> Mapping[str, Any]:
    return MappingProxyType({})


def _mapping(value: object) -> Any:
    if not issubclass(type(value), Mapping):
        return value
    items = list(cast("Mapping[object, Any]", value).items())
    if all(issubclass(type(k), str) for k, _ in items):
        data = {string_data(cast(str, k)): v for k, v in items}
        if len(data) == len(items):
            return MappingProxyType(data)
    return MappingProxyType(dict(items))


def _sequence(value: object) -> Any:
    if issubclass(type(value), str | bytes | bytearray) or not issubclass(type(value), Sequence):
        return value
    return tuple(cast("Sequence[Any]", value))


def _string(value: object) -> Any:
    # By its type alone, so no ``__class__`` of the caller's is read.
    return string_data(cast(str, value)) if issubclass(type(value), str) else value


def _provision(value: object) -> Any:
    """A ``provides`` value: a type's string data, or a mapping whose ``type`` is."""
    if not issubclass(type(value), Mapping):
        return _string(value)
    proxy = cast("Mapping[object, Any]", _mapping(value))
    data = dict(proxy)
    if not all(type(k) is str for k in data) or "type" not in data:
        return proxy
    data["type"] = _string(data["type"])
    if "candidates" in data:
        data["candidates"] = _string(data["candidates"])
    return MappingProxyType(data)


def _strings(value: object) -> Any:
    members = _sequence(value)
    if type(members) is tuple:
        return tuple(_string(v) for v in cast("tuple[object, ...]", members))
    return members


def _set(obj: object, **fields: object) -> None:
    for name, value in fields.items():
        object.__setattr__(obj, name, value)


# ---------------------------------------------------------------------------------------------
# Targets


class TargetRole(enum.StrEnum):
    """How a question's answer bears on one of its latent targets."""

    reflective = "reflective"
    """The answer alone is evidence for or against the target."""
    formative = "formative"
    """The answer supplies a part of the target."""
    unknown = "unknown"
    """Not yet judged."""


@dataclass(frozen=True, slots=True)
class Target:
    """A property of the state a design's questions and rules are about, for a reader:
    ``statement`` says what the design is trying to determine and what counts,
    ``values`` describes its value space, or ``None``."""

    statement: str
    values: str | None = None

    def __post_init__(self) -> None:
        _set(self, statement=_string(self.statement), values=_string(self.values))


def _latent(value: object) -> Any:
    """A latent target as a read-only mapping: its ``target`` as string data, its ``role`` a
    ``TargetRole`` member when it names one."""
    members = _mapping(value)
    if type(members) is not MappingProxyType:
        return cast(object, members)
    snapshot = dict(cast("Mapping[str, Any]", members))
    if "target" in snapshot:
        snapshot["target"] = _string(snapshot["target"])
    role: object = snapshot.get("role")
    if isinstance(role, str) and string_data(role) in TargetRole.__members__:
        snapshot["role"] = TargetRole(string_data(role))
    return MappingProxyType(snapshot)


# ---------------------------------------------------------------------------------------------
# Text, questions and requirements


def _slot(slot: object) -> Any:
    members = _mapping(slot)
    if type(members) is MappingProxyType and "origin" in members:
        snapshot = dict(cast("Mapping[str, Any]", members))
        origin = _mapping(snapshot["origin"])
        if type(origin) is MappingProxyType:
            origin = MappingProxyType(
                {k: _string(v) for k, v in cast("Mapping[str, Any]", origin).items()}
            )
        snapshot["origin"] = origin
        snapshot["name"] = _string(snapshot.get("name"))
        return MappingProxyType(snapshot)
    return cast(object, members)


@dataclass(frozen=True, slots=True)
class Text:
    """A template and its slot list: each slot ``{"name": n, "origin": o}``, ``o`` one of
    ``{"ref": r}``, ``{"param": path}`` or ``{"candidate": r}``. A ref may carry a field path
    through records, ``t:state.sender.name``, but no list index."""

    template: str
    slots: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        slots = _sequence(self.slots)
        if type(slots) is tuple:
            slots = tuple(_slot(s) for s in cast("tuple[object, ...]", slots))
        _set(self, template=_string(self.template), slots=slots)


@dataclass(frozen=True, slots=True)
class Structured:
    """A question's wording as a JSON object or array: each string value a template, rendered
    as a ``Text`` template is from one slot list; keys and other values literal, never
    rendered. An object's members are written, rendered and sent in the order the template
    holds them, a model reading that order. Slots as ``Text``'s."""

    template: Any
    slots: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        slots = _sequence(self.slots)
        if type(slots) is tuple:
            slots = tuple(_slot(s) for s in cast("tuple[object, ...]", slots))
        _set(self, slots=slots)


@dataclass(frozen=True, slots=True)
class Question:
    """A ``q:`` ref's question: its kind, wording and labels. ``labels``
    is ``None``, a sequence of labels, or a mapping of ``fixed``, ``candidates`` and, optionally,
    ``at`` for a candidate-valued label set, ``at`` the number of fixed labels before the
    candidates, by default all of them. A label is its name, a string, or a mapping of exactly
    ``name`` and ``description``, both strings, held as a read-only copy. An extraction question
    also declares ``value_type``, a ``ValueType`` member or its string, held as the member;
    ``is_list``, a boolean; ``text``, a board-ref origin ``{"ref": r}`` naming the string its
    spans index; and ``reference_date``, such an origin or ``None``. A probability kind's
    question leaves those four at ``None``, ``False``, ``None`` and ``None``. ``target`` is the
    target id it asks about, or ``None``; ``latent_targets`` lists the targets it bears on
    without asking about them, each a mapping of ``target`` and ``role``, held as a read-only
    copy whose role is a ``TargetRole`` member."""

    kind: AnswerKind | str
    """An ``AnswerKind`` member or its string, held as the member."""
    wording: "Text | Structured"
    labels: Sequence[str | Mapping[str, str]] | Mapping[str, Any] | None = None
    value_type: ValueType | str | None = None
    is_list: bool = False
    text: Mapping[str, str] | None = None
    reference_date: Mapping[str, str] | None = None
    target: str | None = None
    latent_targets: Sequence[Mapping[str, Any]] = ()

    def __post_init__(self) -> None:
        kind: object = self.kind
        if isinstance(kind, str) and string_data(kind) in AnswerKind.__members__:  # pyright: ignore[reportUnnecessaryIsInstance]
            kind = AnswerKind(string_data(kind))
        value_type: object = self.value_type
        if isinstance(value_type, str) and string_data(value_type) in ValueType.__members__:
            value_type = ValueType(string_data(value_type))
        labels: object = self.labels
        if issubclass(type(labels), Mapping):
            snapshot = dict(cast("Mapping[str, Any]", _mapping(labels)))
            if "fixed" in snapshot:
                snapshot["fixed"] = _labels(snapshot["fixed"])
            if "candidates" in snapshot:
                snapshot["candidates"] = _string(snapshot["candidates"])
            labels = MappingProxyType(snapshot)
        else:
            labels = _labels(labels)
        _set(
            self,
            kind=kind,
            labels=labels,
            value_type=value_type,
            text=_origin(self.text),
            reference_date=_origin(self.reference_date),
            target=_string(self.target),
            latent_targets=_latents(self.latent_targets),
        )


def _latents(value: object) -> Any:
    """A question's latent targets as a tuple of read-only mappings, as ``_latent`` holds each."""
    members = _sequence(value)
    if type(members) is tuple:
        return tuple(_latent(v) for v in cast("tuple[object, ...]", members))
    return members


def _labels(value: object) -> Any:
    """A label list as a tuple: each string its string data, each mapping, a described label's
    ``name`` and ``description``, a read-only copy of it with each string its string data."""
    members = _sequence(value)
    if type(members) is tuple:
        return tuple(_label(v) for v in cast("tuple[object, ...]", members))
    return members


def _label(value: object) -> Any:
    members = _mapping(value)
    if type(members) is MappingProxyType:
        return MappingProxyType(
            {k: _string(v) for k, v in cast("Mapping[Any, Any]", members).items()}
        )
    return _string(cast(object, members))


def label_parts(labels: Sequence[Any]) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    """A well-formed label list's names and descriptions, aligned, ``None`` where a label is
    a bare name; check 10 has passed it."""
    names: list[str] = []
    descriptions: list[str | None] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
            descriptions.append(None)
        else:
            described = cast("Mapping[str, str]", label)
            names.append(described["name"])
            descriptions.append(described["description"])
    return tuple(names), tuple(descriptions)


def _origin(value: object) -> Any:
    """A board-ref origin as a read-only mapping, each string its string data."""
    members = _mapping(value)
    if type(members) is MappingProxyType:
        return MappingProxyType(
            {k: _string(v) for k, v in cast("Mapping[Any, Any]", members).items()}
        )
    return cast(object, members)


@dataclass(frozen=True, slots=True)
class When:
    """A source's function ``when``: the callable and its ``version``."""

    function: Callable[[ReadView], object]
    version: str | None = None

    def __post_init__(self) -> None:
        _set(self, version=_string(self.version))


@dataclass(frozen=True, slots=True)
class SourceRequirement:
    """A source's data-form ``when``: its ``d:`` ref holds one of ``values``."""

    ref: str
    values: Sequence[Any]

    def __post_init__(self) -> None:
        _set(self, ref=_string(self.ref), values=_sequence(self.values))


@dataclass(frozen=True, slots=True)
class Parameter:
    """A parameter leaf: its type and its value."""

    type: str
    value: Any


@dataclass(frozen=True, slots=True)
class Group:
    """A group: the refs it names, to be considered together, and ``reason``, why, each one of
    ``convergent``, ``discriminant``, ``exclusive``, ``exhaustive`` and ``consistent``. It gates
    nothing."""

    refs: Sequence[str]
    reason: Sequence[str]

    def __post_init__(self) -> None:
        _set(self, refs=_strings(self.refs), reason=_strings(self.reason))


# ---------------------------------------------------------------------------------------------
# Components


@dataclass(frozen=True, slots=True)
class Source:
    """A source. ``provides`` maps each ``q:`` ref to its ``Question``
    and each ``t:`` ref to its type, or to a mapping of ``type`` and ``candidates``, ``false``,
    ``true`` for a ``list<string>`` candidate set or the candidate field's name for a list of
    records.
    ``grounded`` names a ``list<string>`` candidate set; each question whose wording holds a
    ``{"candidate": <that set>}`` slot is then asked once per candidate."""

    provides: Mapping[str, Any]
    needs: Sequence[str] = ()
    when: When | None = None
    requirement: SourceRequirement | None = None
    oracle: str | None = None
    timeout: float | None = None
    payload: Any = None
    cost: float = 0
    feedback: bool = False
    grounded: str | None = None

    def __post_init__(self) -> None:
        provides = _mapping(self.provides)
        if type(provides) is MappingProxyType:
            provides = MappingProxyType(
                {k: _provision(v) for k, v in cast("Mapping[Any, Any]", provides).items()}
            )
        _set(
            self,
            provides=provides,
            needs=_strings(self.needs),
            oracle=_string(self.oracle),
            grounded=_string(self.grounded),
        )


@dataclass(frozen=True, slots=True)
class Rule:
    """A function rule: it provides its own ``d:`` ref, its value of type ``outputs``.
    ``target`` is the target id it estimates, or ``None``."""

    provides: str
    outputs: str
    function: Callable[[ReadView], object]
    version: str | None = None
    reads: Sequence[str] = ()
    params: Sequence[str] = ()
    boundaries: Mapping[str, Any] | None = None
    candidates: bool | str = False
    grounded: str | None = None
    target: str | None = None
    form: Literal["function"] = field(default="function", init=False)

    def __post_init__(self) -> None:
        _set(
            self,
            provides=_string(self.provides),
            outputs=_string(self.outputs),
            version=_string(self.version),
            reads=_strings(self.reads),
            params=_strings(self.params),
            boundaries=_mapping(self.boundaries),
            candidates=_string(self.candidates),
            grounded=_string(self.grounded),
            target=_string(self.target),
        )


@dataclass(frozen=True, slots=True, init=False)
class Helper:
    """A table input that calls a helper of the library over one read: ``name`` names the
    helper, ``read`` its read, and ``params`` binds each other argument of its signature to a
    leaf path."""

    name: str
    read: str
    params: Mapping[str, str]

    def __init__(self, name: str, read: str, /, **params: str) -> None:
        _set(self, name=_string(name), read=_string(read), params=_mapping(params))


@dataclass(frozen=True, slots=True)
class Input:
    """A table row's output that passes through the input reading ``ref``."""

    ref: str

    def __post_init__(self) -> None:
        _set(self, ref=_string(self.ref))


@dataclass(frozen=True, slots=True)
class Param:
    """A table row's output that passes through the leaf at ``path``."""

    path: str

    def __post_init__(self) -> None:
        _set(self, path=_string(self.path))


@dataclass(frozen=True, slots=True)
class Table:
    """A rule table: it provides its own ``d:`` ref, its value of type ``outputs``, from its
    ``inputs`` and ``rows`` under the hit policy ``hit``; ``order`` ranks the output values
    under ``P``. Each input is a ref or a ``Helper``; each row a list of its cells, then its
    output entry, the cells alone under ``C#``."""

    provides: str
    outputs: str
    hit: str
    inputs: Sequence[Any]
    rows: Sequence[Any]
    order: Sequence[Any] | None = None
    candidates: bool | str = False
    grounded: str | None = None
    target: str | None = None
    form: Literal["table"] = field(default="table", init=False)

    def __post_init__(self) -> None:
        _set(
            self,
            provides=_string(self.provides),
            outputs=_string(self.outputs),
            hit=_string(self.hit),
            inputs=_sequence(self.inputs),
            rows=_sequence(self.rows),
            order=_sequence(self.order),
            candidates=_string(self.candidates),
            grounded=_string(self.grounded),
            target=_string(self.target),
        )


# ---------------------------------------------------------------------------------------------
# The design


@dataclass(frozen=True, slots=True)
class Design:
    """A design: the state declaration, the policy and the components.

    ``policy`` is the ``d:`` ref of the rule whose estimate is the decision. ``sources``,
    ``rules``, ``types`` and ``groups`` map each declared name to its declaration;
    ``parameters`` maps each leaf's full path to a ``Parameter``; ``targets`` maps each target id
    to a ``Target``; each defaults to empty. ``state`` and ``policy`` are positional; every
    other field is keyword-only, so a field added or removed never rebinds a positional argument.
    """

    state: str
    policy: str
    _: KW_ONLY
    sources: Mapping[str, Source] = field(default_factory=_empty)
    rules: Mapping[str, Rule | Table] = field(default_factory=_empty)
    parameters: Mapping[str, Parameter] = field(default_factory=_empty)
    types: Mapping[str, str] = field(default_factory=_empty)
    groups: Mapping[str, Group] = field(default_factory=_empty)
    name: str | None = None
    targets: Mapping[str, Target] = field(default_factory=_empty)

    def __post_init__(self) -> None:
        types = _mapping(self.types)
        if type(types) is MappingProxyType:
            types = MappingProxyType(
                {k: _string(v) for k, v in cast("Mapping[Any, Any]", types).items()}
            )
        _set(
            self,
            state=_string(self.state),
            policy=_string(self.policy),
            sources=_mapping(self.sources),
            rules=_mapping(self.rules),
            parameters=_mapping(self.parameters),
            groups=_mapping(self.groups),
            types=types,
            name=_string(self.name),
            targets=_mapping(self.targets),
        )


# ---------------------------------------------------------------------------------------------
# InvalidDesign


_BUILT_BY_LOAD = object()


@dataclass(frozen=True, slots=True)
class DesignFailure:
    """One failure load found: its check number, the names or refs it concerns in
    code-point order, and a one-line message. Built only by ``load``."""

    check: int
    names: tuple[str, ...]
    message: str
    token: InitVar[object] = None

    def __post_init__(self, token: object) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("a DesignFailure is built only by load")


def new_failure(check: int, names: Sequence[str], message: str) -> DesignFailure:
    return DesignFailure(check, tuple(sorted(names)), message, _BUILT_BY_LOAD)


def _order(f: DesignFailure) -> tuple[int, list[str], str]:
    return (f.check, list(f.names), f.message)


class InvalidDesign(ValueError):
    """Load refused a design. ``failures`` holds at least one ``DesignFailure``,
    alike ones once, ordered by check number, then names, then message; the string is their
    messages, one per line, in that order. Raised only by ``load``."""

    failures: tuple[DesignFailure, ...]

    def __init__(self, failures: Sequence[DesignFailure], token: object = None) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("an InvalidDesign is raised only by load")
        ordered = tuple(sorted(dict.fromkeys(failures), key=_order))
        if not ordered:
            raise ValueError("an InvalidDesign carries at least one failure")
        self.failures = ordered
        super().__init__("\n".join(f.message for f in ordered))

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy rebuild the error through load's guard, then restore the rest of
        its state, such as its notes; the failures travel only through the rebuild."""
        return (_rebuild_invalid_design, (self.failures,), _state(self.__dict__))

    def __setstate__(self, state: object) -> None:
        """Restore what ``__reduce__`` kept; a ``failures`` entry is ignored, so the checked
        tuple the rebuild set stays."""
        if type(state) is not dict:
            raise TypeError("an InvalidDesign's state is a dict")
        for name, value in _state(cast("dict[str, object]", state)).items():
            setattr(self, name, value)


def _state(given: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in given.items() if name != "failures"}


def _rebuild_invalid_design(failures: Sequence[DesignFailure]) -> InvalidDesign:
    """An ``InvalidDesign`` built again from the failures a pickled or copied one held, each
    a ``DesignFailure`` and no subclass, so nothing else passes for what load found."""
    failures = tuple(failures)
    if not all(type(f) is DesignFailure for f in failures):
        raise TypeError("an InvalidDesign holds only DesignFailure instances")
    return InvalidDesign(failures, _BUILT_BY_LOAD)


# ---------------------------------------------------------------------------------------------
# The loaded design


class UnvaluedParts:
    """The parts of a design document no leaf's value or order reaches, each computed from
    the document at its first read and kept: ``sources``, the decoded ``sources`` object, and
    ``shape``, the document with each leaf's value and order dropped, encoded. Every reader only
    reads them. ``with_parameters`` hands a design it builds the given design's, the two
    documents differing in leaves' values and orders alone."""

    __slots__ = ("_document", "_shape", "_sources")

    def __init__(self, document: bytes) -> None:
        self._document = document
        self._sources: dict[str, Any] | None = None
        self._shape: bytes | None = None

    @property
    def sources(self) -> dict[str, Any]:
        found = self._sources
        if found is None:  # two threads may both decode; each keeps an equal object
            found = self._sources = cast("dict[str, Any]", decode(self._document))["sources"]
        return found

    @property
    def shape(self) -> bytes:
        found = self._shape
        if found is None:
            found = self._shape = _shape_of(self._document)
        return found


def _shape_of(document: bytes) -> bytes:
    """``document`` with each leaf's value and order dropped, encoded."""
    doc = cast("dict[str, Any]", decode(document))
    leaves = cast("dict[str, dict[str, Any]]", doc["parameters"])
    doc["parameters"] = {p: leaf["type"] for p, leaf in leaves.items()}
    return encode(doc, MAX_DOCUMENT_DEPTH)


class LoadedDesign:
    """A design that passed every load check, with what load computed once. Built
    only by ``load``. ``design_hash`` and ``document`` are public; the
    rest is the engine's.

    The private parts: ``_design``, the snapshotted ``Design``; ``_callables``, each function
    component keyed by declared name, or a source's id and ``.when`` for its function ``when``;
    ``_types``, each ``t:`` and ``d:`` ref's declared type, ``t:state`` included, and
    ``_leaf_types``, each leaf's type, each with its definitions resolved; ``_needs``, each
    source's needs with its question slots' refs added; ``_groups``, each group's refs in
    code-point order, and ``_group_reasons`` its reasons; ``_graph``, each node's predecessors;
    ``_providers``, the ref-to-provider map; ``_grid``, the answer grid; ``_labels``, each
    ``q:`` ref's label names and descriptions, aligned, ``None`` where a label is undescribed,
    ``yes`` and ``no`` for ``noul`` and none for ``extraction``, a candidate-valued label set's
    fixed part alone; ``_label_sets``, each candidate-valued question's set ref, and
    ``_label_at`` how many of its fixed labels come before the candidates; ``_candidate_sets``,
    each declared candidate set's candidate field, ``None`` for a ``list<string>`` set;
    ``_grounded``, each per-candidate question's declared ref and its grounding set's ref;
    ``_requirement_bytes``, each source requirement's values as canonical bytes, by source id;
    ``_reads`` and ``_params``, each rule's declared reads and leaf paths, a table's from its
    inputs, cells and outputs; ``_table_bytes``, each table's document object by rule id, read
    by ``_table``; ``_read_leaves``, each leaf whose value a load check read beyond check 10's
    domain and check 8's conformance clauses, and ``_keyed_leaves``, each whose keys alone a
    check read, so a design changing one, or the other's keys, is loaded
    (``_load.with_parameters``); ``_document_form``, whether ``_design`` was built from a
    document, so that its leaf values are in document form, ``bytes`` as base64url, rather than
    the live form a caller's ``Design`` holds; and ``_unvalued``, the document's
    ``UnvaluedParts``, shared with each design ``with_parameters`` builds from this one. The
    shape trees, payloads and leaf values are decoded afresh at each read, so nothing a caller
    changes after load reaches a turn; ``_unvalued``'s decoded sources, decoded once, are
    shared by every design built from this one and are read-only by convention.
    """

    __slots__ = (
        "_callables",
        "_candidate_sets",
        "_design",
        "_document",
        "_document_form",
        "_graph",
        "_grid",
        "_grounded",
        "_group_reasons",
        "_groups",
        "_hash",
        "_keyed_leaves",
        "_label_at",
        "_label_sets",
        "_labels",
        "_leaf_bytes",
        "_leaf_types",
        "_needs",
        "_params",
        "_payload_bytes",
        "_providers",
        "_read_leaves",
        "_reads",
        "_requirement_bytes",
        "_shapes_bytes",
        "_table_bytes",
        "_types",
        "_unvalued",
    )

    _design: Design
    _callables: Mapping[str, Callable[..., Any]]
    _types: Mapping[str, Type]
    _leaf_types: Mapping[str, Type]
    _needs: Mapping[str, frozenset[str]]
    _groups: Mapping[str, tuple[str, ...]]
    _group_reasons: Mapping[str, tuple[str, ...]]
    _graph: Mapping[str, frozenset[str]]
    _providers: Mapping[str, str]
    _grid: Mapping[str, Any]
    _labels: Mapping[str, tuple[tuple[str, ...], tuple[str | None, ...]]]
    _label_sets: Mapping[str, str]
    _label_at: Mapping[str, int]
    _candidate_sets: Mapping[str, str | None]
    _grounded: Mapping[str, str]
    _shapes_bytes: bytes
    _payload_bytes: Mapping[str, bytes]
    _leaf_bytes: Mapping[str, bytes]
    _requirement_bytes: Mapping[str, frozenset[bytes]]
    _document: bytes
    _hash: str
    _reads: Mapping[str, tuple[str, ...]]
    _params: Mapping[str, tuple[str, ...]]
    _table_bytes: Mapping[str, bytes]
    _read_leaves: frozenset[str]
    _keyed_leaves: frozenset[str]
    _document_form: bool
    _unvalued: UnvaluedParts

    def __init__(self, token: object, **parts: object) -> None:
        if token is not _BUILT_BY_LOAD:
            raise TypeError("a LoadedDesign is built only by load")
        for name, value in parts.items():
            object.__setattr__(self, "_" + name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a LoadedDesign is read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a LoadedDesign is read-only")

    @property
    def design_hash(self) -> str:
        """The design hash over ``document``."""
        return self._hash

    @property
    def document(self) -> bytes:
        """The design's canonical bytes."""
        return self._document

    @property
    def _shapes(self) -> dict[str, Any]:
        """Each ref's shape tree, keyed by ``t:state`` and every provided ref."""
        return cast("dict[str, Any]", decode(self._shapes_bytes))

    def _payload(self, source: str) -> Any:
        """A source's payload as decoded from the document."""
        return decode(self._payload_bytes[source])

    def _table(self, key: str) -> dict[str, Any]:
        """A table's document object, by rule id, decoded afresh."""
        return cast("dict[str, Any]", decode(self._table_bytes[key]))

    def _leaf_value(self, path: str) -> Any:
        """A leaf's value as decoded from the document; a ``bytes`` value as ``bytes``."""
        return live_value(decode(self._leaf_bytes[path]), self._leaf_types[path])


# ---------------------------------------------------------------------------------------------
# A leaf's value in its two forms


def document_value(value: object, t: Type) -> object:
    """A conforming leaf value as the document writes it: ``bytes`` as unpadded base64url.
    ``t`` is a leaf type with its definitions resolved: a scalar, or a list or mapping of
    one."""
    if isinstance(t, Scalar) and t.name == "bytes":
        return b64url_encode(cast(bytes, value))
    if isinstance(t, ListOf):
        return [document_value(v, t.item) for v in cast("Sequence[object]", value)]
    if isinstance(t, MappingOf):
        return {k: document_value(v, t.value) for k, v in cast("dict[str, object]", value).items()}
    return value


def live_value(value: object, t: Type) -> object:
    """A leaf value decoded from the document, with each ``bytes`` value as Python ``bytes``."""
    if isinstance(t, Scalar) and t.name == "bytes":
        return b64url_decode(cast(str, value))
    if isinstance(t, ListOf):
        return [live_value(v, t.item) for v in cast("list[object]", value)]
    if isinstance(t, MappingOf):
        return {k: live_value(v, t.value) for k, v in cast("dict[str, object]", value).items()}
    return value
