"""The record views: pure functions from a closed record to a value, for the caller, the
escalation target or a verifier.

``escalation(record)`` returns an undecided turn's five members: the marker's kind, its cause, the
root cause, whether that is a defect, and the record hash; ``None`` when the policy gave a value.
``defects(record)`` returns every entry whose cause is a defect by the same test, whatever the
policy gave.
``transcript(record)``, deprecated, returns every source call and the decision, unworded. Each
of the escalation, the defects and the transcript gives ``None`` for a record before format 8,
which holds no policy's entry, and the escalation and the transcript find the decision as
``decision_entry`` does. ``redact(record, root_key, policy)`` returns the redacted view's
document as canonical bytes: the record's ``log`` and ``unhashed`` with a keyed digest where
each ``content`` or ``identifier`` value was, ``redacted``, the record hash its key came from,
``applied``, the policy applied, each ref's class tree in full, and from format 3 on the
document's ``hash``, open. The caller's policy names each value's class, as ``redact`` says. In
an answer to a candidate-valued choice, each label and ``probabilities`` key that is not a fixed
name takes its set's class, found through the call item the answer's entry names.

Structure is open: refs but a grounded ref's candidate, kinds, cause codes, tool names, origin
tags, positions, usage, the hashes, the codec's name, an exception's ``type``, the call item's
members but ``prompt``, and the design and settings items. Text the engine records with no type
is digested whatever the policy names: a call's ``prompt``, a plan's text, exception text,
``reads`` values and ``unhashed``'s other text, a call's ``extras``, whatever its provider
reports beyond the profile, among it as one value, and any ``calls`` entry member the view does
not know, whole. The one exception to data being digested is the
source's profile, which is structure though a binding writes it freely: the call item's
``elicitation``, ``model_requested``, ``model_reported``, ``revision``, ``provider`` and
``runtime`` strings and ``unhashed``'s ``clients`` strings pass open.

A plan's literal is ``content`` whole, but in a stored record of format 7, whose ledger admits an
origin carrying ``set``, that literal takes its set's applied tree. A grounded ref's candidate is
digested under its set's class wherever the record holds one: an entry's ref, a cause's refs, a
plan's ref origin and a call's ``sums`` and ``truncated``; every other ref passes unchanged.
Each view raises ``ValueError`` before reading anything when
``record`` is not exactly a ``Record``. The continuation record is deferred, so it has no function
here.
"""

import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, cast

from ommat._canonical import encode, hash_value, keyed_digest, turn_key
from ommat._ledger import (
    Record,
    _with_held_order,  # pyright: ignore[reportPrivateUsage]
    decision_entry,
)
from ommat._markers import CauseCode
from ommat._refs import code_point_sorted, declared_ref, entry_ref, grounded_ref, parse_ref
from ommat._typed import is_a
from ommat._values import (
    HASH_FORMAT,
    MAX_DOCUMENT_DEPTH,
    POLICY_FORMAT,
    SHAPE_LEAF,
    SHAPES_FORMAT,
    clipped,
    described,
    is_scalar_string,
    string_data,
    type_name,
)

_OPEN = "open"
_CONTENT = "content"


def _closed(record: object) -> Record:
    """The record, refused by its exact type before anything of it is read."""
    if type(record) is not Record:
        raise ValueError(f"a record view takes a Record, not {type_name(record)}")
    return record


# ---------------------------------------------------------------------------------------------
# The escalation


def escalation(record: Record) -> dict[str, Any] | None:
    """An undecided turn's escalation: ``kind``, the policy's marker's kind; ``cause``, its
    cause; ``root_cause``, the cause reached by following each ``required`` cause to the entry
    its first ref names, until a cause whose code is not ``required`` or a ref with no marker
    entry; ``defect``, whether the root cause is ``raised``, ``invalid-return`` or
    ``undeclared-read`` naming a rule or the codec; and ``record_hash``. ``None`` when the
    policy gave a value, when the log holds no entry at the policy's ref, or for a record
    before format 8."""
    log = _closed(record).log
    entry = decision_entry(log)
    if entry is None or "value" in entry:
        return None
    marker = cast("dict[str, Any]", entry["marker"])
    cause = cast("dict[str, Any]", marker["cause"])
    root = _root_cause(log, cause)
    return {
        "kind": marker["kind"],
        "cause": cause,
        "root_cause": root,
        "defect": _defect(log, root),
        "record_hash": record.record_hash,
    }


_STATE = "t:state"
_CODEC = "codec"
_DEFECT_CODES = frozenset({CauseCode.raised, CauseCode.invalid_return, CauseCode.undeclared_read})
"""The codes a root cause is a defect under when it names a rule or the codec."""


def _defect(log: list[dict[str, Any]], root: dict[str, Any]) -> bool:
    """Whether ``root`` is a defect code whose first ref names a rule, by the ``settings``
    item's providers of ``d:`` refs, or the codec: ``codec``, or ``t:state``, whose value
    only the codec gives."""
    if root["code"] not in _DEFECT_CODES or not root["refs"]:
        return False
    providers = cast("dict[str, str]", log[1]["providers"])
    rules = {rid for ref, rid in providers.items() if ref.startswith("d:")}
    first = root["refs"][0]
    return first in rules or first in (_CODEC, _STATE)


def defects(record: Record) -> list[dict[str, Any]] | None:
    """Every entry item whose marker's cause is a defect by the test ``escalation`` applies to
    its root cause, in log order, whatever the policy gave; ``None`` for a record before
    format 8. A ``required`` cause is never one, so a rule's defect is listed at its own
    entry, not at the entries that required it."""
    log = _closed(record).log
    if not log or log[0].get("format", 0) < POLICY_FORMAT:
        return None
    return [item for _, item in _defect_items(log)]


def _defect_items(log: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    """Each entry item ``defects`` lists, with its log index, for a log of format 8 or later."""
    return [
        (index, item)
        for index, item in enumerate(log)
        if item["item"] == "entry"
        and "marker" in item
        and _defect(log, cast("dict[str, Any]", item["marker"]["cause"]))
    ]


def _root_cause(log: list[dict[str, Any]], cause: dict[str, Any]) -> dict[str, Any]:
    """The cause a ``required`` chain ends at: each step reads the marker at the entry of the
    cause's first ref, its field path dropped and its grounded bracket kept."""
    markers = {
        cast(str, item["ref"]): cast("dict[str, Any]", item["marker"])
        for item in log
        if item["item"] == "entry" and "marker" in item
    }
    seen: set[str] = set()
    while cause["code"] == CauseCode.required and cause["refs"]:
        ref = _entry_ref(cast(str, cause["refs"][0]))
        if ref is None or ref in seen or ref not in markers:
            break
        seen.add(ref)
        cause = cast("dict[str, Any]", markers[ref]["cause"])
    return cause


def _entry_ref(ref: str) -> str | None:
    """The entry a ref is read at: its field path dropped, a grounded bracket kept; ``None``
    for a cause ref that is no ref."""
    try:
        return entry_ref(ref)
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------
# The transcript


def transcript(record: Record) -> dict[str, Any] | None:
    """Every source call of a closed record and its decision, unworded: ``record_hash``,
    ``design``, ``rounds`` of calls, then ``decide``, decide's call and its result: the
    ``board``, each ``d:`` entry but the policy's, and the ``decision``, the policy's entry.
    ``None`` for a record before format 8, or one with no entry at the policy's ref. Each value
    and marker is tagged, and each id is ``om_``, the record hash's first 24 hex digits and
    ``_c<n>`` or ``_d``. Deprecated: it warns, and leaves ``ommat`` in a later minor release;
    render the record with ``ommat.conversation`` instead."""
    warnings.warn(
        "transcript is deprecated; pass the record to an ommat.conversation renderer",
        DeprecationWarning,
        stacklevel=2,
    )
    closed = _closed(record)
    log = closed.log
    decision = decision_entry(log)
    if decision is None:
        return None
    stem = "om_" + closed.record_hash.removeprefix("sha256:")[:24]
    settings = log[1]
    providers = cast("dict[str, str]", settings["providers"])
    written: dict[int, dict[str, Any]] = {}
    for item in log:
        if item["item"] == "entry" and item.get("call") is not None:
            written.setdefault(item["call"], {})[item["ref"]] = _tagged(item)
    rounds: list[list[dict[str, Any]]] = []
    held: dict[int, list[dict[str, Any]]] = {}
    n = 0
    for index, item in enumerate(log):
        if item["item"] != "call":
            continue
        answers = written.get(index, {})
        for source in cast("list[str]", item["sources"]):
            n += 1
            # A per-candidate entry is its declared ref's source's.
            asks = code_point_sorted(r for r in answers if providers.get(declared_ref(r)) == source)
            call = {
                "id": f"{stem}_c{n}",
                "source": source,
                "asks": asks,
                "outcome": deepcopy(item["outcome"]),
                "answers": {ref: answers[ref] for ref in asks},
            }
            if item["round"] not in held:
                held[item["round"]] = []
                rounds.append(held[item["round"]])
            held[item["round"]].append(call)
    board = {
        entry["ref"]: _tagged(entry)
        for entry in log
        if entry["item"] == "entry"
        and entry["ref"].startswith("d:")
        and entry["ref"] != settings["policy"]
    }
    decided = {"id": f"{stem}_d", "board": board, "decision": _tagged(decision)}
    return {
        "record_hash": closed.record_hash,
        "design": {"hash": log[0]["hash"]},
        "rounds": rounds,
        "decide": decided,
    }


def _tagged(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry's value as ``{"value"}`` or its marker as ``{"marker"}``, as recorded."""
    if "value" in entry:
        return {"value": entry["value"]}
    return {"marker": entry["marker"]}


# ---------------------------------------------------------------------------------------------
# The redacted view


def redact(record: Record, root_key: bytes, policy: Mapping[str, Any]) -> bytes:
    """The redacted view of a closed record under the caller's ``policy``, as canonical bytes,
    its digests under the key derived from the record's own hash and the applied policy's.

    ``policy`` maps a ref of the record, or a ref and a field path into its shape
    (``ref.field.path``), to a class name, ``open``, ``identifier`` or ``content``, or an
    object naming fields of the record there, each a class name or such an object; a class
    name covers every value beneath it, and a list or mapping layer has no node of its own, or
    takes a one-member list as ``applied`` writes it, so ``applied`` is a policy too. Keys
    apply from the shortest path to the longest, so a longer one overrides beneath a shorter.
    A candidate set's ref, or its candidate field, takes a class name. A mapping's keys are
    digested whatever the policy names.

    Every value the policy names nothing for is ``content``, at every format. In a record
    before format 9 the ``settings`` item's stored class trees are only a floor: at each place
    the more restrictive of the policy's class and the stored class applies, by ``open`` <
    ``identifier`` < ``content``, so ``{}`` digests every value, and a value is open only where
    the policy and the stored tree both name ``open``.

    Raises ``ValueError``, the first error found in this order: ``record`` not exactly a
    ``Record``, before anything is read; ``root_key`` not an instance of ``bytes`` of length
    32, before the record is read; then ``policy`` not a mapping of that form over the
    record's refs, its keys checked in its iteration order, each with whether a candidate
    set's key holds a class name, and then its values from the shortest key to the longest."""
    closed = _closed(record)
    key = _root_key(root_key)
    record_hash = closed.record_hash
    log = closed.log
    shapes: dict[str, Any] = {}
    stored: dict[str, Any] | None = None
    fields: dict[str, str | None] = {}
    grounded: dict[str, str] = {}
    for item in log:
        if item["item"] == "settings":
            if log[0]["format"] >= SHAPES_FORMAT:
                shapes = cast("dict[str, Any]", item["shapes"])
            else:
                # A tree no entry is held to may be malformed: it is content.
                classes = cast("dict[str, Any]", item["classes"])
                stored = {r: t if _sound(t) else _CONTENT for r, t in classes.items()}
                shapes = {ref: _shape_of(tree) for ref, tree in stored.items()}
            grounded = cast("dict[str, str]", item.get("grounded", {}))
            fields = cast("dict[str, str | None]", item.get("sets", {}))
    applied = _applied(policy, shapes, stored, fields)
    walk = _Walk(turn_key(key, record_hash, hash_value(applied, MAX_DOCUMENT_DEPTH)))
    walk.classes = applied
    walk.grounded = grounded
    walk.fields = fields
    walk.sets = _candidate_sets(log, applied, fields)
    view = {
        "log": [walk.item(item, applied) for item in log],
        "unhashed": walk.unhashed(closed.unhashed),
        "redacted": record_hash,
        "applied": applied,
    }
    if log[0]["format"] >= HASH_FORMAT:  # the document's hash, open
        view["hash"] = record_hash
    return encode(view, MAX_DOCUMENT_DEPTH)


def _root_key(root_key: object) -> bytes:
    """The root key's bytes, read by its type and ``bytes``' own slicing, so a subclass's code
    never runs; ``ValueError`` unless it is an instance of ``bytes`` of length 32."""
    key = bytes.__getitem__(cast(bytes, root_key), slice(None)) if is_a(root_key, bytes) else None
    if key is None or len(key) != 32:
        raise ValueError("the root key is not an instance of bytes of length 32")
    return key


# ---------------------------------------------------------------------------------------------
# The applied policy

_RANK = {_OPEN: 0, "identifier": 1, _CONTENT: 2}
"""The class names from least to most restrictive, by which ``_joined`` takes the floor."""


def _sound(tree: Any) -> bool:
    """Whether a stored class tree is well formed: a class name, an object of such trees, or a
    one-member list of one."""
    if type(tree) is str:
        return tree in _RANK
    if type(tree) is dict:
        return all(_sound(t) for t in cast("dict[str, Any]", tree).values())
    if type(tree) is list:
        members = cast("list[Any]", tree)
        return len(members) == 1 and _sound(members[0])
    return False


def _shape_of(tree: Any) -> Any:
    """A class tree's shape: its structure with ``SHAPE_LEAF`` at every leaf."""
    if type(tree) is dict:
        return {name: _shape_of(node) for name, node in cast("dict[str, Any]", tree).items()}
    if type(tree) is list:
        return [_shape_of(cast("list[Any]", tree)[0])]
    return SHAPE_LEAF


def _filled(shape: Any, name: str) -> Any:
    """A shape with the class ``name`` at every leaf."""
    if type(shape) is dict:
        return {field: _filled(node, name) for field, node in cast("dict[str, Any]", shape).items()}
    if type(shape) is list:
        return [_filled(cast("list[Any]", shape)[0], name)]
    return name


def _joined(a: Any, b: Any) -> Any:
    """Two class trees of one shape joined leaf by leaf, the more restrictive class at each."""
    if type(a) is dict:
        nodes = cast("dict[str, Any]", a)
        return {
            field: _joined(node, cast("dict[str, Any]", b)[field]) for field, node in nodes.items()
        }
    if type(a) is list:
        return [_joined(cast("list[Any]", a)[0], cast("list[Any]", b)[0])]
    return a if _RANK[a] >= _RANK[b] else b


def _applied(
    policy: object,
    shapes: dict[str, Any],
    stored: dict[str, Any] | None,
    sets: dict[str, str | None],
) -> dict[str, Any]:
    """Each ref's class tree in full under ``policy`` over ``content``, joined before format 9
    with the ``stored`` class trees as ``redact`` says; ``ValueError`` for a policy of another
    form or one naming what the record's shapes do not hold."""
    if not isinstance(policy, Mapping):
        raise ValueError(f"the policy is {described(policy)}, not a mapping")
    named: list[tuple[int, str, str, tuple[str, ...], Any]] = []
    seen: set[str] = set()
    for key, given in cast("Mapping[object, Any]", policy).items():
        if not is_scalar_string(key):
            raise ValueError(f"the policy has the key {described(key)}, which is not a string")
        text = string_data(cast(str, key))
        try:
            parsed = parse_ref(text)
        except ValueError:
            raise ValueError(f"the policy's key {clipped(text)} is not a ref") from None
        if parsed.candidate is not None:
            raise ValueError(f"the policy's key {clipped(text)} has a grounded bracket")
        ref = f"{parsed.level}:{parsed.name}"
        if ref not in shapes:
            raise ValueError(f"the policy's key {clipped(text)} names no ref of the record")
        if ref in sets and not parsed.fields and sets[ref] is None and not is_scalar_string(given):
            raise ValueError(
                f"the policy's key {clipped(text)} is a candidate set's, which takes a class name"
            )
        if text in seen:
            raise ValueError(f"the policy has two keys of the string data {clipped(text)}")
        seen.add(text)
        named.append((len(parsed.fields), text, ref, parsed.fields, given))
    applied = {ref: _filled(shape, _CONTENT) for ref, shape in shapes.items()}
    for _, text, ref, path, given in sorted(named, key=lambda n: (n[0], n[1])):
        applied[ref] = _at(applied[ref], shapes[ref], path, given, clipped(text))
    if stored is not None:
        applied = {ref: _joined(tree, stored[ref]) for ref, tree in applied.items()}
    return applied


def _at(tree: Any, shape: Any, path: tuple[str, ...], given: object, key: str) -> Any:
    """``tree`` with ``given`` applied at the field ``path`` names in ``shape``, through any
    mapping layer."""
    if not path:
        return _over(tree, shape, given, f"the policy's value at {key}")
    if type(shape) is list:
        return [_at(cast("list[Any]", tree)[0], cast("list[Any]", shape)[0], path, given, key)]
    if type(shape) is not dict or path[0] not in shape:
        raise ValueError(f"the policy's key {key} names a field the record's shape does not hold")
    nodes = cast("dict[str, Any]", shape)
    out = dict(cast("dict[str, Any]", tree))
    out[path[0]] = _at(out[path[0]], nodes[path[0]], path[1:], given, key)
    return out


def _over(tree: Any, shape: Any, given: object, where: str) -> Any:
    """``tree`` with ``given``, a class name or an object of fields each holding one or such an
    object, applied over it, held to ``shape``."""
    if is_scalar_string(given):
        name = string_data(cast(str, given))
        if name not in _RANK:
            raise ValueError(f"{where} is {clipped(name)}, not open, identifier or content")
        return _filled(shape, name)
    if type(given) is list:  # a mapping layer, as ``applied`` writes it
        members = cast("list[Any]", given)
        if type(shape) is not list or len(members) != 1:
            raise ValueError(f"{where} is a list, where the record's shape is no mapping layer")
        return [_over(cast("list[Any]", tree)[0], cast("list[Any]", shape)[0], members[0], where)]
    if not isinstance(given, Mapping):
        raise ValueError(f"{where} is {described(given)}, not a class name or an object")
    fields = cast("Mapping[object, Any]", given)
    if type(shape) is list:
        return [_over(cast("list[Any]", tree)[0], cast("list[Any]", shape)[0], fields, where)]
    if type(shape) is not dict:
        raise ValueError(f"{where} is an object, where the record's shape is not a record")
    nodes = cast("dict[str, Any]", shape)
    out = dict(cast("dict[str, Any]", tree))
    seen: set[str] = set()
    for field, member in fields.items():
        name = string_data(cast(str, field)) if is_scalar_string(field) else None
        if name is None or name not in nodes:
            raise ValueError(f"{where} names {described(field)}, which is not a field there")
        if name in seen:
            raise ValueError(f"{where} has two keys of the string data {clipped(name)}")
        seen.add(name)
        out[name] = _over(out[name], nodes[name], member, f"{where}.{name}")
    return out


def _candidate_class(classes: dict[str, Any], fields: dict[str, str | None], set_ref: str) -> Any:
    """A candidate's class: the class name a ``list<string>`` set's tree is, or for a set of
    records its candidate field's node, by the ``settings`` item's ``sets``; ``content`` where
    the tree holds no class name for it."""
    tree = classes.get(set_ref, _CONTENT)
    field = fields.get(set_ref)
    if field is not None and type(tree) is dict:
        tree = cast("dict[str, Any]", tree).get(field, _CONTENT)
    return tree if type(tree) is str else _CONTENT


def _candidate_sets(
    log: list[dict[str, Any]], classes: dict[str, Any], fields: dict[str, str | None]
) -> dict[tuple[int, str], tuple[frozenset[str], Any]]:
    """Each candidate-valued choice a call item asked, keyed by the call's log index and its
    ``q:`` ref: its fixed names and its candidates' class; under a set the record holds no
    tree for, ``content``."""
    found: dict[tuple[int, str], tuple[frozenset[str], Any]] = {}
    for index, item in enumerate(log):
        if item["item"] != "call":
            continue
        for ref, question in cast("dict[str, Any]", item["questions"]).items():
            labels: object = cast("dict[str, Any]", question)["labels"]
            if type(labels) is not dict:  # the ledger admits an object only for a choice's
                continue
            declared = cast("dict[str, Any]", labels)
            fixed = frozenset(
                label if type(label) is str else cast("dict[str, str]", label)["name"]
                for label in cast("list[Any]", declared["fixed"])
            )
            try:
                set_ref = declared_ref(cast(str, declared["candidates"]))
            except ValueError:  # a ref outside the grammar names no tree
                tree: Any = _CONTENT
            else:
                tree = _candidate_class(classes, fields, set_ref)
            found[(index, ref)] = (fixed, tree)
    return found


class _Walk:
    """The digests of one record, all under its per-turn key."""

    def __init__(self, key: bytes) -> None:
        self.key = key
        self.sets: dict[tuple[int, str], tuple[frozenset[str], Any]] = {}
        self.grounded: dict[str, str] = {}
        self.classes: dict[str, Any] = {}
        self.fields: dict[str, str | None] = {}

    def ref(self, ref: Any) -> Any:
        """A ref with any grounded bracket's candidate replaced by its keyed digest under the
        candidate's class, of the set the ``settings`` item's ``grounded`` names, open passing;
        any other ref, or a role name, as it is."""
        if type(ref) is not str:
            return ref
        try:
            parsed = parse_ref(ref)
        except ValueError:  # a role name, rule:N, or step:N (only in records before format 8)
            return ref
        if parsed.candidate is None:
            return ref
        declared = f"{parsed.level}:{parsed.name}"
        if _candidate_class(self.classes, self.fields, self.grounded[declared]) == _OPEN:
            return ref
        written = grounded_ref(declared, self.digest(parsed.candidate))
        return written + "".join(f".{field}" for field in parsed.fields)

    def context_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """A plan's context entry, in a record before format 8: its text digested, its marker's
        cause rewritten."""
        out = {**entry, "text": self.text(entry["text"])}
        if entry["marker"] is not None:
            marker = cast("dict[str, Any]", entry["marker"])
            out["marker"] = {**marker, "cause": self.cause(marker["cause"])}
        return out

    def cause(self, cause: dict[str, Any]) -> dict[str, Any]:
        """A cause with each grounded ref of its ``refs`` rewritten."""
        return {**cause, "refs": [self.ref(r) for r in cast("list[Any]", cause["refs"])]}

    def digest(self, value: Any) -> str:
        """A keyed digest of the canonical bytes of a value or of text ``unhashed`` holds,
        encoded as a holder within the document's depth bound: text is held to no size
        bound."""
        return keyed_digest(self.key, encode(value, MAX_DOCUMENT_DEPTH))

    def text(self, value: Any) -> Any:
        """Text the engine records with no type: ``content``, a null passing."""
        return None if value is None else self.digest(value)

    # -- the log -----------------------------------------------------------------------------

    def item(self, item: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        kind = item["item"]
        out = dict(item)
        if kind == "entry":
            out["ref"] = self.ref(item["ref"])
            if "marker" in item:
                marker = cast("dict[str, Any]", item["marker"])
                out["marker"] = {**marker, "cause": self.cause(marker["cause"])}
        elif kind == "call" and type(item["outcome"]) is dict:
            failed = cast("dict[str, Any]", item["outcome"])["failed"]
            out["outcome"] = {"failed": self.cause(failed)}
        elif kind == "cause":
            out["cause"] = self.cause(item["cause"])
        if kind == "entry" and "value" in item:
            tree = classes[declared_ref(item["ref"])]
            # Keyed by the declared ref, as the call item's questions are; a null call: none.
            asked = self.sets.get((cast(int, item["call"]), declared_ref(item["ref"])))
            if asked is None:
                out["value"] = self.value(item["value"], tree)
            else:
                out["value"] = self.answer(item["value"], tree, *asked)
        elif kind == "call":
            out["prompt"] = self.text(item["prompt"])
        elif kind == "plan":  # a plan item: records before format 8 hold one
            out["reason"] = self.text(item["reason"])
            out["reply"] = self.text(item["reply"])
            out["steps"] = [self.step(step, classes) for step in item["steps"]]
            if item["cause"] is not None:  # a cause may name a grounded ref (§ledger)
                out["cause"] = self.cause(item["cause"])
            context = item.get("context")
            if context is not None:
                out["context"] = {
                    **context,
                    "design": self.text(context["design"]),
                    "entries": [self.context_entry(e) for e in context["entries"]],
                }
        elif kind == "verdict":  # a verdict item: records before format 8 hold one
            out["failures"] = [
                {**f, "requirement": self.requirement(f["requirement"], classes)}
                if f["requirement"] is not None
                else f
                for f in cast("list[dict[str, Any]]", item["failures"])
            ]
        # Every other member is open, a cause's refs but their grounded refs: markers,
        # derivation, rows, slots, a plan's context entry's ref and its block's switches, the
        # verdict's other members, the run settings, the policy's ref and the call item's other
        # members.
        if "order" in item:  # derived afresh over the item as the view writes it
            out = _with_held_order({k: v for k, v in out.items() if k != "order"})
        return out

    def answer(self, value: Any, tree: Any, fixed: frozenset[str], set_tree: Any) -> Any:
        """A candidate-valued choice's answer: each label and ``probabilities`` key that is not
        a fixed name takes its set's class, the rest the answer's own tree."""
        if tree != _OPEN:  # an answer under a class of its own is digested whole
            return self.value(value, tree)
        if set_tree == _OPEN:
            return value
        answer = cast("dict[str, Any]", value)

        def label(name: str) -> str:
            return name if name in fixed else self.digest(name)

        return {
            **answer,
            "labels": [label(n) for n in cast("list[str]", answer["labels"])],
            "probabilities": {
                label(n): p for n, p in cast("dict[str, Any]", answer["probabilities"]).items()
            },
        }

    def requirement(self, requirement: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        """A verdict's requirement, in a record before format 8, with its value under the
        applied tree at its ref exactly. A ref with a field path (``t:state.to``) or outside the
        grammar has no tree of its own, so its value is digested whole, and no policy opens it."""
        try:
            tree = classes.get(declared_ref(requirement["ref"]), _CONTENT)
        except ValueError:  # a ref outside the grammar names no tree
            tree = _CONTENT
        return {**requirement, "value": self.value(requirement["value"], tree)}

    def step(self, step: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        """A plan's step, in a record before format 8, its arguments redacted by origin."""
        arguments: dict[str, Any] = {}
        for name, given in cast("dict[str, Any]", step["arguments"]).items():
            if type(given) is not dict:  # the ledger holds each origin an object
                arguments[name] = given
                continue
            origin = cast("dict[str, Any]", given)
            if "set" in origin:  # a candidate-value literal, from format 7: its set's tree
                tree = classes.get(cast(str, origin["set"]), _CONTENT)
                arguments[name] = {**origin, "literal": self.value(origin["literal"], tree)}
            elif "literal" in origin:  # no ref names it: content
                arguments[name] = {**origin, "literal": self.value(origin["literal"], _CONTENT)}
            elif "ref" in origin:  # open, but a grounded ref's candidate
                arguments[name] = {**origin, "ref": self.ref(origin["ref"])}
            else:  # a binding: open
                arguments[name] = origin
        requirements = [self.requirement(r, classes) for r in step["requirements"]]
        return {**step, "arguments": arguments, "requirements": requirements}

    def value(self, value: Any, tree: Any) -> Any:
        """A value redacted beside its full class tree, read from the tree: a record opened
        field by field; a list or mapping whose tree holds no ``open`` and no record one
        digest, any other opened, a mapping's keys digested as ``content``; a null passing
        unless it lies within a value digested whole."""
        if value is None:
            return None
        if type(value) is list:  # a list layer has no node: the tree describes each member
            if not _keeps_open(tree):
                return self.digest(value)
            return [self.value(v, tree) for v in cast("list[Any]", value)]
        if type(tree) is str:
            return value if tree == _OPEN else self.digest(value)
        members = cast("dict[str, Any]", value)  # the ledger fits each value to its tree
        if type(tree) is dict:  # a record
            nodes = cast("dict[str, Any]", tree)
            return {name: self.value(v, nodes[name]) for name, v in members.items()}
        if not _keeps_open(tree):  # a mapping layer, a one-member list
            return self.digest(value)
        beneath = cast("list[Any]", tree)[0]
        return {self.digest(k): self.value(v, beneath) for k, v in members.items()}

    # -- unhashed ----------------------------------------------------------------------------

    def unhashed(self, unhashed: dict[str, Any]) -> dict[str, Any]:
        """``unhashed`` with its text digested: exception text, ``unexpected``, ``warnings`` and
        ``request_id`` each whole, ``extras`` whole, ``invalid``'s ``repr`` text, the ``reads``
        values, an unknown member whole and an unknown ``calls`` entry member whole; member
        names, log-index keys, ``usage``, ``sums``, ``truncated``, ``calibrated`` (before format
        7), ``missing``, ``clients``, times, the engine and codec names and an exception's
        ``type`` are open."""
        out: dict[str, Any] = {}
        for member, value in unhashed.items():
            if member in ("turn", "engine", "codec"):
                out[member] = value
            elif member == "calls":
                out[member] = {k: self.call(v) for k, v in value.items()}
            elif member == "exceptions":
                out[member] = {
                    k: {
                        **v,
                        "message": self.digest(v["message"]),
                        "traceback": self.digest(v["traceback"]),
                    }
                    for k, v in value.items()
                }
            elif member == "reads":
                out[member] = {k: self.digest(v) for k, v in value.items()}
            else:
                out[member] = self.digest(value)
        return out

    def call(self, entry: dict[str, Any]) -> dict[str, Any]:
        """A ``calls`` entry, failing closed: each member ``_CALL_SHOWN`` names open, ``sums``
        and ``truncated`` but for a grounded ref's candidate; ``unexpected``, ``warnings``,
        ``request_id`` and ``extras`` each one digest, and each ``invalid`` text digested; any
        other member digested whole."""
        out = {
            name: value if name in _CALL_SHOWN else self.digest(value)
            for name, value in entry.items()
        }
        if "sums" in entry:
            sums = cast("dict[str, Any]", entry["sums"])
            out["sums"] = {self.ref(k): v for k, v in sums.items()}
        if "truncated" in entry:
            truncated = cast("list[Any]", entry["truncated"])
            out["truncated"] = code_point_sorted(self.ref(r) for r in truncated)
        if "unexpected" in entry:
            out["unexpected"] = self.digest(entry["unexpected"])
        if "warnings" in entry:  # the binding's text, one value
            out["warnings"] = self.digest(entry["warnings"])
        if "extras" in entry:  # whatever the provider reports, its length and order hidden
            out["extras"] = self.digest(entry["extras"])
        if "request_id" in entry:  # the provider's id, which may echo the caller's
            out["request_id"] = self.digest(entry["request_id"])
        if "invalid" in entry:
            invalid = cast("dict[str, Any]", entry["invalid"])
            out["invalid"] = {name: self.digest(text) for name, text in invalid.items()}
        return out


_CALL_SHOWN = frozenset(
    {
        "start",
        "latency",
        "usage",
        "attempts",
        "sums",
        "truncated",
        "calibrated",
        "missing",
        "clients",
    }
)
"""The ``calls`` entry members the view shows: structure, the times and the profile's open
members; ``sums`` and ``truncated`` with a grounded ref's candidate digested."""

_CALL_DIGESTED = frozenset({"unexpected", "warnings", "request_id", "extras", "invalid"})
"""The ``calls`` entry members the view digests, each whole but ``invalid``, whose texts are
each digested."""


def _keeps_open(tree: Any) -> bool:
    """Whether a tree holds an ``open`` name or a record object, so its value is opened."""
    if type(tree) is str:
        return tree == _OPEN
    if type(tree) is dict:
        return True
    return any(_keeps_open(t) for t in cast("list[Any]", tree))
