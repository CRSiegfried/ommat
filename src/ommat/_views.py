"""The record views: pure functions from a closed record to a value, for the caller, the
escalation target or a verifier.

``escalation(record)`` returns an undecided turn's five members: the marker's kind, its cause, the
root cause, whether that is a defect, and the record hash; ``None`` when the policy gave a value.
``defects(record)`` returns every entry whose cause is a defect by the same test, whatever the
policy gave.
``transcript(record)``, deprecated, returns every source call and the decision, unworded. Each
of the escalation, the defects and the transcript gives ``None`` for a record before format 8,
which holds no policy's entry, and the escalation and the transcript find the decision as
``decision_entry`` does. ``redact(record, root_key)`` returns the redacted view's document as
canonical bytes: the record's ``log`` and ``unhashed`` with a keyed digest where each
``content`` or ``identifier`` value was, ``redacted``, the record hash its key came from, and
from format 3 on the document's ``hash``, open. Each value's class is read from the
``settings`` item's class tree for its ref. In an answer to a candidate-valued choice, each
label and ``probabilities`` key that is not a fixed name takes its set's class, found through
the call item the answer's entry names.

This engine's record holds no origin carrying ``set``, which the ledger refuses, so every literal
takes its JSON form's default; it holds a grounded ref only at an entry's ref, a cause's refs and
a call's ``sums`` and ``truncated``, where its candidate is digested under its set's class, and
every other ref passes unchanged. Each view raises ``ValueError`` before reading anything when
``record`` is not exactly a ``Record``. The continuation record is deferred, so it has no function
here.
"""

import warnings
from copy import deepcopy
from typing import Any, cast

from ommat._canonical import encode, keyed_digest, turn_key
from ommat._ledger import (
    Record,
    _with_held_order,  # pyright: ignore[reportPrivateUsage]
    decision_entry,
)
from ommat._markers import CauseCode
from ommat._refs import code_point_sorted, declared_ref, entry_ref, grounded_ref, parse_ref
from ommat._typed import is_a
from ommat._values import HASH_FORMAT, MAX_DOCUMENT_DEPTH, POLICY_FORMAT, type_name

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


def redact(record: Record, root_key: bytes) -> bytes:
    """The redacted view of a closed record, as canonical bytes, its digests under the key
    derived from the record's own hash. Raises ``ValueError`` before reading anything when
    ``record`` is not exactly a ``Record``, then before reading the record when ``root_key`` is
    not an instance of ``bytes`` of length 32."""
    closed = _closed(record)
    key = _root_key(root_key)
    record_hash = closed.record_hash
    walk = _Walk(turn_key(key, record_hash))
    log = closed.log
    classes: dict[str, Any] = {}
    fields: dict[str, str | None] = {}
    for item in log:
        if item["item"] == "settings":
            classes = cast("dict[str, Any]", item["classes"])
            walk.grounded = cast("dict[str, str]", item.get("grounded", {}))
            fields = cast("dict[str, str | None]", item.get("sets", {}))
    walk.classes = classes
    walk.fields = fields
    walk.sets = _candidate_sets(log, classes, fields)
    view = {
        "log": [walk.item(item, classes) for item in log],
        "unhashed": walk.unhashed(closed.unhashed),
        "redacted": record_hash,
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


def candidate_class(classes: dict[str, Any], fields: dict[str, str | None], set_ref: str) -> Any:
    """A candidate's class: the class name a ``list<string>`` set's tree is, or for a set of
    records its candidate field's node, by the ``settings`` item's ``sets``; ``content`` where
    the item holds no class name for it."""
    tree = classes.get(set_ref, _CONTENT)
    field = fields.get(set_ref)
    if field is not None and type(tree) is dict:
        tree = cast("dict[str, Any]", tree).get(field, _CONTENT)
    return tree if type(tree) is str else _CONTENT


def _candidate_sets(
    log: list[dict[str, Any]], classes: dict[str, Any], fields: dict[str, str | None]
) -> dict[tuple[int, str], tuple[frozenset[str], Any]]:
    """Each candidate-valued choice a call item asked, keyed by the call's log index and its
    ``q:`` ref: its fixed names and its candidates' class; under a set the settings item holds
    no tree for, ``content``."""
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
                tree = candidate_class(classes, fields, set_ref)
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
        if candidate_class(self.classes, self.fields, self.grounded[declared]) == _OPEN:
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
        """A verdict's requirement, in a record before format 8, with its value under its ref's
        tree, as that ref's entry is; under a ref the settings item holds no tree for, the value
        is digested whole."""
        try:
            tree = classes.get(declared_ref(requirement["ref"]), "content")
        except ValueError:  # a ref outside the grammar names no tree
            tree = "content"
        return {**requirement, "value": self.value(requirement["value"], tree)}

    def step(self, step: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        """A plan's step, in a record before format 8, its arguments redacted by origin."""
        arguments: dict[str, Any] = {}
        for name, given in cast("dict[str, Any]", step["arguments"]).items():
            if type(given) is not dict:  # the ledger holds each origin an object
                arguments[name] = given
                continue
            origin = cast("dict[str, Any]", given)
            if "set" in origin:  # a candidate-value literal: its set's tree
                tree = classes.get(cast(str, origin["set"]), _CONTENT)
                arguments[name] = {**origin, "literal": self.value(origin["literal"], tree)}
            elif "literal" in origin:  # its JSON form's default
                arguments[name] = {**origin, "literal": self.literal(origin["literal"])}
            elif "ref" in origin:  # open, but a grounded ref's candidate
                arguments[name] = {**origin, "ref": self.ref(origin["ref"])}
            else:  # a binding: open
                arguments[name] = origin
        requirements = [self.requirement(r, classes) for r in step["requirements"]]
        return {**step, "arguments": arguments, "requirements": requirements}

    def literal(self, value: Any) -> Any:
        """A literal argument's default by its JSON form: a string ``content``, any other
        scalar ``open``, a list or object member by member, an object's keys ``content``."""
        if type(value) is str:
            return self.digest(value)
        if type(value) is list:
            return [self.literal(v) for v in cast("list[Any]", value)]
        if type(value) is dict:
            return {
                self.digest(k): self.literal(v) for k, v in cast("dict[str, Any]", value).items()
            }
        return value

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
        ``request_id`` each whole, ``invalid``'s ``repr`` text, the ``reads`` values and an
        unknown member whole; member names, log-index keys, ``usage``, ``sums``, ``truncated``,
        ``calibrated`` (before format 7), ``missing``, ``clients``, times, the engine and codec
        names and an exception's ``type`` are open."""
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
        # ``sums`` and ``truncated`` pass open but for a grounded ref's candidate.
        out = dict(entry)
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
        if "request_id" in entry:  # the provider's id, which may echo the caller's
            out["request_id"] = self.digest(entry["request_id"])
        if "invalid" in entry:
            invalid = cast("dict[str, Any]", entry["invalid"])
            out["invalid"] = {name: self.digest(text) for name, text in invalid.items()}
        return out


def _keeps_open(tree: Any) -> bool:
    """Whether a tree holds an ``open`` name or a record object, so its value is opened."""
    if type(tree) is str:
        return tree == _OPEN
    if type(tree) is dict:
        return True
    return any(_keeps_open(t) for t in cast("list[Any]", tree))
