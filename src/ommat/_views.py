"""The record views: pure functions from a closed record to a value, for the caller, the
escalation target or a verifier.

``escalation(record)`` returns the escalation's four members, or ``None`` for a record that
holds none, reading only the ``plan`` and ``verdict`` items and the record hash.
``handoff(record)`` returns what the turn hands its next reader under the design's context
block, or ``None`` with no block or no ``plan`` item. ``transcript(record)`` returns every
source call and the turn's conclusion, unworded, or ``None`` with no ``plan`` item. ``redact(record,
root_key)`` returns the redacted view's document as canonical bytes: the record's ``log`` and
``unhashed`` with a keyed digest where each ``content`` or ``identifier`` value was,
``redacted``, the record hash its key came from, and from format 3 on the document's ``hash``,
open. Each value's class is read from the ``settings`` item's class tree for its ref. In an
answer to a candidate-valued choice, each label and ``probabilities`` key that is not a fixed
name takes its set's class, found through the call item the answer's entry names.

This engine's record holds no grounded ref and no origin carrying ``set``, since the ledger refuses
both at the write and the read: every ref passes unchanged and every literal takes
its JSON form's default. Each view raises ``ValueError`` before reading anything when
``record`` is not exactly a ``Record``. The continuation record is deferred, so it has no function
here.
"""

from copy import deepcopy
from typing import Any, cast

from ommat._canonical import encode, keyed_digest, turn_key
from ommat._design import Requirement
from ommat._guard import GuardFailure, Verdict
from ommat._ledger import Record, _with_held_order  # pyright: ignore[reportPrivateUsage]
from ommat._markers import PlanKind, VerdictKind
from ommat._refs import code_point_sorted, declared_ref, grounded_ref, parse_ref
from ommat._resolve import board_entries, board_value
from ommat._typed import is_a
from ommat._values import HASH_FORMAT, MAX_DOCUMENT_DEPTH, type_name

_OPEN = "open"


def _closed(record: object) -> Record:
    """The record, refused by its exact type before anything of it is read."""
    if type(record) is not Record:
        raise ValueError(f"a record view takes a Record, not {type_name(record)}")
    return record


# ---------------------------------------------------------------------------------------------
# The escalation


def escalation(record: Record) -> dict[str, Any] | None:
    """The escalation of a closed record: ``cause``, ``failures``, ``reason`` and
    ``record_hash``; ``None`` when the log has no ``verdict`` item, or its plan is not
    ``escalate`` and its verdict passed."""
    log = _closed(record).log
    plans = [item for item in log if item["item"] == "plan"]
    verdicts = [item for item in log if item["item"] == "verdict"]
    if not verdicts:
        return None
    plan, verdict = plans[0], verdicts[0]
    escalated = plan["kind"] == PlanKind.escalate
    if not escalated and verdict["kind"] == VerdictKind.passed:
        return None
    cause: dict[str, Any] | None = None
    if verdict["kind"] == VerdictKind.refused:
        derived = _verdict(verdict).cause
        assert derived is not None  # a refused verdict's cause is never None
        cause = {"code": derived.code.value, "refs": list(derived.refs)}
    elif plan["cause"] is not None:  # the engine's `required` on a plan compile built
        cause = cast("dict[str, Any]", plan["cause"])
    return {
        "cause": cause,
        "failures": verdict["failures"],
        "reason": plan["reason"] if escalated else None,
        "record_hash": record.record_hash,
    }


# ---------------------------------------------------------------------------------------------
# The handoff


def handoff(record: Record) -> dict[str, Any] | None:
    """What a closed record hands its next reader: ``design`` and ``record_hash`` under the
    ``record`` switch, ``plan`` with its kind, text and verdict kind under ``plan`` and its
    steps under ``steps`` once the verdict passed, and ``context``, the entries as recorded;
    ``None`` when the log has no ``plan`` item or the item holds no block."""
    closed = _closed(record)
    log = closed.log
    plan = next((item for item in log if item["item"] == "plan"), None)
    if plan is None or plan.get("context") is None:
        return None
    block = cast("dict[str, Any]", plan["context"])
    verdict = next((item for item in log if item["item"] == "verdict"), None)
    passed = verdict is not None and verdict["kind"] == VerdictKind.passed
    out: dict[str, Any] = {}
    if block["record"]:
        out["design"] = {"name": block["design"], "hash": log[0]["hash"]}
        out["record_hash"] = closed.record_hash
    shown: dict[str, Any] = {}
    if block["plan"]:
        kind = None if verdict is None else verdict["kind"]
        shown.update(kind=plan["kind"], reply=plan["reply"], reason=plan["reason"], verdict=kind)
    if block["steps"] and passed:
        entries = board_entries(log)
        shown["steps"] = [_handed_step(step, entries, n) for n, step in enumerate(plan["steps"], 1)]
    if shown:
        out["plan"] = shown
    out["context"] = block["entries"]
    return out


def _handed_step(step: dict[str, Any], entries: dict[str, object], n: int) -> dict[str, Any]:
    """A step as the handoff gives it: its tool and each argument tagged, a literal as
    ``{"value"}``, a board ref as ``{"ref", "value"}`` and a binding as ``{"step", "path"}``."""
    arguments: dict[str, Any] = {}
    for name, origin in cast("dict[str, Any]", step["arguments"]).items():
        if "literal" in origin:
            arguments[name] = {"value": origin["literal"]}
        elif "ref" in origin:
            ref = cast(str, origin["ref"])
            try:
                value = board_value(entries, ref)
            except (KeyError, TypeError):
                raise ValueError(
                    f"step {n}'s argument {name!r} names {ref!r}, which holds no value in the log"
                ) from None
            arguments[name] = {"ref": ref, "value": value}
        else:
            arguments[name] = {"step": origin["step"], "path": origin["path"]}
    return {"tool": step["tool"], "arguments": arguments}


# ---------------------------------------------------------------------------------------------
# The transcript


def transcript(record: Record) -> dict[str, Any] | None:
    """Every source call of a closed record and its conclusion, unworded: ``record_hash``,
    ``design``, ``rounds`` of calls, ``decide`` and the ``pending`` steps; ``None`` when the log
    has no ``plan`` item. Each value and marker is tagged, and each id is ``om_``, the record
    hash's first 24 hex digits and ``_c<n>``, ``_d`` or ``_s<n>``. ``pending`` holds a passed
    ``execute`` plan's steps with their resolved arguments, and is empty for a plan with a
    binding, which ``run_plan`` runs."""
    closed = _closed(record)
    log = closed.log
    plan = next((item for item in log if item["item"] == "plan"), None)
    if plan is None:
        return None
    verdict = next((item for item in log if item["item"] == "verdict"), None)
    stem = "om_" + closed.record_hash.removeprefix("sha256:")[:24]
    settings = next(item for item in log if item["item"] == "settings")
    providers = cast("dict[str, str]", settings["providers"])
    block = cast("dict[str, Any] | None", plan.get("context"))
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
    entries = board_entries(log)
    steps = cast("list[dict[str, Any]]", plan["steps"])
    board = {
        entry["ref"]: _tagged(entry)
        for entry in log
        if entry["item"] == "entry" and entry["ref"].startswith("d:")
    }
    decided: dict[str, Any] = {
        "id": f"{stem}_d",
        "board": board,
        "plan": {
            "kind": plan["kind"],
            "reason": plan["reason"],
            "reply": plan["reply"],
            "cause": plan["cause"],
            "steps": [
                {**_handed_step(step, entries, i), "requirements": step["requirements"]}
                for i, step in enumerate(steps, 1)
            ],
        },
        "verdict": None
        if verdict is None
        else {"kind": verdict["kind"], "failures": verdict["failures"]},
    }
    if block is not None:
        decided["context"] = block["entries"]
    pending: list[dict[str, Any]] = []
    passed = verdict is not None and verdict["kind"] == VerdictKind.passed
    bound = any("step" in origin for step in steps for origin in step["arguments"].values())
    if passed and plan["kind"] == PlanKind.execute and not bound:  # a binding runs by run_plan
        for i, step in enumerate(decided["plan"]["steps"], 1):
            arguments = {name: deepcopy(tag["value"]) for name, tag in step["arguments"].items()}
            pending.append({"id": f"{stem}_s{i}", "tool": step["tool"], "arguments": arguments})
    return {
        "record_hash": closed.record_hash,
        "design": {"name": None if block is None else block["design"], "hash": log[0]["hash"]},
        "rounds": rounds,
        "decide": decided,
        "pending": pending,
    }


def _tagged(entry: dict[str, Any]) -> dict[str, Any]:
    """An entry's value as ``{"value"}`` or its marker as ``{"marker"}``, as recorded."""
    if "value" in entry:
        return {"value": entry["value"]}
    return {"marker": entry["marker"]}


def _verdict(item: dict[str, Any]) -> Verdict:
    """The guard's ``Verdict`` built from the recorded failures."""
    failures: list[GuardFailure] = []
    for f in cast("list[dict[str, Any]]", item["failures"]):
        recorded = cast("dict[str, Any] | None", f["requirement"])
        requirement = None if recorded is None else Requirement(recorded["ref"], recorded["value"])
        failures.append(GuardFailure(f["step"], requirement, f["ref"]))
    return Verdict(item["kind"], failures)


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
    for item in log:
        if item["item"] == "settings":
            classes = cast("dict[str, Any]", item["classes"])
            walk.grounded = cast("dict[str, str]", item.get("grounded", {}))
    walk.classes = classes
    walk.sets = _candidate_sets(log, classes)
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


def _candidate_sets(
    log: list[dict[str, Any]], classes: dict[str, Any]
) -> dict[tuple[int, str], tuple[frozenset[str], Any]]:
    """Each candidate-valued choice a call item asked, keyed by the call's log index and its
    ``q:`` ref: its fixed names and its set's class, the class name a ``list<string>`` set's
    tree is; under a set the settings item holds no tree for, ``content``."""
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
                tree = classes.get(declared_ref(cast(str, declared["candidates"])), "content")
            except ValueError:  # a ref outside the grammar names no tree
                tree = "content"
            found[(index, ref)] = (fixed, tree if type(tree) is str else "content")
    return found


class _Walk:
    """The digests of one record, all under its per-turn key."""

    def __init__(self, key: bytes) -> None:
        self.key = key
        self.sets: dict[tuple[int, str], tuple[frozenset[str], Any]] = {}
        self.grounded: dict[str, str] = {}
        self.classes: dict[str, Any] = {}

    def ref(self, ref: Any) -> Any:
        """A ref with any grounded bracket's candidate replaced by its keyed digest under the
        class of the set the ``settings`` item's ``grounded`` names, open passing; any other
        ref, or a role name, as it is."""
        if type(ref) is not str:
            return ref
        try:
            parsed = parse_ref(ref)
        except ValueError:  # a role name, rule:N or step:N
            return ref
        if parsed.candidate is None:
            return ref
        declared = f"{parsed.level}:{parsed.name}"
        if self.classes[self.grounded[declared]] == _OPEN:
            return ref
        written = grounded_ref(declared, self.digest(parsed.candidate))
        return written + "".join(f".{field}" for field in parsed.fields)

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
            asked = self.sets.get((cast(int, item["call"]), item["ref"]))  # a null call: none
            if asked is None:
                out["value"] = self.value(item["value"], tree)
            else:
                out["value"] = self.answer(item["value"], tree, *asked)
        elif kind == "call":
            out["prompt"] = self.text(item["prompt"])
        elif kind == "plan":
            out["reason"] = self.text(item["reason"])
            out["reply"] = self.text(item["reply"])
            out["steps"] = [self.step(step, classes) for step in item["steps"]]
            context = item.get("context")
            if context is not None:
                out["context"] = {
                    **context,
                    "design": self.text(context["design"]),
                    "entries": [
                        {**e, "text": self.text(e["text"])}
                        for e in cast("list[dict[str, Any]]", context["entries"])
                    ],
                }
        elif kind == "verdict":
            out["failures"] = [
                {**f, "requirement": self.requirement(f["requirement"], classes)}
                if f["requirement"] is not None
                else f
                for f in cast("list[dict[str, Any]]", item["failures"])
            ]
        # Every other member is open: markers, causes, derivation, rows, slots, a context
        # entry's ref and marker and the block's switches, the verdict's
        # other members, the run settings and the call item's other members.
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
        """A requirement with its value under its ref's tree, as that ref's entry is; under a
        ref the settings item holds no tree for, the value is digested whole."""
        try:
            tree = classes.get(declared_ref(requirement["ref"]), "content")
        except ValueError:  # a ref outside the grammar names no tree
            tree = "content"
        return {**requirement, "value": self.value(requirement["value"], tree)}

    def step(self, step: dict[str, Any], classes: dict[str, Any]) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        for name, origin in cast("dict[str, Any]", step["arguments"]).items():
            if type(origin) is dict and "literal" in origin:  # its JSON form's default
                arguments[name] = {**origin, "literal": self.literal(origin["literal"])}
            else:  # a board ref or a binding: open
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
        ``calibrated``, ``missing``, ``clients``, times, the engine and codec names and an
        exception's ``type`` are open."""
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
            out["truncated"] = [self.ref(r) for r in cast("list[Any]", entry["truncated"])]
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
