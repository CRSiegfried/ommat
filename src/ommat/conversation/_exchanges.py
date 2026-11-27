"""A closed record as the turn history both renderers give: per round the round's source calls
and their answers, then decide's call, whose arguments are the board, and its result, each
named ref's value or why it is undecided. Each tool name is checked here, so the renderers refuse
the same names."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from ommat._canonical import encode
from ommat._design import LoadedDesign
from ommat._ledger import Record
from ommat._refs import code_point_sorted, declared_ref, entry_ref
from ommat._values import POLICY_FORMAT, type_name
from ommat._views import escalation

_DECIDE = "decide"
BOOKKEEPING = "bookkeeping"
BOOKKEEPING_ID = re.compile(r"om_[0-9a-f]{24}_(?:c[1-9][0-9]*|d)")
"""A source call's or decide's id: ``om_``, 24 hex digits, then ``_c<n>`` or ``_d``."""

_ANSWER_KINDS = frozenset({"noul", "choice", "score"})


@dataclass(frozen=True, slots=True)
class Call:
    """One tool call and its result: ``failed`` is true for a source call whose outcome is not
    ``returned``."""

    id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    failed: bool = False
    decide: bool = False


@dataclass(frozen=True, slots=True)
class History:
    """A record's turn history: its hashes and its groups of calls, each group one assistant
    turn. No groups for a record before format 8 or one with no entry at a named ref."""

    record_hash: str
    design_hash: str
    groups: list[list[Call]]


def history(record: object, design: object, refs: object, turn: object = None) -> History:
    """The record's turn history, decide's result each of ``refs``' entries, its board every
    ``d:`` entry but theirs, cut to the declared refs their rules read and every marker when
    ``design`` is the record's loaded design. Raises ``ValueError`` for a ``record`` that is not
    exactly a ``Record``, a ``design`` that is neither ``None`` nor exactly a ``LoadedDesign``, a
    design whose hash is not the record's, or ``refs`` as ``check_refs`` refuses them."""
    if type(record) is not Record:
        raise ValueError(f"a renderer takes a Record, not {type_name(record)}")
    named = check_refs(refs)
    log = record.log
    design_hash = cast(str, log[0]["hash"])
    loaded = _loaded(design, design_hash)
    if log[0].get("format", 0) < POLICY_FORMAT:
        return History(record.record_hash, design_hash, [])
    providers = cast("dict[str, str]", log[1]["providers"])
    for ref in named:
        if ref not in providers:
            raise ValueError(f"{ref!r} is no rule's ref in the record's design")
    entries: dict[str, dict[str, Any]] = {}
    for item in log:
        if item["item"] == "entry":
            entries.setdefault(cast(str, item["ref"]), item)
    if any(ref not in entries for ref in named):
        return History(record.record_hash, design_hash, [])
    reads = None if loaded is None else _reads(loaded, named)
    stem = "om_" + record.record_hash.removeprefix("sha256:")[:24]
    groups = _source_calls(log, stem)
    board = {
        ref: _compact(item)
        for ref, item in entries.items()
        if ref.startswith("d:")
        and ref not in named
        and (reads is None or "marker" in item or declared_ref(ref) in reads)
    }
    result = {ref: _result(record, ref, entries[ref]) for ref in named}
    groups.append([Call(f"{stem}_d", _DECIDE, board, result, decide=True)])
    if turn is not None:
        new = turn_stem(record.record_hash, turn)
        groups = [[replace(call, id=new + call.id[_STEM:]) for call in group] for group in groups]
    return History(record.record_hash, design_hash, groups)


def check_refs(refs: object) -> tuple[str, ...]:
    """``refs`` as a tuple, refused unless a non-empty ``list`` or ``tuple`` of distinct ``d:``
    refs, each exactly a ``str`` naming an entry, with no field path."""
    if type(refs) is not list and type(refs) is not tuple:
        raise ValueError(f"refs must be a list or a tuple of d: refs, not {type_name(refs)}")
    named = tuple(cast("Sequence[object]", refs))
    if not named:
        raise ValueError("refs must name at least one d: ref")
    for ref in named:
        if type(ref) is not str or not ref.startswith("d:"):
            raise ValueError(f"refs must hold d: refs, not {ref!r}")
        if entry_ref(ref) != ref or declared_ref(ref) != ref:
            raise ValueError(f"{ref!r} is not a rule's ref")
    if len(set(named)) != len(named):
        raise ValueError("refs must not repeat a ref")
    return cast("tuple[str, ...]", named)


def _result(record: Record, ref: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Decide's result for one named ref: ``{"value"}``, or ``{"undecided"}`` holding the
    marker's kind and its root cause's code and refs."""
    if "value" in entry:
        return {"value": entry["value"]}
    escalated = cast("dict[str, Any]", escalation(record, ref))
    root = cast("dict[str, Any]", escalated["root_cause"])
    return {
        "undecided": {
            "kind": escalated["kind"],
            "root_cause": root["code"],
            "refs": list(root["refs"]),
        }
    }


_STEM = 27  # "om_" and 24 hex digits


def _loaded(design: object, design_hash: str) -> LoadedDesign | None:
    """The design, refused unless ``None`` or exactly the record's ``LoadedDesign``."""
    if design is None:
        return None
    if type(design) is not LoadedDesign:
        raise ValueError(f"design must be a LoadedDesign or None, not {type_name(design)}")
    if design.design_hash != design_hash:
        raise ValueError(f"design {design.design_hash} is not the record's, {design_hash}")
    return design


def _reads(design: LoadedDesign, refs: tuple[str, ...]) -> frozenset[str]:
    """The declared refs the rules of ``refs`` read, each field path dropped."""
    providers = design._providers  # pyright: ignore[reportPrivateUsage]
    rules = design._reads  # pyright: ignore[reportPrivateUsage]
    return frozenset(declared_ref(entry_ref(r)) for ref in refs for r in rules[providers[ref]])


def _source_calls(log: list[dict[str, Any]], stem: str) -> list[list[Call]]:
    """One group per round that holds a call item: a call per source of each, numbered over
    the whole record, its answers the entries that name the call item."""
    providers = cast("dict[str, str]", log[1]["providers"])
    written: dict[int, dict[str, Any]] = {}
    for item in log:
        if item["item"] == "entry" and item.get("call") is not None:
            written.setdefault(item["call"], {})[item["ref"]] = _compact(item)
    groups: list[list[Call]] = []
    held: dict[int, list[Call]] = {}
    n = 0
    for index, item in enumerate(log):
        if item["item"] != "call":
            continue
        answers = written.get(index, {})
        for source in cast("list[str]", item["sources"]):
            n += 1
            # A per-candidate entry is its declared ref's source's.
            asks = code_point_sorted(r for r in answers if providers.get(declared_ref(r)) == source)
            call = Call(
                f"{stem}_c{n}",
                _source_name(source),
                {},
                {ref: answers[ref] for ref in asks},
                failed=item["outcome"] != "returned",
            )
            if item["round"] not in held:
                held[item["round"]] = []
                groups.append(held[item["round"]])
            held[item["round"]].append(call)
    return groups


def _compact(entry: dict[str, Any]) -> Any:
    """An entry as the history gives it: an answer as its top label and probability, an
    extraction as its value and confidence, any other value as recorded, and a marker as its
    kind and cause code."""
    if "marker" in entry:
        marker = cast("dict[str, Any]", entry["marker"])
        return {"marker": marker["kind"], "cause": marker["cause"]["code"]}
    if not cast(str, entry["ref"]).startswith("q:"):
        return entry["value"]
    answer = cast("dict[str, Any]", entry["value"])
    if answer["kind"] in _ANSWER_KINDS:
        probabilities = cast("dict[str, float]", answer["probabilities"])
        top = max(answer["labels"], key=lambda label: probabilities[label])  # first of a tie
        return [top, probabilities[top]]
    return [answer["value"], answer["confidence"]]  # an extraction


def check_turn(turn: object) -> None:
    """Refuse a ``turn`` that is not ``None``, exactly a ``str`` or an ``int``, or that the
    canonical encoding does not take."""
    if turn is None:
        return
    if type(turn) is not str and type(turn) is not int:
        raise ValueError(f"turn must be a str or an int, not {type(turn).__name__}")
    encode(turn)  # an int past 2^53 - 1 or a str with a surrogate raises ValueError


def turn_stem(record_hash: str, turn: object) -> str:
    """The id stem for a record rendered under ``turn``: ``om_`` and the first 24 hex digits of
    SHA-256 over the canonical JSON of the record hash and the turn, so an id keeps its form and
    its bound whatever the turn holds. ``turn`` is exactly a ``str`` or an ``int``, the canonical
    encoding's: an ``int`` of magnitude at most 2^53 - 1, a ``str`` with no surrogate; anything
    else raises ``ValueError``."""
    check_turn(turn)
    digest = hashlib.sha256(encode([record_hash, turn])).hexdigest()
    return "om_" + digest[:24]


def _source_name(source: str) -> str:
    """A source's tool name: its name with each ``/`` as ``-``, which no ommat name holds."""
    name = source.replace("/", "-")
    if name == _DECIDE:
        raise ValueError(f"source {source!r} renders as {_DECIDE!r}, the name decide's call takes")
    if len(name) > 64:
        raise ValueError(f"source {source!r} renders as a tool name over 64 characters")
    return name
