# pyright: reportPrivateUsage=false
"""The fixture binding: replay's oracle binding, answering each call from the recorded calls
of its sources, so the turn reruns with no provider; ``from_record``'s, answering a caller's
live turn from a record under the oracle names it is bound to; and the stale-question check
both share.

A source is matched to the recorded call item that lists it, never by round or batch, and each
of its refs is answered only from a recorded entry naming that call. The check compares the
source as now asked with the base record, the record the answers were given in: a ref is
stale when the base holds no answer for it (``no-answer``), when a need's value differs from
the base's entry (``needs``), when its question as now asked differs from the call item's, or a
``t:`` ref's declared type from the base design's (``question``), when the source's payload
differs from the base design's (``payload``), or when its ``oracle`` key differs or a scenario
asks it afresh (``oracle``). Offline the fixture gives a stale ref no key and logs it, so the
engine writes ``not-returned`` and replay names it; in a scenario it asks the stale source of the
caller's binding and merges the answers; ``from_record`` refuses a stale ref with
``unsupported``. No answer depends on an earlier call; offline neither binding waits, and each
returns fresh containers with null provenance, a null precision and no truncated refs.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast

from ommat._answer import AnswerKind, ValueType
from ommat._binding import CallResult, OracleBinding, RenderedQuestion, Replaying, SourceCall
from ommat._binding_kit import bounded_warnings
from ommat._canonical import _CanonicalError, encode, ordered
from ommat._decide import EntryRefusal
from ommat._design import LoadedDesign
from ommat._interpret import _record_form, held_identity, warnings_conform
from ommat._ledger import Record, read_record
from ommat._markers import Cause, CauseCode, Failed, Unavailable
from ommat._read_view import _ViewError, walk
from ommat._refs import (
    _RefError,
    candidate_values,
    code_point_sorted,
    declared_ref,
    entry_ref,
    parse_ref,
)
from ommat._typed import decoded, is_a
from ommat._values import KEY_ORDER_FORMAT, clipped, string_data, type_name

_C = CauseCode
_RETURNED_MARKERS = frozenset(
    {
        _C.refused.value,
        _C.invalid_answer.value,
        _C.invalid_return.value,
        _C.unsupported.value,
        # under a returned call, timed-out and raised are the binding's provider request's
        _C.timed_out.value,
        _C.raised.value,
    }
)
_NO_KEY = object()
"""What a ref gets that the fixture leaves out of ``values``; a recorded null is a value."""


class RecordedCalls:
    """The record's call items by the sources they list, and its call-written entries by
    ``(call index, ref)``."""

    def __init__(self, log: list[dict[str, Any]]) -> None:
        self.log = log
        self.calls: dict[str, tuple[int, dict[str, Any]]] = {}
        self.entries: dict[tuple[int, str], dict[str, Any]] = {}
        for index, item in enumerate(log):
            if item["item"] == "call":
                # The ledger refuses a source in two call items, so no source is overwritten.
                for sid in cast("list[str]", item["sources"]):
                    self.calls[sid] = (index, item)
            elif item["item"] == "entry" and item["call"] is not None:
                self.entries[(cast(int, item["call"]), cast(str, item["ref"]))] = item

    def call_of(self, sid: str) -> tuple[int, dict[str, Any]] | None:
        return self.calls.get(sid)


class _Same:
    """An object compared by identity, held so its ``id`` is not reused. A log so held is
    taken as immutable once a binding is built from it, so an evaluation job passes one log
    object, read once, to every run of a record."""

    __slots__ = ("held",)

    def __init__(self, held: object) -> None:
        self.held = held

    def __eq__(self, other: object) -> bool:
        return type(other) is _Same and other.held is self.held

    def __hash__(self) -> int:
        return id(self.held)


@dataclass(frozen=True, slots=True)
class Stale:
    """Why a ref is stale: its clause, ``no-answer``, ``needs``, ``question``, ``payload`` or
    ``oracle``, and a line saying why."""

    clause: str
    why: str


class Base:
    """The base record's calls and entries, read once, against which a source asked under
    ``loaded`` is checked; ``design``, the design the base was decided under, ``loaded`` when
    ``None``, gives each source's payload, ``oracle`` key and ``t:`` types; each source in
    ``asked`` is asked afresh."""

    def __init__(
        self,
        log: list[dict[str, Any]],
        loaded: LoadedDesign,
        *,
        design: LoadedDesign | None = None,
        asked: frozenset[str] = frozenset(),
    ) -> None:
        self.calls = RecordedCalls(log)
        self.entries = {cast(str, i["ref"]): i for i in log if i["item"] == "entry"}
        # A record before format 6 holds no order, so its needs are compared without it.
        self._held = log[0]["format"] >= KEY_ORDER_FORMAT
        # Each candidate set's candidate field, by which its stored labels are rendered; a
        # record before format 7 holds only list<string> sets.
        self._sets = cast("dict[str, str | None]", log[1].get("sets", {}))
        self._loaded = loaded
        self._design = loaded if design is None else design
        self._asked = asked
        self.key = (_Same(log), None if design is None else design.design_hash, asked)
        """How the base was built: its log, by identity, its design's hash, ``None`` for the
        design a source is asked under, and the sources asked afresh. The log is taken as
        immutable from here on: a job passes one log object to every run of a record."""

    def check(self, source: SourceCall) -> tuple[Stale | None, dict[str, Stale]]:
        """Whether ``source`` is stale whole, and why, else each stale ref of it, by the first
        clause that holds: a source with no call item listing it is ``no-answer`` and one whose
        needs differ ``needs``; a question whose kind, labels, value type, list flag or identity
        differ, or a ``t:`` ref whose declared type differs from the base design's, is
        ``question``, and a ref the call wrote no entry for ``no-answer``; every other ref is
        ``payload`` when the source's payload differs from the base design's, then ``oracle``
        when its ``oracle`` key differs or it is asked afresh. A recorded ``Failed`` call is the
        bindings' to judge."""
        found = self.calls.call_of(source.id)
        if found is None:
            return Stale("no-answer", "the record holds no call for it"), {}
        differ = [need for need, value in source.needs.items() if self.differs(need, value)]
        if differ:
            shown = ", ".join(clipped(need) for need in differ)
            return Stale("needs", f"needs differ from the stored turn's: {shown}"), {}
        index, item = found
        stored = cast("dict[str, Any]", item["questions"])
        rest = self._rest(source.id)
        stale: dict[str, Stale] = {}
        for ref in (*source.refs, *source.questions):
            if ref in source.questions:
                why = self._question(ref, source.questions[ref], stored)
                if why:
                    stale[ref] = Stale("question", why)
                    continue
            elif self._retyped(ref):
                stale[ref] = Stale("question", "its declared type differs from the base's")
                continue
            if (index, ref) not in self.calls.entries:
                stale[ref] = Stale("no-answer", "the recorded call wrote no entry for it")
            elif rest is not None:
                stale[ref] = rest
        return None, stale

    def _rest(self, sid: str) -> Stale | None:
        """Why every ref of source ``sid`` is stale by its payload or its oracle, or ``None``."""
        here = self._loaded._design.sources.get(sid)
        there = self._design._design.sources.get(sid)
        if here is not None and there is not None:
            if self._loaded._payload_bytes[sid] != self._design._payload_bytes[sid]:
                return Stale("payload", "its payload differs from the base design's")
            if here.oracle != there.oracle:
                return Stale("oracle", "its oracle key differs from the base design's")
        if sid in self._asked:
            return Stale("oracle", "the scenario asks it afresh")
        return None

    def _retyped(self, ref: str) -> bool:
        """Whether a ``t:`` ref the base design declares has another declared type; a ref it
        does not declare has no entry in its record, so it is ``no-answer``."""
        declared = declared_ref(ref)
        held = self._design._types.get(declared)
        return held is not None and self._loaded._types.get(declared) != held

    def _question(self, ref: str, asked: RenderedQuestion, stored: dict[str, Any]) -> str:
        """Why the question asked at ``ref`` is not the call item's, or empty when it is."""
        declared = declared_ref(ref)
        held = cast("dict[str, Any] | None", stored.get(declared))
        why = _differing(asked, held, self.entries, self._sets)
        if why or held is None:
            return why
        if held_identity(self._loaded, declared) != held["identity"]:
            return "its identity differs from the stored question's"
        return ""

    def differs(self, need: str, value: object) -> bool:
        """Whether a need's live value differs from the record's entry at its base, read through
        its field path, by its ordered encoding, or by canonical bytes for a record before format
        6; a marker, a missing entry or path, a need that is not a ref and a value outside the
        value domain differ."""
        try:
            entry = self.entries.get(entry_ref(need))
            if entry is None or "value" not in entry:
                return True
            stored = walk(entry["value"], parse_ref(need).fields, need)
            live = _record_form(value)
            if self._held:
                return ordered(stored) != ordered(live)
            return encode(stored) != encode(live)
        except (_RefError, _ViewError, _CanonicalError):
            return True


class FixtureBinding(Replaying):
    """An oracle binding answering from ``answers``, a record's calls, under a given design,
    each source checked against ``base``. A ref in ``pinned`` is answered from ``answers`` and
    never checked: an edit set it. A source in ``failed``, whose call an edit failed, is never
    checked against ``base``; its failure comes back as for any recorded ``Failed`` call, whole
    or not at all. A ref in ``unpinned`` is stale, ``no-answer``, unless an earlier clause holds:
    the run gives no answer for it, as a gold record gives none to a question with no gold
    answer.

    Called with a live binding, as a scenario's is, a source with a stale ref is passed whole
    to it in a sub-batch, and so is stale whole, a ref fresh alone taking its source's first
    clause; the rest are answered from ``answers``, and the two merge into the live
    ``CallResult``, a pinned ref of a live source answered from ``answers``. When any ref was
    so answered and the live warnings conform, they are led by ``answered from record <note>``,
    a line per ref so answered and a line per such ref the live ``truncated`` named, which is
    dropped from it. A live result that is not a ``CallResult`` holding a ``dict``, a ``Failed``
    among them, is returned as it is, failing the whole call, and a live value for a
    batchmate's ref fails it ``unexpected-ref``; either way no ref is counted pinned. A wholly
    pinned batch is answered as offline, with null provenance."""

    def __init__(
        self,
        answers: RecordedCalls,
        loaded: LoadedDesign,
        base: Base,
        *,
        pinned: frozenset[str] = frozenset(),
        failed: frozenset[str] = frozenset(),
        unpinned: frozenset[str] = frozenset(),
        note: str = "",
    ) -> None:
        self._recorded = answers
        self._note = note
        self._unpinned = unpinned
        self._types = loaded._types
        self._base = base
        self._pinned = pinned
        self._failed = failed
        self.stale: dict[str, tuple[str, Stale]] = {}
        """Each stale ref, given no key or asked live, with its source's id and why."""
        self.pinned: set[str] = set()
        """Each ref given an answer from the record, a recorded failure given back among them;
        whether the call then stood is read from the record the turn wrote."""
        self.reproduced: set[str] = set()
        """Each ref of a recorded failure given back."""
        self._construction = (_Same(answers.log), base.key, pinned, failed, unpinned)

    def construction(self) -> object:
        """How the binding was built: its answers' log, by identity, its base's key and the
        edit sets; equal between bindings built alike over designs that differ in leaf values
        alone, the design a source is asked under being no part of it."""
        return self._construction

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        return await self.call(batch, seconds, None)

    def effects(self, sources: frozenset[str], refs: frozenset[str]) -> object:
        """What a call of ``sources``, providing ``refs``, noted: each stale ref of theirs with
        its source and why, in the order noted, and the refs given an answer or a recorded
        failure."""
        stale = [(ref, why) for ref, why in self.stale.items() if why[0] in sources]
        return (stale, refs & self.pinned, refs & self.reproduced)

    def restore(self, effects: object) -> None:
        """Note again what ``effects`` says a call noted."""
        stale, pinned, reproduced = cast(
            "tuple[list[tuple[str, tuple[str, Stale]]], frozenset[str], frozenset[str]]",
            effects,
        )
        self.stale.update(stale)
        self.pinned.update(pinned)
        self.reproduced.update(reproduced)

    async def call(
        self, batch: tuple[SourceCall, ...], seconds: float, live: OracleBinding | None
    ) -> object:
        """The batch answered from the record; a source with a stale ref given no key, or,
        with ``live``, passed whole to ``live`` in a sub-batch under the same ``seconds``,
        every ref of it but a pinned one then stale."""
        stale: dict[str, dict[str, Stale]] = {s.id: self._stale(s) for s in batch}
        calls = [self._recorded.call_of(s.id) for s in batch]
        indices = {c[0] for c in calls if c is not None}
        fresh = not any(stale.values())
        if None not in calls and len(indices) == 1:
            item = cast("tuple[int, dict[str, Any]]", calls[0])[1]
            outcome = item["outcome"]
            # A failure comes back only for the call as recorded: every source, none stale.
            whole = {s.id for s in batch} == set(cast("list[str]", item["sources"]))
            if outcome != "returned" and fresh and whole:
                failed = cast("dict[str, Any]", outcome)["failed"]
                refs = [r for s in batch for r in (*s.refs, *s.questions)]
                self.pinned.update(refs)
                self.reproduced.update(refs)
                return Failed(Cause(failed["code"], list(failed["refs"])))
        values: dict[str, Any] = {}
        asked: list[SourceCall] = []
        held: list[str] = []
        for source, found in zip(batch, calls, strict=True):
            mine = stale[source.id]
            refs = (*source.refs, *source.questions)
            if found is not None and found[1]["outcome"] != "returned":
                # A Failed call asked with other sources, without some of its own, or beside a
                # stale batchmate cannot be reproduced whole: what was not pinned is stale.
                why = Stale("no-answer", "its recorded call failed and is not reproduced whole")
                for ref in refs:
                    if ref not in mine and ref not in self._pinned:
                        mine[ref] = why
            if live is not None and mine:
                # Asked live whole, so stale whole: a ref fresh alone takes its source's clause.
                first = next(r for r in refs if r in mine)
                why = Stale(mine[first].clause, f"asked with its source, stale at {first}")
                for ref in refs:
                    if ref not in mine and ref not in self._pinned:
                        mine[ref] = why
                asked.append(source)
            for ref in refs:
                if ref in mine:
                    self.stale[ref] = (source.id, mine[ref])
            if source in asked:
                continue  # its pinned refs are answered after the live call
            if found is None or found[1]["outcome"] != "returned":
                continue  # no call, or a Failed call not reproduced: no key
            for ref in refs:
                if ref not in mine:
                    self._give(ref, found[0], values, held)
        if live is None or not asked:
            self.pinned.update(held)
            return CallResult(values)
        return self._merged(await live(tuple(asked), seconds=seconds), batch, asked, values, held)

    def _merged(
        self,
        result: object,
        batch: tuple[SourceCall, ...],
        asked: list[SourceCall],
        values: dict[str, Any],
        held: list[str],
    ) -> object:
        """The live result with the record's answers merged in, or as it is when it fails the
        call."""
        if type(result) is not CallResult or type(result.values) is not dict:
            return result  # interpret judges it, failing the whole call
        own = {r for s in asked for r in (*s.refs, *s.questions)}
        others = {r for s in batch for r in (*s.refs, *s.questions)} - own
        given = cast("dict[object, object]", result.values)
        named = {k: string_data(cast(str, k)) for k in given if is_a(k, str)}
        wrong = code_point_sorted(ref for ref in named.values() if ref in others)
        if wrong:
            # The live binding answered a ref of a batchmate it was not asked.
            return Failed(Cause(CauseCode.unexpected_ref, wrong))
        for source in asked:
            found = self._recorded.call_of(source.id)
            for ref in (*source.refs, *source.questions):
                if ref in self._pinned:
                    # A pinned ref keeps its set value, or no key when its call is not recorded.
                    given = {k: v for k, v in given.items() if named.get(k) != ref}
                    if found is not None:
                        self._give(ref, found[0], values, held)
        merged: dict[Any, Any] = {**given, **values}
        truncated: object = result.truncated
        dropped: list[str] = []
        if type(truncated) is tuple:
            # Only a pinned question of an asked source is dropped; any other wrong member is
            # left for interpret to judge.
            ours = {q for s in asked for q in s.questions if q in held}
            members = cast("tuple[object, ...]", truncated)
            dropped = [m for m in members if type(m) is str and m in ours]
            truncated = tuple(m for m in members if not (type(m) is str and m in ours))
        warnings: object = result.warnings
        if held and warnings_conform(warnings):
            lines = [f"answered from record {self._note}"]
            lines += [f"{clipped(ref)}: pinned" for ref in held]
            lines += [f"{clipped(ref)}: named truncated, answered from record" for ref in dropped]
            warnings = bounded_warnings([*lines, *warnings])
        self.pinned.update(held)
        return replace(result, values=merged, truncated=truncated, warnings=warnings)

    def _give(self, ref: str, index: int, values: dict[Any, Any], held: list[str]) -> None:
        """The recorded answer to ``ref`` under the call at ``index``, when it has one a binding
        may give, put in ``values`` and listed in ``held``."""
        entry = self._recorded.entries.get((index, ref))
        if entry is not None:
            answer = self._answer(ref, entry)
            if answer is not _NO_KEY:
                values[ref] = answer
                held.append(ref)

    def _stale(self, source: SourceCall) -> dict[str, Stale]:
        """Each stale ref of ``source`` that no edit set, an unpinned ref among them."""
        if source.id in self._failed:
            return {}
        whole, refs = self._base.check(source)
        if whole is not None:
            refs = dict.fromkeys((*source.refs, *source.questions), whole)
        given = Stale("no-answer", "the run gives no answer for it")
        for ref in (*source.refs, *source.questions):
            if ref in self._unpinned:
                refs.setdefault(ref, given)
        return {r: why for r, why in refs.items() if r not in self._pinned}

    def _answer(self, ref: str, entry: Mapping[str, Any]) -> object:
        """The recorded entry's value, typed, a ``q:`` value in the binding's form of its
        recorded kind, or one of the six markers a binding may return; ``_NO_KEY`` for any
        other marker."""
        if "value" in entry:
            value = entry["value"]
            if ref.startswith("q:"):
                # The ledger refuses a q: value that is not an answer's object of its recorded
                # kind, live and read back, so each has that kind's members.
                return _binding_form(cast("dict[str, Any]", value))
            return decoded(value, self._types[declared_ref(ref)])
        marker = cast("dict[str, Any]", entry["marker"])
        code = marker["cause"]["code"]
        if marker["kind"] == "Unavailable" and code in _RETURNED_MARKERS:
            return Unavailable(Cause(code, [ref]))
        return _NO_KEY


def _float(value: object) -> object:
    """An integral number read back as an ``int`` made the ``float`` it was written as."""
    return float(value) if type(value) is int else value


def _binding_form(value: dict[str, Any]) -> dict[str, object]:
    """A recorded answer as a binding returns it, by its recorded kind: a probability kind's
    ``probabilities``, each a ``float``; an extraction answer's five members, ``p_null`` and
    ``confidence`` each a ``float``, under a recorded ``number`` value type each number in
    ``value`` and ``guess`` a ``float``, and each container a fresh list. Interpret validates it
    against the question now asked."""
    if value["kind"] != AnswerKind.extraction:
        probabilities = cast("dict[str, object]", value["probabilities"])
        return {label: _float(p) for label, p in probabilities.items()}
    number = value["value_type"] == ValueType.number

    def member(found: object) -> object:
        if type(found) is list:
            return [member(m) for m in cast("list[object]", found)]
        return _float(found) if number else found

    def pair(found: object) -> object:
        if type(found) is list:
            return [pair(m) for m in cast("list[object]", found)]
        return found

    return {
        "value": member(value["value"]),
        "guess": member(value["guess"]),
        "p_null": _float(value["p_null"]),
        "confidence": _float(value["confidence"]),
        "span": pair(value["span"]),
    }


# ---------------------------------------------------------------------------------------------
# from_record: a caller's live turn answered from a record

_KEPT_MARKERS = frozenset({_C.invalid_answer.value, _C.invalid_return.value})
"""The recorded markers ``from_record`` returns as recorded; any other is ``unsupported``."""


def from_record(record: object, loaded: LoadedDesign) -> OracleBinding:
    """An oracle binding answering a caller's live turn under ``loaded`` from ``record``, a
    ``Record`` or a stored document read through ``read_record``.

    Bound under the oracle names whose sources the record answers, beside live bindings under
    the rest, it answers each asked source from the returned call item that lists it, a
    per-candidate question's stored entry found under its declared ref, each ref
    from the entry that call wrote, in the binding's form. It refuses with ``unsupported``,
    each refusal a line of ``warnings``: every ref of a source the record holds no returned
    call for, or whose ``needs`` differ from the record's entries by their ordered encoding, or
    by canonical bytes for a record before format 6; a
    question whose kind, labels as declared, value type or list flag differ from the call
    item's, a candidate-valued set's labels as its fixed part and its set's stored entry give
    them, or whose set's stored entry is missing or a marker; a ref with no recorded entry
    naming its call; and a ref whose recorded marker is neither ``invalid-answer`` nor
    ``invalid-return``, which come back as recorded. Its provenance is null and its first
    warnings line names the record's hash.

    The checks are the fixture's (``Base.check``): a question's identity under ``loaded`` is
    compared with the call item's too, so a question reworded under another design is refused.
    It cannot see a label set's ref, so a set named otherwise holding the same members is
    answered."""
    if type(loaded) is not LoadedDesign:
        raise EntryRefusal(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
    read = record if type(record) is Record else read_record(record)
    return _FromRecord(read, loaded)


class _FromRecord:
    """``from_record``'s binding: the record's calls and entries, read once."""

    def __init__(self, record: Record, loaded: LoadedDesign) -> None:
        self._base = Base(record.log, loaded)  # read once: every read decodes afresh
        self._types = loaded._types
        self._note = f"answered from record {record.record_hash}"

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        # It waits for nothing, so the call's seconds are not read.
        values: dict[str, object] = {}
        lines: list[str] = []
        for source in batch:
            self._source(source, values, lines)
        return CallResult(values, warnings=bounded_warnings([self._note, *lines]))

    def _source(self, source: SourceCall, values: dict[str, object], lines: list[str]) -> None:
        refs = (*source.refs, *source.questions)
        unsupported = Unavailable(Cause(_C.unsupported))
        found = self._base.calls.call_of(source.id)
        if found is not None and found[1]["outcome"] != "returned":
            code = cast("dict[str, Any]", found[1]["outcome"])["failed"]["code"]
            values.update(dict.fromkeys(refs, unsupported))
            lines.append(f"source {clipped(source.id)}: its recorded call failed with {code}")
            return
        whole, stale = self._base.check(source)
        if whole is not None:
            values.update(dict.fromkeys(refs, unsupported))
            lines.append(f"source {clipped(source.id)}: {whole.why}")
            return
        assert found is not None  # the check finds a source with no call stale whole
        index = found[0]
        for ref in refs:
            if ref in stale:
                values[ref] = unsupported
                lines.append(f"{clipped(ref)}: {stale[ref].why}")
                continue
            entry = self._base.calls.entries[(index, ref)]
            if "value" in entry:
                values[ref] = (
                    _binding_form(cast("dict[str, Any]", entry["value"]))
                    if ref.startswith("q:")
                    else decoded(entry["value"], self._types[declared_ref(ref)])
                )
            else:
                marker = cast("dict[str, Any]", entry["marker"])
                code = cast(str, marker["cause"]["code"])
                if marker["kind"] == "Unavailable" and code in _KEPT_MARKERS:
                    values[ref] = Unavailable(Cause(code, [ref]))
                else:
                    values[ref] = unsupported
                    lines.append(f"{clipped(ref)}: recorded {code}, no answer to give")


def _differing(
    asked: RenderedQuestion,
    stored: dict[str, Any] | None,
    entries: Mapping[str, Mapping[str, Any]],
    sets: Mapping[str, str | None],
) -> str:
    """Why the asked question is not the call item's ``questions`` entry, or empty when it is.
    A candidate-valued set's stored labels are its fixed part, then the candidate values of
    its set's entry in the same record, in order, by the record's ``sets``; a set with no
    stored value refuses the question."""
    if stored is None:
        return "its kind, labels, value type and list flag differ from the stored question's"
    kind = AnswerKind(asked.kind)
    labels: object = None
    if kind is not AnswerKind.extraction:
        labels = [
            name if description is None else {"name": name, "description": description}
            for name, description in zip(asked.labels, asked.descriptions, strict=True)
        ]
    if kind is AnswerKind.noul and all(d is None for d in asked.descriptions):
        labels = None  # an undescribed noul declares no labels
    recorded: object = stored["labels"]
    if type(recorded) is dict:  # a candidate-valued set, as declared: {fixed, candidates}
        declared = cast("dict[str, Any]", recorded)
        set_ref = cast(str, declared["candidates"])
        entry = entries.get(set_ref)
        if entry is None:
            return f"its candidate set {clipped(set_ref)} has no stored entry"
        if "value" not in entry:
            marker = cast("dict[str, Any]", entry["marker"])
            return (
                f"its candidate set {clipped(set_ref)} is stored as "
                f"{marker['kind']} {marker['cause']['code']}"
            )
        value = entry["value"]
        # Anything but a list of the set's form gives no labels: they differ, and nothing
        # raises.
        try:
            listed: list[str] | None = candidate_values(value, sets.get(set_ref))
        except (KeyError, TypeError):
            listed = None
        k = cast(int, declared["at"])
        fixed = cast("list[object]", declared["fixed"])
        recorded = (
            [*fixed[:k], *listed, *fixed[k:]]
            if type(value) is list and listed is not None
            else None
        )
    found: list[str] = []
    if stored["kind"] != kind.value:
        found.append("kind")
    if recorded != labels:
        found.append("labels")
    value_type = None if asked.value_type is None else ValueType(asked.value_type).value
    if stored.get("value_type") != value_type:
        found.append("value type")
    if stored.get("is_list", False) != asked.is_list:
        found.append("list flag")
    return f"its {', '.join(found)} differ from the stored question's" if found else ""


class Keyed:
    """The fixture under one oracle name, asking stale sources of the caller's binding under
    that name."""

    def __init__(self, fixture: FixtureBinding, live: OracleBinding) -> None:
        self._fixture = fixture
        self._live = live

    async def __call__(self, batch: tuple[SourceCall, ...], *, seconds: float) -> object:
        return await self._fixture.call(batch, seconds, self._live)
