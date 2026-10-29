# pyright: reportPrivateUsage=false
"""The switching-value job: for one parameter leaf, the values at which each entry, and the
plan with the verdict, change, found by replaying the turn under each value's design
(§switching-value).

A value's design is the loaded design's document with the leaf, or one key of a numeric
mapping leaf, replaced, loaded with the caller's ``functions``; it is kept nowhere. Replay
under it reports ``design-mismatch``, which is expected and dropped, as is a ``plan`` finding of
context alone. A run that asks outside the record, ``not-in-record``, ``ref-not-in-record`` or
``question-changed``, or writes ``Unavailable`` with ``not-returned`` for a ref whose recorded
entry names a ``Failed`` call, is ``not-determined``; a value whose design fails load is
``not-loading``, with each failure as plain data; the run goes on past both.

A numeric leaf is tried at the leaf's own value, whose result is the record's, and at the
points of §lint-points over the compared values as boundaries, within the leaf's type; each
replay's record adds the compared values it recomputes, until no new point appears. A compared
value is: for a table cell comparing an input against the leaf, that input's value, a bare
read's entry through its field path, or ``mass``, ``size`` or ``extract`` over its read; for a
pick helper's ``bar`` or ``thresholds`` binding, each label's probability; for ``extract``'s
``null_bar``, ``p_null`` when not ``None``; for a ``margin``, the greatest ``float`` whose
decimal does not exceed the top label's lead over the runner-up; for a function component's
``boundaries`` entry naming the leaf, the read's value, or the same role measure of its
answer. A ``boolean`` or ``label`` leaf is tried at every value.

Per entry that some determined run changed, and for the plan with the verdict, the results are
``"carries"`` when every determined run differs from every other, over at least three values
or by a table passing the leaf through; else the classes and, for a numeric leaf, the
switching values and slacks. A switching value lies between two adjacent classes of
determined results: the lower class's greatest when it alone is a compared value, else the
upper class's least; a boundary beside a ``not-determined`` or ``not-loading`` class is none.
A slack's side takes a switching value equal to the leaf's own value, so both are then 0. A
result is ``{"entry": item}``, the item null where the run wrote none. ``unseen`` names, in
code-point order, each function rule, or ``policy``, declaring the leaf with no ``boundaries``
field; it is empty for a finite leaf and for an integer interval tried at every value.

Refused in §switching-value's order; the record checks are ``_edit``'s, and the baseline
replay is ``_diff``'s. The entry written is returned, plain data reaching no ``Record``.
"""

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode, encode, order_of
from ommat._design import InvalidDesign, LoadedDesign
from ommat._load import load
from ommat._read_view import _ViewError, is_candidate_read, walk
from ommat._refs import _RefError, code_point_sorted, parse_ref
from ommat._replay import FindingKind, _replay_with
from ommat._types import Interval, Label, MappingOf, Scalar, Type, render
from ommat._values import MAX_DOCUMENT_DEPTH
from ommat.evaluation._diff import _context_alone, _replayed, _run_sync
from ommat.evaluation._edit import _checked, _declared
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore
from ommat.helpers._numbers import exact
from ommat.helpers._points import _end, axis

_R = RefusalReason
_K = FindingKind
_Log = list[dict[str, Any]]
_Functions = Mapping[str, Callable[..., Any]] | None
_Value = int | float | bool | str
_Result = dict[str, Any] | str

KIND = "switching-value"
_NOT_DETERMINED = "not-determined"
_OUTSIDE = frozenset({_K.not_in_record, _K.ref_not_in_record, _K.question_changed})
_EXPECTED = frozenset({_K.design_mismatch})


async def switching_values(
    record: object,
    loaded: LoadedDesign,
    path: str,
    *,
    functions: _Functions,
    results: EvaluationStore,
) -> dict[str, Any]:
    """The switching values of the leaf at ``path`` for ``record``, per entry and for the plan,
    written to ``results`` under the record hash, the design hash and ``path``; the entry
    written."""
    read, log = _checked(record, loaded)
    t = _leaf_type(path, loaded)
    try:
        load(loaded.document, functions=functions)
    except InvalidDesign as exc:
        raise EvaluationRefusal(
            f"the loaded design's document does not load with functions: {exc}",
            reason=_R.invalid_design,
            error=exc,
        ) from exc
    key = {"record": read.record_hash, "design": loaded.design_hash, "parameter": path}
    if results.lookup(KIND, key) is not None:
        raise EvaluationRefusal(
            f"the store holds a {KIND} entry under {key!r}", reason=_R.held_key, key=[key]
        )
    found = await _replayed(log, loaded, log)
    if found:
        kinds = ", ".join(dict.fromkeys(str(f["kind"]) for f in found))
        raise EvaluationRefusal(
            f"replay under the loaded design differs from the record: {kinds}",
            reason=_R.baseline_differs,
        )
    job = _Job(log, loaded, path, functions)
    value: dict[str, Any]
    if isinstance(t, MappingOf):
        held = cast("dict[str, Any]", loaded._leaf_value(path))
        value = {"unseen": job.unseen(t.value, {}), "keys": {}}
        for k in code_point_sorted(held):
            v0 = _own(held[k], t.value)
            tried, bounds = await job.tried(t.value, v0, k)
            value["keys"][k] = job.summary(t.value, v0, tried, bounds)
    else:
        v0 = _own(loaded._leaf_value(path), t)
        tried, bounds = await job.tried(t, v0, None)
        value = {"unseen": job.unseen(t, tried), **job.summary(t, v0, tried, bounds)}
    entry = {"kind": KIND, "format": 1, "key": key, "value": value}
    results.write(entry)
    return entry


def switching_values_sync(
    record: object,
    loaded: LoadedDesign,
    path: str,
    *,
    functions: _Functions,
    results: EvaluationStore,
) -> dict[str, Any]:
    """``switching_values`` run to completion; ``RuntimeError`` inside a running event loop."""
    return _run_sync(
        "switching_values",
        lambda: switching_values(record, loaded, path, functions=functions, results=results),
    )


def _leaf_type(path: object, loaded: LoadedDesign) -> Type:
    """The leaf's type; ``unknown-path`` for a node, a path not in the tree or a malformed one,
    then ``unsupported-leaf`` for a type the job does not vary."""
    if type(path) is not str or path not in loaded._leaf_types:
        raise EvaluationRefusal(
            f"path {path!r} names no leaf of the parameter tree",
            reason=_R.unknown_path,
            path=path if type(path) is str else None,
        )
    t = loaded._leaf_types[path]
    if not (
        _numeric(t) or _finite(t) is not None or (isinstance(t, MappingOf) and _numeric(t.value))
    ):
        raise EvaluationRefusal(
            f"the leaf at {path} is of type {render(t)}, which the job does not vary",
            reason=_R.unsupported_leaf,
            path=path,
        )
    return t


def _own(value: Any, t: Type) -> Any:
    """The leaf's own value as a tried value: a ``number`` as a ``float``."""
    return float(value) if _numeric(t) and not _integer(t) else value


def _numeric(t: Type) -> bool:
    return isinstance(t, Interval) or (isinstance(t, Scalar) and t.name in ("number", "integer"))


def _integer(t: Type) -> bool:
    return (isinstance(t, Scalar) and t.name == "integer") or (
        isinstance(t, Interval) and t.base == "integer"
    )


def _finite(t: Type) -> list[_Value] | None:
    """Every value of a ``boolean`` or ``label`` leaf, in declared order; ``None`` for any
    other."""
    if isinstance(t, Label):
        return list(t.values)
    if isinstance(t, Scalar) and t.name == "boolean":
        return [False, True]
    return None


class _Job:
    """One run's record, design and leaf; ``tried`` replays, ``summary`` reports."""

    def __init__(self, log: _Log, loaded: LoadedDesign, path: str, functions: _Functions) -> None:
        self.log, self.loaded, self.path, self.functions = log, loaded, path, functions
        self.document = cast("dict[str, Any]", decode(loaded.document))
        self.recorded = _Run.of(log, [])
        self.mine = {i["ref"]: i for i in log if i["item"] == "entry"}

    # --- the values tried ------------------------------------------------------------------

    async def tried(
        self, t: Type, v0: Any, key: str | None
    ) -> tuple[dict[_Value, "_Run | _Result"], set[Fraction]]:
        """Each value tried, the leaf's own first, to its run or its failure, and the compared
        values every run gave."""
        out: dict[_Value, _Run | _Result] = {v0: self.recorded}
        values = _finite(t)
        if values is not None:
            for v in values:
                if v not in out:
                    out[v] = (await self.run(v, key))[0]
            return out, set()
        bounds = self.compared(self.log)
        while new := [v for v in _points(bounds, t) if v not in out]:
            for v in new:
                out[v], rerun = await self.run(v, key)
                if rerun is not None:
                    bounds |= self.compared(rerun)
        return out, bounds

    async def run(self, v: _Value, key: str | None) -> tuple["_Run | _Result", _Log | None]:
        """The run at ``v``, or its failure, and the replay's log when one ran."""
        document = cast("dict[str, Any]", decode(self.loaded.document))
        leaf = document["parameters"][self.path]
        value: Any = v if key is None else {**leaf["value"], key: v}
        leaf["value"] = value
        if "order" in leaf:
            leaf["order"] = order_of(value)
        try:
            design = load(document, functions=self.functions)
        except InvalidDesign as exc:
            failures = [
                {"check": f.check, "names": list(f.names), "message": f.message}
                for f in exc.failures
            ]
            return {"not-loading": failures}, None
        rerun, found = await _replay_with(self.log, design, self.log, observer=None)
        rerun_log = rerun.log
        found = [f for f in found if f["kind"] not in _EXPECTED and not _context_alone(f)]
        if any(f["kind"] in _OUTSIDE for f in found) or self.unreturned(rerun_log):
            return _NOT_DETERMINED, rerun_log
        return _Run.of(self.log, found), rerun_log

    def unreturned(self, rerun: _Log) -> bool:
        """Whether the rerun writes ``Unavailable`` with ``not-returned`` for a ref whose
        recorded entry names a call item recorded ``Failed``."""
        for item in rerun:
            marker = item.get("marker") if item["item"] == "entry" else None
            if marker is None or marker["kind"] != "Unavailable":
                continue
            if marker["cause"]["code"] != "not-returned":
                continue
            call = (self.mine.get(item["ref"]) or {}).get("call")
            if call is not None and self.log[call]["outcome"] != "returned":  # recorded Failed
                return True
        return False

    # --- the compared values ---------------------------------------------------------------

    def compared(self, log: _Log) -> set[Fraction]:
        """The values ``log``'s turn compares the leaf against, exactly."""
        entries = [i for i in log if i["item"] == "entry"]
        out: list[object] = []
        for table in self.tables():
            inputs = cast("list[dict[str, Any]]", table["inputs"])
            for n, i in enumerate(inputs):
                for role, bound in cast("dict[str, Any]", i["params"]).items():
                    if bound == self.path:
                        for v in _read_values(i["read"], entries):
                            out += _measure(_ROLES.get((i["helper"], role)), v, ())
                if any(_names(row["cells"][n], self.path) for row in table["rows"]):
                    for v in _read_values(i["read"], entries):
                        out += self.input_value(i, v)
        for boundaries in self.boundary_fields():
            for read, entry in boundaries.items():
                values = _read_values(read, entries)
                if isinstance(entry, list):
                    if self.path in entry:
                        out += values
                    continue
                roles = cast("dict[str, Any]", entry)
                for role, members in roles.items():
                    if role == "mass":
                        for m in cast("list[dict[str, Any]]", members):
                            if self.path in m["bounds"]:
                                out += [x for v in values for x in _measure("mass", v, m["labels"])]
                    elif self.path in members:
                        out += [x for v in values for x in _measure(role, v, ())]
        return {exact(cast(float, x)) for x in out if type(x) in (int, float)}

    def tables(self) -> list[dict[str, Any]]:
        policy = self.document["policy"]
        found = [r for r in self.document["rules"].values() if r["form"] == "table"]
        return found + ([policy] if policy["form"] == "table" else [])

    def functions_declaring(self) -> dict[str, dict[str, Any]]:
        """Each function rule, by id, and a function policy, as ``policy``."""
        policy = self.document["policy"]
        found = {n: r for n, r in self.document["rules"].items() if r["form"] == "function"}
        return {**found, **({"policy": policy} if policy["form"] == "function" else {})}

    def boundary_fields(self) -> list[dict[str, Any]]:
        return [
            c["boundaries"]
            for c in self.functions_declaring().values()
            if c["boundaries"] is not None
        ]

    def input_value(self, i: dict[str, Any], v: object) -> list[object]:
        """A table input's value over its read's value ``v``."""
        helper = i["helper"]
        if helper is None:
            return [v]
        if helper == "mass":
            labels = self.document["parameters"][i["params"]["labels"]]["value"]
            return _measure("mass", v, labels)
        if helper == "size":
            return [len(cast("list[object]", v))] if type(v) is list else []
        if helper == "extract":
            return _measure("value", v, ())
        return []

    # --- the report ------------------------------------------------------------------------

    def unseen(self, t: Type, tried: Mapping[_Value, object]) -> list[str]:
        """The function components declaring the leaf with no ``boundaries`` field, unless
        every value of the leaf was tried."""
        if _finite(t) is not None:
            return []
        lo, hi = (
            (int(t.lo), int(t.hi)) if isinstance(t, Interval) and t.base == "integer" else (1, 0)
        )
        if lo <= hi and all(n in tried for n in range(lo, hi + 1)):
            return []
        return code_point_sorted(
            name
            for name, c in self.functions_declaring().items()
            if self.path in c["params"] and c["boundaries"] is None
        )

    def summary(
        self, t: Type, v0: Any, tried: Mapping[_Value, "_Run | _Result"], bounds: set[Fraction]
    ) -> dict[str, Any]:
        """``{"entries", "plan"}`` over the values tried."""
        runs = [r for r in tried.values() if isinstance(r, _Run)]
        refs = code_point_sorted({ref for r in runs for ref in r.entries})
        values = _finite(t)
        ordered = (
            values if values is not None else sorted(tried, key=lambda v: exact(cast(float, v)))
        )
        out: dict[str, Any] = {"entries": {}, "plan": None}
        for ref in [*refs, None]:
            results = [_result(tried[v], ref, self.mine) for v in ordered]
            through = ref is not None and self.passes(ref)
            if _carries(results, through):
                summary: _Result = "carries"
            elif values is not None:
                summary = {"classes": _partition(ordered, results)}
            else:
                summary = _numeric_summary(ordered, results, v0, bounds, _integer(t))
            if ref is None:
                out["plan"] = summary
            else:
                out["entries"][ref] = summary
        return out

    def passes(self, ref: str) -> bool:
        """Whether a table providing ``ref`` passes the leaf through in some row."""
        rules = self.document["rules"]
        return any(
            r["form"] == "table"
            and r["provides"] == _declared(ref)
            and any(row.get("output") == {"param": self.path} for row in r["rows"])
            for r in rules.values()
        )


class _Run:
    """One determined run: each changed entry's item by ref, and the plan and verdict."""

    def __init__(self, entries: dict[str, Any], plan: Any, verdict: Any) -> None:
        self.entries, self.plan, self.verdict = entries, plan, verdict

    @classmethod
    def of(cls, log: _Log, found: list[dict[str, Any]]) -> "_Run":
        changed = {f["kind"]: f["replayed"] for f in found if f["kind"] in (_K.plan, _K.verdict)}
        first = {
            k: next((i for i in log if i["item"] == k.value), None) for k in (_K.plan, _K.verdict)
        }
        return cls(
            {f["ref"]: f["replayed"] for f in found if f["kind"] is _K.entry},
            changed.get(_K.plan, first[_K.plan]),
            changed.get(_K.verdict, first[_K.verdict]),
        )


def _result(run: "_Run | _Result", ref: str | None, mine: Mapping[str, Any]) -> _Result:
    """A run's result for the entry at ``ref``, or for the plan at ``None``."""
    if not isinstance(run, _Run):
        return run
    if ref is None:
        return {"plan": run.plan, "verdict": run.verdict}
    if ref in run.entries:
        return {"entry": run.entries[ref]}
    return {"entry": mine.get(ref)}


def _determined(result: _Result) -> bool:
    return isinstance(result, dict) and "not-loading" not in result


def _key(result: _Result) -> bytes:
    """A result's identity: a plan compared without its ``context`` (§context-block)."""
    if isinstance(result, dict) and result.get("plan") is not None:
        plan = {k: v for k, v in result["plan"].items() if k not in ("context", "order")}
        result = {**result, "plan": plan}
    return encode(result, MAX_DOCUMENT_DEPTH)


def _carries(results: list[_Result], through: bool) -> bool:
    """Whether every determined result differs from every other, over three or more, or two
    or more when a table passes the leaf through."""
    keys = [_key(r) for r in results if _determined(r)]
    return len(keys) >= (2 if through else 3) and len(set(keys)) == len(keys)


def _partition(values: list[_Value], results: list[_Result]) -> list[dict[str, Any]]:
    """A finite leaf's classes, each value with every other giving the same result, in
    declared order."""
    classes: dict[bytes, dict[str, Any]] = {}
    for v, r in zip(values, results, strict=True):
        classes.setdefault(_key(r), {"values": [], "result": r})["values"].append(v)
    return list(classes.values())


def _numeric_summary(
    values: list[_Value], results: list[_Result], v0: Any, bounds: set[Fraction], integer: bool
) -> dict[str, Any]:
    """The classes of adjacent values with one result, the switching values between two
    determined classes, and the slacks about ``v0``."""
    classes: list[dict[str, Any]] = []
    for v, r in zip(values, results, strict=True):
        if classes and _key(classes[-1]["result"]) == _key(r):
            classes[-1]["greatest"] = v
        else:
            classes.append({"least": v, "greatest": v, "result": r})
    switching: list[_Value] = []
    for a, b in itertools.pairwise(classes):
        if _determined(a["result"]) and _determined(b["result"]):
            g, lo = a["greatest"], b["least"]
            seen = exact(cast(float, g)) in bounds and exact(cast(float, lo)) not in bounds
            switching.append(g if seen else lo)
    x0 = exact(v0)
    below = [s for s in switching if exact(cast(float, s)) <= x0]
    above = [s for s in switching if exact(cast(float, s)) >= x0]

    def slack(s: _Value) -> int | float:
        d = exact(cast(float, s)) - x0
        return int(d) if integer else float(d)

    return {
        "switching": switching,
        "slacks": [slack(below[-1]) if below else None, slack(above[0]) if above else None],
        "classes": classes,
    }


def _points(bounds: set[Fraction], t: Type) -> list[_Value]:
    """§lint-points's axis over ``bounds`` within ``t``, rounded once, each point once."""
    if not bounds:
        return []
    integer = _integer(t)
    lo = hi = None
    if isinstance(t, Interval):
        lo, hi = exact(t.lo), exact(t.hi)
    elif not integer:
        lo, hi = _end(min(bounds), -1), _end(max(bounds), 1)
    return list(
        dict.fromkeys(
            int(x) if integer else float(x) for x in axis(bounds, lo, hi, integer=integer)
        )
    )


# ---------------------------------------------------------------------------------------------
# Reading the values compared

_ROLES: dict[tuple[str | None, str], str] = {
    ("pick_with_margin", "bar"): "bar",
    ("pick_with_margin", "margin"): "margin",
    ("pick_with_thresholds", "thresholds"): "thresholds",
    ("pick_with_thresholds", "margin"): "margin",
    ("extract", "null_bar"): "null",
}


def _read_values(read: str, entries: _Log) -> list[object]:
    """The values of the entries a read names, a grounded declared ref's each, through its
    field path; a marker, a ``candidate`` read and a path the entry does not hold give none."""
    if is_candidate_read(read):
        return []
    try:
        parsed = parse_ref(read)
    except _RefError:
        return []
    declared = f"{parsed.level}:{parsed.name}"
    out: list[object] = []
    for e in entries:
        if "value" not in e or _declared(e["ref"]) != declared:
            continue
        try:
            out.append(walk(e["value"], parsed.fields, read))
        except _ViewError:
            continue
    return out


def _measure(role: str | None, v: object, labels: Iterable[str]) -> list[object]:
    """What a role compares in an answer ``v``: each label's probability for ``bar`` and
    ``thresholds``; the margin probe; ``mass`` over ``labels``; ``p_null``, ``confidence`` or
    ``value`` of an extraction."""
    if type(v) is not dict:
        return []
    answer = cast("dict[str, Any]", v)
    probs = cast("dict[str, float] | None", answer.get("probabilities"))
    if role in ("bar", "thresholds") and probs is not None:
        return list(probs.values())
    if role == "margin" and probs is not None:
        return [_margin_probe(probs)]
    if role == "mass" and probs is not None:
        total = sum((exact(probs[k]) for k in labels if k in probs), Fraction(0))
        return [float(min(total, Fraction(1)))]
    member = {"null": "p_null", "confidence": "confidence", "value": "value"}.get(role or "")
    if member is not None and "p_null" in answer:
        return [answer[member]]
    return []


def _margin_probe(probs: Mapping[str, float]) -> float:
    """The greatest ``float`` whose decimal does not exceed the top label's lead over the
    runner-up, 0 for a one-label answer's, measured as §choice-helpers measures it."""
    exacts = sorted((exact(p) for p in probs.values()), reverse=True)
    lead = exacts[0] - (exacts[1] if len(exacts) > 1 else Fraction(0))
    f = float(lead)
    while exact(f) > lead:
        f = math.nextafter(f, -math.inf)
    return f


def _names(cell: object, path: str) -> bool:
    """Whether a cell compares its input against the leaf at ``path``."""
    if type(cell) is not dict:
        return False
    ((tag, member),) = cast("dict[str, Any]", cell).items()
    if tag == "between":
        return any(o == {"param": path} for o in cast("list[object]", member))
    return tag in (">=", ">", "<=", "<") and member == {"param": path}
