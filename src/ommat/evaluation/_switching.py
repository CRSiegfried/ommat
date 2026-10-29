# pyright: reportPrivateUsage=false
"""The switching-value job: for one parameter leaf, the values at which each entry, and the
plan with the verdict, change, found by replaying the turn under each value's design
(§switching-value).

A value's design is the loaded design's document with the leaf, or one key of a numeric
mapping leaf, replaced, loaded with the caller's ``functions``; it is kept nowhere. Replay
under it reports ``design-mismatch``, expected and dropped, as is ``helpers-mismatch``, a note
on the helper version. A run that asks outside the record, ``not-in-record``,
``ref-not-in-record`` or ``question-changed``, or writes ``Unavailable`` with
``not-returned`` for a ref whose recorded entry names a ``Failed`` call, is ``not-determined``;
a value whose design fails load is ``not-loading``, each failure as plain data; the job goes
on past both.

A numeric leaf is tried at its own value, whose result is the record's, and in two rounds at
the points of §lint-points' axis within the leaf's type: round 0 over the compared values the
record holds, round 1 over those and the new ones round 0's replays recompute. On an integer
axis each compared value also adds the integers either side of it, clipped to the interval. A
compared value first recomputed in round 1 is not probed, and the component comparing it joins
``unseen``. With no compared value the axis is the interval's two bounds, else +-(2^53 - 1).
A compared value is: for a table cell comparing an input against the leaf, that input's value,
a bare read's entry through its field path, a ``candidate`` read's each member of its set's
entry, or ``mass``, ``size`` or ``extract`` over its read; for a pick helper's ``bar`` or
``thresholds`` binding, each label's probability; for ``extract``'s ``null_bar``, ``p_null``
when not ``None``; for a ``margin``, the greatest ``float`` whose decimal does not exceed the
top label's lead; for a function component's ``boundaries`` entry naming the leaf, the read's
value, or the same role measure of its answer. Only an ``int`` or ``float`` is one. A
``boolean`` or ``label`` leaf is tried at every value.

``entries`` lists each ref whose entry some run changed in value, or in marker kind and cause
code, a run outside the record included; a result compares those alone, so a change of
``rows``, ``derivation`` or ``round`` is none. A plan result is compared without its
``context``. A result is ``"carries"`` when at some value tried a table row giving its output,
under ``F`` the first matched, outputs ``Param`` of the leaf, or for the plan the policy
table's row names a plan declaration holding the leaf; else it gets its classes, and for a
numeric leaf the switching values between adjacent classes and the slacks. Between two classes
the switching value is, on an integer axis, the upper class's least; on a number axis the
lower class's greatest when it alone of the two facing values is a compared value, else the
upper class's least. A ``not-loading`` class's edges are so located only to the values tried.
The slacks are the switching values bounding the leaf's own class, each minus the leaf's
value, null on a side with none; one no ``float`` holds, or for an integer leaf past
2^53 - 1 in magnitude, is refused ``unrepresentable-slack`` before the write.

``unseen`` is empty only when every way the leaf reaches a component is a comparison against a
probed value; it names, in code-point order, each component the leaf reaches otherwise: a
function component declaring it whose ``boundaries`` field does not name it; a ``C+``, ``C<``
or ``C>`` table with a ``Param`` row of it; each component reading an entry that holds the
leaf, one a table's row outputs ``Param`` of or a function component declaring it provides,
one level deep; each component of a value round 1 found; each table comparing a ``candidate``
read against the leaf whose set holds no value; and each component a failure names when a
value tried does not load. It is empty for a ``boolean`` or ``label`` leaf, every value of
which was tried.

Refused in §switching-value's order; the record checks are ``_edit``'s, the baseline replay
and the change of an entry ``_diff``'s, shared with the deciding inputs. The entry written is
returned, plain data reaching no ``Record``.
"""

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from fractions import Fraction
from typing import Any, cast

from ommat._canonical import decode, encode
from ommat._design import InvalidDesign, LoadedDesign
from ommat._load import load
from ommat._read_view import _ViewError, is_candidate_read, walk
from ommat._refs import _RefError, code_point_sorted, parse_ref
from ommat._replay import FindingKind, _differ, _replay_with
from ommat._types import Label, MappingOf, Scalar, Type, _numeric, render
from ommat._values import MAX_INTEGER
from ommat.evaluation._diff import OUTSIDE, _context_alone, _run_sync, baseline, changed, outcome
from ommat.evaluation._edit import _checked, _declared
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason
from ommat.evaluation._store import EvaluationStore
from ommat.helpers._choice import _lead
from ommat.helpers._numbers import exact
from ommat.helpers._points import numeric_axis

_R = RefusalReason
_K = FindingKind
_Log = list[dict[str, Any]]
_Doc = dict[str, Any]
_Functions = Mapping[str, Callable[..., Any]] | None
_Value = int | float | bool | str
_Result = dict[str, Any] | str
_Compared = dict[Fraction, set[str]]  # each compared value to the components comparing it

KIND = "switching-value"
_NOT_DETERMINED = "not-determined"


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
    await baseline(log, loaded)
    job = _Job(log, loaded, path, functions)
    value: dict[str, Any]
    if isinstance(t, MappingOf):
        held = cast("dict[str, Any]", loaded._leaf_value(path))
        keys = {k: await job.summary(t.value, held[k], k) for k in code_point_sorted(held)}
        value = {"unseen": job.unseen(t.value), "keys": keys}
    else:
        summary = await job.summary(t, loaded._leaf_value(path), None)
        value = {"unseen": job.unseen(t), **summary}
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
    inner = t.value if isinstance(t, MappingOf) else None
    if not (_numeric(t) or _finite(t) is not None or (inner is not None and _numeric(inner))):
        raise EvaluationRefusal(
            f"the leaf at {path} is of type {render(t)}, which the job does not vary",
            reason=_R.unsupported_leaf,
            path=path,
        )
    return t


def _finite(t: Type) -> list[_Value] | None:
    """Every value of a ``boolean`` or ``label`` leaf, in declared order; ``None`` for any
    other."""
    if isinstance(t, Label):
        return list(t.values)
    if isinstance(t, Scalar) and t.name == "boolean":
        return [False, True]
    return None


class _Run:
    """One run: each changed entry's item by ref, the plan and verdict, and whether it asked
    outside the record, so that it determines nothing."""

    def __init__(self, log: _Log, found: list[dict[str, Any]], outside: bool = False) -> None:
        self.outside = outside
        self.entries = {f["ref"]: f["replayed"] for f in found if f["kind"] is _K.entry}
        self.changed = {f["ref"] for f in found if f["kind"] is _K.entry and changed(f)}
        items = {f["kind"]: f["replayed"] for f in found if f["kind"] in (_K.plan, _K.verdict)}
        self.plan = items.get(_K.plan, _first(log, "plan"))
        self.verdict = items.get(_K.verdict, _first(log, "verdict"))


def _points(bounds: set[Fraction], t: Type, integer: bool) -> list[_Value]:
    """§lint-points' axis over ``bounds`` within ``t``, ascending; on an integer axis each
    boundary also adds the integers either side of it, clipped to the interval, so the least
    integer above every boundary is tried."""
    points: set[_Value] = set(numeric_axis(sorted(bounds), t))
    if integer:
        numeric = cast("tuple[str, tuple[int, int] | None]", _numeric(t))
        lo, hi = numeric[1] or (-MAX_INTEGER, MAX_INTEGER)
        points |= {max(lo, min(n, hi)) for b in bounds for n in (math.floor(b), math.floor(b) + 1)}
    return sorted(points, key=lambda v: exact(cast(float, v)))


def _first(log: _Log, kind: str) -> dict[str, Any] | None:
    return next((i for i in log if i["item"] == kind), None)


class _Job:
    """One record, design and leaf: ``summary`` replays and reports one varied value's
    results, ``unseen`` names what the probes could not see."""

    def __init__(self, log: _Log, loaded: LoadedDesign, path: str, functions: _Functions) -> None:
        self.log, self.loaded, self.path, self.functions = log, loaded, path, functions
        self.document = cast("_Doc", decode(loaded.document))
        self.mine = {i["ref"]: i for i in log if i["item"] == "entry"}
        self.late: set[str] = set()  # components comparing values only round 1 found, or unread

    def unseen(self, t: Type) -> list[str]:
        if _finite(t) is not None:
            return []
        return code_point_sorted(_hidden(self.document, self.loaded, self.path) | self.late)

    async def summary(self, t: Type, own: Any, key: str | None) -> dict[str, Any]:
        """``{"entries", "plan"}`` over the values tried for the leaf, or its key ``key``."""
        values = _finite(t)
        numeric = _numeric(t)
        integer = numeric is not None and numeric[0] == "integer"
        if numeric is not None and not integer:
            own = float(own)  # a number leaf's own value as its points are held
        tried: dict[_Value, _Run | _Result] = {own: _Run(self.log, [])}
        bounds: set[Fraction] = set()
        if values is not None:
            for v in values:
                if v not in tried:
                    tried[v] = (await self.run(v, key))[0]
            ordered = values
        else:
            found: _Compared = self.compared(self.log)
            for last in (False, True):  # round 0, then round 1
                bounds |= set(found)
                found = {}
                for v in _points(bounds, t, integer):
                    if v not in tried:
                        tried[v], rerun = await self.run(v, key)
                        if not isinstance(tried[v], _Run):
                            continue  # not loading: no log to read
                        for x, names in self.compared(rerun).items():
                            found.setdefault(x, set()).update(names)
                if last:
                    self.late |= {n for x, names in found.items() if x not in bounds for n in names}
            ordered = sorted(tried, key=lambda v: exact(cast(float, v)))
        out: dict[str, Any] = {"entries": {}, "plan": None}
        refs = code_point_sorted(
            {r for run in tried.values() if isinstance(run, _Run) for r in run.changed}
        )
        for ref in [*refs, None]:
            results = [_result(tried[v], ref, self.mine) for v in ordered]
            if any(self.carries(ref, r) for r in results):
                summary: _Result = "carries"
            elif values is not None:
                summary = {"classes": _partition(ordered, results)}
            else:
                summary = _numeric_summary(ordered, results, own, bounds, integer, self.path)
            if ref is None:
                out["plan"] = summary
            else:
                out["entries"][ref] = summary
        return out

    async def run(self, v: _Value, key: str | None) -> tuple[_Run | _Result, _Log]:
        """The run at ``v``, or its failure, and the replay's log, empty when none ran."""
        document = cast("_Doc", decode(self.loaded.document))
        leaf = document["parameters"][self.path]
        leaf["value"] = v if key is None else {**leaf["value"], key: v}
        try:
            design = load(document, functions=self.functions)
        except InvalidDesign as exc:
            failures = [
                {"check": f.check, "names": list(f.names), "message": f.message}
                for f in exc.failures
            ]
            components = _components(self.document)
            self.late |= {n for f in exc.failures for n in f.names if n in components}
            return {"not-loading": failures}, []
        rerun, found = await _replay_with(self.log, design, self.log, observer=None)
        rerun_log = rerun.log
        outside = any(f["kind"] in OUTSIDE for f in found)
        return _Run(
            self.log, found, outside or _unreturned(self.mine, self.log, rerun_log)
        ), rerun_log

    def compared(self, log: _Log) -> _Compared:
        """``_compared`` over ``log``, a table whose ``candidate`` read finds no set noted
        unseen."""
        found, blind = _compared(self.document, self.path, log)
        self.late |= blind
        return found

    def carries(self, ref: str | None, result: _Result) -> bool:
        """Whether a result's item passes the leaf through: a table row it matched outputs
        ``Param`` of the leaf, or for the plan, the policy table's row names a plan declaration
        holding the leaf. Under ``F`` only the first matched row gives the output."""
        if not isinstance(result, dict) or "not-loading" in result:
            return False  # not determined, or not loading
        item = result["entry"] if ref is not None else result["plan"]
        table = (
            next(
                (r for r in self.document["rules"].values() if r["provides"] == _declared(ref)),
                None,
            )
            if ref is not None
            else self.document["policy"]
        )
        if item is None or table is None or not item.get("rows"):  # a function's has none
            return False
        rows = cast("list[int]", item["rows"])
        for n in rows[:1] if table.get("hit", "F") == "F" else rows:
            row = table["rows"][n - 1]
            held = row["output"] if ref is not None else self.document["plans"][row["plan"]]
            if _contains(held, {"param": self.path}):
                return True
        return False


def _unreturned(mine: Mapping[str, Any], log: _Log, rerun: _Log) -> bool:
    """Whether ``rerun`` writes ``Unavailable`` with ``not-returned`` for a ref whose entry in
    ``log``, ``mine`` by ref, names a call item recorded ``Failed``."""
    for item in rerun:
        marker = item.get("marker") if item["item"] == "entry" else None
        if marker is None or marker["kind"] != "Unavailable":
            continue
        if marker["cause"]["code"] != "not-returned":
            continue
        at = cast("dict[str, Any] | None", mine.get(item["ref"]))
        call = None if at is None else at["call"]
        if call is not None and log[call]["outcome"] != "returned":  # recorded Failed
            return True
    return False


def _result(run: _Run | _Result, ref: str | None, mine: Mapping[str, Any]) -> _Result:
    """A run's result for the entry at ``ref``, or for the plan at ``None``."""
    if not isinstance(run, _Run):
        return run
    if run.outside:
        return _NOT_DETERMINED
    if ref is None:
        return {"plan": run.plan, "verdict": run.verdict}
    return {"entry": run.entries[ref] if ref in run.entries else mine.get(ref)}


def _same(a: _Result, b: _Result) -> bool:
    """Whether two results are one: items by canonical bytes, a plan without its
    ``context``."""
    if isinstance(a, dict) and isinstance(b, dict) and "plan" in a and "plan" in b:
        plan = {"kind": _K.plan, "recorded": a["plan"], "replayed": b["plan"]}
        return not _differ(a["verdict"], b["verdict"]) and (
            not _differ(a["plan"], b["plan"]) or _context_alone(plan)
        )
    if isinstance(a, dict) and isinstance(b, dict) and "entry" in a and "entry" in b:
        x, y = a["entry"], b["entry"]  # by value, or marker kind and code, alone
        return x is y is None or (x is not None and y is not None and outcome(x) == outcome(y))
    return encode(a) == encode(b)


def _partition(values: list[_Value], results: list[_Result]) -> list[dict[str, Any]]:
    """A finite leaf's classes, each value with every other giving the same result, in
    declared order."""
    classes: list[dict[str, Any]] = []
    for v, r in zip(values, results, strict=True):
        held = next((c for c in classes if _same(c["result"], r)), None)
        if held is None:
            classes.append({"values": [v], "result": r})
        else:
            held["values"].append(v)
    return classes


def _numeric_summary(
    values: list[_Value],
    results: list[_Result],
    own: Any,
    bounds: set[Fraction],
    integer: bool,
    path: str,
) -> dict[str, Any]:
    """The classes of adjacent values with one result, the switching value between each two,
    and the slacks of the leaf's own class."""
    classes: list[dict[str, Any]] = []
    for v, r in zip(values, results, strict=True):
        if classes and _same(classes[-1]["result"], r):
            classes[-1]["greatest"] = v
        else:
            classes.append({"least": v, "greatest": v, "result": r})
    switching: list[_Value] = []
    for a, b in itertools.pairwise(classes):
        g, lo = a["greatest"], b["least"]
        facing = not integer and exact(g) in bounds and exact(lo) not in bounds
        switching.append(g if facing else lo)
    at = next(
        i for i, c in enumerate(classes) if exact(c["least"]) <= exact(own) <= exact(c["greatest"])
    )
    x0 = exact(own)

    def slack(s: _Value) -> int | float:
        d = exact(cast(float, s)) - x0
        try:
            out = int(d) if integer else float(d)
        except OverflowError:
            out = math.inf
        if (integer and abs(out) > MAX_INTEGER) or not math.isfinite(out):
            raise EvaluationRefusal(
                f"a slack of the leaf at {path}, {s} minus {own}, is past what a slack holds",
                reason=_R.unrepresentable_slack,
                path=path,
            )
        return out

    below = slack(switching[at - 1]) if at > 0 else None
    above = slack(switching[at]) if at < len(switching) else None
    return {"switching": switching, "slacks": [below, above], "classes": classes}


# ---------------------------------------------------------------------------------------------
# The compared values


_ROLES: dict[tuple[str | None, str], str] = {
    ("pick_with_margin", "bar"): "bar",
    ("pick_with_margin", "margin"): "margin",
    ("pick_with_thresholds", "thresholds"): "thresholds",
    ("pick_with_thresholds", "margin"): "margin",
    ("extract", "null_bar"): "null",
}


def _hidden(document: _Doc, loaded: LoadedDesign, path: str) -> set[str]:
    """The components the leaf reaches other than by a comparison the probes see: a function
    component declaring it whose ``boundaries`` field does not name it; a table aggregating
    it under ``C+``, ``C<`` or ``C>`` beside other values; and each component reading an entry
    that holds the leaf, one a table row outputs ``Param`` of, or a function component
    declaring the leaf provides."""
    out: set[str] = set()
    valued: set[str] = set()
    for name, c in _components(document).items():
        if c["form"] == "function":
            if path in c["params"]:
                valued.add(c.get("provides", ""))
                if not _contains(c["boundaries"], path):
                    out.add(name)
        elif any(_contains(row.get("output"), {"param": path}) for row in c["rows"]):
            valued.add(c.get("provides", ""))
            if c.get("hit") in ("C+", "C<", "C>"):
                out.add(name)
    readers = loaded._reads
    out |= {n for n, reads in readers.items() if any(_declared(r) in valued for r in reads)}
    return out


def _components(document: _Doc) -> dict[str, dict[str, Any]]:
    """Each rule by id, then the policy as ``policy``."""
    return {**document["rules"], "policy": document["policy"]}


def _compared(document: _Doc, path: str, log: _Log) -> tuple[_Compared, set[str]]:
    """The values ``log``'s turn compares the leaf at ``path`` against, exactly, each to the
    components comparing it; and each table comparing a ``candidate`` read whose set ``log``
    holds no value of."""
    entries = [i for i in log if i["item"] == "entry"]
    out: _Compared = {}
    blind: set[str] = set()

    def add(name: str, values: Iterable[object]) -> None:
        for x in values:
            if type(x) in (int, float):
                out.setdefault(exact(cast(float, x)), set()).add(name)

    for name, c in _components(document).items():
        if c["form"] == "table":
            for n, i in enumerate(cast("list[dict[str, Any]]", c["inputs"])):
                roles = [_ROLES.get((i["helper"], r)) for r, b in i["params"].items() if b == path]
                cells = any(_compares(row["cells"][n], path) for row in c["rows"])
                if not roles and not cells:
                    continue  # the leaf does not reach this input
                held = _read_values(i["read"], entries, c.get("grounded"))
                if held is None:
                    blind.add(name)
                    continue
                for role in roles:
                    add(name, (x for v in held for x in _measure(role, v, ())))
                if cells:
                    add(name, (x for v in held for x in _input(document, i, v)))
        elif c["boundaries"] is not None:
            for read, entry in cast("dict[str, Any]", c["boundaries"]).items():
                values = _read_values(read, entries, None) or []
                if isinstance(entry, list):
                    if path in entry:
                        add(name, values)
                    continue
                for role, members in cast("dict[str, Any]", entry).items():
                    if role == "mass":
                        for m in cast("list[dict[str, Any]]", members):
                            if path in m["bounds"]:
                                add(
                                    name,
                                    (x for v in values for x in _measure(role, v, m["labels"])),
                                )
                    elif path in members:
                        add(name, (x for v in values for x in _measure(role, v, ())))
    return out, blind


def _input(document: _Doc, i: dict[str, Any], v: object) -> list[object]:
    """A table input's value over its read's value ``v``."""
    helper = i["helper"]
    if helper is None:
        return [v]
    if helper == "mass":
        return _measure("mass", v, document["parameters"][i["params"]["labels"]]["value"])
    if helper == "size":
        return [len(cast("list[object]", v))] if type(v) is list else []
    if helper == "extract":
        return _measure("value", v, ())
    return []


def _read_values(read: str, entries: _Log, grounded: str | None) -> list[object] | None:
    """The values of the entries a read names, a grounded declared ref's each, through its
    field path, a marker and a path the entry does not hold giving none. A ``candidate`` read
    of a table grounded over ``grounded`` takes each member of that set's entry through its
    field path, the input as recorded; ``None`` when the entry holds no value."""
    if is_candidate_read(read):
        held = next((e for e in entries if e["ref"] == grounded), None)
        if held is None or "value" not in held:
            return None
        members, fields = cast("list[object]", held["value"]), tuple(read.split(".")[1:])
    else:
        try:
            parsed = parse_ref(read)
        except _RefError:
            return []
        declared = f"{parsed.level}:{parsed.name}"
        members = [e["value"] for e in entries if "value" in e and _declared(e["ref"]) == declared]
        fields = parsed.fields
    out: list[object] = []
    for m in members:
        try:
            out.append(walk(m, fields, read))
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
    runner-up, as the pick helpers measure it."""
    exacts = {label: exact(p) for label, p in probs.items()}
    lead = _lead(exacts, max(exacts, key=exacts.__getitem__))
    f = float(lead)
    while exact(f) > lead:
        f = math.nextafter(f, -math.inf)
    return f


def _compares(cell: object, path: str) -> bool:
    """Whether a cell compares its input against the leaf at ``path``."""
    if type(cell) is not dict:
        return False
    ((tag, member),) = cast("dict[str, Any]", cell).items()
    if tag == "between":
        return any(o == {"param": path} for o in cast("list[object]", member))
    return tag in (">=", ">", "<=", "<") and member == {"param": path}


def _contains(value: object, target: object) -> bool:
    """Whether ``target`` is ``value`` or stands anywhere within it, through dicts and lists."""
    stack = [value]
    while stack:
        item = stack.pop()
        if item == target:
            return True
        if type(item) is dict:
            stack += cast("dict[str, object]", item).values()
        elif type(item) is list:
            stack += cast("list[object]", item)
    return False
