"""Decision tables: a table's members in the document's tagged form, read from what a design
gives; a table's evaluation over its input values under its hit policy and the marker
contract; and the overlap test load runs for ``U`` and ``A``.

The tagged form is §table-notation's. An input is ``{"read", "helper", "params"}``; a rule
table's row ``{"cells", "output"}`` and a policy table's ``{"cells", "plan"}``; a cell ``"-"``
or a one-member object tagged ``is``, ``one_of``, ``marker``, ``>=``, ``>``, ``<=``, ``<``,
``between`` or ``in``; an operand or an output entry an object tagged ``literal``, ``param`` or
``ref``, or an output ``{"marker": "Unsure"}``. A cell given as a string is read once by the
notation, its form decided by how it begins.

Evaluation takes the inputs' values, already read and their helpers called, so a raise in a
helper never reaches it. It tests every row in table order, so ``rows`` lists every match,
and records for each row that failed whether every failing cell, and every value condition
the row holds, rested on an input holding ``Unavailable``: rule 2's test. A comparison over a
value that is not a number is ``Incomparable``, which load's typing makes unreachable.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

from ommat._design import Helper, Input, Param, PlanDeclaration
from ommat._markers import Cause, CauseCode, Unasked, Unavailable, Undetermined, Unsure
from ommat._plan import Origin, Step
from ommat._read_view import ReadView
from ommat._refs import is_declared_name, is_ref
from ommat._values import in_domain, string_data

ANY = "-"
MARKERS: dict[str, type[Undetermined]] = {
    "Unsure": Unsure,
    "Unavailable": Unavailable,
    "Unasked": Unasked,
}
_KIND_NAMES = {v: k for k, v in MARKERS.items()}
COMPARISONS = (">=", ">", "<=", "<")
CELL_TAGS = frozenset({"is", "one_of", "marker", "markers", *COMPARISONS, "between", "in"})
ANY_MARKER = "Undetermined"
OPERAND_TAGS = frozenset({"literal", "param", "ref"})
OUTPUT_TAGS = frozenset({"literal", "param", "ref", "marker"})
HELPERS: dict[str, tuple[str, ...]] = {
    "pick_with_margin": ("bar", "margin"),
    "pick_with_thresholds": ("thresholds", "margin"),
    "size": (),
    "mass": ("labels",),
    "extract": ("null_bar",),
}
"""The helpers a table's input may call, each with its arguments after the read, in order."""
REFUSED_HELPERS = frozenset(
    {
        "pick_each_with_margin",
        "pick_each_with_thresholds",
        "determined",
        "agree",
        "collect",
        "first",
        "priority",
        "unique",
    }
)
"""Names of the library a table's input may not call: the per-candidate picks and
``determined`` (check 6), and the combinators, which a hit policy stands for."""
_INTEGER = re.compile(r"-?(0|[1-9][0-9]*)")
_IN = re.compile(r"in +([qtd]:\S*)")
_IN_START = re.compile(r"in +[qtd]:")


class NotationError(ValueError):
    """A member a table's notation or tagged form cannot read; check 10's."""


class Incomparable(Exception):
    """A comparison over a value that is not a number, or an ``in`` cell over one that is not
    a list: what load's typing leaves no design to reach."""


# ---------------------------------------------------------------------------------------------
# The tagged form, from what a design gives


def _level_prefixed(text: str) -> bool:
    return len(text) > 2 and text[1] == ":" and text[0] in "qtd"


def _operand(text: str, *, ref: bool) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise NotationError("an operand is empty")
    if _INTEGER.fullmatch(text):
        return {"literal": int(text)}
    if _level_prefixed(text):
        if not ref:
            raise NotationError(f"an interval's operand {text!r} is a ref")
        if not is_ref(text):
            raise NotationError(f"the operand {text!r} is not a ref")
        return {"ref": text}
    if not is_declared_name(text):
        raise NotationError(f"the operand {text!r} is not an integer, a ref or a leaf path")
    return {"param": text}


def _notation(text: str) -> Any:
    """A cell written in the notation, in the tagged form."""
    cell = text.strip()
    if cell == ANY:
        return ANY
    if cell in MARKERS:
        return {"marker": cell}
    if cell == ANY_MARKER:
        return {"markers": list(MARKERS)}
    kinds = [k.strip() for k in cell.split(" or ")]
    if len(kinds) > 1 and all(k in MARKERS for k in kinds):
        return _markers(kinds)
    for op in COMPARISONS:
        if cell.startswith(op):  # ">=" and "<=" are tried before ">" and "<"
            return {op: _operand(cell[len(op) :], ref=True)}
    if cell.startswith("["):
        if not cell.endswith("]") or cell.count(",") != 1:
            raise NotationError(f"the cell {text!r} is not an interval [X, Y]")
        lo, hi = cell[1:-1].split(",")
        return {"between": [_operand(lo, ref=False), _operand(hi, ref=False)]}
    if cell.startswith("{"):
        members = [m.strip() for m in cell[1:-1].split(",")] if cell.endswith("}") else []
        if len(members) < 2 or not all(m and not set(m) & set("{}") for m in members):
            raise NotationError(f"the cell {text!r} is not a set of two or more labels")
        return {"one_of": members}
    found = _IN.fullmatch(cell)
    if found is None and _IN_START.match(cell):
        raise NotationError(f"the cell {text!r} is not in and one ref")
    if found is not None:
        if not is_ref(found.group(1)):
            raise NotationError(f"the cell {text!r} names no ref")
        return {"in": found.group(1)}
    return {"is": cell}


def cell_form(value: object) -> Any:
    """A cell in the tagged form, from the tagged mapping, a marker class, a ``bool`` or an
    ``int``, or a string in the notation."""
    if type(value) is bool or type(value) is int:
        return {"is": value}
    if isinstance(value, type) and value in _KIND_NAMES:
        return {"marker": _KIND_NAMES[cast("type[Undetermined]", value)]}
    if value is Undetermined:
        return {"markers": list(MARKERS)}
    if type(value) is tuple:
        members = cast("tuple[object, ...]", value)
        if all(isinstance(v, type) and v in _KIND_NAMES for v in members):
            return _markers([_KIND_NAMES[cast("type[Undetermined]", v)] for v in members])
    if isinstance(value, str):
        return _notation(string_data(value))
    if isinstance(value, Mapping):
        return _tagged_cell(cast("Mapping[object, Any]", value))
    raise NotationError(f"a cell of type {type(cast(object, value)).__name__} is not a cell")


def _tagged_cell(cell: Mapping[object, Any]) -> Any:
    if len(cell) != 1:
        raise NotationError("a tagged cell is not one tag and its value")
    ((key, value),) = cell.items()
    tag = string_data(key) if isinstance(key, str) else None
    if tag not in CELL_TAGS:
        raise NotationError("a tagged cell's tag is not a cell's")
    if tag == "is":
        if not _scalar(value):
            raise NotationError("an is cell's value is not a label, a boolean or an integer")
        return {"is": _plain(value)}
    if tag == "one_of":
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            raise NotationError("a one_of cell's value is not a list")
        members = list(cast("Sequence[object]", value))
        if not members or not all(_scalar(m) for m in members):
            raise NotationError("a one_of cell lists no value, or one of another form")
        return {"one_of": [_plain(m) for m in members]}
    if tag == "marker":
        if not isinstance(value, str) or string_data(value) not in MARKERS:
            raise NotationError("a marker cell names no marker kind")
        return {"marker": string_data(value)}
    if tag == "markers":
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            raise NotationError("a markers cell is not a list of marker kinds")
        kinds = list(cast("Sequence[object]", value))
        if not all(isinstance(k, str) and string_data(k) in MARKERS for k in kinds):
            raise NotationError("a markers cell lists something that is not a marker kind")
        return _markers([string_data(cast(str, k)) for k in kinds])
    if tag == "in":
        if not isinstance(value, str):
            raise NotationError("an in cell does not name a ref")
        return {"in": string_data(value)}
    if tag == "between":
        if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
            raise NotationError("a between cell is not two operands")
        bounds = list(cast("Sequence[object]", value))
        if len(bounds) != 2:
            raise NotationError("a between cell is not two operands")
        return {"between": [_tagged_operand(b, ref=False) for b in bounds]}
    return {tag: _tagged_operand(value, ref=True)}


def _markers(kinds: list[str]) -> dict[str, Any]:
    """A cell listing two or three marker kinds; one kind is a marker cell's."""
    if len(kinds) < 2:
        raise NotationError("a markers cell lists fewer than two marker kinds")
    return {"markers": kinds}


def _scalar(value: object) -> bool:
    return isinstance(value, str) or type(value) is bool or type(value) is int


def _plain(value: object) -> Any:
    return string_data(value) if isinstance(value, str) else value


def _tagged_operand(value: object, *, ref: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or len(cast("Mapping[object, Any]", value)) != 1:
        raise NotationError("an operand is not one tag and its value")
    ((key, member),) = cast("Mapping[object, Any]", value).items()
    tag = string_data(key) if isinstance(key, str) else None
    if tag == "literal" and type(member) is int:
        return {"literal": member}
    if tag == "param" and isinstance(member, str):
        return {"param": string_data(member)}
    if tag == "ref" and ref and isinstance(member, str):
        return {"ref": string_data(member)}
    raise NotationError("an operand is not a literal integer, a param or a ref")


def input_form(value: object) -> dict[str, Any]:
    """An input in the tagged form, from a ref or a ``Helper``."""
    if isinstance(value, str):
        return {"read": string_data(value), "helper": None, "params": {}}
    if type(value) is Helper:
        params = cast(object, value.params)
        name, read = cast(object, value.name), cast(object, value.read)
        if not isinstance(params, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in cast("Mapping[object, object]", params).items()
        ):
            raise NotationError("a helper's params do not bind each argument to a leaf path")
        if not isinstance(name, str) or not isinstance(read, str):
            raise NotationError("a helper input's name or read is not a string")
        bound = cast("Mapping[str, str]", params)
        return {
            "read": string_data(read),
            "helper": string_data(name),
            "params": {string_data(k): string_data(v) for k, v in bound.items()},
        }
    raise NotationError(f"an input of type {type(value).__name__} is not a ref or a Helper")


def output_form(value: object) -> dict[str, Any]:
    """A rule table's output entry in the tagged form: ``Unsure``, ``Input``, ``Param``, or any
    other value as a literal."""
    if value is Unsure:
        return {"marker": "Unsure"}
    if type(value) is Input:
        ref = cast(object, value.ref)
        if not isinstance(ref, str) or not in_domain(ref):
            raise NotationError("an Input's ref is not a string")
        return {"ref": ref}
    if type(value) is Param:
        path = cast(object, value.path)
        if not isinstance(path, str) or not in_domain(path):
            raise NotationError("a Param's path is not a string")
        return {"param": path}
    if isinstance(value, type | Undetermined):
        raise NotationError("an output entry is a marker other than the class Unsure")
    return {"literal": value}


def row_form(row: object, *, hit: object, policy: bool) -> dict[str, Any]:
    """A row in the tagged form: its cells, then its output entry or its plan's name, the
    cells alone under ``C#``."""
    if isinstance(row, NotationError):
        raise row
    if isinstance(row, str | bytes | bytearray) or not isinstance(row, Sequence):
        raise NotationError("a row is not a list")
    members = list(cast("Sequence[object]", row))
    if policy:
        if not members or not isinstance(members[-1], str):
            raise NotationError("a policy table's row does not end with a plan's name")
        return {"cells": [cell_form(c) for c in members[:-1]], "plan": string_data(members[-1])}
    if hit == "C#":
        return {"cells": [cell_form(c) for c in members], "output": None}
    if not members:
        raise NotationError("a row holds no output entry")
    return {"cells": [cell_form(c) for c in members[:-1]], "output": output_form(members[-1])}


def python_row(row: Mapping[str, Any], *, policy: bool, hit: object = None) -> Any:
    """A document row as a design written in Python holds it: its tagged cells, then its
    output entry or its plan's name. A row not of the tagged form gives a ``NotationError``,
    which check 10 reports: a cell that is neither ``"-"`` nor a tagged object, a plan's name
    that is not a string, or an output that is not one tag, null only under ``C#``."""
    cells = row.get("cells")
    if not isinstance(cells, list) or not all(
        c == ANY or isinstance(c, dict) for c in cast("list[object]", cells)
    ):
        return NotationError("a row's cells are not a list of tagged cells")
    out = list(cast("list[Any]", cells))
    if policy:
        plan = row.get("plan")
        if not isinstance(plan, str):
            return NotationError("a row's plan is not a name")
        return [*out, plan]
    output = row.get("output")
    if (output is None) != (hit == "C#"):
        return NotationError("a row's output is null where the hit policy is not C#, or not")
    if output is None:
        return out
    if not isinstance(output, dict) or len(cast("dict[str, Any]", output)) != 1:
        return NotationError("a row's output is not one tag and its value")
    ((tag, member),) = cast("dict[str, Any]", output).items()
    if tag == "literal":
        return [*out, member]
    if tag == "ref" and isinstance(member, str):
        return [*out, Input(member)]
    if tag == "param" and isinstance(member, str):
        return [*out, Param(member)]
    if tag == "marker" and member == "Unsure":
        return [*out, Unsure]
    return NotationError("a row's output is not a literal, a ref, a param or Unsure")


def python_input(value: object) -> Any:
    """A document input as a design written in Python holds it; another form kept whole."""
    if not isinstance(value, dict):
        return value
    members = cast("dict[str, Any]", value)
    if set(members) != {"read", "helper", "params"}:
        return cast(object, members)
    read, helper, params = members["read"], members["helper"], members["params"]
    if helper is None and params == {}:
        return read
    if isinstance(helper, str) and isinstance(read, str) and isinstance(params, dict):
        return Helper(helper, read, **cast("dict[str, str]", params))
    return cast(object, members)


# ---------------------------------------------------------------------------------------------
# Evaluation


def read_inputs(
    inputs: Sequence[Mapping[str, Any]], view: ReadView, leaves: Mapping[str, Any]
) -> list[object]:
    """Each input's value, in input order: a bare read's value or marker, or what its helper
    returns over its read, each other argument the leaf its binding names. A helper's raise
    propagates."""
    from ommat.helpers import _choice  # the library, at the call: past the package's imports

    out: list[object] = []
    for i in inputs:
        helper = i["helper"]
        if helper is None:
            out.append(view.read(i["read"]))
            continue
        args = [leaves[i["params"][name]] for name in HELPERS[helper]]
        out.append(getattr(_choice, helper)(view, i["read"], *args))
    return out


def plan_conditions(plan: PlanDeclaration) -> list[tuple[str, tuple[int, int]]]:
    """A plan declaration's board refs a row naming it needs held, each with its order key
    for rule 2's naming: its steps' board-ref origins, by step and then by argument name in
    code-point order, then the slots of ``reason`` and of ``reply``, in slot-list order."""
    out: list[tuple[str, tuple[int, int]]] = []
    for step in plan.steps:
        arguments = cast(Step, step).arguments
        for name in sorted(arguments):
            origin = arguments[name]
            if isinstance(origin, Origin) and origin.ref is not None:
                out.append((origin.ref, (1, len(out))))
    for text in (plan.reason, plan.reply):
        for slot in () if text is None else text.slots:
            tag, value = next(iter(slot["origin"].items()))
            if tag == "ref":
                out.append((cast(str, value), (2, len(out))))
    return out


@dataclass(frozen=True, slots=True)
class Condition:
    """A value a row needs held besides its cells: a pass-through's input, or in the policy
    table a plan's board-ref origin or slot. ``at`` orders it for rule 2's naming."""

    ref: str
    value: object
    at: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Outcome:
    """A table's result: ``value``, a value or a marker for a rule table; ``plan``, the
    winning row's plan's name for the policy table, or ``None`` with ``value`` the marker;
    ``rows``, every matching row's position from 1."""

    value: object
    plan: str | None
    rows: list[int]


def _marker(value: object) -> bool:
    return isinstance(value, Undetermined)


def _number(value: object) -> bool:
    return type(value) is int or type(value) is float


def _equal(a: object, b: object) -> bool:
    if _number(a) and _number(b):
        return a == b
    return type(a) is type(b) and a == b


def _bound(
    operand: Mapping[str, Any], values: Mapping[str, object], leaves: Mapping[str, Any]
) -> Any:
    ((tag, member),) = operand.items()
    if tag == "literal":
        return member
    if tag == "param":
        return leaves[member]
    return values[member]


class _Table:
    def __init__(
        self,
        reads: Sequence[str],
        values: Sequence[object],
        leaves: Mapping[str, Any],
    ) -> None:
        self.reads = list(reads)
        self.values = list(values)
        self.by_read = dict(zip(self.reads, self.values, strict=True))
        self.leaves = leaves

    def cell(self, cell: Any, i: int) -> tuple[bool, list[int]]:
        """Whether the cell over input ``i`` matches, and when not, the inputs holding
        ``Unavailable`` it tested, in input order: none when it failed on anything else."""
        value = self.values[i]
        if cell == ANY:
            return True, []
        ((tag, member),) = cast("dict[str, Any]", cell).items()
        unavailable = [i] if type(value) is Unavailable else []
        if tag == "marker":  # no value could match it, so an outage changes nothing
            return type(value) is MARKERS[member], []
        if tag == "markers":
            return any(type(value) is MARKERS[k] for k in member), []
        other = member if tag == "in" else member.get("ref") if tag in COMPARISONS else None
        if other is not None:
            # A cell over two inputs fails on an outage only when every input it tests that is
            # not Unavailable holds a value, so the withheld value alone could have changed it.
            j = self.reads.index(other)
            pair = ((i, value), (j, self.values[j]))
            held = all(type(v) is Unavailable or not (_marker(v) or v is None) for _, v in pair)
            unavailable = sorted({k for k, v in pair if type(v) is Unavailable}) if held else []
        if _marker(value) or value is None:
            return False, unavailable
        if tag == "is":
            return _equal(value, member), []
        if tag == "one_of":
            return any(_equal(value, m) for m in member), []
        if tag == "in":
            j = self.reads.index(member)
            listed = self.values[j]
            if _marker(listed) or listed is None:
                return False, [j] if type(listed) is Unavailable else []
            if type(listed) is not list and type(listed) is not tuple:
                raise Incomparable(member)
            return any(_equal(value, m) for m in cast("Sequence[object]", listed)), []
        if not _number(value):
            raise Incomparable(self.reads[i])
        if tag == "between":
            lo, hi = (_bound(b, self.by_read, self.leaves) for b in member)
            return bool(lo <= cast(float, value) <= hi), []
        other = _bound(member, self.by_read, self.leaves)
        if "ref" in member:
            j = self.reads.index(member["ref"])
            if _marker(other) or other is None:
                return False, [j] if type(other) is Unavailable else []
            if not _number(other):
                raise Incomparable(member["ref"])
        x, y = cast(float, value), cast(float, other)
        if tag == ">=":
            return x >= y, []
        if tag == ">":
            return x > y, []
        if tag == "<=":
            return x <= y, []
        return x < y, []

    def row(
        self, cells: Sequence[Any], conditions: Sequence[Condition]
    ) -> tuple[bool, tuple[int, int] | None, str | None]:
        """Whether a row matches; when not, whether it failed only on ``Unavailable``, as the
        first such input's order key and ref, else ``None``."""
        failed = False
        only = True
        named: list[tuple[tuple[int, int], str]] = []
        for i, cell in enumerate(cells):
            matched, unavailable = self.cell(cell, i)
            if matched:
                continue
            failed = True
            if not unavailable:
                only = False
            named += [((0, j), self.reads[j]) for j in unavailable]
        for condition in conditions:
            if not _marker(condition.value):
                continue
            failed = True
            if type(condition.value) is Unavailable:
                named.append((condition.at, condition.ref))
            else:
                only = False
        if not failed:
            return True, None, None
        if not only:
            return False, None, None
        at, ref = min(named)
        return False, at, ref

    def no_match(self) -> Undetermined:
        """Rule 1: the inputs' marker of highest precedence, naming its first input; else
        ``Unsure`` with ``abstained`` naming the inputs."""
        for kind in (Unavailable, Unsure, Unasked):
            for read, value in zip(self.reads, self.values, strict=True):
                if type(value) is kind:
                    return kind(Cause(CauseCode.required, [read]))
        return self.abstained()

    def abstained(self) -> Unsure:
        return Unsure(Cause(CauseCode.abstained, list(dict.fromkeys(self.reads))))

    def output(self, output: Mapping[str, Any]) -> object:
        ((tag, member),) = output.items()
        if tag == "literal":
            return member
        if tag == "marker":
            return self.abstained()
        if tag == "param":
            return self.leaves[member]
        return self.by_read[member]


def evaluate(
    rows: Sequence[Mapping[str, Any]],
    hit: str,
    order: Sequence[Any] | None,
    reads: Sequence[str],
    values: Sequence[object],
    leaves: Mapping[str, Any],
    conditions: Mapping[int, Sequence[Condition]] | None = None,
) -> Outcome:
    """A table over its inputs' ``values``, in input order, each read once: its result under
    ``hit`` and the marker contract. ``conditions`` gives the policy table's plan origins and
    slots by row position from 0; a rule table's pass-through is a condition of its own row.
    Raises ``Incomparable``."""
    t = _Table(reads, values, leaves)
    policy = conditions is not None
    matched: list[int] = []
    failed: list[tuple[int, str]] = []  # rows that failed only on Unavailable, with the ref
    for k, row in enumerate(rows):
        held = list(conditions.get(k, ())) if conditions is not None else []
        output = row.get("output")
        if not policy and isinstance(output, dict) and "ref" in output:
            ref = cast(str, output["ref"])
            j = t.reads.index(ref)
            held.append(Condition(ref, t.values[j], (0, j)))
        ok, at, ref = t.row(row["cells"], held)
        if ok:
            matched.append(k)
        elif at is not None:
            assert ref is not None
            failed.append((k, ref))
    positions = [k + 1 for k in matched]

    def unavailable(ref: str) -> Outcome:
        return Outcome(Unavailable(Cause(CauseCode.required, [ref])), None, positions)

    if hit.startswith("C"):
        if failed:
            return unavailable(failed[0][1])
        if not matched:
            if hit in ("C+", "C#"):
                return Outcome(0, None, positions)
            return Outcome(t.no_match(), None, positions)
        if hit == "C#":
            return Outcome(len(matched), None, positions)
        outs = [t.output(rows[k]["output"]) for k in matched]
        if hit == "C+":
            return Outcome(_sum(cast("list[int | float]", outs)), None, positions)
        pick = min if hit == "C<" else max
        return Outcome(pick(cast("list[float]", outs)), None, positions)
    if not matched:
        return Outcome(t.no_match(), None, positions)
    if hit == "P":
        assert order is not None
        ranks = [_rank(order, rows[k]["output"]) for k in matched]
        winner = matched[ranks.index(min(ranks))]
        best = min(ranks)
        above = [(k, r) for k, r in failed if _rank(order, rows[k]["output"]) < best]
        if above:
            return unavailable(above[0][1])
    else:
        winner = matched[0]
        if hit == "F":
            before = [(k, r) for k, r in failed if k < winner]
            if before:
                return unavailable(before[0][1])
    if policy:
        return Outcome(None, cast(str, rows[winner]["plan"]), positions)
    return Outcome(t.output(rows[winner]["output"]), None, positions)


def _sum(outs: list[int | float]) -> int | float:
    """The sum as ``collect`` takes it: an ``int`` when every output is one, else the exact
    sum over their decimals rounded once; one no finite ``float`` holds is ``inf``, which no
    type allows."""
    if all(type(v) is int for v in outs):
        return sum(cast("list[int]", outs))
    total = sum((Fraction(repr(v)) for v in outs), Fraction(0))
    try:
        return float(total)
    except OverflowError:
        return math.inf


def _rank(order: Sequence[Any], output: Mapping[str, Any]) -> int:
    value = output["literal"]
    return next(i for i, v in enumerate(order) if _equal(v, value))


# ---------------------------------------------------------------------------------------------
# Overlap, for U and A


@dataclass(frozen=True, slots=True)
class Axis:
    """What load knows of an input's values for the overlap test: whether they are integers,
    and the interval its type declares, each end ``None`` when unbounded."""

    integer: bool = False
    lo: Fraction | None = None
    hi: Fraction | None = None


@dataclass(frozen=True, slots=True)
class _Span:
    lo: Fraction | None
    lo_closed: bool
    hi: Fraction | None
    hi_closed: bool


_ALL = "all"  # every value, null included
_SOME = "some"  # some values, none named: a cell over two inputs


def _exact(x: object) -> Fraction:
    return Fraction(x) if type(x) is int else Fraction(repr(x))


class _Set:
    """The states a cell or a row's cells admit for one input: marker kinds, and values."""

    __slots__ = ("markers", "values")

    def __init__(self, markers: frozenset[str], values: Any) -> None:
        self.markers = markers
        self.values = values  # _ALL, _SOME, a list of values, a _Span, or None for none

    def meet(self, other: "_Set", axis: Axis) -> "_Set":
        return _Set(self.markers & other.markers, _meet(self.values, other.values, axis))

    def empty(self, axis: Axis) -> bool:
        return not self.markers and _empty(self.values, axis)


def _meet(a: Any, b: Any, axis: Axis) -> Any:
    if a is None or b is None:
        return None
    if a == _ALL:
        return b
    if b == _ALL:
        return a
    if a == _SOME:
        return b
    if b == _SOME:
        return a
    if isinstance(a, list) and isinstance(b, list):
        bs = cast("list[object]", b)
        return [x for x in cast("list[object]", a) if any(_equal(x, y) for y in bs)]
    if isinstance(a, list) or isinstance(b, list):
        listed = cast("list[object]", a if isinstance(a, list) else b)
        span = cast(_Span, b if isinstance(a, list) else a)
        return [x for x in listed if _number(x) and _within(x, span)]
    return _span_meet(cast(_Span, a), cast(_Span, b))


def _within(x: object, s: _Span) -> bool:
    v = _exact(x)
    if s.lo is not None and (v < s.lo or (v == s.lo and not s.lo_closed)):
        return False
    return not (s.hi is not None and (v > s.hi or (v == s.hi and not s.hi_closed)))


def _span_meet(a: _Span, b: _Span) -> _Span:
    if a.lo is None or (b.lo is not None and (b.lo > a.lo or (b.lo == a.lo and not b.lo_closed))):
        lo, lo_closed = b.lo, b.lo_closed
    else:
        lo, lo_closed = a.lo, a.lo_closed
    if a.hi is None or (b.hi is not None and (b.hi < a.hi or (b.hi == a.hi and not b.hi_closed))):
        hi, hi_closed = b.hi, b.hi_closed
    else:
        hi, hi_closed = a.hi, a.hi_closed
    return _Span(lo, lo_closed, hi, hi_closed)


def _empty(values: Any, axis: Axis) -> bool:
    if values is None:
        return True
    if values in (_ALL, _SOME):
        return False
    if isinstance(values, list):
        return not values
    s = _span_meet(cast(_Span, values), _Span(axis.lo, True, axis.hi, True))
    if s.lo is None or s.hi is None:
        return False
    if axis.integer:
        least = s.lo.__ceil__() if s.lo_closed else s.lo.__floor__() + 1
        return not (least < s.hi or (least == s.hi and s.hi_closed))
    return s.lo > s.hi or (s.lo == s.hi and not (s.lo_closed and s.hi_closed))


_EVERY = frozenset(MARKERS)
_SPLITS = {(">=", "<"), ("<", ">="), (">", "<="), ("<=", ">")}


def _cell_set(cell: Any, leaves: Mapping[str, Any]) -> _Set:
    if cell == ANY:
        return _Set(_EVERY, _ALL)
    ((tag, member),) = cast("dict[str, Any]", cell).items()
    if tag == "marker":
        return _Set(frozenset({member}), None)
    if tag == "markers":
        return _Set(frozenset(member), None)
    if tag == "is":
        return _Set(frozenset(), [member])
    if tag == "one_of":
        return _Set(frozenset(), list(member))
    if tag == "in" or (tag in COMPARISONS and "ref" in member):
        return _Set(frozenset(), _SOME)
    if tag == "between":
        lo, hi = (_exact(_bound(b, {}, leaves)) for b in member)
        return _Set(frozenset(), _Span(lo, True, hi, True))
    x = _exact(_bound(member, {}, leaves))
    spans = {
        ">=": _Span(x, True, None, True),
        ">": _Span(x, False, None, True),
        "<=": _Span(None, True, x, True),
        "<": _Span(None, True, x, False),
    }
    return _Set(frozenset(), spans[tag])


def _held(cells: Sequence[Any], output: object, reads: Sequence[str]) -> set[int]:
    """The inputs a row needs to hold a value besides its cells: a pass-through's, and the
    other input of each cell over two."""
    out: set[int] = set()
    for cell in cells:
        if cell == ANY:
            continue
        ((tag, member),) = cast("dict[str, Any]", cell).items()
        if tag == "in":
            out.add(reads.index(member))
        elif tag in COMPARISONS and "ref" in member:
            out.add(reads.index(member["ref"]))
    if isinstance(output, dict) and "ref" in output:
        out.add(reads.index(cast("dict[str, str]", output)["ref"]))
    return out


def overlap(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    reads: Sequence[str],
    axes: Sequence[Axis],
    leaves: Mapping[str, Any],
) -> bool:
    """Whether some assignment of the inputs makes both rows match (§decision-tables)."""
    held = _held(a["cells"], a.get("output"), reads) | _held(b["cells"], b.get("output"), reads)
    for i, (x, y) in enumerate(zip(a["cells"], b["cells"], strict=True)):
        if _split(x, y):
            return False
        s = _cell_set(x, leaves).meet(_cell_set(y, leaves), axes[i])
        if i in held:
            s = _Set(frozenset(), s.values)
        if s.empty(axes[i]):
            return False
    return True


def _split(x: Any, y: Any) -> bool:
    if x == ANY or y == ANY:
        return False
    ((tx, mx),) = cast("dict[str, Any]", x).items()
    ((ty, my),) = cast("dict[str, Any]", y).items()
    if tx not in COMPARISONS or ty not in COMPARISONS:
        return False
    if "ref" not in mx or "ref" not in my or mx["ref"] != my["ref"]:
        return False
    return (tx, ty) in _SPLITS
