"""What the outline's modules share: the fixed key sets, ref parsing, and the small readers that
take a document's members as plain dicts and lists whatever the document holds."""

import hashlib
import json
import re
from fractions import Fraction
from typing import Any, TypeGuard, cast

from ommat._canonical import encode, ordered, sha256_ref
from ommat._documents import read_document

OUTLINE_VERSION = 7
CUT = 120  # prompts, warnings and long values are cut to this many characters, then "…"
STATE = "t:state"
STATE_NODE = "state"
# The roles _Loader.graph turns into edges; "candidate", "boundary", "group" and "policy" are not.
EDGE_ROLES = (
    "need",
    "slot",
    "label-set",
    "grounding",
    "text",
    "read",
    "helper",
    "source-requirement",
)
_REF = re.compile(r"^([qtd]):([^.\[\]]+)")

NODE_KEYS = (
    "id",
    "kind",
    "policy",
    "name",
    "refs",
    "round",
    "wavefront",
    "predecessors",
    "fan_in",
    "fan_out",
    "steps_to_policy",
    "type",
    # a source's
    "oracle",
    "needs",
    "requirement",
    "cost",
    "timeout",
    "feedback",
    "grounded",
    "payload",
    "when",
    "questions",
    "tool_results",
    # a rule's
    "form",
    "provides",
    "outputs",
    "reads",
    "params",
    "boundaries",
    "candidates",
    "target",
    "module",
    "qualname",
    "version",
    "source",
    "body",
    "hit",
    "order",
    "inputs",
    "rows",
)
QUESTION_KEYS = (
    "ref",
    "kind",
    "labels",
    "label_set",
    "value_type",
    "is_list",
    "text",
    "reference_date",
    "target",
    "latent_targets",
    "wording",
)
ENTRY_KEYS = (
    "ref",
    "index",
    "round",
    "call",
    "holds",
    "value",
    "marker",
    "derivation",
    "rows",
    "policy",
    "declared_reads",
    "declared_params",
    "declared_inputs",
    "wording",
    "target",
    "latent_targets",
    "bounds",
    "bounds_reason",
    "nearest",
)
BOUND_KEYS = (
    "role",
    "param",
    "bound",
    "by",
    "key",
    "labels",
    "measure",
    "measured",
    "distance",
    "side",
    "per_label",
)
CALL_KEYS = (
    "index",
    "round",
    "sources",
    "outcome",
    "provider",
    "model_requested",
    "model_reported",
    "revision",
    "runtime",
    "elicitation",
    "precision",
    "claimed",
    "snapshot",
    "prompt",
    "questions",
    "unhashed",
)
UNHASHED_KEYS = (
    "start",
    "latency",
    "attempts",
    "usage",
    "request_id",
    "missing",
    "warnings",
    "truncated",
    "clients",
    "invalid",
    "traceback",
)


def parsed(document: object, what: str) -> tuple[dict[str, Any], str | None, str | None]:
    """A document as a dict, its sha256 and where the hash came from: the bytes given, or
    recomputed over the canonical encoding of a parsed dict, both ``None`` when the encoding
    refuses the dict."""
    if isinstance(document, (bytes, bytearray)):
        data = bytes(document)
        return read_document(data, what), "sha256:" + hashlib.sha256(data).hexdigest(), "bytes"
    if isinstance(document, dict):
        held = cast("dict[str, Any]", document)
        try:
            return held, sha256_ref(encode(held)), "recomputed"
        except ValueError:  # a dict the canonical encoding refuses has no hash
            return held, None, None
    raise ValueError(f"{what} must be bytes or a dict, not {type(document).__name__}")


def fixed(keys: tuple[str, ...], values: dict[str, Any]) -> dict[str, Any]:
    """Every key of a fixed shape, in its order, ``None`` where ``values`` holds none."""
    return {k: values.get(k) for k in keys}


def obj(x: object) -> dict[str, Any]:
    """``x`` when it is a dict, else an empty one."""
    return cast("dict[str, Any]", x) if isinstance(x, dict) else {}


def arr(x: object) -> list[Any]:
    """``x`` when it is a list, else an empty one."""
    return cast("list[Any]", x) if isinstance(x, list) else []


def base_of(ref: object) -> str | None:
    """A ref's declared base: level and name, without a grounded bracket or a field path."""
    if not isinstance(ref, str):
        return None
    m = _REF.match(ref)
    return f"{m[1]}:{m[2]}" if m else None


def fields_of(ref: str) -> list[str]:
    """A ref's field path, after any grounded bracket."""
    m = _REF.match(ref)
    if not m:
        return []
    rest = ref[m.end() :]
    if rest.startswith("["):
        i = 2
        while i < len(rest) and rest[i] != '"':
            i += 2 if rest[i] == "\\" else 1
        rest = rest[i + 2 :]
    return [f for f in rest.split(".") if f]


def is_candidate_read(ref: object) -> bool:
    return isinstance(ref, str) and (ref == "candidate" or ref.startswith("candidate."))


def tested_reads(cells: object, inputs: object) -> list[tuple[int, str]]:
    """A table row's tests: each input's index and read whose cell is not ``-``, its read a
    ref and no candidate read, in input order."""
    from ommat._table import ANY

    ins = [obj(i) for i in arr(inputs)]
    out: list[tuple[int, str]] = []
    for k, cell in enumerate(arr(cells)):
        read = ins[k].get("read") if k < len(ins) else None
        if cell != ANY and isinstance(read, str) and not is_candidate_read(read):
            out.append((k, read))
    return out


def text_of(value: object) -> str:
    """A value as the ordered encoding writes it, each object in the order it holds; plain JSON
    when the ordered encoding refuses it."""
    try:
        return ordered(value)
    except ValueError:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def number(x: object) -> TypeGuard[int | float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_dict(x: object) -> bool:
    """Whether ``x`` is a dict, without narrowing its type: ``obj`` gives it typed."""
    return isinstance(x, dict)


def is_list(x: object) -> bool:
    """Whether ``x`` is a list, without narrowing its type: ``arr`` gives it typed."""
    return isinstance(x, list)


def exact(x: float) -> Fraction:
    """The shortest decimal that reads back as ``x``, exactly, as the choice helpers compare."""
    return Fraction(repr(x))


def as_float(f: Fraction) -> float | int:
    return int(f) if f.denominator == 1 and abs(f) < 2**53 else float(f)


def held(value: object, order: object) -> Any:
    """Data rebuilt in the order its order object gives: each object counted from 0 in pre-order
    as the canonical encoding writes it, members in canonical (UTF-16) key order unless the
    order object lists their positions."""
    positions_of = obj(order)
    count = [0]

    def walk(v: object) -> Any:
        if isinstance(v, list):
            return [walk(x) for x in cast("list[Any]", v)]
        if not isinstance(v, dict):
            return v
        members = cast("dict[str, Any]", v)
        index = str(count[0])
        count[0] += 1
        keys = sorted(members, key=lambda k: k.encode("utf-16-be"))
        rebuilt = {k: walk(members[k]) for k in keys}
        positions = positions_of.get(index)
        if isinstance(positions, list) and sorted(cast("list[int]", positions)) == list(
            range(len(keys))
        ):
            return {keys[i]: rebuilt[keys[i]] for i in cast("list[int]", positions)}
        return rebuilt

    return walk(value)
