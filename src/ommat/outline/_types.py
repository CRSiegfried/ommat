"""The document's type texts parsed into one shape, a target's values text split, and a table's
tagged cells written back in the notation of §decision-tables."""

import contextlib
import json
import re
from typing import Any, cast

from ommat._table import ANY, ANY_MARKER, KNOWN, MARKERS, cell_form
from ommat.outline._common import obj, text_of

_SCALARS = ("string", "boolean", "bytes")


def _split_top(text: str, sep: str) -> list[str]:
    """``text`` split at each ``sep`` outside brackets and quotes."""
    parts: list[str] = []
    depth, quoted, start, i = 0, False, 0, 0
    while i < len(text):
        c = text[i]
        if quoted:
            if c == "\\":
                i += 1
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c in "{[<(":
            depth += 1
        elif c in "}]>)":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            parts.append(text[start:i])
            start = i + len(sep)
            i += len(sep)
            continue
        i += 1
    parts.append(text[start:])
    return parts


def parse_type(text: object) -> dict[str, Any] | None:
    """A type as the document spells it, parsed: ``{text, kind, labels, range, of, fields}``.
    ``kind`` is label, number, integer, string, boolean, bytes, list, mapping, record,
    nullable, named (a ``types`` name) or unparsed. ``range`` is ``[lo, hi]`` of a bounded
    number or integer (an open end ``None``); ``of`` the element, value or non-null type;
    ``fields`` a record's ``[{name, type}]`` in declared order. ``None`` for no type text."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    out: dict[str, Any] = {
        "text": text,
        "kind": "unparsed",
        "labels": None,
        "range": None,
        "of": None,
        "fields": None,
    }
    alts = _split_top(t, " or ")
    if len(alts) == 2 and alts[1].strip() == "null":
        out.update(kind="nullable", of=parse_type(alts[0]))
        return out
    if t.startswith("label{") and t.endswith("}"):
        with contextlib.suppress(ValueError):
            out.update(kind="label", labels=json.loads("[" + t[6:-1] + "]"))
        return out
    if t.startswith("list<") and t.endswith(">"):
        out.update(kind="list", of=parse_type(t[5:-1]))
        return out
    if t.startswith("mapping<") and t.endswith(">"):
        kv = _split_top(t[8:-1], ",")
        out.update(kind="mapping", of=parse_type(kv[-1].strip()))
        return out
    if t.startswith("record{") and t.endswith("}"):
        fields: list[dict[str, Any]] = []
        for part in _split_top(t[7:-1], ","):
            if part.strip():
                name, _, ftype = part.partition(":")
                fields.append({"name": name.strip(), "type": parse_type(ftype.strip())})
        out.update(kind="record", fields=fields)
        return out
    m = re.fullmatch(r"(number|integer)(?:\s*\[\s*([^,\]]*)\s*,\s*([^\]]*)\s*\])?", t)
    if m:
        rng = None
        if m[2] is not None:
            rng = [json.loads(x) if x.strip() else None for x in (m[2], m[3])]
        out.update(kind=m[1], range=rng)
        return out
    if t in _SCALARS:
        out["kind"] = t
    elif re.fullmatch(r"[A-Za-z_][\w/]*", t):
        out["kind"] = "named"
    return out


def target_values(text: object) -> list[Any] | None:
    """A target's ``values`` text as a list: a label type's labels, else the text split at
    commas and at " or ". ``None`` for no text."""
    if not isinstance(text, str):
        return None
    t = parse_type(text)
    if t and t["kind"] == "label" and t["labels"] is not None:
        return cast("list[Any]", t["labels"])
    return [p.strip() for chunk in text.split(",") for p in chunk.split(" or ") if p.strip()]


def cell_text(cell: object) -> str:
    """A tagged cell in the notation of §decision-tables, checked to read back to the same
    tagged cell through ``_table.cell_form``; a bool or int value cell as its JSON (no notation
    string reads as one); otherwise the tagged cell as JSON."""
    tagged = obj(cell)
    if list(tagged) == ["is"] and not isinstance(tagged["is"], str):
        return text_of(tagged["is"])
    text = _cell_guess(cell)
    if text is not None:
        try:
            if _same_cell(cell_form(text), cell):
                return text
        except ValueError:
            pass
    return text_of(cell)


def _same_cell(a: object, b: object) -> bool:
    """Two tagged cells alike, a ``markers`` or ``one_of`` list compared as a set: the document
    holds it sorted, the notation reads it in its own order."""
    x, y = obj(a), obj(b)
    if isinstance(a, dict) and isinstance(b, dict) and list(x) == list(y) and len(x) == 1:
        ((tag, v),) = x.items()
        if tag in ("markers", "one_of") and isinstance(v, list) and isinstance(y[tag], list):
            mine = cast("list[Any]", v)
            theirs = cast("list[Any]", y[tag])
            return sorted(map(text_of, mine)) == sorted(map(text_of, theirs))
    return a == b


def _operand(o: object) -> str:
    held = obj(o)
    if "param" in held:
        return str(held["param"])
    if "ref" in held:
        return str(held["ref"])
    if "literal" in held:
        return text_of(held["literal"])
    return text_of(held)


def _cell_guess(cell: object) -> str | None:
    if cell == ANY:
        return ANY
    tagged = obj(cell)
    if not isinstance(cell, dict) or len(tagged) != 1:
        return None
    ((tag, v),) = tagged.items()
    if tag == "known":
        return KNOWN
    if tag == "marker":
        return str(v)
    if tag == "markers":
        kinds = cast("list[Any]", v)
        if set(kinds) == set(MARKERS):
            return ANY_MARKER
        return " or ".join(str(k) for k in kinds)
    if tag == "is":
        return v if isinstance(v, str) else None
    if tag == "one_of":
        return "{" + ", ".join(str(x) for x in cast("list[Any]", v)) + "}"
    if tag in (">=", ">", "<=", "<"):
        return f"{tag} {_operand(v)}"
    if tag == "between":
        pair = cast("list[Any]", v)
        return f"[{_operand(pair[0])}, {_operand(pair[1])}]"
    if tag == "in":
        return f"in {v}"
    return None


def output_text(output: object) -> str | None:
    """A row's tagged output in the notation: a literal, a marker, ``Input(ref)`` or
    ``Param(path)``."""
    if output is None:
        return None
    o = obj(output)
    if "literal" in o:
        v = o["literal"]
        return v if isinstance(v, str) else text_of(v)
    if "marker" in o:
        return str(o["marker"])
    if "ref" in o:
        return f"Input({o['ref']})"
    if "param" in o:
        return f"Param({o['param']})"
    return text_of(output)


def input_text(i: dict[str, Any]) -> str:
    """A table input in the notation: its read, or its helper over the read with its bindings."""
    if i.get("helper") is None:
        return str(i.get("read"))
    params = ", ".join(f"{k}={v}" for k, v in obj(i.get("params")).items())
    return f"{i['helper']}({i.get('read')}{', ' + params if params else ''})"
