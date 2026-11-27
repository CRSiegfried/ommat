"""What the turn outline computes from its entries alone: the lineage from the decision back to
what it rests on, the bounds nearest to being crossed, and one summary line."""

from typing import Any

from ommat.outline._common import (
    arr,
    as_float,
    base_of,
    exact,
    obj,
    tested_reads,
    text_of,
)


def _hit_reads(e: dict[str, Any], design: dict[str, Any], provider: object) -> list[str] | None:
    """The reads of a table's inputs whose cell in a row the entry hit is not ``-``, in row
    and then input order; ``None`` when the entry's rows name no row of a table."""
    rule = obj(obj(design.get("rules")).get(provider)) if isinstance(provider, str) else {}
    rows, inputs = arr(rule.get("rows")), [obj(i) for i in arr(rule.get("inputs"))]
    hit = [n for n in arr(e["rows"]) if isinstance(n, int) and 1 <= n <= len(rows)]
    if rule.get("form") != "table" or not hit:
        return None
    out: list[str] = []
    for n in hit:
        for _, read in tested_reads(obj(rows[n - 1]).get("cells"), inputs):
            out += [] if read in out else [read]
    return out


def _followed(
    e: dict[str, Any], design: dict[str, Any] | None, providers: dict[str, Any]
) -> list[tuple[str, str]]:
    """The refs an entry rests on that its record names, each with how it was found: a
    derivation's ``rested_on``, or its ``refs`` where that is empty; else, for a table's entry
    and paired, the hit rows' inputs when its rows name some; and then a marker's cause refs.
    Empty when the record names none, whatever the entry declares."""
    out: list[tuple[str, str]] = []
    derivations = [obj(d) for d in arr(e["derivation"])]
    if derivations:
        for d in derivations:
            rested = [r for r in arr(d.get("rested_on")) if isinstance(r, str)]
            if rested:
                out += [(r, "rested_on") for r in rested]
            else:
                out += [(r, "refs") for r in arr(d.get("refs")) if isinstance(r, str)]
    elif e["rows"] is not None and design is not None:
        reads = _hit_reads(e, design, providers.get(base_of(e["ref"]) or ""))
        if reads is not None:
            out += [(r, "row") for r in reads]
    if e["holds"] != "value":
        cause = obj(obj(e["marker"]).get("cause"))
        out += [(r, "cause") for r in arr(cause.get("refs")) if isinstance(r, str)]
    return out


def _declared(e: dict[str, Any]) -> list[tuple[str, str]]:
    """The reads an entry's design declares for its rule, each with ``declared``."""
    return [(r["ref"], "declared") for r in arr(e["declared_reads"])]


def lineage_of(
    entries: list[dict[str, Any]],
    root: str | None,
    design: dict[str, Any] | None,
    providers: dict[str, Any],
) -> list[dict[str, Any]]:
    """From the entry at ``root``, the ref the caller names (the turn outline names the
    record's policy), back to what it rests on, depth first, each entry once: each
    ``{ref, holds, via, from, index}``, ``via`` how it was found from the entry at ``from``,
    both ``None`` for the root's. A ref no entry holds closes its branch, ``holds`` and
    ``index`` ``None``. Empty with no ``root`` or no entry at it.

    The walk has two phases. The first follows only what the record names (``rested_on``,
    ``refs``, ``row``, ``cause``). The second then follows, from each entry the first found no
    such hop for, the reads its design declares, each ``declared``, and from the entries
    those reach what their records name, so a ref a record names is never taken by a
    declared hop."""
    by_ref = {e["ref"]: e for e in entries}

    def find(ref: str) -> dict[str, Any] | None:
        return by_ref.get(ref) or by_ref.get(base_of(ref) or "")

    if root is None or find(root) is None:
        return []
    lineage: list[dict[str, Any]] = []
    seen: set[str] = set()
    bare: list[dict[str, Any]] = []  # entries the record names no hop for, in walk order

    def walk(stack: list[tuple[str, str | None, str | None]], declared: bool) -> None:
        while stack:
            ref, via, parent = stack.pop()
            e = find(ref)
            key = str(e["ref"]) if e is not None else ref
            if key in seen:
                continue
            seen.add(key)
            lineage.append(
                {
                    "ref": key,
                    "holds": None if e is None else e["holds"],
                    "via": via,
                    "from": parent,
                    "index": None if e is None else e["index"],
                }
            )
            if e is not None:
                followed = _followed(e, design, providers)
                if not followed:
                    if declared:
                        followed = _declared(e)
                    else:
                        bare.append(e)
                stack += [(r, how, key) for r, how in reversed(followed)]

    walk([(root, None, None)], False)
    for e in bare:
        walk([(r, "declared", str(e["ref"])) for r, _ in reversed(_declared(e))], True)
    return lineage


def precise_refs(lineage: list[dict[str, Any]]) -> set[str]:
    """The refs reached from the policy's entry by hops the record names alone: the policy's,
    and each whose hop is not ``declared`` and whose ``from`` is one of them."""
    out: set[str] = set()
    for p in lineage:
        if p["from"] is None or (p["via"] != "declared" and p["from"] in out):
            out.add(p["ref"])
    return out


def root_chain(lineage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The lineage's first branch: the policy's entry and each next entry found from the one
    before, to the first that rests on nothing new, the root. It stops before a ``declared``
    hop, since a declared read is not a record of what the turn rested on."""
    chain = lineage[:1]
    for item in lineage[1:]:
        if item["from"] != chain[-1]["ref"] or item["via"] == "declared":
            break
        chain.append(item)
    return chain


def _nearest_of(b: dict[str, Any]) -> tuple[Any, Any]:
    """A bound's distance and side, a threshold's the nearest of its labels', the first on a
    tie."""
    if b["distance"] is not None:
        return b["distance"], b["side"]
    best: tuple[Any, Any] = (None, None)
    for x in arr(b["per_label"]):
        if x["distance"] is not None and (
            best[0] is None or abs(exact(x["distance"])) < abs(exact(best[0]))
        ):
            best = (x["distance"], x["side"])
    return best


def nearest_bounds(
    entries: list[dict[str, Any]], lineage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every bound of every entry, each ``{ref, param, role, by, distance, side, on_lineage}``,
    nearest first: by absolute distance, a bound with none last, then ref and param.
    ``on_lineage`` holds for an entry reached by hops the record names alone."""
    on = precise_refs(lineage)
    out: list[dict[str, Any]] = []
    for e in entries:
        for b in arr(e["bounds"]):
            distance, side = _nearest_of(b)
            out.append(
                {
                    "ref": e["ref"],
                    "param": b["param"],
                    "role": b["role"],
                    "by": b["by"],
                    "distance": distance,
                    "side": side,
                    "on_lineage": e["ref"] in on,
                }
            )

    def key(m: dict[str, Any]) -> tuple[Any, ...]:
        far = m["distance"] is None
        return (far, 0 if far else abs(exact(m["distance"])), m["ref"], m["param"], m["role"])

    return sorted(out, key=key)


def ties_of(near: list[dict[str, Any]] | None) -> int | None:
    """How many of the nearest bounds lie as far as the first, 1 when none does, ``None``
    unpaired or when no bound has a distance."""
    if not near or near[0]["distance"] is None:
        return None
    first = abs(exact(near[0]["distance"]))
    return sum(1 for m in near if m["distance"] is not None and abs(exact(m["distance"])) == first)


def _hex(h: object) -> str:
    return h[7:19] if isinstance(h, str) and h.startswith("sha256:") and len(h) >= 19 else "-"


def summary_nearest(o: dict[str, Any]) -> dict[str, Any] | None:
    """The one bound the summary names: of the bounds with a distance, the nearest of the
    lineage's first branch, else of the lineage, else of all; ``None`` when none has one."""
    misses: list[dict[str, Any]] = o["nearest_bounds"] or []
    measured = [m for m in misses if m["distance"] is not None]
    on_chain = {p["ref"] for p in root_chain(o["lineage"])}
    return next(
        (m for m in measured if m["ref"] in on_chain),
        next((m for m in measured if m["on_lineage"]), measured[0] if measured else None),
    )


def summary_of(o: dict[str, Any]) -> str:
    """One line of fixed columns, two spaces apart: the record hash's first 12 hex digits,
    status, decision (``-`` for a record that names no policy), root, defect, rounds, nearest
    bound (the nearest of the lineage's first branch, else of the lineage, else of all) and the
    design hash's 12 digits."""
    d = o["decision"]
    if d is None:
        # A record that names no policy names no decision to miss.
        decided = "-" if o["policy"] is None else "no-decision"
    elif d["holds"] == "value":
        decided = "decided " + text_of(d["value"]).replace("  ", " \\u0020")
    else:
        cause = obj(obj(d["marker"]).get("cause"))
        refs = ", ".join(map(str, arr(cause.get("refs"))))
        decided = f"undecided {d['holds']} {cause.get('code')} ({refs})"
    chain = root_chain(o["lineage"])
    esc = o["escalation"]
    defect = "-" if esc is None else ("yes" if esc.get("defect") else "no")
    rounds = sum(1 for r in o["rounds"] if isinstance(r["round"], int) and r["round"] >= 1)
    near = summary_nearest(o)
    nearest = "nearest -"
    if near is not None:
        nearest = (
            f"nearest {text_of(near['distance'])} {near['side'] or '-'} "
            f"{near['param']}@{near['ref']}"
        )
    return "  ".join(
        (
            _hex(o["record_hash"]),
            str(o["status"]),
            decided,
            f"root {chain[-1]['ref'] if chain else '-'}",
            f"defect {defect}",
            f"rounds {rounds}",
            nearest,
            f"design {_hex(o['design_hash'])}",
        )
    )


def gap(distance: object) -> str:
    """A distance's absolute value, as the ordered encoding writes it."""
    return text_of(as_float(abs(exact(distance)))) if isinstance(distance, (int, float)) else "-"
