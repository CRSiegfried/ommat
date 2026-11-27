"""The turn outline: one record document as a plain dict of fixed shape, round by round, with
the design-derived members of each entry when the design it ran under is given."""

from fractions import Fraction
from typing import Any

from ommat import InvalidRecord, Record, defects, escalation, read_record
from ommat._refs import key_path
from ommat.outline._common import (
    BOUND_KEYS,
    CALL_KEYS,
    ENTRY_KEYS,
    OUTLINE_VERSION,
    STATE,
    UNHASHED_KEYS,
    arr,
    as_float,
    base_of,
    exact,
    fields_of,
    fixed,
    is_dict,
    is_list,
    number,
    obj,
    parsed,
)
from ommat.outline._design import params_of, providers_of, reads_of, wording_of
from ommat.outline._path import lineage_of, nearest_bounds, summary_nearest, summary_of, ties_of
from ommat.outline._review import review_of
from ommat.outline._text import why_of

# --- bounds and distances -------------------------------------------------------------------


def _project(value: object, fields: list[str]) -> object:
    for f in fields:
        if not is_dict(value) or f not in obj(value):
            return None
        value = obj(value)[f]
    return value


def _bound_value(leaves: dict[str, Any], path: str) -> object:
    """A bound's value: its leaf's, or for a key path, ``leaf.key``, its leaf's value at the key;
    ``None`` where the document holds neither."""
    keyed = key_path(path)
    if keyed is None:
        return obj(leaves.get(path)).get("value")
    return obj(obj(leaves.get(keyed[0])).get("value")).get(keyed[1])


def bounds_of(design: dict[str, Any], base: str) -> list[dict[str, Any]]:
    """Every bound the paired design declares for a read: the rules' ``boundaries`` fields, a
    table's helper bindings and its comparisons over a bare input against a leaf. Each holds
    its role, leaf path, leaf value, the component naming it, the key a plain bound bounds and
    a mass bound's labels."""
    leaves = obj(design.get("parameters"))
    out: list[dict[str, Any]] = []

    def put(role: str, path: object, by: str, key: str, labels: object = None) -> None:
        if not isinstance(path, str):
            return
        b = {
            "role": role,
            "param": path,
            "bound": _bound_value(leaves, path),
            "by": by,
            "key": key if role == "plain" else None,
            "labels": labels,
        }
        if not any(all(x[k] == b[k] for k in ("role", "param", "key", "labels")) for x in out):
            out.append(b)

    for rid, rule in obj(design.get("rules")).items():
        r = obj(rule)
        for key, entry in obj(r.get("boundaries")).items():
            if base_of(key) != base:
                continue
            if is_list(entry):
                for p in arr(entry):
                    put("plain", p, rid, key)
                continue
            for role in ("bar", "thresholds", "margin", "null", "confidence", "value"):
                for p in arr(obj(entry).get(role)):
                    put(role, p, rid, key)
            for m in arr(obj(entry).get("mass")):
                for p in arr(obj(m).get("bounds")):
                    put("mass", p, rid, key, labels=arr(obj(m).get("labels")))
        if r.get("form") != "table":
            continue
        inputs = [obj(i) for i in arr(r.get("inputs"))]
        for i in inputs:
            if base_of(i.get("read")) != base or i.get("helper") is None:
                continue
            params = obj(i.get("params"))
            for name, role in (
                ("bar", "bar"),
                ("margin", "margin"),
                ("thresholds", "thresholds"),
                ("null_bar", "null"),
            ):
                put(role, params.get(name), rid, str(i.get("read")))
        for row in arr(r.get("rows")):
            for n, cell in enumerate(arr(obj(row).get("cells"))):
                if n >= len(inputs) or inputs[n].get("helper") is not None:
                    continue
                if base_of(inputs[n].get("read")) != base:
                    continue
                for tag, m in obj(cell).items():
                    for o in arr(m) if tag == "between" else [m]:
                        if tag in (">=", ">", "<=", "<", "between") and "param" in obj(o):
                            put("plain", obj(o)["param"], rid, str(inputs[n]["read"]))
    return out


def side_of(role: str, distance: Fraction) -> str | None:
    """``short`` when the value falls short of clearing the bound, ``clear`` when it clears it.
    For every role but two, short is a negative distance. A ``null`` bound (extract's
    ``null_bar``) is cleared from below, as extract abstains at or above it, so there short is a
    distance of 0 or more. A plain bound's direction is the rule's, so it has no side."""
    if role == "plain":
        return None
    if role == "null":
        return "short" if distance >= 0 else "clear"
    return "short" if distance < 0 else "clear"


def measured_bounds(entry: dict[str, Any], bounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each bound beside what the entry's value measures against it and the signed distance,
    measured minus bound, both in the choice helpers' exact decimal arithmetic. A measure the
    value cannot give is ``None``, its bound shown alone."""
    v = entry.get("value") if entry.get("holds") == "value" else None
    value = obj(v)
    out: list[dict[str, Any]] = []
    probs = (
        obj(value.get("probabilities")) if isinstance(value.get("probabilities"), dict) else None
    )
    for b in bounds:
        m: dict[str, Any] = fixed(BOUND_KEYS, b)
        bound = b["bound"]
        measured: Fraction | None = None
        if probs is not None:
            ps = sorted((exact(x) for x in probs.values() if number(x)), reverse=True)
            if b["role"] == "margin" and ps:
                measured, m["measure"] = ps[0] - (ps[1] if len(ps) > 1 else 0), "lead"
            elif b["role"] == "bar" and ps:
                measured, m["measure"] = ps[0], "top probability"
            elif b["role"] == "mass":
                labels: list[str] = b["labels"] or []
                total = sum(
                    (exact(probs[lab]) for lab in labels if number(probs.get(lab))), Fraction(0)
                )
                measured, m["measure"] = min(Fraction(1), total), "mass of " + "+".join(labels)
            elif b["role"] == "thresholds" and is_dict(bound):
                m["measure"] = "probability per label"
                per_label: list[dict[str, Any]] = []
                for lab, t in obj(bound).items():
                    if not number(t):
                        continue
                    p = probs.get(lab)
                    d = (exact(p) if number(p) else Fraction(0)) - exact(t)
                    per_label.append(
                        {
                            "label": lab,
                            "threshold": t,
                            "probability": p,
                            "distance": as_float(d),
                            "side": side_of("thresholds", d),
                        }
                    )
                m["per_label"] = per_label
        elif value.get("kind") == "extraction":
            if b["role"] == "null" and number(value.get("p_null")):
                measured, m["measure"] = exact(value["p_null"]), "p_null"
            elif b["role"] == "confidence" and number(value.get("confidence")):
                measured, m["measure"] = exact(value["confidence"]), "confidence"
        elif b["role"] == "plain":
            x = _project(v, fields_of(b["key"]))
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                measured, m["measure"] = exact(x), b["key"]
        if measured is not None:
            m["measured"] = as_float(measured)
            if number(bound):
                d = measured - exact(bound)
                m["distance"], m["side"] = as_float(d), side_of(b["role"], d)
        out.append(m)
    return out


def nearest_of(bounds: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The bound with the smallest absolute distance, a threshold's per-label ones included;
    the first in bound order on a tie. ``None`` when no bound has a distance."""
    best: dict[str, Any] | None = None
    for b in bounds or []:
        found: list[tuple[Any, Any, Any]] = []
        if b["distance"] is not None:
            found.append((b["distance"], b["bound"], b["side"]))
        found += [(x["distance"], x["threshold"], x["side"]) for x in arr(b["per_label"])]
        for d, bound, side in found:
            if best is None or abs(exact(d)) < abs(exact(best["distance"])):
                best = {
                    "param": b["param"],
                    "bound": bound,
                    "by": b["by"],
                    "distance": d,
                    "side": side,
                }
    return best


def why_unbounded(design: dict[str, Any], base: str) -> str:
    """Why a paired entry has no bounds: the function rules reading it that hold no boundaries
    field, and the tables reading it that bind no leaf to it, or that no rule reads it."""
    rules = obj(design.get("rules"))
    readers = sorted(
        rid for rid, r in rules.items() if any(base_of(x["ref"]) == base for x in reads_of(obj(r)))
    )
    if not readers:
        return "no rule reads it"
    parts: list[str] = []
    functions = [r for r in readers if obj(rules[r]).get("form") != "table"]
    tables = [r for r in readers if obj(rules[r]).get("form") == "table"]
    if functions:
        word = "rule" if len(functions) == 1 else "rules"
        parts.append(f"no boundaries field on {word} {', '.join(functions)}")
    if tables:
        word = "table {} binds" if len(tables) == 1 else "tables {} bind"
        parts.append(word.format(", ".join(tables)) + " no leaf to it")
    return "; ".join(parts)


# --- the turn outline -----------------------------------------------------------------------


def _source_inputs(s: dict[str, Any], ref: str) -> list[dict[str, Any]]:
    """The declared inputs of one ref a source provides, each ``{ref, roles}``, a ref under two
    roles listed once: the source's needs, grounding and requirement, and that question's own
    slot, text, reference-date and label-set refs."""
    found: list[tuple[object, str]] = [(r, "need") for r in arr(s.get("needs"))]
    q = obj(obj(s.get("provides")).get(ref))
    for slot in arr(obj(q.get("wording")).get("slots")):
        o = obj(obj(slot).get("origin"))
        if isinstance(o.get("ref"), str):
            found.append((o["ref"], "slot"))
    for member in ("text", "reference_date"):
        given = obj(q.get(member))
        if isinstance(given.get("ref"), str):
            found.append((given["ref"], member))
    labels = obj(q.get("labels"))
    if isinstance(labels.get("candidates"), str):
        found.append((labels["candidates"], "label-set"))
    if isinstance(s.get("grounded"), str):
        found.append((s["grounded"], "grounding"))
    requirement = obj(s.get("requirement"))
    if isinstance(requirement.get("ref"), str):
        found.append((requirement["ref"], "requirement"))
    out: list[dict[str, Any]] = []
    for r, role in found:
        have = next((x for x in out if x["ref"] == r), None)
        if have is None:
            out.append({"ref": r, "roles": [role]})
        elif role not in have["roles"]:
            have["roles"].append(role)
    return out


def _call(idx: int, item: dict[str, Any], unhashed: dict[str, Any]) -> dict[str, Any]:
    kept = obj(obj(unhashed.get("calls")).get(str(idx)))
    questions = None
    if "questions" in item:
        questions = [
            {"ref": r, "identity": obj(q).get("identity"), "kind": obj(q).get("kind")}
            for r, q in obj(item["questions"]).items()
        ]
    return fixed(
        CALL_KEYS,
        {**item, "index": idx, "questions": questions, "unhashed": fixed(UNHASHED_KEYS, kept)},
    )


def _holds(item: dict[str, Any]) -> str:
    if "marker" in item:
        return str(obj(item["marker"]).get("kind"))
    return "value"


def _entry(idx: int, item: dict[str, Any], policy: str | None) -> dict[str, Any]:
    values = {
        "ref": item.get("ref"),
        "index": idx,
        "round": item.get("round"),
        "call": item.get("call"),
        "holds": _holds(item),
        "value": item.get("value"),
        "marker": item.get("marker"),
        "derivation": item.get("derivation"),
        "rows": item.get("rows"),
        "policy": policy is not None and item.get("ref") == policy,
    }
    return fixed(ENTRY_KEYS, values)


def _pair_entry(e: dict[str, Any], design: dict[str, Any], providers: dict[str, Any]) -> None:
    """Fill an entry's design-derived members. Applicable to the entry's provider: a list or
    value, possibly empty; not applicable (a rule's members on a source's entry, and the
    reverse): ``None``. ``bounds`` is a list for every entry of a paired turn."""
    sources, rules = obj(design.get("sources")), obj(design.get("rules"))
    base = base_of(e["ref"])
    provider = providers.get(base) if base else None
    if provider in rules:
        rule = obj(rules[provider])
        e["declared_reads"] = reads_of(rule)
        e["declared_params"] = params_of(rule)
        e["target"] = rule.get("target")
    elif provider in sources and base is not None:
        s = obj(sources[provider])
        e["declared_inputs"] = _source_inputs(s, base)
        q = obj(obj(s.get("provides")).get(base))
        if "kind" in q:
            e["wording"] = wording_of(q)
            e["target"] = q.get("target")
            e["latent_targets"] = q.get("latent_targets") or []
    e["bounds"] = measured_bounds(e, bounds_of(design, base)) if base else []
    if not e["bounds"] and base:
        e["bounds_reason"] = why_unbounded(design, base)
    e["nearest"] = nearest_of(e["bounds"])


def turn_outline(record_document: object, design_document: object = None) -> dict[str, Any]:
    """One record document, ``bytes`` or parsed, as a plain dict of fixed shape: header,
    settings, state, rounds of calls and entries, the decision and, when undecided, the
    escalation. With the design document it ran under, each entry gains its declared reads,
    inputs and leaves, each question its wording and targets, and each entry the bounds its
    design declares for it, with the distance to each; ``absent`` lists the refs the design
    provides that no entry wrote, ``review`` sorts each source and table row by what the turn
    did with it, and ``nearest_bounds`` lists every bound nearest first. The
    ``lineage`` from the decision back to what it rests on and the one-line ``summary`` are
    computed from the documents alone. A record ``read_record`` refuses is read raw."""
    doc, _, _ = parsed(record_document, "the record")
    record: Record | None = None
    read_error: str | None = None
    try:
        record = read_record(record_document)
        log = record.log
        read_by = "read_record"
    except InvalidRecord as exc:  # shown as the raw document
        log = [obj(i) for i in arr(doc.get("log")) if is_dict(i)]
        read_by, read_error = "raw", str(exc)
    unhashed = obj(doc.get("unhashed"))
    first = log[0] if log and log[0].get("item") == "design" else {}
    settings = next((i for i in log if i.get("item") == "settings"), obj(None))
    given_policy = settings.get("policy")
    policy = given_policy if isinstance(given_policy, str) else None
    entries = {
        i["ref"]: i for i in log if i.get("item") == "entry" and isinstance(i.get("ref"), str)
    }
    cause = next((i for i in log if i.get("item") == "cause"), None)

    if log and log[-1].get("item") == "cause":
        status, reason = "unfinished", "the log ends at a cause item"
    elif policy is not None and policy not in entries:
        status, reason = "unfinished", "the log holds no entry at the policy's ref"
    else:
        status, reason = "finished", None

    design: dict[str, Any] | None = None
    matches: bool | None = None
    design_name = None
    if design_document is not None:
        design, ddigest, _ = parsed(design_document, "the design")
        design_name = design.get("name")
        if ddigest is not None and first.get("hash") is not None:
            matches = ddigest == first["hash"]

    rounds: dict[int, dict[str, Any]] = {}

    def col(n: object) -> dict[str, Any]:
        key = n if isinstance(n, int) else -1
        empty: dict[str, Any] = {
            "round": n,
            "closed": False,
            "cost": None,
            "deadline": None,
            "calls": [],
            "entries": [],
        }
        return rounds.setdefault(key, empty)

    legacy: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    for idx, item in enumerate(log):
        kind = item.get("item")
        if kind == "call":
            c = _call(idx, item, unhashed)
            col(item.get("round"))["calls"].append(c)
            all_calls.append(c)
        elif kind == "entry":
            e = _entry(idx, item, policy)
            col(item.get("round"))["entries"].append(e)
            all_entries.append(e)
        elif kind == "round":
            r = col(item.get("round"))
            r.update(closed=True, cost=item.get("cost"), deadline=item.get("deadline"))
        elif kind in ("plan", "verdict"):
            legacy.append({"index": idx, **{k: v for k, v in item.items() if k != "order"}})

    providers = obj(settings.get("providers"))
    absent = None
    design_providers = providers
    if design is not None:
        design_providers = providers or providers_of(design)
        for e in all_entries:
            _pair_entry(e, design, design_providers)
        written = {base_of(r) for r in entries}
        absent = sorted(r for r in providers_of(design) if base_of(r) not in written)
    lineage = lineage_of(all_entries, policy, design, design_providers)

    st = entries.get(STATE)
    decision = None
    if policy is not None and policy in entries:
        d = entries[policy]
        decision = {
            "ref": policy,
            "round": d.get("round"),
            "holds": _holds(d),
            "value": d.get("value"),
            "marker": d.get("marker"),
            "rows": d.get("rows"),
            "derivation": d.get("derivation"),
        }
    esc, found_defects = None, None
    if record is not None:
        esc = None if policy is None else escalation(record, policy)
        found = defects(record)
        found_defects = None if found is None else [x["ref"] for x in found]
    turn = obj(unhashed.get("turn"))
    outline: dict[str, Any] = {
        "outline_version": OUTLINE_VERSION,
        "kind": "turn",
        "read_by": read_by,
        "read_error": read_error,
        "record_hash": doc.get("hash"),
        "design_hash": first.get("hash"),
        "record_format": first.get("format"),
        "helpers": first.get("helpers"),
        "engine": unhashed.get("engine"),
        "turn": {"start": turn.get("start"), "end": turn.get("end")} if turn else None,
        "status": status,
        "status_reason": reason,
        "settings": obj(settings.get("settings")) or None,
        "policy": policy,
        "providers": providers or None,
        "grounded": settings.get("grounded"),
        "sets": settings.get("sets"),
        "paired": design is not None,
        "design_name": design_name,
        "design_hash_matches": matches,
        "state": None
        if st is None
        else {"holds": _holds(st), "value": st.get("value"), "marker": st.get("marker")},
        "rounds": [rounds[k] for k in sorted(rounds)],
        "cause": None if cause is None else cause.get("cause"),
        "legacy_items": legacy,
        "decision": decision,
        "escalation": esc,
        "defects": found_defects,
        "absent": absent,
        "review": review_of(design, design_providers, all_entries, all_calls),
        "lineage": lineage,
        "nearest_bounds": None if design is None else nearest_bounds(all_entries, lineage),
    }
    outline["ties"] = ties_of(outline["nearest_bounds"])
    outline["summary"] = summary_of(outline)
    outline["summary_nearest"] = summary_nearest(outline)
    outline["why"] = why_of(outline)
    return outline
