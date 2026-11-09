"""The text form of an outline: one node, call or entry per line, detail lines indented two
more spaces, no colour and no wrapping, the same bytes for the same outline."""

import json
import re
from typing import Any

from ommat.outline._common import CUT, STATE, arr, base_of, is_dict, is_list, number, obj, text_of
from ommat.outline._path import gap, root_chain


def cut(text: str, n: int = CUT) -> str:
    """``text`` on one line, cut to ``n`` characters with ``…``."""
    text = text.replace("\n", "\\n")
    return text if len(text) <= n else text[: n - 1] + "…"


def short(h: object) -> str:
    """A sha256 ref to its first 12 hex digits."""
    if isinstance(h, str) and h.startswith("sha256:") and len(h) > 19:
        return h[:19] + "…"
    return str(h)


def p2(x: object) -> str:
    return f"{x:.2f}" if number(x) else text_of(x)


def _q(s: object) -> str:
    return json.dumps(s, ensure_ascii=False) if isinstance(s, str) else text_of(s)


def _type_text(t: object) -> str:
    return str(obj(t)["text"]) if is_dict(t) else ""


def _wording_lines(w: object, pad: str) -> list[str]:
    if not w:
        return []
    wording = obj(w)
    t = wording["template"]
    out = [f"{pad}wording {cut(_q(t) if isinstance(t, str) else text_of(t))}"]
    for s in arr(wording.get("slots")):
        origin = obj(obj(s).get("origin"))
        tag, v = next(iter(origin.items())) if len(origin) == 1 else ("?", text_of(origin))
        out.append(f"{pad}slot {obj(s).get('name')} <- {tag} {v}")
    return out


def _target_lines(target: object, latent: object, pad: str, word: str = "asks") -> list[str]:
    out = [f"{pad}{word} {target}"] if target else []
    for lt in arr(latent):
        out.append(f"{pad}bears on {obj(lt).get('target')} ({obj(lt).get('role')})")
    return out


def render_design(o: dict[str, Any]) -> str:
    """The text form of a design outline."""
    lines: list[str] = []
    head = ["design", _q(o["name"]) if o["name"] is not None else "unnamed"]
    if o["format"] is not None:
        head.append(f"format {o['format']}")
    if o["hash"] is not None:
        head.append(o["hash"] + ("" if o["hash_source"] == "bytes" else " (recomputed)"))
    if o["policy_node"] is not None:
        head.append(f"policy {o['policy'] or o['policy_node']}")
    lines.append("  ".join(head))
    if o["other_members"]:
        lines.append(
            "  not shown: "
            + ", ".join(o["other_members"])
            + " (members this outline does not read)"
        )
    if o["parameters"]:
        lines.append("parameters")
        for p in o["parameters"]:
            lines.append(f"  {p['path']}  {_type_text(p['type'])}  = {cut(text_of(p['value']))}")
    if o["types"]:
        lines.append("types")
        for t in o["types"]:
            lines.append(f"  {t['name']} = {_type_text(t['type'])}")
    for n in o["nodes"]:
        lines.extend(_design_node(n, o["policy_node"] is not None))
    lines.append("edges (the dependency graph)")
    dependency = [e for e in o["edges"] if e["dependency"]]
    lines.extend(f"  {e['from']} -> {e['to']}  {e['role']} {e['ref']}" for e in dependency)
    if not dependency:
        lines.append("  none")
    state = [e for e in o["edges"] if not e["dependency"]]
    if state:
        lines.append("state edges (not in the dependency graph)")
        lines.extend(f"  {e['from']} -> {e['to']}  {e['role']} {e['ref']}" for e in state)
    if o["requirements"]:
        lines.append("requirements")
        by_id = {n["id"]: n for n in o["nodes"]}
        for r in o["requirements"]:
            req = obj(by_id[r["source"]]["requirement"])
            lines.append(
                f"  {r['source']} requires {req.get('ref')} in {text_of(req.get('values'))}"
                f" from {r['provider'] or 'none'}"
            )
            lines.append(f"    decided from {', '.join(r['decided_from']) or 'nothing'}")
            lines.append(f"    behind {', '.join(r['behind']) or 'nothing'}")
    for g in o["groups"]:
        lines.append(
            f"group {g['name']}  refs {', '.join(g['refs'] or [])}  "
            f"reason {', '.join(g['reason'] or [])}"
        )
    for t in o["targets"]:
        extra = f"  values {cut(t['values_text'])}" if t["values_text"] is not None else ""
        lines.append(f"target {t['id']}  {_q(cut(str(t['statement'])))}{extra}")
        for q in t["questions"]:
            how = "asked by" if q["role"] == "asks" else str(q["role"])
            lines.append(f"  {how} {q['ref']} ({q['node']})")
        if t["estimated_by"]:
            lines.append(f"  estimated by {', '.join(t['estimated_by'])}")
        if t["readers"]:
            lines.append(f"  read by {', '.join(t['readers'])}")
        if t["path"]:
            lines.append(f"  path {' -> '.join(t['path'])}")
    if o["classes"]:
        lines.append("classes  " + ", ".join(f"{c['ref']} {c['class']}" for c in o["classes"]))
    return "\n".join(lines) + "\n"


def _fn(n: dict[str, Any]) -> str:
    parts: list[str] = []
    if n.get("module") is not None or n.get("qualname") is not None:
        parts.append(f"{n.get('module')}.{n.get('qualname')}")
    if n.get("version") is not None:
        parts.append(f"version {n['version']}")
    if n.get("source") is not None:
        parts.append(f"source {short(n['source'])}")
    return "  ".join(parts)


def _question_lines(q: dict[str, Any]) -> list[str]:
    bits = [f"  {q['ref']}", q["kind"] or ""]
    labels: list[dict[str, Any]] = q["labels"] or []
    described = any(lab["description"] is not None for lab in labels)
    if labels and not described:
        bits.append("labels " + ", ".join(str(lab["name"]) for lab in labels))
    if q["label_set"] is not None:
        bits.append("label set " + text_of(q["label_set"]))
    if q["kind"] == "extraction":
        bits.append(f"{q['value_type']}{' list' if q['is_list'] else ''}")
        if q["text"]:
            bits.append(f"text {obj(q['text']).get('ref', text_of(q['text']))}")
        if q["reference_date"]:
            bits.append(f"reference_date {obj(q['reference_date']).get('ref')}")
    out = ["  ".join(bits)]
    if described:
        for lab in labels:
            desc = f"  {_q(cut(lab['description']))}" if lab["description"] is not None else ""
            out.append(f"    label {lab['name']}{desc}")
    out.extend(_wording_lines(q["wording"], "    "))
    out.extend(_target_lines(q["target"], q["latent_targets"], "    "))
    return out


def _source_lines(n: dict[str, Any], preds: str) -> list[str]:
    out: list[str] = []
    head = [f"[source] {n['name']}"]
    oracle = n["oracle"]
    head.append(f"oracle {oracle if isinstance(oracle, str) else text_of(oracle)}")
    if n["needs"]:
        head.append("needs " + ", ".join(n["needs"]))
    for k in ("cost", "timeout"):
        if n[k] is not None:
            head.append(f"{k} {n[k]}")
    if n["feedback"]:
        head.append("feedback")
    out.append("  ".join(head) + preds)
    if n["when"]:
        out.append(f"  when {_fn(n['when'])}")
    if n["requirement"]:
        r = obj(n["requirement"])
        out.append(f"  requirement {r.get('ref')} in {text_of(r.get('values'))}")
    if n["grounded"]:
        out.append(f"  grounded on {n['grounded']}")
    if n["payload"] is not None:
        out.append(f"  payload {cut(text_of(n['payload']))}")
    for q in n["questions"]:
        out.extend(_question_lines(q))
    for t in n["tool_results"]:
        extra = "  candidates" if t["candidates"] else ""
        out.append(f"  {t['ref']}  tool result  {_type_text(t['type'])}{extra}")
    return out


def _design_node(n: dict[str, Any], has_policy: bool) -> list[str]:
    if n["kind"] == "state":
        return [f"[state]  {STATE}  {_type_text(n['type'])}"]
    preds = f"  predecessors {', '.join(n['predecessors'])}" if n["predecessors"] else ""
    if has_policy and not n["policy"]:
        steps = n["steps_to_policy"]
        preds += "  no path to the policy" if steps is None else f"  to policy {steps}"
    if n["kind"] == "source":
        return _source_lines(n, preds)
    out: list[str] = []
    tag = "[rule]  "
    policy = " [policy]" if n["policy"] else ""
    provides = f"-> {n['provides']}  " if n["provides"] else ""
    outputs = _type_text(n["outputs"])
    if n["form"] == "table":
        out.append(f"{tag} {n['name']}  table {n['hit']}  {provides}{outputs}{preds}{policy}")
        if n["order"] is not None:
            out.append(f"  order {', '.join(map(str, n['order']))}")
        aggregate = n["hit"] == "C#"
        cols = ["#", *(i["text"] for i in n["inputs"])] + ([] if aggregate else ["out"])
        rows = [
            [str(r["n"]), *(c["text"] for c in r["cells"])]
            + ([] if aggregate else [str(r["output"]["text"])])
            for r in n["rows"]
        ]
        widths = [max(len(c[k]) for c in [cols, *rows]) for k in range(len(cols))]
        for row in [cols, *rows]:
            cells = [row[k].ljust(widths[k]) for k in range(len(row))]
            if not aggregate:
                cells = [*cells[:-1], "|", cells[-1]]
            out.append("  " + "  ".join(cells).rstrip())
        if n["params"]:
            out.append("  params " + ", ".join(f"{p['param']} ({p['via']})" for p in n["params"]))
    else:
        out.append(f"{tag} {n['name']}  {n['form'] or 'function'}  {_fn(n)}{preds}{policy}")
        reads = ", ".join(r["ref"] for r in n["reads"]) or "nothing"
        out.append(f"  reads {reads}  {provides}{outputs}".rstrip())
        if n["params"]:
            out.append(f"  params {', '.join(p['param'] for p in n['params'])}")
        for key, entry in obj(n["boundaries"]).items():
            out.append(f"  boundaries {key} {_bounds_text(entry)}")
        out.append(f"  body {n['body']}")
    if n["target"]:
        out.append(f"  estimates {n['target']}")
    if n["grounded"]:
        out.append(f"  grounded on {n['grounded']}")
    if n["candidates"]:
        out.append("  candidates")
    return out


def _bounds_text(entry: object) -> str:
    if is_list(entry):
        return ", ".join(map(str, arr(entry)))
    parts: list[str] = []
    for role, v in obj(entry).items():
        if role == "mass":
            for m in arr(v):
                parts.append(
                    f"mass {'+'.join(obj(m).get('labels', []))} "
                    f"{', '.join(obj(m).get('bounds', []))}"
                )
        elif v:
            parts.append(f"{role} {', '.join(map(str, arr(v)))}")
    return "  ".join(parts) or "none"


def _value_text(v: object) -> str:
    """A value for the text form. A distribution: each label and its probability to two
    decimals, in label order. An extraction: value always; guess only when it differs from
    value; p_null and confidence when not ``None``; span only when not ``None``."""
    value = obj(v)
    if isinstance(value.get("probabilities"), dict):
        probs = obj(value["probabilities"])
        given = [x if isinstance(x, str) else obj(x).get("name") for x in arr(value.get("labels"))]
        labels = given or list(probs)
        shown = "  ".join(f"{lab} {p2(probs.get(str(lab)))}" for lab in labels)
        return f"{value.get('kind', '')}  " + shown
    if value.get("kind") == "extraction":
        bits = [
            f"extraction {value.get('value_type')}{' list' if value.get('is_list') else ''}",
            f"value {cut(text_of(value.get('value')), 60)}",
        ]
        if value.get("guess") != value.get("value"):
            bits.append(f"guess {cut(text_of(value.get('guess')), 60)}")
        for k in ("p_null", "confidence"):
            if value.get(k) is not None:
                bits.append(f"{k} {p2(value[k])}")
        if value.get("span") is not None:
            bits.append(f"span {text_of(value['span'])}")
        return "  ".join(bits)
    return cut(text_of(v))


def _cause_text(c: object) -> str:
    cause = obj(c)
    refs = f" ({', '.join(cause['refs'])})" if cause.get("refs") else ""
    return f"{cause.get('code')}{refs}"


def _marker_text(m: object) -> str:
    return f"{obj(m).get('kind')} {_cause_text(obj(m).get('cause'))}"


def render_turn(o: dict[str, Any], *, detail: bool = False) -> str:
    """The text form of a turn outline; each call's detail lines folded into one, or with
    ``detail`` each on its own line."""
    lines: list[str] = []
    head = [f"turn {o['record_hash']}"]
    for k, word in (
        ("design_hash", "design"),
        ("record_format", "format"),
        ("helpers", "helpers"),
        ("engine", "engine"),
    ):
        if o[k] is not None:
            head.append(f"{word} {o[k]}")
    head.append(o["status"])
    lines.append("  ".join(head))
    by_index = {e["index"]: e for r in o["rounds"] for e in r["entries"]}
    lines.append(_why_line(o, by_index))
    if o["status_reason"]:
        lines.append(f"  {o['status']}: {o['status_reason']}")
    if o["read_by"] == "raw":
        lines.append(f"  read raw: read_record refused it: {cut(o['read_error'] or '')}")
    if o["paired"]:
        name = _q(o["design_name"]) if o["design_name"] is not None else "unnamed"
        match = o["design_hash_matches"]
        note = {True: " (hash matches)", False: " (hash differs)"}.get(
            match, "  (hash not checked)"
        )
        lines.append(f"paired with design {name}{note}")
    else:
        lines.append("no design paired: declared reads, wording and bounds not shown")
    if o["turn"]:
        lines.append(f"time  {o['turn']['start']} .. {o['turn']['end']}")
    if o["settings"]:
        shown = "  ".join(f"{k} {text_of(v)}" for k, v in o["settings"].items())
        lines.append("settings  " + shown)
    if o["policy"]:
        lines.append(f"policy  {o['policy']}")
    if o["providers"]:
        lines.append("providers  " + ", ".join(f"{r} {p}" for r, p in o["providers"].items()))
    for k in ("grounded", "sets"):
        if o[k]:
            lines.append(f"{k}  {cut(text_of(o[k]))}")
    if o["state"]:
        st = o["state"]
        shown = _marker_text(st["marker"]) if st["holds"] != "value" else text_of(st["value"])
        lines.append(f"state  {shown}")  # whole, never cut
    providers: dict[str, Any] = o["providers"] or {}
    on_lineage = {p["ref"] for p in o["lineage"]}
    for r in o["rounds"]:
        if not r["calls"] and all(e["ref"] == STATE for e in r["entries"]):
            continue  # round 0 with the state's entry alone, shown on the state line
        lines.append(f"round {r['round']}")
        for c in r["calls"]:
            lines.extend(_call_lines(c, detail=detail))
        for e in r["entries"]:
            if e["ref"] != STATE:  # shown once, on the state line
                lines.extend(_entry_lines(e, providers, e["ref"] in on_lineage))
        if r["closed"]:
            deadline = "yes" if r["deadline"] else "no"
            lines.append(f"  cost {text_of(r['cost'])}  deadline {deadline}")
        elif r["round"] != 0:  # round 0 holds the state's entry and never a round item
            lines.append("  no round item: the round did not close")
    for item in o["legacy_items"]:
        rest = {k: v for k, v in item.items() if k not in ("index", "item")}
        lines.append(f"{item['item']} (log {item['index']})  {cut(text_of(rest))}")
    if o["cause"] is not None:
        lines.append(f"cause  {_cause_text(o['cause'])}")
    d = o["decision"]
    if o["policy"] is None:
        lines.append("decision  none: the record holds no policy (before format 8)")
    elif d is None:
        lines.append(f"decision  {o['policy']}: no entry")
    elif d["holds"] != "value":
        lines.append(f"decision  {d['ref']} undecided: {_marker_text(d['marker'])}")
    else:
        rows = f"  rows {', '.join(map(str, d['rows']))}" if d["rows"] is not None else ""
        lines.append(f"decision  {d['ref']} = {cut(text_of(d['value']))}{rows}")
    lines.extend(_lineage_lines(o["lineage"], by_index))
    if o["nearest_bounds"] is not None:
        lines.extend(_nearest_lines(o["nearest_bounds"], o["ties"]))
    esc = o["escalation"]
    if esc:
        lines.append(
            f"escalation  {esc['kind']}  cause {_cause_text(esc['cause'])}"
            f"  root {_cause_text(esc['root_cause'])}"
            f"  defect {'yes' if esc['defect'] else 'no'}"
        )
    if o["defects"]:
        lines.append(f"defects  {', '.join(o['defects'])}")
    if o["absent"]:
        lines.append(f"absent  {', '.join(o['absent'])}")
    lines.extend(_review_lines(o["review"]))
    return "\n".join(lines) + "\n"


def _review_lines(review: dict[str, Any] | None) -> list[str]:
    """Each source by what the turn did with it, and each table entry's rows by result."""
    if review is None:
        return []
    out = ["sources"]
    for s in review["sources"]:
        line = f"  {s['outcome']}  {s['source']}"
        if s["outcome"] == "asked":
            line += f"  round {s['round']}  call {s['call']}"
        elif s["outcome"] == "not asked":
            cause = _cause_text(s["cause"]) if s["cause"] else "-"
            line += f"  round {s['round']}  {s['why'] or '-'}: {s['marker'] or '-'} {cause}"
        out.append(line)
    for tb in review["tables"]:
        out.append(f"table {tb['ref']}  hit {tb['hit']}")
        for r in tb["rows"]:
            chosen = " (its output)" if r["chosen"] else ""
            out.append(f"  row {r['n']}  {r['result']}{chosen}  -> {r['output'] or '-'}")
    return out


_WARNING = re.compile(r"^(\S+): (\S+) (.*)$", re.S)


def _warning_lines(warnings: object) -> list[str]:
    """Warnings folded by code, ``<source>: <code> <message>``: one line per code in first-seen
    order, with its count and its first message. A warning of no such form is its own code."""
    groups: dict[str, list[str]] = {}
    for w in arr(warnings):
        m = _WARNING.match(str(w))
        code, message = (m[2], m[3]) if m else (str(w), "")
        groups.setdefault(code, []).append(message)
    return [cut(f"    warning {code} x{len(ms)}  {ms[0]}".rstrip()) for code, ms in groups.items()]


def _brief(e: dict[str, Any] | None) -> str:
    """An entry in a few words: a marker's kind and code; a distribution's top label and its
    probability, the first in label order on a tie; any other value cut to 60 characters."""
    if e is None:
        return "no entry"
    if e["holds"] != "value":
        return f"{e['holds']} {obj(obj(e['marker']).get('cause')).get('code')}"
    value = obj(e["value"])
    probs = obj(value.get("probabilities"))
    if probs:
        given = [x if isinstance(x, str) else obj(x).get("name") for x in arr(value.get("labels"))]
        labels = [str(lab) for lab in given or list(probs) if number(probs.get(str(lab)))]
        if labels:
            top = max(labels, key=lambda lab: probs[lab])  # max keeps the first of a tie
            return f"{top} {p2(probs[top])}"
    if value.get("kind") == "extraction":
        return cut(text_of(value.get("value")), 60)
    return cut(text_of(e["value"]), 60)


def why_of(o: dict[str, Any]) -> str:
    """The text form's why line without its ``why`` label."""
    by_index = {e["index"]: e for r in o["rounds"] for e in r["entries"]}
    return _why_line(o, by_index).removeprefix("why  ")


def _why_line(o: dict[str, Any], by_index: dict[Any, dict[str, Any]]) -> str:
    """The decision in one line: its value and ref, or the lineage's first branch down to its
    root, the root's nearest bound, the root and whether the escalation is a defect."""
    d = o["decision"]
    if d is None:
        held = "the record holds no policy" if o["policy"] is None else f"no entry at {o['policy']}"
        return f"why  no decision: {held}"
    if d["holds"] == "value":
        return f"why  decided {cut(text_of(d['value']))} by {d['ref']}"
    chain = root_chain(o["lineage"])
    text = " <- ".join(f"{p['ref']} {_brief(by_index.get(p['index']))}" for p in chain)
    root = chain[-1]["ref"] if chain else "-"
    root_entry = by_index.get(chain[-1]["index"]) if chain else None
    if root_entry is not None and root_entry.get("bounds") == []:
        text += f" no bound declared ({root_entry['bounds_reason']})"
    elif root_entry is not None and root_entry.get("bounds") is None:
        text += " (no design paired)"
    misses: list[dict[str, Any]] = o["nearest_bounds"] or []
    near = next((m for m in misses if m["ref"] == root and m["distance"] is not None), None)
    if near is not None:
        text += f" {near['side'] or 'distance'} {near['param']} by {gap(near['distance'])}"
        # the tie count is the first nearest bound's: shown only when this bound lies as near
        if (o["ties"] or 0) > 1 and gap(near["distance"]) == gap(misses[0]["distance"]):
            n = o["ties"] - 1
            text += f" (tied with {n} {'other' if n == 1 else 'others'})"
    esc = o["escalation"]
    defect = "-" if esc is None else ("yes" if esc.get("defect") else "no")
    return f"why  undecided: {text} (root {root}, defect {defect})"


def _lineage_lines(lineage: list[dict[str, Any]], by_index: dict[Any, dict[str, Any]]) -> list[str]:
    """The lineage, one entry per line, each after the first with how it was found and, where
    that is not the line before, from which entry."""
    if not lineage:
        return []
    out = ["lineage"]
    for k, p in enumerate(lineage):
        shown = f"{p['ref']}  {_brief(by_index.get(p['index']))}"
        if k == 0:
            out.append(f"  {shown}")
            continue
        origin = "" if p["from"] == lineage[k - 1]["ref"] else f" from {p['from']}"
        out.append(f"  <- {shown}  ({p['via']}{origin})")
    return out


NEAR_SHOWN = 8


def _nearest_lines(near: list[dict[str, Any]], ties: int | None) -> list[str]:
    """The nearest bounds, at most eight, each starred when its entry is on the lineage."""
    if not near:
        return ["nearest bounds  none"]
    out = ["nearest bounds"]
    if ties is not None and ties > 1:
        first = near[0]
        side = f", {first['side']}" if first["side"] else ""
        out.append(f"  ({ties} bounds tie at {text_of(first['distance'])}{side})")
    for m in near[:NEAR_SHOWN]:
        distance = "-" if m["distance"] is None else text_of(m["distance"])
        side = f" {m['side']}" if m["side"] else ""
        star = " *" if m["on_lineage"] else ""
        out.append(
            f"  {m['ref']}  {m['role']} {m['param']}  {distance}{side}  (by {m['by']}){star}"
        )
    return out


def _detail_line(c: dict[str, Any]) -> str | None:
    """A call's prompt, request id, clients, missing members and warnings in one line: the
    prompt's length, the count of missing members and of warnings, each warning code with its
    count."""
    u = c["unhashed"]
    parts: list[str] = []
    if c["prompt"] is not None:
        prompt = c["prompt"] if isinstance(c["prompt"], str) else text_of(c["prompt"])
        parts.append(f"prompt {len(prompt)} chars")
    for k in ("request_id", "clients"):
        if u[k]:
            parts.append(k)
    if u["missing"]:
        missing: object = u["missing"]
        parts.append(f"missing {len(arr(missing) or obj(missing)) or 1}")
    warnings = [str(w) for w in arr(u["warnings"])]
    if warnings:
        codes: dict[str, int] = {}
        for w in warnings:
            m = _WARNING.match(w)
            codes[m[2] if m else w] = codes.get(m[2] if m else w, 0) + 1
        counted = ", ".join(f"{code} x{n}" for code, n in codes.items())
        parts.append(f"warnings {len(warnings)} ({counted})")
    return "    call detail  " + ", ".join(parts) if parts else None


def _call_lines(c: dict[str, Any], *, detail: bool = False) -> list[str]:
    outcome = c["outcome"]
    if not isinstance(outcome, str):
        outcome = "failed " + _cause_text(obj(outcome).get("failed"))
    head = [f"  call {c['index']}", ", ".join(c["sources"] or []), outcome]
    for k in (
        "provider",
        "model_requested",
        "model_reported",
        "revision",
        "elicitation",
        "precision",
        "snapshot",
    ):
        if c[k] is not None:
            shown = c[k] if isinstance(c[k], str) else text_of(c[k])
            head.append(f"{k.replace('model_', 'model ')} {shown}")
    if c["claimed"]:
        head.append("claimed " + ", ".join(c["claimed"]))
    u = c["unhashed"]
    if u["latency"] is not None:
        head.append(f"latency {u['latency']:.6f}s")
    if u["attempts"] is not None:
        head.append(f"attempts {u['attempts']}")
    if u["usage"] is not None:
        head.append("usage " + " ".join(f"{k} {v}" for k, v in u["usage"].items()))
    out = ["  ".join(head)]
    for q in arr(c["questions"]):
        out.append(f"    asks {q['ref']}  {q['kind']}  identity {short(q['identity'])}")
    if c["runtime"] is not None:
        out.append(f"    runtime {text_of(c['runtime'])}")
    if not detail:
        folded = _detail_line(c)
        out.extend([folded] if folded else [])
        out.extend(f"    {k} {cut(text_of(u[k]))}" for k in ("truncated", "invalid") if u[k])
    else:
        if c["prompt"] is not None:
            out.append(f"    prompt {cut(_q(c['prompt']))}")
        for k in ("missing", "truncated", "invalid", "clients", "request_id"):
            if u[k]:
                out.append(f"    {k} {cut(text_of(u[k]))}")
        out.extend(_warning_lines(u["warnings"]))
    if u["traceback"]:
        out.append(f"    traceback {cut(str(u['traceback']))}")
    return out


def _entry_lines(e: dict[str, Any], providers: dict[str, Any], on_lineage: bool) -> list[str]:
    shown = _marker_text(e["marker"]) if e["holds"] != "value" else _value_text(e["value"])
    tail: list[str] = []
    if e["call"] is not None:
        tail.append(f"(call {e['call']})")
    if e["policy"]:
        tail.append("[policy]")
    star = " *" if on_lineage else ""
    out = [f"  {e['ref']}{star}  {shown}" + ("  " + "  ".join(tail) if tail else "")]
    for given in arr(e["derivation"]):
        d = obj(given)
        out.append(
            f"    derivation {d.get('helper')} refs {', '.join(d.get('refs', []))}"
            f"  rested on {', '.join(d.get('rested_on', [])) or 'nothing'}"
        )
    if e["rows"] is not None:
        out.append(f"    rows {', '.join(map(str, e['rows'])) or 'none'}")
    provider = providers.get(base_of(e["ref"]) or "", "?")
    if e["declared_reads"] is not None:
        reads = ", ".join(r["ref"] for r in e["declared_reads"]) or "nothing"
        out.append(f"    declared reads {reads}  (rule {provider})")
        if e["declared_params"]:
            leaves = ", ".join(p["param"] for p in e["declared_params"])
            out.append(f"    declared params {leaves}")
    if e["declared_inputs"] is not None:
        ins = ", ".join(f"{i['ref']} ({', '.join(i['roles'])})" for i in e["declared_inputs"])
        out.append(f"    declared inputs {ins or 'nothing'}  (source {provider})")
    out.extend(_wording_lines(e["wording"], "    "))
    word = "estimates" if e["declared_reads"] is not None else "asks"
    out.extend(_target_lines(e["target"], e["latent_targets"], "    ", word))
    if e["bounds"] is not None:
        value = obj(e["value"]) if e["holds"] == "value" else {}
        answer = "probabilities" in value or value.get("kind") == "extraction"
        if not e["bounds"] and answer:
            out.append(f"    bounds none declared ({e['bounds_reason']})")
        for b in e["bounds"]:
            out.append("    " + _bound_text(b))
    return out


def _bound_text(b: dict[str, Any]) -> str:
    what = b["role"]
    if b["role"] == "plain":
        what = f"bound on {b['key']}"
    elif b["role"] == "mass":
        what = f"mass {'+'.join(b['labels'] or [])}"
    text = f"{what} {b['param']} = {cut(text_of(b['bound']), 60)}  (by {b['by']})"
    for x in arr(b["per_label"]):
        text += (
            f"  {x['label']} {text_of(x['probability'])} vs {text_of(x['threshold'])} "
            f"distance {text_of(x['distance'])} {x['side']}"
        )
    if b["measured"] is not None:
        text += f"  {b['measure']} {text_of(b['measured'])}"
    if b["distance"] is not None:
        text += f"  distance {text_of(b['distance'])}"
    if b["side"] is not None:
        text += f"  {b['side']}"
    return text
