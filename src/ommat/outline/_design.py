"""The design outline: one design document as a plain dict of fixed shape, its nodes by round
and wavefront and its edges as the loaded design's graph builds them."""

from typing import Any

from ommat.outline._common import (
    EDGE_ROLES,
    NODE_KEYS,
    OUTLINE_VERSION,
    QUESTION_KEYS,
    STATE,
    STATE_NODE,
    arr,
    base_of,
    fixed,
    held,
    is_candidate_read,
    is_dict,
    is_list,
    obj,
    parsed,
    tested_reads,
)
from ommat.outline._layout import fingerprint_of, layout_of, short_of
from ommat.outline._reach import reach
from ommat.outline._types import cell_text, input_text, output_text, parse_type, target_values

KNOWN_DESIGN = (
    "classes",
    "format",
    "groups",
    "name",
    "parameters",
    "policy",
    "rules",
    "sources",
    "state",
    "targets",
    "types",
)


def reads_of(rule: dict[str, Any]) -> list[dict[str, str]]:
    """A component's declared reads, each ``{ref, via}``: a function's ``reads`` via
    ``declared``, a table input via ``table-input`` or, with a helper, ``helper``; a candidate
    read given as the grounding set it reads from."""
    grounded = rule.get("grounded")
    out: list[dict[str, str]] = []
    if rule.get("form") == "table":
        pairs = [
            (obj(i).get("read"), "table-input" if obj(i).get("helper") is None else "helper")
            for i in arr(rule.get("inputs"))
        ]
    else:
        pairs = [(r, "declared") for r in arr(rule.get("reads"))]
    for ref, via in pairs:
        if is_candidate_read(ref) and isinstance(grounded, str):
            ref = grounded
        if isinstance(ref, str) and not any(x["ref"] == ref for x in out):
            out.append({"ref": ref, "via": via})
    return out


def params_of(rule: dict[str, Any]) -> list[dict[str, str]]:
    """A component's declared leaves, each ``{param, via}``: a function's ``params`` via
    ``declared``; a table's helper bindings via ``helper`` and the leaves its cells and outputs
    name via ``table-cell``."""
    if rule.get("form") != "table":
        pairs = [(p, "declared") for p in arr(rule.get("params"))]
    else:
        pairs = [
            (p, "helper")
            for i in arr(rule.get("inputs"))
            for p in obj(obj(i).get("params")).values()
        ]
        for row in arr(rule.get("rows")):
            for cell in arr(obj(row).get("cells")):
                for tag, m in obj(cell).items():
                    for o in arr(m) if tag == "between" else [m]:
                        if isinstance(obj(o).get("param"), str):
                            pairs.append((obj(o)["param"], "table-cell"))
            output = obj(obj(row).get("output"))
            if isinstance(output.get("param"), str):
                pairs.append((output["param"], "table-cell"))
    out: list[dict[str, str]] = []
    for p, via in pairs:
        if isinstance(p, str) and not any(x["param"] == p for x in out):
            out.append({"param": p, "via": via})
    return out


def _function_of(member: object) -> dict[str, Any] | None:
    m = obj(member)
    return {k: m.get(k) for k in ("module", "qualname", "version", "source")} if m else None


def _labels(labels: object) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """A question's labels as ``[{name, description}]``, or its candidate-valued label set."""
    if is_dict(labels):
        return None, obj(labels)
    if not is_list(labels):
        return None, None
    out: list[dict[str, Any]] = []
    for x in arr(labels):
        if isinstance(x, str):
            out.append({"name": x, "description": None})
        else:
            out.append({"name": obj(x).get("name"), "description": obj(x).get("description")})
    return out, None


def wording_of(q: dict[str, Any]) -> dict[str, Any] | None:
    """A question's wording, its template in its declared order and its slots."""
    if "wording" not in q:
        return None
    w = obj(q["wording"])
    return {"template": held(w.get("template"), w.get("order")), "slots": w.get("slots")}


def _question(ref: str, q: dict[str, Any]) -> dict[str, Any]:
    labels, label_set = _labels(q.get("labels"))
    return fixed(
        QUESTION_KEYS,
        {**q, "ref": ref, "labels": labels, "label_set": label_set, "wording": wording_of(q)},
    )


def _rule_node(rule: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "form",
        "provides",
        "boundaries",
        "candidates",
        "grounded",
        "target",
        "module",
        "qualname",
        "version",
        "source",
        "hit",
        "order",
    )
    node: dict[str, Any] = {k: rule.get(k) for k in keys}
    node["outputs"] = parse_type(rule.get("outputs"))
    node["reads"] = reads_of(rule)
    node["params"] = params_of(rule)
    if rule.get("form") == "table":
        node["inputs"] = [
            {
                "read": obj(i).get("read"),
                "helper": obj(i).get("helper"),
                "params": obj(i).get("params"),
                "text": input_text(obj(i)),
            }
            for i in arr(rule.get("inputs"))
        ]
        hit = rule.get("hit")
        rows: list[dict[str, Any]] = []
        for n, row in enumerate(arr(rule.get("rows")), 1):
            r = obj(row)
            output = (
                None
                if hit == "C#"
                else {"cell": r.get("output"), "text": output_text(r.get("output"))}
            )
            cells = [{"cell": c, "text": cell_text(c)} for c in arr(r.get("cells"))]
            tests = [
                {"input": k, "read": read, "node": None, "cell": cells[k]["text"]}
                for k, read in tested_reads(r.get("cells"), rule.get("inputs"))
            ]
            rows.append({"n": n, "cells": cells, "output": output, "tests": tests})
        node["rows"] = rows
    else:
        node["body"] = "not in the document"
    return node


def _uses(doc: dict[str, Any]) -> list[tuple[str, str, object]]:
    """Each (owner, role, ref) the dependency graph is built from, as the loader finds them in
    the document: a source's grounding, needs, wording slot refs, candidate-valued label sets,
    extraction text and reference-date refs and requirement; a rule's grounding, reads and
    table inputs (a helper input's role is ``helper``). Candidate reads and candidate slots are
    not uses, and a cell's or output's refs are held to the inputs, so add no edge. Owners are
    bare names; a format-9 policy's is ``policy``."""
    out: list[tuple[str, str, object]] = []
    for sid, source in obj(doc.get("sources")).items():
        s = obj(source)
        if isinstance(s.get("grounded"), str):
            out.append((sid, "grounding", s["grounded"]))
        for ref in arr(s.get("needs")):
            out.append((sid, "need", ref))
        for provided in obj(s.get("provides")).values():
            q = obj(provided)
            if "kind" not in q:
                continue  # a tool result
            for slot in arr(obj(q.get("wording")).get("slots")):
                origin = obj(obj(slot).get("origin"))
                if isinstance(origin.get("ref"), str):
                    out.append((sid, "slot", origin["ref"]))
            labels = obj(q.get("labels"))
            if isinstance(labels.get("candidates"), str):
                out.append((sid, "label-set", labels["candidates"]))
            for member in ("text", "reference_date"):
                origin = obj(q.get(member))
                if isinstance(origin.get("ref"), str):
                    out.append((sid, "text", origin["ref"]))
        req = obj(s.get("requirement"))
        if isinstance(req.get("ref"), str):
            out.append((sid, "source-requirement", req["ref"]))
    for rid, rule in obj(doc.get("rules")).items():
        r = obj(rule)
        if isinstance(r.get("grounded"), str):
            out.append((rid, "grounding", r["grounded"]))
        if r.get("form") == "table":
            for given in arr(r.get("inputs")):
                i = obj(given)
                if not is_candidate_read(i.get("read")):
                    role = "read" if i.get("helper") is None else "helper"
                    out.append((rid, role, i.get("read")))
        else:
            for ref in arr(r.get("reads")):
                if not is_candidate_read(ref):
                    out.append((rid, "read", ref))
    policy = doc.get("policy")
    if is_dict(policy):  # a format-9 policy: a function with reads and no provides
        for ref in arr(obj(policy).get("reads")):
            out.append(("policy", "read", ref))
    return out


def providers_of(doc: dict[str, Any]) -> dict[str, str]:
    """Each provided ref to its source's or rule's bare name, from the document."""
    out: dict[str, str] = {}
    for sid, s in obj(doc.get("sources")).items():
        for ref in obj(obj(s).get("provides")):
            out.setdefault(ref, sid)
    for rid, r in obj(doc.get("rules")).items():
        provides = obj(r).get("provides")
        if isinstance(provides, str):
            out.setdefault(provides, rid)
    return out


def _node_ids(doc: dict[str, Any]) -> dict[str, str]:
    """Each bare source or rule name to its node id, ``source:<name>`` or ``rule:<name>``; a
    format-9 policy is the rule ``rule:policy``."""
    ids = {sid: f"source:{sid}" for sid in obj(doc.get("sources"))}
    ids.update({rid: f"rule:{rid}" for rid in obj(doc.get("rules"))})
    if is_dict(doc.get("policy")):
        ids["policy"] = "rule:policy"
    return ids


def _policy_name(doc: dict[str, Any], ids: dict[str, str]) -> str | None:
    """The bare name of the rule whose estimate is the decision: the rule providing the
    design's policy ref, or a format-9 policy."""
    policy = doc.get("policy")
    if is_dict(policy):
        return "policy"
    return next(
        (r for r, x in obj(doc.get("rules")).items() if obj(x).get("provides") == policy), None
    )


def edges_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The edges: the dependency graph's (``dependency`` true), from the provider of each use's
    base to its owner, as the loaded design's graph builds it; and each use of the state
    (``dependency`` false), from ``state``, which that graph leaves out. One per (from, to,
    role, ref), sorted."""
    providers, ids = providers_of(doc), _node_ids(doc)
    found: set[tuple[str, str, str, str, bool]] = set()
    for owner, role, ref in _uses(doc):
        base = base_of(ref)
        if role not in EDGE_ROLES or base is None or owner not in ids:
            continue
        assert isinstance(ref, str)  # base_of gives a base for a str alone
        if base == STATE:
            found.add((STATE_NODE, ids[owner], role, ref, False))
        elif base in providers:
            found.add((ids[providers[base]], ids[owner], role, ref, True))
    return [
        {"from": f, "to": t, "role": r, "ref": ref, "dependency": e}
        for f, t, r, ref, e in sorted(found)
    ]


def topological(
    names: list[str], ids: dict[str, str], edges: list[dict[str, Any]]
) -> list[list[str]]:
    """Ready sets over the dependency edges: each the nodes whose predecessors are all done, in
    code-point order of their bare names. A node left by a cycle closes the list as its own
    set."""
    back = {v: k for k, v in ids.items()}
    preds = {
        n: {
            back[e["from"]]
            for e in edges
            if e["dependency"] and e["to"] == ids[n] and e["from"] in back
        }
        for n in names
    }
    done: set[str] = set()
    ready: list[list[str]] = []
    while len(done) < len(names):
        found = sorted(n for n in names if n not in done and preds[n] <= done)
        if not found:
            ready.append(sorted(n for n in names if n not in done))
            break
        ready.append(found)
        done.update(found)
    return ready


def rounds_of(
    order: list[str], ids: dict[str, str], edges: list[dict[str, Any]]
) -> dict[str, tuple[int, int]]:
    """Each node's round and wavefront, by §round-numbering, over ``order``, a topological
    order. A rule's round is its latest predecessor's (0 with none), a source's one more (1
    with none). A source's wavefront is 0, as is the state's; a rule's is one more than the
    largest among its predecessors in its own round, a source counting 0. A predecessor not
    yet placed, one a cycle leaves, is passed over."""
    back = {v: k for k, v in ids.items()}
    at: dict[str, tuple[int, int]] = {}
    for n in order:
        preds = sorted(
            {back[e["from"]] for e in edges if e["dependency"] and e["to"] == ids[n]} & set(at)
        )
        latest = max((at[p][0] for p in preds), default=None)
        if ids[n].startswith("source:"):
            at[n] = (1 if latest is None else latest + 1, 0)
            continue
        r = 0 if latest is None else latest
        at[n] = (r, 1 + max((at[p][1] for p in preds if at[p][0] == r), default=0))
    return at


def _class_summary(tree: object) -> str:
    if isinstance(tree, str):
        return tree
    leaves: list[str] = []

    def walk(t: object) -> None:
        if is_dict(t):
            for v in obj(t).values():
                walk(v)
        elif is_list(t):
            for v in arr(t):
                walk(v)
        else:
            leaves.append(str(t))

    walk(tree)
    counts = {c: leaves.count(c) for c in sorted(set(leaves))}
    return "tree: " + ", ".join(f"{c} ×{k}" for c, k in counts.items())  # noqa: RUF001


def design_outline(document: object) -> dict[str, Any]:
    """One design document, ``bytes`` or parsed, as a plain dict of fixed shape: its header,
    parameters, nodes by round and wavefront, edges, a ref-to-node map, groups, targets and
    classes; and, computed from these, a layout, a fingerprint and a short label per ref.
    ``ValueError`` for a document that is neither, or bytes that are not one strict JSON
    object."""
    doc, digest, hash_source = parsed(document, "the design")
    sources, rules = obj(doc.get("sources")), obj(doc.get("rules"))
    policy = doc.get("policy")
    ids = _node_ids(doc)
    edges = edges_of(doc)
    names = list(ids)
    placed = rounds_of([n for w in topological(names, ids, edges) for n in w], ids, edges)
    order = sorted(names, key=lambda n: (*placed[n], n))
    policy_name = _policy_name(doc, ids)
    if policy_name in order and not any(e["from"] == ids[policy_name] for e in edges):
        order.remove(policy_name)
        order.append(policy_name)  # the policy last, as nothing reads it

    def counts(node_id: str) -> dict[str, Any]:
        preds = {e["from"] for e in edges if e["dependency"] and e["to"] == node_id}
        return {
            "predecessors": sorted(preds),
            "fan_in": sum(1 for e in edges if e["to"] == node_id),
            "fan_out": sum(1 for e in edges if e["from"] == node_id),
        }

    nodes: list[dict[str, Any]] = []
    if "state" in doc:
        state = {
            "id": STATE_NODE,
            "kind": "state",
            "policy": False,
            "name": STATE,
            "refs": [STATE],
            "round": 0,
            "wavefront": 0,
            "type": parse_type(doc["state"]),
            **counts(STATE_NODE),
        }
        nodes.append(fixed(NODE_KEYS, state))
    for n in order:
        node_id = ids[n]
        kind = node_id.split(":", 1)[0]
        values: dict[str, Any] = {
            "id": node_id,
            "kind": kind,
            "policy": n == policy_name,
            "name": n,
            "round": placed[n][0],
            "wavefront": placed[n][1],
            **counts(node_id),
        }
        if kind == "source":
            s = obj(sources[n])
            for k in ("oracle", "needs", "requirement", "cost", "timeout", "feedback", "grounded"):
                values[k] = s.get(k)
            values["payload"] = held(s.get("payload"), s.get("payload_order"))
            values["when"] = _function_of(s.get("when"))
            provided = [(r, obj(q)) for r, q in obj(s.get("provides")).items()]
            values["refs"] = [r for r, _ in provided]
            values["questions"] = [_question(r, q) for r, q in provided if "kind" in q]
            values["tool_results"] = [
                {"ref": r, "type": parse_type(q.get("type")), "candidates": q.get("candidates")}
                for r, q in provided
                if "kind" not in q
            ]
        else:
            rule = obj(policy) if n == "policy" and is_dict(policy) else obj(rules[n])
            values.update(_rule_node(rule))
            provides = rule.get("provides")
            values["refs"] = [provides] if isinstance(provides, str) else []
        nodes.append(fixed(NODE_KEYS, values))

    ref_node: dict[str, str] = {STATE: STATE_NODE} if "state" in doc else {}
    for node in nodes:
        for r in node["refs"]:
            ref_node.setdefault(r, node["id"])

    targets: list[dict[str, Any]] = [
        {
            "id": k,
            "statement": obj(v).get("statement"),
            "values": target_values(obj(v).get("values")),
            "values_text": obj(v).get("values"),
        }
        for k, v in sorted(obj(doc.get("targets")).items())
    ]
    policy_node = ids[policy_name] if policy_name else None
    steps, requirements, links = reach(nodes, edges, ref_node, policy_node, targets)
    for node in nodes:
        node["steps_to_policy"] = steps[node["id"]]
    for t in targets:
        t.update(links[t["id"]])
    for node in nodes:
        for row in map(obj, arr(node["rows"])):
            for test in map(obj, arr(row["tests"])):
                test["node"] = ref_node.get(base_of(test["read"]) or "")

    return {
        "outline_version": OUTLINE_VERSION,
        "kind": "design",
        "name": doc.get("name"),
        "format": doc.get("format"),
        "hash": digest,
        "hash_source": hash_source,
        "policy": policy if isinstance(policy, str) else None,
        "policy_node": policy_node,
        "state": parse_type(doc.get("state")),
        "parameters": [
            {
                "path": path,
                "type": parse_type(obj(p).get("type")),
                "value": held(obj(p).get("value"), obj(p).get("order")),
            }
            for path, p in sorted(obj(doc.get("parameters")).items())
        ],
        "types": [
            {"name": k, "type": parse_type(v)} for k, v in sorted(obj(doc.get("types")).items())
        ],
        "nodes": nodes,
        "edges": edges,
        "ref_node": dict(sorted(ref_node.items())),
        "groups": [
            {"name": g, "refs": obj(v).get("refs"), "reason": obj(v).get("reason")}
            for g, v in sorted(obj(doc.get("groups")).items())
        ],
        "targets": targets,
        "classes": [
            {"ref": k, "class": _class_summary(v)}
            for k, v in sorted(obj(doc.get("classes")).items())
        ],
        "requirements": requirements,
        "other_members": sorted(k for k in doc if k not in KNOWN_DESIGN),
        "layout": layout_of(nodes, edges),
        "fingerprint": fingerprint_of(nodes, edges),
        "short": short_of(ref_node, edges),
    }
