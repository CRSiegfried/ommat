"""What the design outline computes from its nodes and edges alone: a layered layout, a
fingerprint that erases names, and a short label for each ref."""

import re
from fractions import Fraction
from typing import Any

from ommat._canonical import encode, ordered, sha256_ref
from ommat.outline._common import STATE, STATE_NODE, arr

SWEEPS = 3  # forward and backward barycentre sweeps
WL_ROUNDS = 3  # Weisfeiler-Lehman iterations


def _dependency(edges: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [e for e in edges if e["dependency"] and e["from"] in ids and e["to"] in ids]


def layout_of(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Each node's column and row. A column is one (round, wavefront), in that order: the state
    alone, a round's sources, or one rule wavefront. Each column starts in the nodes' order;
    then, for each of three sweeps, forward then backward, each column is sorted by the mean
    row of its predecessors (forward) or successors (backward) over the dependency edges, a
    node with none keeping its row, ties by the row each holds, which starts as that order, so
    a sweep that finds every node of a column alike leaves it as the last sweep left it."""
    seed = {n["id"]: k for k, n in enumerate(nodes)}
    keys = sorted({(int(n["round"]), int(n["wavefront"])) for n in nodes})
    col_of = {k: c for c, k in enumerate(keys)}
    columns: list[list[str]] = [[] for _ in keys]
    kinds: list[list[str]] = [[] for _ in keys]
    for n in nodes:
        c = col_of[(int(n["round"]), int(n["wavefront"]))]
        columns[c].append(str(n["id"]))
        if n["kind"] not in kinds[c]:
            kinds[c].append(str(n["kind"]))
    preds: dict[str, set[str]] = {i: set() for i in seed}
    succs: dict[str, set[str]] = {i: set() for i in seed}
    for e in _dependency(edges, set(seed)):
        if e["from"] != e["to"]:
            preds[e["to"]].add(e["from"])
            succs[e["from"]].add(e["to"])
    row = {i: k for col in columns for k, i in enumerate(col)}

    def sort(col: list[str], near: dict[str, set[str]]) -> None:
        def key(i: str) -> tuple[Fraction, int, int, str]:
            rows = [row[j] for j in near[i]]
            centre = Fraction(sum(rows), len(rows)) if rows else Fraction(row[i])
            return centre, row[i], seed[i], i

        col.sort(key=key)
        row.update({i: k for k, i in enumerate(col)})

    for _ in range(SWEEPS):
        for col in columns[1:]:
            sort(col, preds)
        for col in reversed(columns[:-1]):
            sort(col, succs)
    return {
        "columns": [
            {"round": r, "wavefront": w, "kinds": sorted(kinds[c]), "nodes": columns[c]}
            for c, (r, w) in enumerate(keys)
        ],
        "nodes": {
            n["id"]: {
                "col": col_of[(n["round"], n["wavefront"])],
                "row": row[n["id"]],
                "round": n["round"],
                "wavefront": n["wavefront"],
            }
            for n in nodes
        },
    }


def role_of(node: dict[str, Any]) -> str:
    """A node's typed role, its name erased: its kind, whether it is the policy, its form,
    question kinds with their counts, tool-result count, and whether it holds boundaries, a
    target and a requirement."""
    questions: list[dict[str, Any]] = node["questions"] or []
    kinds = sorted(str(q["kind"]) for q in questions)
    counted = ",".join(f"{k}:{kinds.count(k)}" for k in sorted(set(kinds))) or "-"
    target = node["target"] is not None or any(q["target"] is not None for q in questions)

    def yes(x: object) -> str:
        return "yes" if x else "no"

    return ";".join(
        (
            f"kind={node['kind']}",
            f"policy={yes(node['policy'])}",
            f"form={node['form'] or '-'}",
            f"questions={counted}",
            f"tools={len(arr(node['tool_results']))}",
            f"boundaries={yes(node['boundaries'])}",
            f"target={yes(target)}",
            f"requirement={yes(node['requirement'])}",
        )
    )


def fingerprint_of(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    """The design's shape with its names erased: each node's role, refined by three
    Weisfeiler-Lehman iterations over the dependency edges, each message an edge's direction, role
    and the colour across it; ``hash`` is over the sorted final colours, ``roles`` the sorted
    initial ones."""
    roles = {n["id"]: role_of(n) for n in nodes}
    dependency = _dependency(edges, set(roles))
    colour = dict(roles)
    for _ in range(WL_ROUNDS):
        colour = {
            i: sha256_ref(
                encode(
                    [
                        c,
                        sorted([e["role"], colour[e["from"]]] for e in dependency if e["to"] == i),
                        sorted([e["role"], colour[e["to"]]] for e in dependency if e["from"] == i),
                    ]
                )
            )
            for i, c in colour.items()
        }
    return {
        "hash": sha256_ref(ordered(sorted(colour.values())).encode("utf-8")),
        "roles": sorted(roles.values()),
    }


_SEP = re.compile(r"[/.]")


def short_of(ref_node: dict[str, str], edges: list[dict[str, Any]]) -> dict[str, str]:
    """Each ref the design provides, and each ``t:state`` path it uses, to its last path
    segment; refs that share a label are each lengthened by the segment before, until every
    label is unique, a ref of one segment giving its whole ref."""
    refs = set(ref_node)
    refs.update(e["ref"] for e in edges if e["from"] == STATE_NODE)
    refs = {r for r in refs if r == STATE or r in ref_node or r.startswith(STATE + ".")}
    choices: dict[str, list[str]] = {}
    for ref in refs:
        name = ref.split(":", 1)[1] if ":" in ref else ref
        cuts = [m.end() for m in _SEP.finditer(name)]
        choices[ref] = [name[c:] for c in reversed(cuts)] + [name, ref]
    level = dict.fromkeys(refs, 0)
    while True:
        seen: dict[str, list[str]] = {}
        for ref in refs:
            seen.setdefault(choices[ref][level[ref]], []).append(ref)
        moved = False
        for alike in seen.values():
            if len(alike) > 1:
                for ref in alike:
                    if level[ref] < len(choices[ref]) - 1:
                        level[ref] += 1
                        moved = True
        if not moved:
            break
    return {ref: choices[ref][level[ref]] for ref in sorted(refs)}
