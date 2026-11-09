"""What reaches what over a design's dependency edges: each node's steps to the policy, each
source requirement's provider and the nodes it is decided from and holds back, and each
target's questions, estimating rules, readers and path to the policy. Every list is in the
outline's ``nodes`` order."""

from collections import deque
from collections.abc import Iterable
from typing import Any

from ommat.outline._common import arr, base_of, obj


def _closure(start: str, step: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    todo = [start]
    while todo:
        for m in step.get(todo.pop(), ()):
            if m not in seen:
                seen.add(m)
                todo.append(m)
    return seen


def reach(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    ref_node: dict[str, str],
    policy_node: str | None,
    targets: list[dict[str, Any]],
) -> tuple[dict[str, int | None], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """``steps_to_policy`` by node id, the ``requirements`` list, and each target's links by
    target id."""
    order = {n["id"]: i for i, n in enumerate(nodes)}
    by_id = {n["id"]: n for n in nodes}
    deps = [e for e in edges if e["dependency"] and e["from"] in order and e["to"] in order]
    preds: dict[str, set[str]] = {}
    succs: dict[str, set[str]] = {}
    for e in deps:
        preds.setdefault(e["to"], set()).add(e["from"])
        succs.setdefault(e["from"], set()).add(e["to"])

    def in_order(ids: Iterable[str]) -> list[str]:
        return sorted(set(ids), key=order.__getitem__)

    steps: dict[str, int | None] = dict.fromkeys(order)
    if policy_node in order:
        assert policy_node is not None
        steps[policy_node] = 0
        todo = deque([policy_node])
        while todo:
            x = todo.popleft()
            for p in preds.get(x, ()):
                if steps[p] is None and p != policy_node:
                    steps[p] = _known(steps[x]) + 1
                    todo.append(p)

    requirements: list[dict[str, Any]] = []
    gated = [n for n in nodes if n["kind"] == "source" and isinstance(n.get("requirement"), dict)]
    for n in sorted(gated, key=lambda n: (n["round"], n["id"])):
        base = base_of(obj(n["requirement"]).get("ref"))
        found = ref_node.get(base) if base is not None else None
        provider = found if found in order else None
        decided: set[str] = set() if provider is None else {provider, *_closure(provider, preds)}
        requirements.append(
            {
                "source": n["id"],
                "provider": provider,
                "decided_from": in_order(decided),
                "behind": in_order(_closure(n["id"], succs)),
            }
        )

    def nearest(ids: list[str]) -> str | None:
        reached = [i for i in ids if steps[i] is not None]
        return min(reached, key=lambda i: (steps[i], i)) if reached else None

    def path_from(start: str) -> list[str]:
        path = [start]
        while steps[path[-1]] != 0:
            here = _known(steps[path[-1]])
            path.append(min(s for s in succs[path[-1]] if steps[s] == here - 1))
        return path

    links: dict[str, dict[str, Any]] = {}
    for t in targets:
        tid = t["id"]
        questions: list[dict[str, Any]] = []
        for n in nodes:
            for q in map(obj, arr(n.get("questions"))):
                if q.get("target") == tid:
                    questions.append({"ref": q["ref"], "node": n["id"], "role": "asks"})
                for lt in map(obj, arr(q.get("latent_targets"))):
                    if lt.get("target") == tid:
                        questions.append({"ref": q["ref"], "node": n["id"], "role": lt.get("role")})
        estimated_by = [n["id"] for n in nodes if n["kind"] == "rule" and n.get("target") == tid]
        asked = {q["ref"] for q in questions}
        readers = in_order(
            e["to"]
            for e in deps
            if base_of(e["ref"]) in asked
            and by_id[e["to"]]["kind"] == "rule"
            and e["to"] not in estimated_by
        )
        start = nearest(estimated_by) or nearest(readers)
        links[tid] = {
            "questions": questions,
            "estimated_by": estimated_by,
            "readers": readers,
            "path": path_from(start) if start is not None else None,
        }
    return steps, requirements, links


def _known(x: int | None) -> int:
    assert x is not None  # a node on the walk was reached
    return x
