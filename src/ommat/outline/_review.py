"""What a paired turn did, sorted for review: each source asked or not asked and why, and each
table's rows matched or not. It sorts what the record and its design hold and adds nothing the
record does not carry: no cell's result, and no reason for a ref no entry wrote."""

from typing import Any

from ommat.outline._common import arr, base_of, obj, tested_reads
from ommat.outline._types import cell_text, output_text

# Why a source wrote markers and no call: the cause codes each class takes.
_NEED = ("need-undetermined", "need-unavailable")
_WHEN = ("when-false", "when-failed", "undeclared-read")
_STOPPED = ("round-bound", "deadline", "settled", "over-budget", "cancelled")


def _why_not_asked(source: dict[str, Any], sid: str, cause: dict[str, Any]) -> str:
    """``requirement`` when the cause names the source requirement's ref, ``when`` when it is a
    function ``when``'s, ``need`` for a need not determined, ``stopped`` when the turn stopped
    asking, else ``other``."""
    code, refs = cause.get("code"), arr(cause.get("refs"))
    requirement = obj(source.get("requirement")).get("ref")
    if code in (*_NEED, "when-false") and requirement in refs:
        return "requirement"
    if code in _WHEN and sid in refs:
        return "when"
    if code in _NEED:
        return "need"
    if code in _STOPPED:
        return "stopped"
    return "other"


def _sources(
    design: dict[str, Any],
    providers: dict[str, Any],
    entries: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Each source the design declares, those asked first in call order, then those not asked
    in entry order, then those that wrote nothing in design order."""
    out: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for at, (sid, source) in enumerate(obj(design.get("sources")).items()):
        s = obj(source)
        refs = [r for r, p in providers.items() if p == sid]
        bases = {base_of(r) for r in refs}
        written = [e for e in entries if base_of(e["ref"]) in bases]
        call = next((c for c in calls if sid in arr(c["sources"])), None)
        item: dict[str, Any] = {
            "source": sid,
            "refs": refs,
            "requirement": s.get("requirement"),
            "when": s.get("when") is not None,
            "outcome": "no entry",
            "call": None,
            "round": None,
            "entries": [e["index"] for e in written],
            "marker": None,
            "cause": None,
            "why": None,
        }
        if call is not None:
            item.update(outcome="asked", call=call["index"], round=call["round"])
            key = (0, call["index"])
        elif written:
            first = written[0]
            cause = obj(obj(first["marker"]).get("cause"))
            item.update(
                outcome="not asked",
                round=first["round"],
                marker=first["holds"] if first["holds"] != "value" else None,
                cause=cause or None,
                why=_why_not_asked(s, sid, cause) if cause else None,
            )
            key = (1, first["index"])
        else:
            key = (2, at)
        out.append((key, item))
    return [item for _, item in sorted(out, key=lambda x: x[0])]


def _chosen(rule: dict[str, Any], entry: dict[str, Any], matched: list[int]) -> int | None:
    """The matched row whose output the entry holds: the first under every single-hit policy,
    the highest-ranked under ``P``; ``None`` under a collect policy, with no row matched, or when
    a row that could not be tested made the result ``Unavailable``."""
    hit = rule.get("hit")
    cause = obj(obj(entry["marker"]).get("cause"))
    if not matched or not isinstance(hit, str) or hit.startswith("C"):
        return None
    if entry["holds"] == "Unavailable" and cause.get("code") == "required":
        return None
    if hit != "P":
        return matched[0]
    order = arr(rule.get("order"))
    rows = arr(rule.get("rows"))

    def rank(n: int) -> int:
        out = obj(obj(rows[n - 1]).get("output"))
        value = next(iter(out.values()), None) if len(out) == 1 else None
        return order.index(value) if value in order else len(order)

    return min(matched, key=rank)


def _tables(
    design: dict[str, Any], providers: dict[str, Any], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Each table entry of the turn, in entry order, with every row of its table: ``matched``
    when the entry's rows name it, ``untested`` when a read it tests holds ``Unavailable`` or no
    entry, else ``not matched``. Each test gives the entry its read found."""
    rules = obj(design.get("rules"))
    by_base = {base_of(e["ref"]): e for e in entries}
    out: list[dict[str, Any]] = []
    for e in entries:
        rid = providers.get(base_of(e["ref"]) or "")
        rule = obj(rules.get(rid)) if isinstance(rid, str) else {}
        held: object = e["rows"]
        if rule.get("form") != "table" or held is None:
            continue
        inputs = [obj(i) for i in arr(rule.get("inputs"))]
        matched = [n for n in arr(held) if isinstance(n, int)]
        chosen = _chosen(rule, e, matched)
        rows: list[dict[str, Any]] = []
        for n, row in enumerate(arr(rule.get("rows")), 1):
            r = obj(row)
            cells = arr(r.get("cells"))
            tests: list[dict[str, Any]] = []
            for k, read in tested_reads(cells, inputs):
                found = by_base.get(base_of(read))
                tests.append(
                    {
                        "input": k,
                        "read": read,
                        "helper": inputs[k].get("helper"),
                        "cell": cell_text(cells[k]),
                        "entry": None if found is None else found["index"],
                        "holds": None if found is None else found["holds"],
                    }
                )
            if n in matched:
                result = "matched"
            elif any(t["holds"] in (None, "Unavailable") for t in tests):
                result = "untested"
            else:
                result = "not matched"
            rows.append(
                {
                    "n": n,
                    "output": None if rule.get("hit") == "C#" else output_text(r.get("output")),
                    "result": result,
                    "chosen": n == chosen,
                    "tests": tests,
                }
            )
        out.append(
            {
                "ref": e["ref"],
                "rule": rid,
                "index": e["index"],
                "hit": rule.get("hit"),
                "policy": e["policy"],
                "rows": rows,
            }
        )
    return out


def review_of(
    design: dict[str, Any] | None,
    providers: dict[str, Any],
    entries: list[dict[str, Any]],
    calls: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """``{sources, tables}`` for a paired turn; ``None`` unpaired."""
    if design is None:
        return None
    return {
        "sources": _sources(design, providers, entries, calls),
        "tables": _tables(design, providers, entries),
    }
