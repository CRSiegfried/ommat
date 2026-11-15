# pyright: reportPrivateUsage=false
"""The outline: every stored fixture prints as text and as JSON, each outline has its fixed
shape, edges are the loaded design's graph, bounds and margins measure as the choice helpers
compare, the views it holds are the engine's, and the command pairs and skips as the lens's."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import designs
import outline_showcase as showcase
from designs import full
from ledger_logs import LOGS, fresh, rehashed
from ommat import Record, decide_sync, defects, escalation, load, read_record
from ommat._canonical import ordered
from ommat.outline import design_outline, render_design, render_turn, turn_outline
from ommat.outline.__main__ import main
from ommat.outline._common import base_of
from ommat.outline._path import precise_refs, root_chain

TESTS = Path(__file__).parent
FORMAT_10 = TESTS / "format_10"
FORMAT_9 = TESTS / "format_9"
IDIA = TESTS / "idia_logs"

NODE_KEYS = {
    *("id", "kind", "policy", "name", "refs", "round", "wavefront", "predecessors"),
    *("fan_in", "fan_out", "steps_to_policy", "type"),
    *("oracle", "needs", "requirement", "cost", "timeout", "feedback", "grounded", "payload"),
    *("when", "questions", "tool_results"),
    *("form", "provides", "outputs", "reads", "params", "boundaries", "candidates", "target"),
    *("module", "qualname", "version", "source", "body", "hit", "order", "inputs", "rows"),
}
QUESTION_KEYS = {
    *("ref", "kind", "labels", "label_set", "value_type", "is_list", "text"),
    *("reference_date", "target", "latent_targets", "wording"),
}
ENTRY_KEYS = {
    *("ref", "index", "round", "call", "holds", "value", "marker", "derivation", "rows"),
    *("policy", "declared_reads", "declared_params", "declared_inputs", "wording", "target"),
    *("latent_targets", "bounds", "bounds_reason", "nearest"),
}
BOUND_KEYS = {
    *("role", "param", "bound", "by", "key", "labels", "measure", "measured", "distance"),
    *("side", "per_label"),
}
CALL_KEYS = {
    *("index", "round", "sources", "outcome", "provider", "model_requested", "model_reported"),
    *("revision", "runtime", "elicitation", "precision", "claimed", "snapshot", "prompt"),
    *("questions", "unhashed"),
}
PAIRED = (
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
"""The design-derived members of an entry."""


def ledger(name: str) -> bytes:
    """A format-8 ledger log as a record document, read as tests/test_views.py reads one."""
    log = json.loads(json.dumps(fresh(LOGS[name])))
    return read_record(rehashed({"log": log, "unhashed": {}})).document


@pytest.fixture(scope="module")
def shown() -> dict[str, Any]:
    """The showcase design, a turn it decides and a turn it leaves undecided."""
    loaded = showcase.loaded()
    return {
        "design": loaded.document,
        "decided": showcase.turn(loaded, *showcase.DECIDED),
        "undecided": showcase.turn(loaded, *showcase.UNDECIDED),
        "level": showcase.turn(loaded, 0.6, showcase.DECIDED[1]),
    }


def fixtures() -> list[tuple[str, bytes | None, bytes | None]]:
    """Every stored fixture, a design, a record or both, the record paired with the design."""
    return [
        (
            "format_10",
            (FORMAT_10 / "design.json").read_bytes(),
            (FORMAT_10 / "record.json").read_bytes(),
        ),
        (
            "format_9",
            (FORMAT_9 / "design.json").read_bytes(),
            (FORMAT_9 / "record.json").read_bytes(),
        ),
        ("idia", (IDIA / "design.json").read_bytes(), (IDIA / "turn.record.json").read_bytes()),
        ("full", load(full()).document, None),
        ("normal-8", None, ledger("normal-8")),
        ("broken-8", None, ledger("broken-8")),
        ("cancelled-8", None, ledger("cancelled-8")),
    ]


def entries(outline: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for r in outline["rounds"] for e in r["entries"]]


def entry(outline: dict[str, Any], ref: str) -> dict[str, Any]:
    (found,) = [e for e in entries(outline) if e["ref"] == ref]
    return found


def bound(e: dict[str, Any], role: str) -> dict[str, Any]:
    (found,) = [b for b in e["bounds"] if b["role"] == role]
    return found


def check_shape(outline: dict[str, Any]) -> None:
    """Every node, question, entry, call and bound holds its fixed key set."""
    for node in outline.get("nodes", []):
        assert set(node) == NODE_KEYS
        questions: list[dict[str, Any]] = node["questions"] or []
        for q in questions:
            assert set(q) == QUESTION_KEYS
    for r in outline.get("rounds", []):
        for c in r["calls"]:
            assert set(c) == CALL_KEYS
        for e in r["entries"]:
            assert set(e) == ENTRY_KEYS
            bounds: list[dict[str, Any]] = e["bounds"] or []
            for b in bounds:
                assert set(b) == BOUND_KEYS
    for p in outline.get("lineage", []):
        assert set(p) == {"ref", "holds", "via", "from", "index"}
    near: list[dict[str, Any]] = outline.get("nearest_bounds") or []
    for m in near:
        assert set(m) == {"ref", "param", "role", "by", "distance", "side", "on_lineage"}


# --- every fixture prints --------------------------------------------------------------------


@pytest.mark.parametrize(("name", "design", "record"), fixtures(), ids=lambda x: str(x)[:12])
def test_every_fixture_prints_as_text_and_json(
    name: str, design: bytes | None, record: bytes | None
) -> None:
    for outline, render in (
        *([(design_outline(design), render_design)] if design is not None else []),
        *([(turn_outline(record, design), render_turn)] if record is not None else []),
    ):
        check_shape(outline)
        text = render(outline)
        assert text.endswith("\n")
        assert text == render(json.loads(ordered(outline)))  # the text is the outline's
        assert json.loads(ordered(outline)) == outline
        assert outline["outline_version"] == 8
    if record is not None and design is not None:
        assert turn_outline(record, design)["design_hash_matches"] is True


def test_the_showcase_prints_and_round_trips(shown: dict[str, Any]) -> None:
    for outline in (
        design_outline(shown["design"]),
        turn_outline(shown["decided"].document, shown["design"]),
        turn_outline(shown["undecided"].document, shown["design"]),
    ):
        check_shape(outline)
        assert json.loads(ordered(outline)) == outline
        render = render_design if outline["kind"] == "design" else render_turn
        assert render(outline) == render(json.loads(ordered(outline)))


def test_the_text_is_alike_across_two_builds(shown: dict[str, Any]) -> None:
    record = shown["undecided"].document
    assert render_design(design_outline(shown["design"])) == render_design(
        design_outline(shown["design"])
    )
    assert render_turn(turn_outline(record, shown["design"])) == render_turn(
        turn_outline(record, shown["design"])
    )
    assert ordered(turn_outline(record, shown["design"])) == ordered(
        turn_outline(json.loads(record), json.loads(shown["design"]))
    )


# --- the design outline ----------------------------------------------------------------------


def test_the_showcase_design_holds_its_boundaries_table_requirement_and_targets(
    shown: dict[str, Any],
) -> None:
    o = design_outline(shown["design"])
    assert [n["id"] for n in o["nodes"]] == [
        "state",
        "source:ask",
        "rule:urgency",
        "source:desk",
        "rule:act",
    ]
    assert [(n["round"], n["wavefront"]) for n in o["nodes"]] == [
        *((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
    ]
    assert [n["id"] for n in o["nodes"] if n["policy"]] == [o["policy_node"]] == ["rule:act"]
    assert {n["kind"] for n in o["nodes"]} == {"state", "source", "rule"}
    desk = o["nodes"][3]
    assert desk["requirement"] == {"ref": "d:urgency", "values": ["yes"]}
    act = o["nodes"][4]
    assert act["form"] == "table"
    assert act["body"] is None
    assert [i["text"] for i in act["inputs"]] == [
        "d:urgency",
        "pick_with_margin(q:tone, bar=tone/bar, margin=tone/margin)",
        "t:ticket",
    ]
    assert [[c["text"] for c in r["cells"]] for r in act["rows"]] == [
        ["yes", "-", "Known"],
        ["no", "{neutral, warm}", "-"],  # the document holds it sorted
        ["no", "cold", "-"],
        ["Undetermined", "-", "-"],
    ]
    assert act["params"] == [
        {"param": "tone/bar", "via": "helper"},
        {"param": "tone/margin", "via": "helper"},
    ]
    urgency = o["nodes"][2]
    assert urgency["body"] == "not in the document"
    assert urgency["reads"] == [{"ref": "q:urgent", "via": "declared"}]
    assert o["targets"] == [
        {
            "id": "tg:urgent",
            "statement": "The mail needs an answer today",
            "values": ["yes", "no"],
            "values_text": "yes or no",
            "questions": [
                {"ref": "q:tone", "node": "source:ask", "role": "unknown"},
                {"ref": "q:urgent", "node": "source:ask", "role": "asks"},
            ],
            "estimated_by": ["rule:urgency"],
            "readers": ["rule:act"],
            "path": ["rule:urgency", "rule:act"],
        }
    ]
    assert [n["steps_to_policy"] for n in o["nodes"]] == [None, 1, 1, 1, 0]
    assert o["requirements"] == [
        {
            "source": "source:desk",
            "provider": "rule:urgency",
            "decided_from": ["source:ask", "rule:urgency"],
            "behind": ["rule:act"],
        }
    ]
    assert [
        [(x["input"], x["read"], x["node"], x["cell"]) for x in r["tests"]] for r in act["rows"]
    ] == [
        [(0, "d:urgency", "rule:urgency", "yes"), (2, "t:ticket", "source:desk", "Known")],
        [(0, "d:urgency", "rule:urgency", "no"), (1, "q:tone", "source:ask", "{neutral, warm}")],
        [(0, "d:urgency", "rule:urgency", "no"), (1, "q:tone", "source:ask", "cold")],
        [(0, "d:urgency", "rule:urgency", "Undetermined")],
    ]
    text = render_design(o)
    assert "predecessors rule:urgency  to policy 1\n" in text
    assert (
        "requirements\n"
        '  source:desk requires d:urgency in ["yes"] from rule:urgency\n'
        "    decided from source:ask, rule:urgency\n"
        "    behind rule:act\n"
    ) in text
    assert (
        'target tg:urgent  "The mail needs an answer today"  values yes or no\n'
        "  unknown q:tone (source:ask)\n"
        "  asked by q:urgent (source:ask)\n"
        "  estimated by rule:urgency\n"
        "  read by rule:act\n"
        "  path rule:urgency -> rule:act\n"
    ) in text
    assert "  source:desk -> rule:act  read t:ticket" in text
    assert (
        '[rule]   act  table F  -> d:act  label{"escalate", "file", "reply"}  '
        "predecessors rule:urgency, source:ask, source:desk [policy]"
    ) in text
    assert "  rule:urgency -> source:desk  source-requirement d:urgency" in text
    assert '  requirement d:urgency in ["yes"]' in text
    assert (
        "  4  Undetermined  -                                                           -" in text
    )


def _graph_of(outline: dict[str, Any]) -> dict[str, frozenset[str]]:
    name = {n["id"]: n["name"] for n in outline["nodes"]}
    return {
        n["name"]: frozenset(name[a] for a in n["predecessors"])
        for n in outline["nodes"]
        if n["kind"] != "state"
    }


@pytest.mark.parametrize("which", ["showcase", "full", "idia"])
def test_the_dependency_edges_are_the_loaded_designs_graph(which: str) -> None:
    loaded = {
        "showcase": showcase.loaded,
        "full": lambda: load(full()),
        "idia": lambda: load((IDIA / "design.json").read_bytes()),
    }[which]()
    o = design_outline(loaded.document)
    assert _graph_of(o) == dict(loaded._graph)
    # Topological: each node after every node it follows.
    seen: set[str] = {"state"}
    for n in o["nodes"][1:]:
        assert set(n["predecessors"]) <= seen
        seen.add(n["id"])
    # Each node's round and wavefront by §round-numbering.
    at = {n["id"]: n for n in o["nodes"]}
    for n in o["nodes"][1:]:
        preds = [at[p] for p in n["predecessors"]]
        latest = max((p["round"] for p in preds), default=None)
        if n["kind"] == "source":
            assert (n["round"], n["wavefront"]) == (1 if latest is None else latest + 1, 0)
        else:
            assert n["round"] == (latest or 0)
            same = [p["wavefront"] for p in preds if p["round"] == n["round"]]
            assert n["wavefront"] == 1 + max(same, default=0)
    for column in {(n["round"], n["wavefront"]) for n in o["nodes"]}:
        names = [
            n["name"]
            for n in o["nodes"]
            if (n["round"], n["wavefront"]) == column and not n["policy"]
        ]
        assert names == sorted(names)


def paired_turns() -> list[tuple[str, bytes | dict[str, Any], bytes]]:
    """Every stored fixture's record with the design it ran under, and the showcase's turns."""
    loaded = showcase.loaded()
    shown = [
        (name, loaded.document, showcase.turn(loaded, *answers).document)
        for name, answers in (("decided", showcase.DECIDED), ("undecided", showcase.UNDECIDED))
    ]
    stored = [(n, d, r) for n, d, r in fixtures() if d is not None and r is not None]
    return [*stored, *shown]


@pytest.mark.parametrize(("name", "design", "record"), paired_turns(), ids=lambda x: str(x)[:12])
def test_each_entry_sits_in_its_nodes_round(
    name: str, design: bytes | dict[str, Any], record: bytes
) -> None:
    d = design_outline(design)
    t = turn_outline(record, design)
    rounds = {n["id"]: n["round"] for n in d["nodes"]}
    checked = 0
    for e in entries(t):
        node = d["ref_node"].get(base_of(e["ref"]))
        if node is None:  # its provider is carried or absent from the design
            continue
        assert e["round"] == rounds[node], e["ref"]
        checked += 1
    assert checked


def test_a_wavefront_is_in_code_point_order() -> None:
    def source(ref: str) -> dict[str, Any]:
        return {"provides": {ref: {"kind": "noul"}}, "needs": [], "oracle": "x"}

    document = {
        "policy": "d:x",
        "sources": {"b": source("q:b"), "a": source("q:a"), "B": source("q:B"), "é": source("q:e")},
        "rules": {"x": {"form": "function", "provides": "d:x", "reads": ["q:a", "q:b", "q:B"]}},
    }
    o = design_outline(document)
    assert [n["name"] for n in o["nodes"]] == ["B", "a", "b", "é", "x"]
    assert [(n["round"], n["wavefront"]) for n in o["nodes"]] == [(1, 0)] * 4 + [(1, 1)]


def test_rules_over_the_state_alone_fire_in_round_0() -> None:
    def rule(ref: str, reads: list[str]) -> dict[str, Any]:
        return {"form": "function", "provides": ref, "reads": reads}

    document = {
        "policy": "d:z",
        "state": "string",
        "sources": {"s": {"provides": {"q:s": {"kind": "noul"}}, "needs": ["d:x"], "oracle": "o"}},
        "rules": {
            "x": rule("d:x", ["t:state"]),
            "y": rule("d:y", ["d:x"]),
            "z": rule("d:z", ["q:s", "d:y"]),
        },
    }
    o = design_outline(document)
    assert [(n["id"], n["round"], n["wavefront"]) for n in o["nodes"]] == [
        ("state", 0, 0),
        ("rule:x", 0, 1),
        ("rule:y", 0, 2),
        ("source:s", 1, 0),
        ("rule:z", 1, 1),  # d:y is of round 0, so q:s alone counts
    ]


def test_a_parsed_design_gets_its_hash_recomputed() -> None:
    data = (FORMAT_10 / "design.json").read_bytes()
    from_bytes, from_dict = design_outline(data), design_outline(json.loads(data))
    assert from_bytes["hash_source"] == "bytes"
    assert from_dict["hash_source"] == "recomputed"
    assert from_dict["hash"] == from_bytes["hash"]
    assert {**from_dict, "hash_source": "bytes"} == from_bytes


# --- the turn outline ------------------------------------------------------------------------


def test_the_undecided_tone_and_urgency_fall_short_of_bar_and_margin(
    shown: dict[str, Any],
) -> None:
    o = turn_outline(shown["undecided"].document, shown["design"])
    tone = entry(o, "q:tone")
    assert bound(tone, "bar")["distance"] == -0.05
    assert bound(tone, "bar")["side"] == "short"
    assert bound(tone, "margin")["measured"] == 0.05
    assert bound(tone, "margin")["distance"] == -0.05
    assert bound(tone, "margin")["side"] == "short"
    urgent = entry(o, "q:urgent")
    assert bound(urgent, "bar")["distance"] == -0.05
    assert bound(urgent, "margin")["measured"] == 0.1
    assert bound(urgent, "margin")["distance"] == -0.1
    assert {bound(urgent, r)["side"] for r in ("bar", "margin")} == {"short"}
    assert urgent["nearest"] == {
        "param": "urgent/bar",
        "bound": 0.6,
        "by": "urgency",
        "distance": -0.05,
        "side": "short",
    }
    assert o["decision"] is not None
    assert o["decision"]["holds"] == "Unsure"


def test_the_decided_tone_clears_bar_and_margin(shown: dict[str, Any]) -> None:
    o = turn_outline(shown["decided"].document, shown["design"])
    tone = entry(o, "q:tone")
    assert (bound(tone, "bar")["distance"], bound(tone, "bar")["side"]) == (0.3, "clear")
    assert (bound(tone, "margin")["distance"], bound(tone, "margin")["side"]) == (0.6, "clear")
    assert o["decision"] is not None
    assert o["decision"]["value"] == "escalate"
    assert o["escalation"] is None
    assert "  q:tone  choice  warm 0.10  cold 0.80  neutral 0.10  (call 3)" in render_turn(o)


def test_a_lead_equal_to_its_margin_is_clear(shown: dict[str, Any]) -> None:
    o = turn_outline(shown["level"].document, shown["design"])
    urgent = entry(o, "q:urgent")
    for role in ("bar", "margin"):
        assert (bound(urgent, role)["distance"], bound(urgent, role)["side"]) == (0, "clear")
    assert entry(o, "d:urgency")["value"] == "yes"  # the helper's own >= agrees


BOUNDED: dict[str, Any] = {
    "format": 10,
    "name": "bounded",
    "policy": "d:pick",
    "state": "record{}",
    "parameters": {
        "tone/th": {"type": "mapping<string, number>", "value": {"warm": 0.5, "cold": 0.3}},
        "tone/mass": {"type": "number [0, 1]", "value": 0.5},
        "date/null": {"type": "number [0, 1]", "value": 0.3},
        "date/conf": {"type": "number [0, 1]", "value": 0.7},
        "size/max": {"type": "integer", "value": 10},
    },
    "sources": {
        "ask": {
            "oracle": "llm",
            "needs": [],
            "provides": {
                "q:tone": {"kind": "choice", "labels": ["warm", "cold", "neutral"]},
                "q:date": {"kind": "extraction", "value_type": "date", "is_list": False},
            },
        },
        "look": {
            "oracle": "db",
            "needs": [],
            "provides": {"t:size": {"type": "record{n: integer}"}},
        },
    },
    "rules": {
        "pick": {
            "form": "function",
            "provides": "d:pick",
            "outputs": 'label{"a"}',
            "reads": ["q:tone", "q:date", "t:size.n"],
            "params": ["tone/th", "tone/mass", "date/null", "date/conf", "size/max"],
            "boundaries": {
                "q:tone": {
                    "thresholds": ["tone/th"],
                    "mass": [{"labels": ["warm", "neutral"], "bounds": ["tone/mass"]}],
                },
                "q:date": {"null": ["date/null"], "confidence": ["date/conf"]},
                "t:size.n": ["size/max"],
            },
        }
    },
    "classes": {"t:state": {"a": "content", "b": "content", "c": "open"}},
}
"""A design whose one rule declares a threshold, a mass, a null, a confidence and a plain
bound, read raw, with a record of the same shape, to measure each kind of bound."""


def _bounded_record() -> dict[str, Any]:
    def entry(ref: str, value: object, call: int | None = 3) -> dict[str, Any]:
        return {"item": "entry", "ref": ref, "round": 1, "call": call, "value": value}

    tone = {
        "kind": "choice",
        "labels": ["warm", "cold", "neutral"],
        "probabilities": {"warm": 0.45, "cold": 0.4, "neutral": 0.15},
    }
    date = {
        "kind": "extraction",
        "value_type": "date",
        "is_list": False,
        "value": "2026-03-03",
        "guess": "2026-03-03",
        "p_null": 0.3,
        "confidence": 0.9,
        "span": None,
    }
    providers = {"q:tone": "ask", "q:date": "ask", "t:size": "look", "d:pick": "pick"}
    return {
        "hash": "sha256:" + "0" * 64,
        "log": [
            {"item": "design", "hash": "sha256:" + "1" * 64, "format": 8, "helpers": 4},
            {"item": "settings", "policy": "d:pick", "providers": providers},
            entry("t:state", {}, None),
            entry("q:tone", tone),
            entry("q:date", date),
            entry("t:size", {"n": 12}),
            entry("d:pick", "a", None),
        ],
    }


def test_each_kind_of_bound_measures_and_sides_as_its_helper() -> None:
    o = turn_outline(_bounded_record(), BOUNDED)
    assert o["read_by"] == "raw"
    tone = entry(o, "q:tone")
    per_label = bound(tone, "thresholds")["per_label"]
    assert [(x["label"], x["distance"], x["side"]) for x in per_label] == [
        ("warm", -0.05, "short"),
        ("cold", 0.1, "clear"),
    ]
    mass = bound(tone, "mass")
    assert (mass["measured"], mass["distance"], mass["side"]) == (0.6, 0.1, "clear")
    assert tone["nearest"] == {
        "param": "tone/th",
        "bound": 0.5,
        "by": "pick",
        "distance": -0.05,
        "side": "short",
    }
    date = entry(o, "q:date")
    # extract abstains at or above its null bar, so a p_null equal to it is short of clearing it
    assert (bound(date, "null")["distance"], bound(date, "null")["side"]) == (0, "short")
    confidence = bound(date, "confidence")
    assert (confidence["distance"], confidence["side"]) == (0.2, "clear")
    plain = bound(entry(o, "t:size"), "plain")
    assert (plain["key"], plain["measured"], plain["distance"], plain["side"]) == (
        "t:size.n",
        12,
        2,
        None,
    )
    text = render_turn(o)
    assert '  q:date  extraction date  value "2026-03-03"  p_null 0.30  confidence 0.90' in text
    assert "guess" not in text
    assert "span" not in text
    assert "    bound on t:size.n size/max = 10  (by pick)  t:size.n 12  distance 2\n" in text
    assert "warm 0.45 vs 0.5 distance -0.05 short  cold 0.4 vs 0.3 distance 0.1 clear" in text
    design = design_outline(BOUNDED)
    assert "classes" not in design
    assert design["other_members"] == ["classes"]
    assert "\nclasses  " not in render_design(design)
    check_shape(o)
    check_shape(design)


def test_a_format_11_design_outline_has_no_classes_key_and_no_classes_line() -> None:
    document = load(full()).document
    assert "classes" not in json.loads(document)
    o = design_outline(document)
    assert "classes" not in o
    assert "classes" not in o["other_members"]
    assert "classes" not in render_design(o)
    check_shape(o)


def test_escalation_and_defects_are_the_engines_views(shown: dict[str, Any]) -> None:
    records: list[Record] = [shown["undecided"], read_record(ledger("broken-8"))]
    for record in records:
        o = turn_outline(record.document)
        assert o["escalation"] == escalation(record)
        found = defects(record)
        assert found is not None
        assert o["defects"] == [x["ref"] for x in found]
        esc = escalation(record)
        assert esc is not None
        assert f"escalation  {esc['kind']}  cause {esc['cause']['code']}" in render_turn(o)
    broken = turn_outline(ledger("broken-8"))
    assert broken["defects"]
    assert f"defects  {', '.join(broken['defects'])}" in render_turn(broken)


def test_an_unpaired_turn_holds_no_design_derived_member(shown: dict[str, Any]) -> None:
    alone = turn_outline(shown["undecided"].document)
    paired = turn_outline(shown["undecided"].document, shown["design"])
    assert alone["paired"] is False
    assert alone["absent"] is None
    assert alone["design_hash_matches"] is None
    for e in entries(alone):
        assert all(e[k] is None for k in PAIRED)
    # A record alone and paired differ only by the paired members.
    drop = ("paired", "design_name", "design_hash_matches", "absent", "rounds")
    drop += ("lineage", "nearest_bounds", "ties", "summary", "summary_nearest", "why", "review")
    assert {k: v for k, v in alone.items() if k not in drop} == {
        k: v for k, v in paired.items() if k not in drop
    }
    for a, p in zip(entries(alone), entries(paired), strict=True):
        assert {k: v for k, v in a.items() if k not in PAIRED} == {
            k: v for k, v in p.items() if k not in PAIRED
        }
    assert "no design paired" in render_turn(alone)


def test_a_paired_entry_with_no_boundaries_says_why() -> None:
    o = turn_outline(
        (FORMAT_10 / "record.json").read_bytes(), (FORMAT_10 / "design.json").read_bytes()
    )
    spam = entry(o, "q:spam")
    assert spam["bounds"] == []
    assert spam["bounds_reason"] == "no boundaries field on rule route"
    assert spam["nearest"] is None
    assert entry(o, "d:route")["declared_inputs"] is None  # a source's member, on a rule's entry
    assert spam["declared_reads"] is None
    assert "    bounds none declared (no boundaries field on rule route)" in render_turn(o)


def test_a_record_before_format_8_prints_entries_and_calls_alone() -> None:
    o = turn_outline((FORMAT_9 / "record.json").read_bytes())
    assert o["record_format"] == 7
    assert o["policy"] is None
    assert o["decision"] is None
    assert o["escalation"] is None
    assert o["defects"] is None
    assert entries(o)
    assert any(r["calls"] for r in o["rounds"])
    assert "decision  none: the record holds no policy (before format 8)" in render_turn(o)


def test_a_cancelled_turn_prints_what_is_there() -> None:
    o = turn_outline(ledger("cancelled-8"))
    assert o["status"] == "unfinished"
    assert o["cause"] is not None
    assert o["decision"] is None


def test_a_record_read_record_refuses_is_read_raw() -> None:
    document = json.loads((FORMAT_10 / "record.json").read_bytes())
    document["hash"] = "sha256:" + "0" * 64
    o = turn_outline(document)
    assert o["read_by"] == "raw"
    assert o["read_error"]
    assert o["escalation"] is None
    assert o["defects"] is None
    assert "  read raw: read_record refused it: " in render_turn(o)


def test_a_document_of_another_type_is_refused() -> None:
    with pytest.raises(ValueError, match="the design must be bytes or a dict"):
        design_outline("{}")
    with pytest.raises(ValueError, match="the record is not JSON"):
        turn_outline(b"{")


# --- the command -----------------------------------------------------------------------------


def _folder(tmp_path: Path) -> Path:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "design.json").write_bytes((FORMAT_10 / "design.json").read_bytes())
    (tmp_path / "turn.json").write_bytes((FORMAT_10 / "record.json").read_bytes())
    return tmp_path


def test_the_command_prints_a_record_beside_its_design(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path)
    assert main([str(folder)]) == 0
    out = capsys.readouterr().out
    assert f'== {(folder / "design.json").as_posix()}\ndesign  "format 10"' in out
    assert f"== {(folder / 'turn.json').as_posix()}\nturn sha256:" in out
    assert 'paired with design "format 10" (hash matches)\n' in out


def test_the_command_prints_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _folder(tmp_path)
    assert main([str(folder), "--json"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [x["outline"]["kind"] for x in lines] == ["design", "turn"]
    assert lines[1]["outline"]["paired"] is True
    assert lines[1]["outline"] == turn_outline(
        (FORMAT_10 / "record.json").read_bytes(), (FORMAT_10 / "design.json").read_bytes()
    )


def test_the_command_writes_a_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = _folder(tmp_path / "in")
    out = tmp_path / "outline.txt"
    assert main([str(folder / "turn.json"), "-o", str(out)]) == 0
    assert capsys.readouterr().out == ""
    assert "no design paired" in out.read_text(encoding="utf-8")


def test_the_command_skips_what_is_no_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path)
    (folder / "notes.json").write_bytes(b'{"hello": 1}')
    (folder / "broken.json").write_bytes(b"{")
    (folder / "classifier.json").write_bytes((IDIA / "classifier.json").read_bytes())
    assert main([str(folder)]) == 0
    captured = capsys.readouterr()
    assert "skipped notes.json" not in captured.err  # named by its path as given
    assert f"skipped {(folder / 'notes.json').as_posix()} is not an ommat record or design" in (
        captured.err
    )
    assert f"skipped {(folder / 'broken.json').as_posix()} is not JSON" in captured.err
    assert f"skipped {(folder / 'classifier.json').as_posix()}" in captured.err
    assert "notes.json" not in captured.out


def test_the_command_stops_quietly_when_the_reader_closes_the_pipe(tmp_path: Path) -> None:
    folder = _folder(tmp_path)
    for i in range(200):
        (folder / f"turn{i}.json").write_bytes((FORMAT_10 / "record.json").read_bytes())
    command = [sys.executable, "-m", "ommat.outline", str(folder)]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        assert child.stdout is not None
        assert child.stderr is not None
        child.stdout.readline()
        child.stdout.close()
        error = child.stderr.read().decode()
        child.wait(timeout=60)
    assert "Traceback" not in error
    assert "BrokenPipeError" not in error


def test_the_command_exits_1_on_nothing_printable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.json").write_bytes(b'{"hello": 1}')
    assert main([str(tmp_path)]) == 1
    assert "no ommat record or design found" in capsys.readouterr().err
    assert main([str(tmp_path / "missing")]) == 1
    assert "no such file or folder" in capsys.readouterr().err


def test_the_command_refuses_an_output_it_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path)
    assert main([str(folder), "-o", str(folder / "turn.json")]) == 1
    assert "is one of the files read" in capsys.readouterr().err


# --- what the outline computes ---------------------------------------------------------------


def crossings(o: dict[str, Any], at: dict[str, tuple[int, int]]) -> int:
    """How many pairs of dependency edges cross as straight lines between (col, row) points;
    edges that share a node, or that join the same two nodes, cross nothing."""
    pairs = sorted({(e["from"], e["to"]) for e in o["edges"] if e["dependency"]})

    def turn(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
        d = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (d > 0) - (d < 0)

    count = 0
    for k, (a, b) in enumerate(pairs):
        for c, d in pairs[k + 1 :]:
            if {a, b} & {c, d}:
                continue
            p, q, r, t = at[a], at[b], at[c], at[d]
            count += turn(p, q, r) * turn(p, q, t) < 0 and turn(r, t, p) * turn(r, t, q) < 0
    return count


def seeded(o: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Each node at its (round, wavefront) column and its place among that column's nodes in
    the outline's order."""
    columns = sorted({(n["round"], n["wavefront"]) for n in o["nodes"]})
    rows: dict[int, int] = {}
    at: dict[str, tuple[int, int]] = {}
    for n in o["nodes"]:
        col = columns.index((n["round"], n["wavefront"]))
        at[n["id"]] = (col, rows.get(col, 0))
        rows[col] = rows.get(col, 0) + 1
    return at


def laid(o: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {i: (p["col"], p["row"]) for i, p in o["layout"]["nodes"].items()}


def test_the_layout_uncrosses_idia_and_crosses_no_more_elsewhere() -> None:
    idia = design_outline((IDIA / "design.json").read_bytes())
    assert (crossings(idia, seeded(idia)), crossings(idia, laid(idia))) == (9, 0)
    assert [
        (c["round"], c["wavefront"], c["kinds"], c["nodes"]) for c in idia["layout"]["columns"]
    ] == [
        (0, 0, ["state"], ["state"]),
        (1, 0, ["source"], ["source:registry", "source:details", "source:timing", "source:triage"]),
        (
            *(1, 1, ["rule"]),
            [
                *("rule:noticed", "rule:places", "rule:seen_at"),
                *("rule:asset", "rule:hazard", "rule:severity"),
            ],
        ),
        (1, 2, ["rule"], ["rule:route"]),
    ]
    for document in (showcase.loaded().document, load(full()).document):
        o = design_outline(document)
        assert crossings(o, laid(o)) <= crossings(o, seeded(o))
    for o in (idia, design_outline(showcase.loaded().document)):
        assert o["layout"]["nodes"]["state"] == {"col": 0, "row": 0, "round": 0, "wavefront": 0}
        assert [n["id"] for n in o["nodes"]] == list(o["layout"]["nodes"])
        kind = {n["id"]: n["kind"] for n in o["nodes"]}
        for col, column in enumerate(o["layout"]["columns"]):
            assert column["kinds"] == sorted({kind[i] for i in column["nodes"]})
            assert len(column["kinds"]) == 1
            for row, i in enumerate(column["nodes"]):
                assert o["layout"]["nodes"][i] == {
                    "col": col,
                    "row": row,
                    "round": column["round"],
                    "wavefront": column["wavefront"],
                }
        keys = [(c["round"], c["wavefront"]) for c in o["layout"]["columns"]]
        assert keys == sorted(set(keys))
    again = design_outline(json.loads((IDIA / "design.json").read_bytes()))
    assert again["layout"] == idia["layout"]


SHOWCASE_NAMES = {
    **{"ask": "s1", "desk": "s2", "urgency": "r1", "act": "r2"},
    **{"q:urgent": "q:a", "q:tone": "q:b", "t:ticket": "t:c", "d:urgency": "d:d"},
    **{"d:act": "d:e", "tg:urgent": "tg:f"},
}


def renamed(document: bytes, names: dict[str, str]) -> dict[str, Any]:
    """The document with every string, key or value, that ``names`` holds renamed."""

    def walk(v: object) -> Any:
        if isinstance(v, dict):
            return {names.get(k, k): walk(x) for k, x in cast("dict[str, Any]", v).items()}
        if isinstance(v, list):
            return [walk(x) for x in cast("list[Any]", v)]
        return names.get(v, v) if isinstance(v, str) else v

    return cast("dict[str, Any]", walk(json.loads(document)))


def test_the_fingerprint_erases_names_and_holds_the_shape(shown: dict[str, Any]) -> None:
    o = design_outline(shown["design"])
    other = renamed(shown["design"], SHOWCASE_NAMES)
    assert design_outline(other)["short"] != o["short"]  # the names did change
    assert design_outline(other)["fingerprint"] == o["fingerprint"]
    assert o["fingerprint"]["hash"].startswith("sha256:")
    assert (
        "kind=rule;policy=yes;form=table;questions=-;tools=0;boundaries=no;target=no;"
        "requirement=no" in (o["fingerprint"]["roles"])
    )
    bounded = json.loads(shown["design"])
    bounded["rules"]["act"]["boundaries"] = {"t:ticket": ["tone/bar"]}
    changed = design_outline(bounded)["fingerprint"]
    assert changed["hash"] != o["fingerprint"]["hash"]
    assert changed["roles"] != o["fingerprint"]["roles"]


def test_a_short_label_is_the_last_segment_lengthened_on_a_collision() -> None:
    o = design_outline((IDIA / "design.json").read_bytes())
    short = o["short"]
    assert list(short) == sorted(short)
    assert short["t:state.report"] == "report"
    assert short["t:open_orders"] == "open_orders"
    assert short["d:route"] == "route"
    # q:triage/hazard and d:hazard share "hazard": the one lengthens by a segment, the other,
    # of one segment, gives its whole ref.
    assert (short["q:triage/hazard"], short["d:hazard"]) == ("triage/hazard", "d:hazard")
    assert len(set(short.values())) == len(short)
    document: dict[str, Any] = {
        "policy": "d:x",
        "sources": {
            "a": {"provides": {"q:a/b/c": {"kind": "noul"}, "q:d/b/c": {"kind": "noul"}}},
            "b": {"provides": {"q:e/c": {"kind": "noul"}}},
        },
        "rules": {"x": {"form": "function", "provides": "d:x", "reads": []}},
    }
    assert design_outline(document)["short"] == {
        "d:x": "x",
        "q:a/b/c": "a/b/c",
        "q:d/b/c": "d/b/c",
        "q:e/c": "e/c",
    }


def path_of(o: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    return [(p["ref"], p["via"], p["from"]) for p in o["lineage"]]


def test_the_lineage_walks_back_from_the_decision(shown: dict[str, Any]) -> None:
    undecided = turn_outline(shown["undecided"].document, shown["design"])
    assert path_of(undecided) == [
        ("d:act", None, None),
        ("d:urgency", "row", "d:act"),  # row 4 constrains d:urgency alone
        ("q:urgent", "rested_on", "d:urgency"),
        ("q:tone", "cause", "d:act"),
        ("t:ticket", "cause", "d:act"),
    ]
    assert [p["holds"] for p in undecided["lineage"]] == [
        *("Unsure", "Unsure", "value", "value", "Unasked"),
    ]
    for p in undecided["lineage"]:
        assert (
            entries(undecided)[[e["index"] for e in entries(undecided)].index(p["index"])]["ref"]
            == (p["ref"])
        )
    decided = turn_outline(shown["decided"].document, shown["design"])
    assert path_of(decided) == [
        ("d:act", None, None),
        ("d:urgency", "row", "d:act"),
        ("q:urgent", "rested_on", "d:urgency"),
        ("t:ticket", "row", "d:act"),  # row 1 constrains d:urgency and t:ticket
    ]
    # Unpaired, a table's rows name no inputs, so the marker's cause alone is followed.
    alone = turn_outline(shown["undecided"].document)
    assert path_of(alone)[:2] == [("d:act", None, None), ("d:urgency", "cause", "d:act")]
    idia = turn_outline(
        (IDIA / "turn.record.json").read_bytes(), (IDIA / "design.json").read_bytes()
    )
    assert path_of(idia) == [("d:route", None, None), ("q:triage/hazard", "cause", "d:route")]
    broken = turn_outline(ledger("broken-8"))
    assert path_of(broken) == [("d:route", None, None), ("t:state", "cause", "d:route")]
    assert turn_outline(ledger("cancelled-8"))["lineage"] == []
    assert turn_outline((FORMAT_9 / "record.json").read_bytes())["lineage"] == []


def test_a_plain_rule_s_declared_reads_follow_the_record_s_own_hops() -> None:
    record, design = showcase.plain_turn().document, load(showcase.PLAIN).document
    paired = turn_outline(record, design)
    # Neither rule names what it read, so the design's declared reads are followed: the
    # policy's in declared order, then those of the entries they reach.
    assert path_of(paired) == [
        ("d:verdict", None, None),
        ("d:flag", "declared", "d:verdict"),
        ("q:urgent", "declared", "d:flag"),
        ("t:state", "declared", "d:verdict"),
    ]
    # The root stops before a declared hop, so it is the policy's own entry.
    assert [p["ref"] for p in root_chain(paired["lineage"])] == ["d:verdict"]
    assert paired["summary"].split("  ")[3] == "root d:verdict"
    # Unpaired, a record holds no declared reads, so the lineage is the policy's entry alone.
    assert path_of(turn_outline(record)) == [("d:verdict", None, None)]


def test_a_ref_the_record_names_keeps_its_hop_when_a_declared_read_also_reaches_it() -> None:
    idia = turn_outline(
        (IDIA / "turn.record.json").read_bytes(), (IDIA / "design.json").read_bytes()
    )
    # d:route declares many reads, among them d:hazard, whose own reads reach q:triage/hazard;
    # the marker's cause names q:triage/hazard, which keeps its ``cause`` hop.
    assert ("q:triage/hazard", "cause", "d:route") in path_of(idia)
    assert all(via != "declared" for _, via, _ in path_of(idia))


def test_only_hops_the_record_names_put_an_entry_on_the_lineage() -> None:
    record, design = showcase.plain_turn().document, load(showcase.PLAIN).document
    o = turn_outline(record, design)
    assert precise_refs(o["lineage"]) == {"d:verdict"}
    marked = turn_outline(
        showcase.turn(showcase.loaded(), *showcase.DECIDED).document, showcase.loaded().document
    )
    assert precise_refs(marked["lineage"]) == {p["ref"] for p in marked["lineage"]}
    assert "d:flag *" not in render_turn(o)
    assert "d:verdict *" in render_turn(o)


def test_the_nearest_bounds_are_every_bound_nearest_first(shown: dict[str, Any]) -> None:
    o = turn_outline(shown["decided"].document, shown["design"])
    found = o["nearest_bounds"]
    assert [(m["ref"], m["param"], m["distance"], m["on_lineage"]) for m in found] == [
        ("q:urgent", "urgent/bar", 0.25, True),
        ("q:tone", "tone/bar", 0.3, False),
        ("q:urgent", "urgent/margin", 0.5, True),
        ("q:tone", "tone/margin", 0.6, False),
    ]
    undecided = turn_outline(shown["undecided"].document, shown["design"])
    assert [(m["ref"], m["param"]) for m in undecided["nearest_bounds"]] == [
        ("q:tone", "tone/bar"),  # a tie at 0.05 by ref, then by param
        ("q:tone", "tone/margin"),
        ("q:urgent", "urgent/bar"),
        ("q:urgent", "urgent/margin"),
    ]
    bounded = turn_outline(_bounded_record(), BOUNDED)
    thresholds = next(m for m in bounded["nearest_bounds"] if m["role"] == "thresholds")
    assert (thresholds["distance"], thresholds["side"]) == (-0.05, "short")  # its nearest label
    assert turn_outline(shown["undecided"].document)["nearest_bounds"] is None


def test_the_summary_is_one_line_of_fixed_columns(shown: dict[str, Any]) -> None:
    o = turn_outline(shown["undecided"].document, shown["design"])
    record_hash, design_hash = o["record_hash"][7:19], o["design_hash"][7:19]
    assert o["summary"].split("  ") == [
        record_hash,
        "finished",
        "undecided Unsure abstained (d:urgency, q:tone, t:ticket)",
        "root q:urgent",
        "defect no",
        "rounds 2",
        "nearest -0.05 short urgent/bar@q:urgent",
        f"design {design_hash}",
    ]
    assert turn_outline(shown["undecided"].document, shown["design"])["summary"] == o["summary"]
    decided = turn_outline(shown["decided"].document, shown["design"])["summary"].split("  ")
    assert decided[2:7] == [
        'decided "escalate"',
        "root q:urgent",
        "defect -",
        "rounds 2",
        "nearest 0.25 clear urgent/bar@q:urgent",
    ]
    assert turn_outline(shown["decided"].document)["summary"].split("  ")[6] == "nearest -"
    broken = turn_outline(ledger("broken-8"))["summary"].split("  ")
    assert broken[2:6] == [
        "undecided Unavailable invalid-return (t:state)",
        "root t:state",
        "defect yes",
        "rounds 0",
    ]
    cancelled = turn_outline(ledger("cancelled-8"))["summary"].split("  ")
    assert cancelled[1:4] == ["unfinished", "no-decision", "root -"]
    for _, design, record in fixtures():
        if record is not None:
            assert len(turn_outline(record, design)["summary"].split("  ")) == 8


def test_ties_count_the_bounds_as_near_as_the_first(shown: dict[str, Any]) -> None:
    undecided = turn_outline(shown["undecided"].document, shown["design"])
    assert undecided["ties"] == 3  # tone/bar, tone/margin and urgent/bar, all 0.05
    assert turn_outline(shown["decided"].document, shown["design"])["ties"] == 1
    assert turn_outline(shown["undecided"].document)["ties"] is None
    lines = render_turn(undecided).splitlines()
    assert lines[lines.index("nearest bounds") + 1] == "  (3 bounds tie at -0.05, short)"
    assert not any(
        "bounds tie" in x
        for x in render_turn(turn_outline(shown["decided"].document, shown["design"])).splitlines()
    )


def test_the_why_line_shows_the_tie_only_when_the_root_s_bound_lies_as_near(
    shown: dict[str, Any],
) -> None:
    # ``ties`` counts the bounds as near as the first nearest bound, which need not be the
    # root's: a root bound further off carries no tie count.
    undecided = turn_outline(shown["undecided"].document, shown["design"])
    first = undecided["nearest_bounds"][0]
    assert first["ref"] == "q:tone"
    first["distance"] = -0.01
    why = render_turn(undecided).splitlines()[1]
    assert "q:urgent yes 0.55 short urgent/bar by 0.05 (root q:urgent" in why
    assert "tied with" not in why


def test_the_text_says_why_and_shows_the_lineage_and_the_nearest_bounds(
    shown: dict[str, Any],
) -> None:
    text = render_turn(turn_outline(shown["undecided"].document, shown["design"]))
    lines = text.splitlines()
    assert lines[1] == (
        "why  undecided: d:act Unsure abstained <- d:urgency Unsure abstained <- q:urgent yes "
        "0.55 short urgent/bar by 0.05 (tied with 2 others) (root q:urgent, defect no)"
    )
    assert lines[2] == 'paired with design "showcase" (hash matches)'
    at = lines.index("decision  d:act undecided: Unsure abstained (d:urgency, q:tone, t:ticket)")
    assert lines[at + 1 : at + 13] == [
        "lineage",
        "  d:act  Unsure abstained",
        "  <- d:urgency  Unsure abstained  (row)",
        "  <- q:urgent  yes 0.55  (rested_on)",
        "  <- q:tone  warm 0.45  (cause from d:act)",
        "  <- t:ticket  Unasked need-undetermined  (cause from d:act)",
        "nearest bounds",
        "  (3 bounds tie at -0.05, short)",
        "  q:tone  bar tone/bar  -0.05 short  (by act) *",
        "  q:tone  margin tone/margin  -0.05 short  (by act) *",
        "  q:urgent  bar urgent/bar  -0.05 short  (by urgency) *",
        "  q:urgent  margin urgent/margin  -0.1 short  (by urgency) *",
    ]
    assert "  d:urgency *  Unsure abstained (q:urgent)" in lines
    assert "  q:urgent *  noul  yes 0.55  no 0.45  (call 3)" in lines
    decided = render_turn(turn_outline(shown["decided"].document, shown["design"]))
    assert decided.splitlines()[1] == 'why  decided "escalate" by d:act'
    assert "  q:tone  choice  warm 0.10  cold 0.80  neutral 0.10  (call 3)" in decided  # no star
    idia = turn_outline(
        (IDIA / "turn.record.json").read_bytes(), (IDIA / "design.json").read_bytes()
    )
    assert "\nnearest bounds  none\n" in render_turn(idia)
    idia_why = render_turn(idia).splitlines()[1]
    assert (
        "q:triage/hazard yes 0.58 no bound declared (no boundaries field on rule hazard) ("
        in idia_why
    )
    assert idia_why.endswith("(root q:triage/hazard, defect no)")
    assert (
        "(no design paired) (root"
        in render_turn(turn_outline((IDIA / "turn.record.json").read_bytes())).splitlines()[1]
    )
    alone = render_turn(turn_outline(shown["undecided"].document))
    assert "nearest bounds" not in alone
    assert "(defect no)" not in alone
    assert "why  no decision: the record holds no policy" in render_turn(
        turn_outline((FORMAT_9 / "record.json").read_bytes())
    )
    differs = json.loads(shown["design"])
    differs["name"] = "other"
    assert 'paired with design "other" (hash differs)' in render_turn(
        turn_outline(shown["undecided"].document, differs)
    )


def test_a_calls_detail_members_fold_into_one_line() -> None:
    o = turn_outline((IDIA / "turn.record.json").read_bytes(), (IDIA / "design.json").read_bytes())
    folded, detail = render_turn(o), render_turn(o, detail=True)
    assert (
        "    call detail  prompt 1136 chars, request_id, clients, missing 2, warnings 7 "
        "(no-snapshot x1, unchecked-limit x6)\n"
    ) in folded
    assert "missing 2, extras 2, warnings 2" in folded  # the classifier's passes, per request
    words = ("    prompt ", "    request_id ", "    clients ", "    missing ", "    extras ")
    for word in (*words, "    warning "):
        assert word not in folded
        assert word in detail
    assert "    call detail  " not in detail
    assert [x for x in detail.splitlines() if not x.startswith("    ") or "asks" in x] == [
        x for x in folded.splitlines() if not x.startswith("    ") or "asks" in x
    ]


def test_the_command_prints_one_summary_line_per_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path)
    assert main([str(folder), "--oneline"]) == 0
    out = capsys.readouterr().out
    o = turn_outline(
        (FORMAT_10 / "record.json").read_bytes(), (FORMAT_10 / "design.json").read_bytes()
    )
    assert out == o["summary"] + "\n"
    assert main([str(folder), "--detail"]) == 0
    assert render_turn(o, detail=True) in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(folder), "--oneline", "--json"])


def _rule(provides: str, reads: list[str], target: str | None = None) -> dict[str, Any]:
    return {"form": "function", "provides": provides, "reads": reads, "target": target}


def _ask(
    *refs: str, requirement: str | None = None, links: dict[str, dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A source asking a noul question per ref, its target and latent targets from ``links``."""
    links = links or {}
    return {
        "oracle": "llm",
        "requirement": None if requirement is None else {"ref": requirement, "values": ["yes"]},
        "provides": {
            r: {
                "kind": "noul",
                "target": links.get(r, {}).get("target"),
                "latent_targets": links.get(r, {}).get("latent", []),
            }
            for r in refs
        },
    }


def _doc(
    sources: dict[str, Any], rules: dict[str, Any], policy: str | None = "d:p"
) -> dict[str, Any]:
    return {"format": 10, "name": "t", "policy": policy, "sources": sources, "rules": rules}


def _steps(o: dict[str, Any]) -> dict[str, int | None]:
    return {n["id"]: n["steps_to_policy"] for n in o["nodes"]}


def test_steps_to_policy_are_the_fewest_dependency_edges_and_null_off_the_path() -> None:
    o = design_outline(
        _doc(
            {"s": _ask("q:x")},
            {
                "a": _rule("d:a", ["q:x"]),
                "b": _rule("d:b", ["d:a"]),
                "p": _rule("d:p", ["d:a", "d:b"]),
                "z": _rule("d:z", ["d:a"]),
            },
        )
    )
    assert _steps(o) == {
        "source:s": 2,
        "rule:a": 1,
        "rule:b": 1,
        "rule:p": 0,
        "rule:z": None,
    }
    assert "  no path to the policy" in render_design(o)


def test_the_state_and_a_design_with_no_policy_node_have_no_steps() -> None:
    doc = _doc({"s": {**_ask("q:x"), "needs": ["t:state.a"]}}, {"a": _rule("d:a", ["q:x"])})
    doc["state"] = "record{a: string}"
    o = design_outline(doc)
    assert o["policy_node"] is None
    assert set(_steps(o).values()) == {None}
    assert "to policy" not in render_design(o)
    assert "no path to the policy" not in render_design(o)
    with_policy = design_outline({**doc, "policy": "d:a"})
    assert _steps(with_policy)["state"] is None


def test_a_cyclic_document_still_outlines() -> None:
    o = design_outline(
        _doc(
            {}, {"a": _rule("d:a", ["d:b"]), "b": _rule("d:b", ["d:a"]), "p": _rule("d:p", ["d:a"])}
        )
    )
    assert _steps(o) == {"rule:a": 1, "rule:b": 2, "rule:p": 0}


def test_requirements_are_by_round_then_id_with_their_provider_and_reach() -> None:
    o = design_outline(
        _doc(
            {
                "first": _ask("q:f"),
                "z_late": _ask("q:z", requirement="d:gate"),
                "a_late": _ask("q:y", requirement="d:gate"),
                "later": _ask("q:l", requirement="d:z"),
                "ghost": _ask("q:g", requirement="d:missing"),
            },
            {
                "gate": _rule("d:gate", ["q:f"]),
                "z": _rule("d:z", ["q:z"]),
                "p": _rule("d:p", ["d:z", "q:y", "q:l", "q:g"]),
            },
        )
    )
    reqs = {r["source"]: r for r in o["requirements"]}
    assert [r["source"] for r in o["requirements"]] == [
        "source:ghost",
        "source:a_late",
        "source:z_late",
        "source:later",
    ]
    assert reqs["source:ghost"] == {
        "source": "source:ghost",
        "provider": None,
        "decided_from": [],
        "behind": ["rule:p"],
    }
    assert reqs["source:a_late"]["provider"] == "rule:gate"
    assert reqs["source:a_late"]["decided_from"] == ["source:first", "rule:gate"]
    assert reqs["source:z_late"]["behind"] == ["rule:z", "source:later", "rule:p"]
    assert "source:later" not in reqs["source:later"]["behind"]
    assert reqs["source:later"]["decided_from"] == [
        "source:first",
        "rule:gate",
        "source:z_late",
        "rule:z",
    ]
    assert '  source:ghost requires d:missing in ["yes"] from none' in render_design(o)


def test_a_target_s_questions_estimators_readers_and_path() -> None:
    doc = _doc(
        {
            "s": _ask(
                "q:b",
                "q:a",
                links={
                    "q:b": {"latent": [{"target": "tg:t", "role": "formative"}]},
                    "q:a": {"target": "tg:t", "latent": [{"target": "tg:t", "role": "reflective"}]},
                },
            )
        },
        {
            "est": _rule("d:est", ["q:a"], target="tg:t"),
            "rb": _rule("d:rb", ["q:b"]),
            "ra": _rule("d:ra", ["q:a"]),
            "p": _rule("d:p", ["d:est", "d:ra", "d:rb"]),
        },
    )
    doc["targets"] = {"tg:t": {"statement": "s", "values": None}}
    (t,) = design_outline(doc)["targets"]
    assert t["questions"] == [
        {"ref": "q:b", "node": "source:s", "role": "formative"},
        {"ref": "q:a", "node": "source:s", "role": "asks"},
        {"ref": "q:a", "node": "source:s", "role": "reflective"},
    ]
    assert t["estimated_by"] == ["rule:est"]
    assert t["readers"] == ["rule:ra", "rule:rb"]  # the estimator is not a reader
    assert t["path"] == ["rule:est", "rule:p"]


def test_a_target_s_path_falls_back_to_its_readers_and_breaks_ties_by_id() -> None:
    doc = _doc(
        {"s": _ask("q:a", links={"q:a": {"target": "tg:t"}})},
        {
            "est": _rule("d:est", ["q:a"], target="tg:t"),  # no path to the policy
            "rb": _rule("d:rb", ["q:a"]),
            "ra": _rule("d:ra", ["q:a"]),
            "m2": _rule("d:m2", ["d:ra", "d:rb"]),
            "m1": _rule("d:m1", ["d:ra", "d:rb"]),
            "p": _rule("d:p", ["d:m1", "d:m2"]),
        },
    )
    doc["targets"] = {"tg:t": {"statement": "s", "values": None}, "tg:none": {"statement": "n"}}
    none, t = design_outline(doc)["targets"]
    assert t["path"] == ["rule:ra", "rule:m1", "rule:p"]
    assert none == {
        **none,
        "questions": [],
        "estimated_by": [],
        "readers": [],
        "path": None,
    }


def test_a_row_s_tests_skip_any_cell_and_a_candidate_read() -> None:
    table = {
        "form": "table",
        "provides": "d:p",
        "hit": "F",
        "inputs": [{"read": "q:a"}, {"read": "candidate.score"}, {"read": "d:x"}],
        "rows": [
            {"cells": [{"is": "yes"}, {"is": 1}, "-"], "output": {"literal": "go"}},
            {"cells": ["-", "-", {"is": "no"}], "output": {"literal": "stop"}},
        ],
    }
    o = design_outline(_doc({"s": _ask("q:a")}, {"x": _rule("d:x", ["q:a"]), "p": table}))
    (p,) = [n for n in o["nodes"] if n["id"] == "rule:p"]
    assert [r["tests"] for r in p["rows"]] == [
        [{"input": 0, "read": "q:a", "node": "source:s", "cell": "yes"}],
        [{"input": 2, "read": "d:x", "node": "rule:x", "cell": "no"}],
    ]


def test_review_sorts_each_source_by_what_the_turn_did(shown: dict[str, Any]) -> None:
    decided = turn_outline(shown["decided"].document, shown["design"])["review"]
    undecided = turn_outline(shown["undecided"].document, shown["design"])["review"]
    assert [(s["source"], s["outcome"]) for s in decided["sources"]] == [
        ("ask", "asked"),
        ("desk", "asked"),
    ]
    desk = undecided["sources"][1]
    assert (desk["source"], desk["outcome"], desk["why"]) == ("desk", "not asked", "requirement")
    assert desk["marker"] == "Unasked"
    assert desk["cause"] == {"code": "need-undetermined", "refs": ["d:urgency"]}
    assert desk["requirement"] == {"ref": "d:urgency", "values": ["yes"]}
    assert desk["call"] is None
    assert turn_outline(shown["decided"].document)["review"] is None


def test_review_gives_each_row_of_a_table_entry_its_result(shown: dict[str, Any]) -> None:
    decided = turn_outline(shown["decided"].document, shown["design"])
    (act,) = decided["review"]["tables"]
    assert (act["ref"], act["rule"], act["hit"], act["policy"]) == ("d:act", "act", "F", True)
    assert [(r["n"], r["result"], r["chosen"]) for r in act["rows"]] == [
        (1, "matched", True),
        (2, "not matched", False),
        (3, "not matched", False),
        (4, "not matched", False),
    ]
    by_index = {e["index"]: e for r in decided["rounds"] for e in r["entries"]}
    tests = act["rows"][0]["tests"]
    assert [(t["read"], t["cell"], t["holds"]) for t in tests] == [
        ("d:urgency", "yes", "value"),
        ("t:ticket", "Known", "value"),
    ]
    assert all(by_index[t["entry"]]["ref"] == t["read"] for t in tests)
    assert act["rows"][1]["tests"][1]["helper"] == "pick_with_margin"
    undecided = turn_outline(shown["undecided"].document, shown["design"])
    (act,) = undecided["review"]["tables"]
    assert [(r["n"], r["result"], r["chosen"]) for r in act["rows"]][3] == (4, "matched", True)


def test_why_and_the_summary_s_bound_are_the_text_form_s(shown: dict[str, Any]) -> None:
    for turn in ("decided", "undecided"):
        o = turn_outline(shown[turn].document, shown["design"])
        assert render_turn(o).splitlines()[1] == "why  " + o["why"]
    o = turn_outline(shown["undecided"].document, shown["design"])
    near = o["summary_nearest"]
    assert near is not None
    assert o["summary"].split("  ")[6] == (
        f"nearest {near['distance']} {near['side']} {near['param']}@{near['ref']}"
    )
    text = render_turn(o).splitlines()
    assert text[text.index("sources") :][:3] == [
        "sources",
        "  asked  ask  round 1  call 3",
        "  not asked  desk  round 2  requirement: Unasked need-undetermined (d:urgency)",
    ]
    assert "  row 4  matched (its output)  -> Unsure" in text


def test_a_key_path_bound_is_its_leafs_value_at_the_key() -> None:
    loaded = designs.parts()
    record = decide_sync(loaded, {"email": 0.4, "phone": 0.6}, {}).record
    state = entry(turn_outline(record.document, loaded.document), "t:state")
    by_param = {b["param"]: b for b in state["bounds"]}
    assert by_param["part/at.email"]["bound"] == 0.5
    assert by_param["part/at.email"]["distance"] == -0.1
    # A document whose leaf lacks the key gives the bound as null.
    document = json.loads(loaded.document)
    document["parameters"]["part/at"]["value"] = {"phone": 0.5}
    state = entry(turn_outline(record.document, document), "t:state")
    assert {b["param"]: b["bound"] for b in state["bounds"]}["part/at.email"] is None


def test_a_calls_extras_is_outlined_and_survives_redaction() -> None:
    from ommat import redact
    from test_provenance import turn as provenance_turn

    record = provenance_turn(attempts=2, extras=({"passes": 2}, {"note": "x"}))
    o = turn_outline(record.document)
    (call,) = [c for r in o["rounds"] for c in r["calls"]]
    assert call["unhashed"]["extras"] == [{"passes": 2}, {"note": "x"}]
    assert "extras 2" in render_turn(o)
    assert '    extras [{"passes":2},{"note":"x"}]' in render_turn(o, detail=True)
    view = redact(record, bytes(32), {})
    r = turn_outline(view)
    (shown,) = [c for rnd in r["rounds"] for c in rnd["calls"]]
    assert str(shown["unhashed"]["extras"]).startswith("hmac-sha256:")
    assert "extras 1" in render_turn(r)
    assert '    extras "hmac-sha256:' in render_turn(r, detail=True)
    bare = turn_outline(provenance_turn(attempts=1).document)
    (none,) = [c for rnd in bare["rounds"] for c in rnd["calls"]]
    assert none["unhashed"]["extras"] is None
    assert "extras" not in render_turn(bare)
