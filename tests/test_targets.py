"""Targets: a design declaring them loads and round-trips byte for byte, a target may stand
under several roles, load refuses an undeclared id, a self-link, a repeated link and a member of
another form, and a question's identity leaves its targets out."""

import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ommat import (
    CallResult,
    Design,
    InvalidDesign,
    Question,
    ReadView,
    Rule,
    Source,
    SourceCall,
    Table,
    Target,
    TargetRole,
    Text,
    decide_sync,
    load,
)
from ommat import types as t
from ommat.helpers import pick_with_margin

FORMAT_6 = Path(__file__).parent / "format_6"


def urgent(r: ReadView) -> object:
    return pick_with_margin(r, "q:ask/urgent", 0.6, 0.2)


def act(r: ReadView) -> str:
    return "act" if r.read("d:urgent") == "yes" else "hold"


DESIGN = Design(
    name="targets",
    state=t.record(request=t.string),
    sources={
        "ask": Source(
            {
                "q:ask/urgent": Question(
                    "noul",
                    Text(
                        "Does {request} need action today?",
                        [{"name": "request", "origin": {"ref": "t:state.request"}}],
                    ),
                )
            },
            needs=["t:state"],
        ),
    },
    rules={
        "urgent": Rule(
            "d:urgent", t.label("no", "yes"), urgent, version="1", reads=["q:ask/urgent"]
        ),
        "act": Rule("d:act", t.label("act", "hold"), act, version="1", reads=["d:urgent"]),
    },
    policy="d:act",
)

STATE = {"request": "a cancelled flight"}


async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"q:ask/urgent": {"yes": 0.9, "no": 0.1}})


TARGETS = {
    "tg:urgency": Target("Whether the traveller needs action today", values="yes or no"),
    "tg:distress": Target("How upset the traveller is"),
}


def targeted(
    target: object = "tg:urgency",
    latent: object = ({"target": "tg:distress", "role": "reflective"},),
    rule_target: object = "tg:urgency",
    targets: Mapping[str, Any] | None = None,
) -> Design:
    """The design above with its question, its rule and the design targeted."""
    source = DESIGN.sources["ask"]
    question = source.provides["q:ask/urgent"]
    asked = dataclasses.replace(question, target=target, latent_targets=latent)  # type: ignore[arg-type]
    rule = dataclasses.replace(DESIGN.rules["urgent"], target=rule_target)  # type: ignore[arg-type]
    return dataclasses.replace(
        DESIGN,
        sources={"ask": dataclasses.replace(source, provides={"q:ask/urgent": asked})},
        rules={**DESIGN.rules, "urgent": rule},
        targets=TARGETS if targets is None else targets,
    )


def failures(design: Design | dict[str, Any]) -> list[tuple[int, tuple[str, ...], str]]:
    """Each failure load raises on a design or a document: its check, names and message."""
    with pytest.raises(InvalidDesign) as raised:
        load(design)
    return [(f.check, f.names, f.message) for f in raised.value.failures]


def test_a_design_declaring_targets_writes_them_and_round_trips_byte_for_byte() -> None:
    loaded = load(targeted())
    doc = json.loads(loaded.document)
    assert doc["format"] == 10
    assert doc["targets"] == {
        "tg:distress": {"statement": "How upset the traveller is", "values": None},
        "tg:urgency": {
            "statement": "Whether the traveller needs action today",
            "values": "yes or no",
        },
    }
    asked = doc["sources"]["ask"]["provides"]["q:ask/urgent"]
    assert asked["target"] == "tg:urgency"
    assert asked["latent_targets"] == [{"role": "reflective", "target": "tg:distress"}]
    assert doc["rules"]["urgent"]["target"] == "tg:urgency"
    again = load(loaded.document)
    assert again.document == loaded.document
    assert again.design_hash == loaded.design_hash


@pytest.mark.parametrize(
    "other",
    [
        targeted(target=None),
        targeted(latent=()),
        targeted(latent=({"target": "tg:distress", "role": "formative"},)),
        targeted(rule_target=None),
        targeted(targets={**TARGETS, "tg:distress": Target("How calm the traveller is")}),
        targeted(targets={**TARGETS, "tg:distress": Target("How upset", values="0 to 3")}),
    ],
)
def test_every_target_member_is_in_the_design_hash(other: Design) -> None:
    assert load(other).design_hash != load(targeted()).design_hash


def test_a_role_is_held_as_its_member_and_latent_targets_are_written_unordered() -> None:
    latent = (
        {"target": "tg:distress", "role": "unknown"},
        {"target": "tg:distress", "role": TargetRole.formative},
    )
    design = targeted(latent=latent)
    held = design.sources["ask"].provides["q:ask/urgent"].latent_targets
    assert [m["role"] for m in held] == [TargetRole.unknown, TargetRole.formative]
    asked = json.loads(load(design).document)["sources"]["ask"]["provides"]["q:ask/urgent"]
    assert asked["latent_targets"] == [
        {"role": "formative", "target": "tg:distress"},
        {"role": "unknown", "target": "tg:distress"},
    ]


def test_a_target_listed_under_two_roles_loads() -> None:
    latent = (
        {"target": "tg:distress", "role": "reflective"},
        {"target": "tg:distress", "role": "formative"},
    )
    asked = json.loads(load(targeted(latent=latent)).document)["sources"]["ask"]["provides"]
    assert asked["q:ask/urgent"]["latent_targets"] == [
        {"role": "formative", "target": "tg:distress"},
        {"role": "reflective", "target": "tg:distress"},
    ]


def test_a_design_with_targets_and_no_link_loads() -> None:
    doc = json.loads(load(targeted(target=None, latent=(), rule_target=None)).document)
    assert set(doc["targets"]) == set(TARGETS)


@pytest.mark.parametrize(
    ("design", "names", "where"),
    [
        (targeted(target="tg:missing"), ("q:ask/urgent",), 'source "ask"\'s question'),
        (
            targeted(latent=({"target": "tg:missing", "role": "unknown"},)),
            ("q:ask/urgent",),
            "latent_targets",
        ),
        (targeted(rule_target="tg:missing"), ("urgent",), 'rule "urgent"'),
    ],
)
def test_check_3_refuses_an_undeclared_target_id(
    design: Design, names: tuple[str, ...], where: str
) -> None:
    ((check, named, message),) = failures(design)
    assert (check, named) == (3, names)
    assert where in message
    assert message.endswith('names the target "tg:missing", which is not declared')


def test_check_3_refuses_every_link_when_no_target_is_declared() -> None:
    assert {(c, n) for c, n, _ in failures(targeted(targets={}))} == {
        (3, ("q:ask/urgent",)),
        (3, ("urgent",)),
    }


def test_check_1_refuses_a_questions_own_target_as_latent() -> None:
    latent = ({"target": "tg:urgency", "role": "reflective"},)
    assert failures(targeted(latent=latent)) == [
        (
            1,
            ("q:ask/urgent",),
            'source "ask"\'s question "q:ask/urgent" lists its own target "tg:urgency" as latent',
        )
    ]


def test_check_1_refuses_a_link_repeated_under_one_role() -> None:
    latent = (
        {"target": "tg:distress", "role": "formative"},
        {"target": "tg:distress", "role": TargetRole.formative},
    )
    assert failures(targeted(latent=latent)) == [
        (
            1,
            ("q:ask/urgent",),
            'source "ask"\'s question "q:ask/urgent" lists the target "tg:distress" twice as '
            "formative",
        )
    ]


@pytest.mark.parametrize(
    ("design", "says"),
    [
        (
            targeted(latent=({"target": "tg:distress", "role": "causal"},)),
            "hold a role that is not reflective, formative or unknown",
        ),
        (targeted(target=7), "'s target is not a string"),
        (targeted(rule_target=["tg:urgency"]), 'rule "urgent"\'s target is not a string'),
        (
            targeted(latent=({"target": "tg:distress"},)),
            "hold a member that is not a target and a role",
        ),
        (
            targeted(latent=({"target": 3, "role": "unknown"},)),
            "latent_targets' target is not a string",
        ),
        (targeted(latent="tg:distress"), "latent_targets are not a list"),
        (targeted(latent=("tg:distress",)), "hold a member that is not a target and a role"),
        (
            targeted(latent=({"role": "unknown"},)),
            "hold a member that is not a target and a role",
        ),
        (targeted(targets={"tg:urgency": "urgent"}), 'targets maps "tg:urgency" to no Target'),
        (
            targeted(targets={**TARGETS, "tg:urgency": Target(None)}),  # type: ignore[arg-type]
            'target "tg:urgency"\'s statement is not a string',
        ),
        (
            targeted(targets={**TARGETS, "tg:urgency": Target("s", values=2)}),  # type: ignore[arg-type]
            'target "tg:urgency"\'s values is not a string',
        ),
    ],
)
def test_check_10_refuses_a_target_member_of_another_form(design: Design, says: str) -> None:
    found = failures(design)
    assert any(c == 10 and says in m for c, _, m in found), found


def test_a_questions_identity_leaves_its_targets_out() -> None:
    """The stored format-6 record asked this question before targets existed; the identity a
    design gives it, targets declared, is the one that record holds."""
    stored = json.loads((FORMAT_6 / "record.json").read_bytes())
    (call,) = [i for i in stored["log"] if i["item"] == "call"]
    expected = call["questions"]["q:ask/urgent"]["identity"]
    for design in (DESIGN, targeted(), targeted(target=None, latent=())):
        record = decide_sync(load(design), STATE, {"default": binding}).record
        (asked,) = [i for i in record.log if i["item"] == "call"]
        assert asked["questions"]["q:ask/urgent"]["identity"] == expected


def test_targets_leave_the_decision_and_the_entries_unchanged() -> None:
    plain = decide_sync(load(DESIGN), STATE, {"default": binding})
    with_targets = decide_sync(load(targeted()), STATE, {"default": binding})
    assert with_targets.estimate == plain.estimate == "act"
    entries = [[i for i in d.record.log if i["item"] == "entry"] for d in (plain, with_targets)]
    assert entries[0] == entries[1]


def test_a_question_and_a_rule_default_to_no_target() -> None:
    question = Question("noul", Text("Is it urgent?"))
    assert (question.target, question.latent_targets) == (None, ())
    assert Rule("d:x", "string", urgent).target is None
    assert Table("d:x", "string", "U", [], []).target is None


@pytest.mark.parametrize(
    ("path", "member", "says"),
    [
        (("targets", "tg:urgency"), "values", 'target "tg:urgency" has no member'),
        (
            ("sources", "ask", "provides", "q:ask/urgent", "latent_targets", 0),
            "role",
            "latent target has no member",
        ),
        (("rules", "urgent"), "target", 'rule "urgent" has no member "target"'),
    ],
)
def test_check_11_holds_a_documents_target_members(
    path: tuple[str | int, ...], member: str, says: str
) -> None:
    doc: dict[str, Any] = json.loads(load(targeted()).document)
    at: Any = doc
    for step in path:
        at = at[step]
    del at[member]
    with pytest.raises(InvalidDesign) as raised:
        load(doc)
    assert [f.check for f in raised.value.failures] == [11]
    assert says in str(raised.value)


def document(**edits: Any) -> dict[str, Any]:
    """The targeted design's document, its question's and rule's target members set as given."""
    doc: dict[str, Any] = json.loads(load(targeted()).document)
    asked = doc["sources"]["ask"]["provides"]["q:ask/urgent"]
    for member in ("target", "latent_targets"):
        if member in edits:
            asked[member] = edits[member]
    if "rule_target" in edits:
        doc["rules"]["urgent"]["target"] = edits["rule_target"]
    if "targets" in edits:
        doc["targets"] = edits["targets"]
    return doc


@pytest.mark.parametrize(
    ("edits", "check", "says"),
    [
        ({"target": "tg:missing"}, 3, 'names the target "tg:missing", which is not declared'),
        ({"rule_target": "tg:missing"}, 3, 'names the target "tg:missing", which is not declared'),
        (
            {"latent_targets": [{"role": "reflective", "target": "tg:urgency"}]},
            1,
            'lists its own target "tg:urgency" as latent',
        ),
        (
            {
                "latent_targets": [
                    {"role": "formative", "target": "tg:distress"},
                    {"role": "formative", "target": "tg:distress"},
                ]
            },
            1,
            'lists the target "tg:distress" twice as formative',
        ),
        (
            {"latent_targets": [{"role": "causal", "target": "tg:distress"}]},
            10,
            "hold a role that is not reflective, formative or unknown",
        ),
        ({"target": 7}, 10, "'s target is not a string"),
        (
            {
                "targets": {
                    "tg:distress": {"statement": 1, "values": None},
                    **{"tg:urgency": {"statement": "s", "values": None}},
                }
            },
            10,
            'target "tg:distress"\'s statement is not a string',
        ),
    ],
)
def test_a_documents_target_slips_are_refused_as_a_designs_are(
    edits: dict[str, Any], check: int, says: str
) -> None:
    found = failures(document(**edits))
    assert any(c == check and says in m for c, _, m in found), found


def test_a_tables_target_is_held_to_its_form() -> None:
    table = Table("d:urgent", 'label{"no", "yes"}', "U", [], [], target="tg:urgency")
    assert table.target == "tg:urgency"
    design = dataclasses.replace(
        targeted(rule_target=None), rules={**DESIGN.rules, "urgent": table}
    )
    assert load(design).design_hash.startswith("sha256:")
    unknown = dataclasses.replace(table, target="tg:nowhere")
    found = failures(dataclasses.replace(design, rules={**design.rules, "urgent": unknown}))
    assert {c for c, _, _ in found} == {3}
    bad = dataclasses.replace(table, target=3)  # type: ignore[arg-type]
    found = failures(dataclasses.replace(design, rules={**design.rules, "urgent": bad}))
    assert any(c == 10 and 'rule "urgent"\'s target is not a string' in m for c, _, m in found)


def test_a_role_names_its_member_and_any_other_string_stays_a_string() -> None:
    latent = ({"target": "tg:distress", "role": "reflective"}, {"target": "x", "role": "causal"})
    held = Question("noul", Text("Upset?"), latent_targets=latent).latent_targets
    assert held[0]["role"] is TargetRole.reflective
    assert type(held[1]["role"]) is str
    assert [str(r) for r in TargetRole] == ["reflective", "formative", "unknown"]


def test_a_design_declares_no_target_and_a_target_no_values_by_default() -> None:
    assert dict(dataclasses.replace(DESIGN).targets) == {}
    assert Target("s").values is None
