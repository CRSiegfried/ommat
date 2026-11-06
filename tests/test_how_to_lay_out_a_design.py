"""The how-to's example runs as the guide says: docs/how-to/lay_out_a_design. Each Python block in
the guide is a passage of one of the example's files, the driver prints the guide's output, and
the two checks catch what they are for."""

import dataclasses
import re
import types
from pathlib import Path

import pytest

import refund_design
import refund_run
from ommat import Design, Parameter, decide_sync, load, replay_sync
from ommat import types as t
from ommat.bindings import fixed
from refund_checks import map_disagrees, moved_nothing

DOCS = Path(__file__).resolve().parent.parent / "docs" / "how-to"
GUIDE = DOCS / "lay-out-a-design.md"
EXAMPLE = DOCS / "lay_out_a_design"


def _blocks(language: str) -> list[str]:
    text = GUIDE.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)


def _contiguous(block: str, text: str) -> bool:
    """Whether the block's lines stand together in ``text``, at any one indent."""
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    source = [line.strip() for line in text.splitlines() if line.strip()]
    width = len(lines)
    return any(source[i : i + width] == lines for i in range(len(source) - width + 1))


def test_each_python_block_is_a_passage_of_the_example() -> None:
    files = [p.read_text(encoding="utf-8") for p in EXAMPLE.glob("*.py")]
    blocks = _blocks("python")
    assert blocks
    for block in blocks:
        assert any(_contiguous(block, text) for text in files), block


def test_the_driver_prints_the_guide_s_output(capsys: pytest.CaptureFixture[str]) -> None:
    refund_run.main()
    (shown,) = _blocks("text")
    assert capsys.readouterr().out == shown


def test_the_map_check_names_a_ref_missing_from_either_side() -> None:
    edited = types.ModuleType("edited")
    edited.__doc__ = (refund_design.__doc__ or "").replace(
        "- ``d:reason``: why", "- ``d:reasons``: why"
    )
    assert map_disagrees(edited, refund_design.DESIGN) == {
        "not provided": ["d:reasons"],
        "not in the map": ["d:reason"],
    }


def _renamed_within_window() -> Design:
    """The design with ``within_window`` copied under another name, as a tidy that renames
    a rule function leaves it."""
    old = refund_design.within_window
    new = types.FunctionType(old.__code__, old.__globals__, "order_is_young")
    new.__qualname__ = "order_is_young"
    rules = dict(refund_design.DESIGN.rules)
    rules["within_window"] = dataclasses.replace(rules["within_window"], function=new)
    return dataclasses.replace(refund_design.DESIGN, rules=rules)


def test_a_renamed_rule_function_moves_nothing_but_the_hash() -> None:
    renamed = _renamed_within_window()
    assert load(renamed).document != load(refund_design.DESIGN).document
    assert moved_nothing(refund_design.DESIGN, renamed)
    shorter = {**refund_design.BARS, "bars/window_days": Parameter(t.integer(0, 365), 14)}
    changed = dataclasses.replace(refund_design.DESIGN, parameters=shorter)
    assert not moved_nothing(refund_design.DESIGN, changed)


def test_a_record_replays_under_the_tidied_design_with_only_a_design_mismatch() -> None:
    model = fixed(
        {
            "q:refund": {"yes": 0.9, "no": 0.1},
            "q:reason": {"damaged": 0.9, "late": 0.05, "changed_mind": 0.05},
        }
    )
    decision = decide_sync(
        load(refund_design.DESIGN),
        {"message": "It arrived broken", "order_id": "A-1"},
        {"default": model, refund_design.ORDERS: fixed({"t:order_age": 3})},
    )
    _, findings = replay_sync(decision.record, load(_renamed_within_window()))
    assert [f["kind"] for f in findings] == ["design-mismatch"]
