"""The tutorial's example runs as the tutorial says: docs/tutorials/write_a_design."""

import pytest

import helpdesk_design
import helpdesk_run
from ommat import load


def test_the_kept_document_loads_to_the_same_hash() -> None:
    loaded = load(helpdesk_design.DESIGN)
    rules = helpdesk_design.RULES
    functions = {name: rule.function for name, rule in rules.items()}
    assert load(loaded.document, functions=functions).design_hash == loaded.design_hash


def test_the_driver_decides_and_replays(capsys: pytest.CaptureFixture[str]) -> None:
    results = helpdesk_run.main()
    assert [decision for decision, _ in results] == ["route_delivery_with_order", "route_delivery"]
    assert [findings for _, findings in results] == [[], []]
    assert "replay findings: []" in capsys.readouterr().out
