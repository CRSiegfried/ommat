# pyright: reportPrivateUsage=false
"""The lints over grounded forms: the grid holds nothing for a cell over a candidate read, and
the dead-question lint grounds a ``t:`` source's entries."""

from typing import Any

import test_grounded_rules as rules
import test_grounded_sources as sources
from ommat.helpers import dead_questions


def test_a_cell_over_a_candidate_read_adds_nothing_to_the_grid() -> None:
    assert dict(rules.LOADED._grid) == {}


def test_the_dead_question_lint_grounds_a_t_sources_entries() -> None:
    values: dict[str, list[Any]] = {
        "t:state.people": [[sources.ANA, sources.BO]],
        sources.CITY: ["Oslo", None],
        sources.INFO: [{"city": "x"}],
    }
    report = dead_questions(sources.LOADED, ["d:act"], values=values)
    assert report["questions"] == {sources.OUT: {"status": "not-on-grid", "witness": None}}
    assert report["points"] > 0
