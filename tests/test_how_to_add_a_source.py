"""The how-to's example runs as the guide says: docs/how-to/add_a_source. Each Python block in
the guide is a passage of one of the example's files, and the driver prints the guide's
output."""

# pyright: reportPrivateUsage=false

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest

import reports_run
from ommat import (
    AnswerKind,
    Cause,
    PlanKind,
    RenderedQuestion,
    SourceCall,
    Unavailable,
    ValueType,
    decide_sync,
    load,
)
from reports_design import DESIGN
from span_binding import SpanBinding
from street_spotter import StreetSpotter

DOCS = Path(__file__).resolve().parent.parent / "docs" / "how-to"
GUIDE = DOCS / "add-a-source.md"
EXAMPLE = DOCS / "add_a_source"


def _blocks(language: str) -> list[str]:
    text = GUIDE.read_text(encoding="utf-8")
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.MULTILINE | re.DOTALL)


def test_each_python_block_is_a_passage_of_the_example() -> None:
    files = [p.read_text(encoding="utf-8") for p in EXAMPLE.glob("*.py")]
    blocks = _blocks("python")
    assert blocks
    for block in blocks:
        assert any(_contiguous(block, text) for text in files), block


def _contiguous(block: str, text: str) -> bool:
    """Whether the block's lines stand together in ``text``, at any one indent."""
    lines = [line for line in block.splitlines() if line.strip()]
    source = [line for line in text.splitlines() if line.strip()]
    width = len(lines)
    strip = [line.strip() for line in lines]
    return any(
        [s.strip() for s in source[i : i + width]] == strip for i in range(len(source) - width + 1)
    )


def test_the_driver_prints_what_the_guide_shows(capsys: pytest.CaptureFixture[str]) -> None:
    results = reports_run.main()
    (shown,) = _blocks("text")
    assert capsys.readouterr().out == shown
    assert [plan.kind for plan, _ in results] == [
        PlanKind.inform,
        PlanKind.inform,
        PlanKind.escalate,
    ]
    assert [calls for _, calls in results] == [0, 0, 0]


def test_the_idia_path_binds_an_idia_provider() -> None:
    pytest.importorskip("idia")
    from idia import Dispatcher, RetryPolicy
    from idia.providers.scripted import Scripted

    import idia_path
    from test_idia_binding import answer_all, caps

    leaf = Scripted(capabilities=caps(), script=answer_all)
    bindings = idia_path.idia_bindings(Dispatcher(leaf, retry=RetryPolicy.without_delay()))
    decision = decide_sync(load(DESIGN), {"text": "The lamp on Quay Street is out."}, bindings)
    assert decision.plan.kind is PlanKind.escalate


class _Failing(StreetSpotter):
    """A model that fails on one label and answers the rest."""

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        if labels == ["lamp"]:
            raise RuntimeError("out of memory")
        return super().predict(text, labels, threshold)


def _batch(*labels: str) -> tuple[SourceCall, ...]:
    questions = {
        f"q:places/{label}": RenderedQuestion(
            AnswerKind.extraction, label, (), value_type=ValueType.text, text="t:state.text"
        )
        for label in labels
    }
    return (SourceCall("places", None, {"t:state.text": reports_run.TEXT}, (), questions),)


def test_a_model_error_marks_its_own_ref_raised() -> None:
    result = asyncio.run(SpanBinding(_Failing())(_batch("lamp", "street"), seconds=5))
    assert result.values["q:places/lamp"] == Unavailable(Cause("raised"))
    assert result.values["q:places/street"]["value"] == "Quay Street"
    assert "q:places/lamp: raised: RuntimeError: out of memory" in result.warnings
    assert result.attempts == 2


def test_a_call_that_waits_out_its_seconds_never_starts_the_model() -> None:
    model = StreetSpotter()
    binding = SpanBinding(model)
    with binding._lock:
        result = asyncio.run(binding(_batch("street"), seconds=0.01))
    assert result.values["q:places/street"] == Unavailable(Cause("timed-out"))
    assert model.calls == 0
    assert result.snapshot is None
