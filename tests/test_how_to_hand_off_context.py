"""The how-to's example runs as the guide says: docs/how-to/hand_off_context. Each Python
block in the guide is a passage of one of the example's files, and the driver prints the
guide's output."""

import re
from pathlib import Path

import pytest

import handoff_run

DOCS = Path(__file__).resolve().parent.parent / "docs" / "how-to"
GUIDE = DOCS / "hand-off-context.md"
EXAMPLE = DOCS / "hand_off_context"


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
    handoff_run.main()
    (shown,) = _blocks("text")
    assert capsys.readouterr().out == shown
