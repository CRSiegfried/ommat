"""A stored format-8 design, as the step-19 engine wrote it: check 11 refuses it, naming its
format."""

import json
from pathlib import Path

import pytest

from ommat import InvalidDesign, load

FORMAT_8 = Path(__file__).parent / "format_8"


def test_check_11_refuses_a_stored_format_8_design_naming_its_format() -> None:
    data = (FORMAT_8 / "design.json").read_bytes()
    assert json.loads(data)["format"] == 8
    with pytest.raises(InvalidDesign) as raised:
        load(data)
    assert [f.check for f in raised.value.failures] == [11]
    assert "the document's format 8 is not 12" in str(raised.value)
