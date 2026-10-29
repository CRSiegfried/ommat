"""A stored format-8 design, as the step-19 engine wrote it, read under its own rules: its tables
hold no ``known`` cell, which check 11 refuses before format 9."""

import json
from pathlib import Path

import pytest

from format_8_design import DESIGN
from ommat import InvalidDesign, load

FORMAT_8 = Path(__file__).parent / "format_8"
FORMAT_8_DESIGN = "sha256:2b951550cc69b5a5a31ad1d283e8f4a422dab422a307710b1636daa038beca1c"


def test_a_stored_format_8_design_loads_to_its_own_hash() -> None:
    data = (FORMAT_8 / "design.json").read_bytes()
    loaded = load(data)
    assert loaded.design_hash == FORMAT_8_DESIGN
    assert loaded.document == data
    assert json.loads(data)["format"] == 8


def test_the_same_design_built_now_is_format_9_under_another_hash() -> None:
    loaded = load(DESIGN)
    assert json.loads(loaded.document)["format"] == 9
    assert loaded.design_hash != FORMAT_8_DESIGN


@pytest.mark.parametrize(
    ("where", "shown"), [("rules", 'rule "level"'), ("policy", "the policy table")]
)
def test_check_11_refuses_a_format_8_document_holding_a_known_cell(where: str, shown: str) -> None:
    doc = json.loads((FORMAT_8 / "design.json").read_bytes())
    table = doc["rules"]["level"] if where == "rules" else doc["policy"]
    table["rows"][0]["cells"][0] = {"known": True}
    with pytest.raises(InvalidDesign) as raised:
        load(doc)
    assert 11 in [f.check for f in raised.value.failures]
    assert f"{shown} holds a Known cell, which format 8 does not have" in str(raised.value)
