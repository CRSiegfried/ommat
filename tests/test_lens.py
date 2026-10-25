import hashlib
import json
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

import per_candidate as example
from ommat import decide_sync, load
from ommat.lens import page
from ommat.lens.__main__ import main

DESIGN = load(example.DESIGN)


def _record(text: str = "Lamps out on Quay Street.") -> bytes:
    state = {"text": text, "streets": ["Quay Street", "Elm Row"]}
    return decide_sync(DESIGN, state, {"default": example.binding}).record.document


def _data(html: str) -> dict[str, Any]:
    found = re.search(r'<script id="lens-data" type="application/json">(.*?)</script>', html, re.S)
    assert found is not None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def test_the_page_embeds_each_record_and_keys_each_design_by_its_hash() -> None:
    record = _record()
    data = _data(page({"turn.json": record}, [DESIGN.document]))
    assert [r["name"] for r in data["records"]] == ["turn.json"]
    assert data["records"][0]["record"] == json.loads(record)
    key = "sha256:" + hashlib.sha256(DESIGN.document).hexdigest()
    assert data["designs"] == {key: json.loads(DESIGN.document)}
    # The key is the hash the record's design item holds, so the page pairs them.
    assert json.loads(record)["log"][0]["hash"] == key


def test_a_sequence_of_records_is_named_by_position() -> None:
    data = _data(page([_record(), _record("Elm Row is dark.")]))
    assert [r["name"] for r in data["records"]] == ["record 1", "record 2"]
    assert data["designs"] == {}


def test_no_text_in_a_record_closes_the_page_s_script() -> None:
    html = page([_record("</script><script>alert(1)</script><!--")])
    template = files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
    assert html.count("</script>") == template.count("</script>")
    assert "alert(1)" in _data(html)["records"][0]["record"]["log"][2]["value"]["text"]


def test_the_page_loads_nothing_from_the_network() -> None:
    template = files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
    assert not re.search(r"https?://|//[a-z0-9.-]+\.[a-z]{2,}/|@import|\bsrc=", template, re.I)


@pytest.mark.parametrize(
    ("records", "designs", "message"),
    [
        (["{}"], (), "must be bytes"),
        ([b"not json"], (), "is not JSON"),
        ([b"[1]"], (), "is not a JSON object"),
        ([b'{"log": {}}'], (), "holds no 'log' list"),
        ([], [b"[]"], "design 1 is not a JSON object"),
    ],
)
def test_page_refuses_what_is_not_a_record_or_design(
    records: list[object], designs: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        page(records, designs)  # type: ignore[arg-type]


def test_the_command_writes_a_page_for_every_record_under_a_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logs = tmp_path / "logs"
    (logs / "older").mkdir(parents=True)
    (logs / "b.json").write_bytes(_record())
    (logs / "older" / "a.jsonl").write_bytes(_record("one") + b"\n" + _record("two") + b"\n")
    (logs / "design.json").write_bytes(DESIGN.document)
    (logs / "notes.json").write_text('{"note": 1}')
    (logs / "readme.txt").write_text("not read")
    out = tmp_path / "lens.html"
    assert main([str(logs), "-o", str(out)]) == 0
    data = _data(out.read_text(encoding="utf-8"))
    assert [r["name"] for r in data["records"]] == [
        "b.json",
        "older/a.jsonl · line 1",
        "older/a.jsonl · line 2",
    ]
    assert len(data["designs"]) == 1
    assert data["skipped"] == ["notes.json"]
    captured = capsys.readouterr()
    assert "wrote" in captured.out
    assert "skipped notes.json" in captured.err


def test_the_command_takes_one_record_file(tmp_path: Path) -> None:
    (tmp_path / "turn.json").write_bytes(_record())
    out = tmp_path / "lens.html"
    assert main([str(tmp_path / "turn.json"), "-o", str(out)]) == 0
    assert [r["name"] for r in _data(out.read_text(encoding="utf-8"))["records"]] == ["turn.json"]


def test_the_command_exits_1_with_no_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "design.json").write_bytes(DESIGN.document)
    assert main([str(tmp_path), "-o", str(tmp_path / "lens.html")]) == 1
    assert main([str(tmp_path / "missing"), "-o", str(tmp_path / "lens.html")]) == 1
    assert not (tmp_path / "lens.html").exists()
    assert "no ommat record found" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the path")
def test_the_page_s_script_parses(tmp_path: Path) -> None:
    html = page([_record()], [DESIGN.document])
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    script = tmp_path / "page.js"
    script.write_text(scripts[0], encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True)
