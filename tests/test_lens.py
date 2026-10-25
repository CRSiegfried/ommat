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


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (b'{"log": [], "x": NaN}', "NaN is not a JSON number"),
        (b'{"log": [], "x": -Infinity}', "-Infinity is not a JSON number"),
        (b'{"log": [], "x": ' + b"1" * 5000 + b"}", "not readable JSON"),
        (b"[" * 100_000 + b"]" * 100_000, "nests too deep|not JSON"),
        (b'{"log": [], "x": "\xff"}', "not UTF-8"),
    ],
)
def test_page_refuses_a_document_that_is_not_strict_json(document: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        page([document])


@pytest.mark.parametrize(
    ("records", "skipped"),
    [(b"{}", ()), ("{}", ()), (None, ()), (5, ()), ([], "abc"), ([], [1])],
)
def test_page_refuses_arguments_of_the_wrong_type(records: object, skipped: object) -> None:
    with pytest.raises(ValueError, match="must be"):
        page(records, skipped=skipped)  # type: ignore[arg-type]


def test_the_data_is_ascii_and_keeps_a_lone_surrogate_as_an_escape() -> None:
    html = page([b'{"log": [], "x": "\\ud800 \\u2028 caf\\u00e9"}'])
    html.encode("utf-8")
    found = re.search(r'<script id="lens-data" type="application/json">(.*?)</script>', html, re.S)
    assert found is not None
    found.group(1).encode("ascii")
    assert _data(html)["records"][0]["record"]["x"] == "\ud800 \u2028 caf\u00e9"


def test_no_raw_angle_bracket_is_left_in_the_data() -> None:
    html = page([_record("<!-- <script> </script")])
    found = re.search(r'<script id="lens-data" type="application/json">(.*?)</script>', html, re.S)
    assert found is not None
    assert "<" not in found.group(1)


def test_the_command_refuses_to_overwrite_a_file_it_reads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "r.json"
    stored = _record()
    record.write_bytes(stored)
    assert main([str(record), "-o", str(record)]) == 1
    assert main([str(tmp_path), "-o", str(record)]) == 1
    assert record.read_bytes() == stored
    assert "is one of the files read" in capsys.readouterr().err


def test_the_command_skips_a_bad_file_and_names_why(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "good.json").write_bytes(_record())
    (tmp_path / "nan.json").write_bytes(b'{"log": [], "x": NaN}')
    (tmp_path / "latin.json").write_bytes(b'{"log": [], "x": "\xff"}')
    (tmp_path / "deep.json").write_bytes(b"[" * 100_000 + b"]" * 100_000)
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert sorted(_data(out.read_text(encoding="utf-8"))["skipped"]) == [
        "deep.json",
        "latin.json",
        "nan.json",
    ]
    err = capsys.readouterr().err
    assert "nan.json" in err
    assert "NaN is not a JSON number" in err
    assert "latin.json is not UTF-8 text" in err


def test_the_command_reads_a_linked_file_once(tmp_path: Path) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    (tmp_path / "link.json").symlink_to(tmp_path / "r.json")
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert len(_data(out.read_text(encoding="utf-8"))["records"]) == 1


def test_the_command_notes_a_design_that_matches_no_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    pretty = json.dumps(json.loads(DESIGN.document), indent=2).encode()
    (tmp_path / "pretty.json").write_bytes(pretty)
    assert main([str(tmp_path), "-o", str(tmp_path / "lens.html")]) == 0
    assert "design pretty.json matches no record" in capsys.readouterr().err


def test_the_command_reports_an_output_it_cannot_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    assert main([str(tmp_path / "r.json"), "-o", str(tmp_path / "no" / "lens.html")]) == 1
    assert "cannot write" in capsys.readouterr().err


def test_the_jsonl_reader_takes_crlf_lines(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_bytes(_record() + b"\r\n" + _record("two") + b"\r\n")
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert len(_data(out.read_text(encoding="utf-8"))["records"]) == 2
