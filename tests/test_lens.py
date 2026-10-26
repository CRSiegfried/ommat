import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

import per_candidate as example
from format_6_design import DESIGN as FORMAT_6
from format_6_design import STATE as FORMAT_6_STATE
from format_6_design import binding as format_6_binding
from ommat import LoadedDesign, Parameter, Target, decide_sync, load
from ommat import types as t
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


def test_the_command_exits_1_with_neither_a_record_nor_a_design(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.json").write_text('{"note": 1}')
    assert main([str(tmp_path), "-o", str(tmp_path / "lens.html")]) == 1
    assert main([str(tmp_path / "missing"), "-o", str(tmp_path / "lens.html")]) == 1
    assert not (tmp_path / "lens.html").exists()
    assert "no ommat record or design found" in capsys.readouterr().err


def test_the_command_writes_a_page_for_a_design_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "design.json").write_bytes(DESIGN.document)
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    data = _data(out.read_text(encoding="utf-8"))
    assert data["records"] == []
    assert list(data["designs"].values()) == [json.loads(DESIGN.document)]
    assert main([str(tmp_path / "design.json"), "-o", str(out)]) == 0
    # With no record, a design matching none is the point, not a slip to note.
    assert "matches no record" not in capsys.readouterr().err


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
        (b"[" * 100_000 + b"]" * 100_000, "nests deeper than 128"),
        (b'{"log": [], "x": "\xff"}', "not UTF-8"),
        (b'{"log": [], "x": 1e999}', "out of a JSON number's range"),
        (b'{"log": ' + b"[" * 130 + b"]" * 130 + b"}", "nests deeper than 128"),
        (b'{"log": [], "log": 3}', "the key 'log' is repeated"),
        ('{"log": []}'.encode("utf-16"), "not UTF-8"),
        (b'\xef\xbb\xbf{"log": []}', "not JSON"),
        (b'{"log": [], "x": "\xed\xa0\x80"}', "not UTF-8"),
        (b'{"log": [], "x": 9007199254740993}', "names no double exactly"),
    ],
    ids=[
        "nan",
        "infinity",
        "long integer",
        "deep",
        "latin-1",
        "1e999",
        "129 deep",
        "repeated key",
        "utf-16",
        "bom",
        "surrogate",
        "inexact integer",
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
    (tmp_path / "inf.json").write_bytes(b'{"log": [], "x": 1e999}')
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert sorted(_data(out.read_text(encoding="utf-8"))["skipped"]) == [
        "deep.json",
        "inf.json",
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


def test_the_command_names_a_folder_it_cannot_open(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.json").write_bytes(_record())
    locked.chmod(0)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("this user reads any folder")
        out = tmp_path / "lens.html"
        assert main([str(tmp_path), "-o", str(out)]) == 0
        assert _data(out.read_text(encoding="utf-8"))["skipped"] == ["locked/"]
        assert "skipped locked/" in capsys.readouterr().err
    finally:
        locked.chmod(0o755)


def test_the_command_reads_a_hard_link_once_and_will_not_write_over_one(tmp_path: Path) -> None:
    stored = _record()
    (tmp_path / "r.json").write_bytes(stored)
    os.link(tmp_path / "r.json", tmp_path / "copy.json")
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert len(_data(out.read_text(encoding="utf-8"))["records"]) == 1
    os.link(tmp_path / "r.json", tmp_path / "out.html")
    assert main([str(tmp_path), "-o", str(tmp_path / "out.html")]) == 1
    assert (tmp_path / "r.json").read_bytes() == stored


def test_a_design_that_matches_its_record_is_not_noted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    (tmp_path / "design.json").write_bytes(DESIGN.document)
    assert main([str(tmp_path), "-o", str(tmp_path / "lens.html")]) == 0
    assert "matches no record" not in capsys.readouterr().err


@pytest.mark.parametrize(
    "first", [b'{"item": "design", "hash": [1]}', b'{"hash": {}}', b"[]", b"3", b"null"]
)
def test_the_command_shows_a_record_whose_design_item_is_malformed(
    tmp_path: Path, first: bytes
) -> None:
    (tmp_path / "r.json").write_bytes(b'{"log": [' + first + b"]}")
    (tmp_path / "design.json").write_bytes(DESIGN.document)
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert len(_data(out.read_text(encoding="utf-8"))["records"]) == 1


def test_an_exact_large_integer_reads_as_ommat_reads_it() -> None:
    from ommat.lens import read_document

    assert read_document(b'{"log": [], "n": 9007199254740992}', "x")["n"] == 2.0**53


_SETUP = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[2], "utf8");
const data = /<script id="lens-data" type="application\/json">(.*?)<\/script>/s.exec(html)[1];
const js = /<script>(.*?)<\/script>/s.exec(html)[1];
const els = {};
const mk = (id) => ({ id, innerHTML: "", textContent: "", value: "", hidden: false, disabled: false,
  dataset: {}, classList: { toggle() {}, contains() { return false; } }, setAttribute() {},
  addEventListener() {}, querySelectorAll() { return []; },
  insertAdjacentHTML(w, h) { this.innerHTML += h; }, remove() {}, focus() {},
  closest() { return null; } });
const doc = { getElementById: (id) => els[id] || (els[id] = mk(id)),
  addEventListener() {}, activeElement: null };
doc.getElementById("lens-data").textContent = data;
const ctx = vm.createContext(
  { document: doc, window: {}, console, TextEncoder, TextDecoder, JSON });
vm.runInContext(js, ctx);
"""

_HARNESS = (
    _SETUP
    + r"""
const shown = vm.runInContext(`deck.map((d, i) => { showAt(i); return {
  name: d.name, guess: d.guess, paired: !!d.design,
  inputs: Object.fromEntries(
    [...state.inputs].map(([k, v]) => [k, v.map((x) => [x.node, x.src])])) }; })`, ctx);
process.stdout.write(JSON.stringify(shown));
"""
)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the path")
def test_the_page_traces_lineage_from_the_record_and_its_design(tmp_path: Path) -> None:
    logs = Path(__file__).parent / "idia_logs"
    triage = (logs / "turn.record.json").read_bytes()
    triage_design = (logs / "design.json").read_bytes()
    html = page({"triage": triage, "lamps": _record()}, [triage_design])
    (tmp_path / "page.html").write_text(html, encoding="utf-8")
    (tmp_path / "harness.js").write_text(_HARNESS, encoding="utf-8")
    run = subprocess.run(
        ["node", str(tmp_path / "harness.js"), str(tmp_path / "page.html")],
        capture_output=True,
        text=True,
        check=True,
    )
    shown = {r["name"]: r for r in json.loads(run.stdout)}

    triage_view = shown["triage"]
    assert triage_view["paired"]
    assert triage_view["guess"] is None
    inputs = {node: {tuple(i) for i in ins} for node, ins in triage_view["inputs"].items()}
    # A rule's reads: a link the record and the design both give is shown once, as the
    # record's, which is the stronger evidence.
    assert ("e:q:triage/hazard", "record") in inputs["e:d:hazard"]
    assert ("e:q:triage/hazard", "design") not in inputs["e:d:hazard"]
    # An answer comes from its call, in the record; the call needs what its source needs.
    call = next(n for n, _ in inputs["e:q:triage/hazard"])
    assert call.startswith("c:")
    assert ("e:t:state", "design") in inputs[call]
    # The policy reads every estimate and the registry's value.
    assert {n for n, _ in inputs["plan"]} >= {
        "e:d:asset",
        "e:d:hazard",
        "e:d:severity",
        "e:t:open_orders",
    }

    # With no design for it, a record shows only its own links, and the grounded entries
    # still lead to their candidate set.
    lamps_view = shown["lamps"]
    assert lamps_view["guess"] == "folder"
    lamps = {node: {tuple(i) for i in ins} for node, ins in lamps_view["inputs"].items()}
    assert ("e:d:streets", "record") in lamps['e:q:lamps/out["Elm Row"]']


def _targeted() -> LoadedDesign:
    """The format-6 design with targets on its question and rule, and its rule's pick declared
    as boundaries over two parameters."""
    source = FORMAT_6.sources["ask"]
    asked = dataclasses.replace(
        source.provides["q:ask/urgent"],
        target="tg:urgency",
        latent_targets=({"target": "tg:distress", "role": "reflective"},),
    )
    roles = {"bar": ["urgent/bar"], "thresholds": [], "margin": ["urgent/margin"], "mass": []}
    rule = dataclasses.replace(
        FORMAT_6.rules["urgent"], target="tg:urgency", boundaries={"q:ask/urgent": roles}
    )
    return load(
        dataclasses.replace(
            FORMAT_6,
            sources={"ask": dataclasses.replace(source, provides={"q:ask/urgent": asked})},
            rules={"urgent": rule},
            targets={
                "tg:urgency": Target("Whether the traveller needs action today"),
                "tg:distress": Target("How upset the traveller is"),
                "tg:cost": Target("What the fix would cost"),
            },
            parameters={
                "urgent/bar": Parameter(t.number(0, 1), 0.6),
                "urgent/margin": Parameter(t.number(0, 1), 0.2),
            },
        )
    )


def _run(html: str, tmp_path: Path, expression: str) -> Any:
    """``expression``'s JSON value, evaluated in the page's script under node after it ran."""
    (tmp_path / "page.html").write_text(html, encoding="utf-8")
    (tmp_path / "harness.js").write_text(_SETUP + _EVALUATE, encoding="utf-8")
    run = subprocess.run(
        ["node", str(tmp_path / "harness.js"), str(tmp_path / "page.html"), expression],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(run.stdout)


_EVALUATE = r"""
process.stdout.write(JSON.stringify(vm.runInContext(process.argv[3], ctx)));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the path")
def test_a_design_alone_opens_on_the_design_view_with_its_declared_links(tmp_path: Path) -> None:
    targeted = _targeted()
    shown = _run(
        page([], [targeted.document]),
        tmp_path,
        """(() => {
          const ins = (n) => (dstate.inputs.get(n) || []).map((x) => x.node);
          const kinds = Object.fromEntries([...dstate.nodes].map(([k, v]) => [k, v.kind]));
          const out = { tab, kinds,
            asks: dstate.links.asks.get("tg:urgency"), bears: dstate.links.bears.get("tg:distress"),
            est: dstate.links.est.get("tg:urgency"), question: ins("e:q:ask/urgent"),
            rule: ins("e:d:urgent"), policy: ins("plan"), cols: $("cols").innerHTML };
          select("g:tg:urgency"); out.urgency = $("lineage").innerHTML;
          select("g:tg:cost"); out.cost = $("lineage").innerHTML;
          select("e:q:ask/urgent"); out.asked = $("lineage").innerHTML;
          return out; })()""",
    )
    assert shown["tab"] == "design"
    assert shown["kinds"] == {
        "e:t:state": "state",
        "e:q:ask/urgent": "question",
        "e:d:urgent": "rule",
        "g:tg:cost": "target",
        "g:tg:distress": "target",
        "g:tg:urgency": "target",
        "p:urgent/bar": "param",
        "p:urgent/margin": "param",
        "plan": "policy",
    }
    assert shown["asks"] == ["q:ask/urgent"]
    assert shown["bears"] == [{"ref": "q:ask/urgent", "role": "reflective"}]
    assert shown["est"] == ["d:urgent"]
    assert "e:t:state" in shown["question"]
    assert set(shown["rule"]) == {"e:q:ask/urgent", "p:urgent/bar", "p:urgent/margin"}
    assert shown["policy"] == ["e:d:urgent"]
    assert "Asked about by" in shown["urgency"]
    assert "q:ask/urgent" in shown["urgency"]
    # A target nothing declares a link to says so, and never calls the rest unrelated.
    assert shown["cost"].count("None declared") == 3
    assert "no link declared" in shown["cost"]
    assert "unrelated" not in shown["cost"] + shown["urgency"]
    assert "bears on · reflective" in shown["asked"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the path")
def test_a_turn_shows_targets_and_the_boundaries_its_design_declares(tmp_path: Path) -> None:
    targeted = _targeted()
    plain = load(FORMAT_6)
    record = decide_sync(targeted, FORMAT_6_STATE, {"default": format_6_binding}).record
    other = decide_sync(plain, FORMAT_6_STATE, {"default": format_6_binding}).record
    identity = json.loads(record.document)["log"][3]["questions"]["q:ask/urgent"]["identity"]
    shown = _run(
        page({"a": record.document, "b": other.document}, [targeted.document, plain.document]),
        tmp_path,
        """(() => {
          const out = {};
          showAt(0); out.targeted = $("cols").innerHTML;
          select("e:q:ask/urgent"); out.jump = $("lineage").innerHTML;
          setTab("design", designNode(selected)); out.design = [tab, selected, designKey];
          setTab("turn", turnNode(selected)); out.back = [tab, selected];
          showAt(1); out.plain = $("cols").innerHTML;
          setTab("turns"); out.byDesign = $("turnsView").innerHTML;
          turnsBy = IDENTITY; renderTurns(); out.byQuestion = $("turnsView").innerHTML;
          return out; })()""".replace("IDENTITY", json.dumps(identity)),
    )
    assert "asks tg:urgency" in shown["targeted"]
    assert "bears on tg:distress (reflective)" in shown["targeted"]
    assert "estimates tg:urgency" in shown["targeted"]
    assert shown["targeted"].count('class="tick"') == 2  # the bar, on each label's track
    assert "lead 80.0%" in shown["targeted"]
    assert "margin 20.0%" in shown["targeted"]
    assert "bar 60.0%" in shown["targeted"]
    assert identity[7:15] in shown["targeted"]
    assert 'data-jump="design"' in shown["jump"]
    key = "sha256:" + hashlib.sha256(targeted.document).hexdigest()
    assert shown["design"] == ["design", "e:q:ask/urgent", key]
    assert shown["back"] == ["turn", "e:q:ask/urgent"]
    assert "no target declared" in shown["plain"]
    assert "no boundaries declared" in shown["plain"]
    assert 'class="tick"' not in shown["plain"]
    # Each turn under its own design's hash, the two designs apart.
    assert shown["byDesign"].count('class="group"') == 2
    assert shown["byDesign"].count("data-row=") == 2
    # One question asked under both designs has one identity: targets stay out of it.
    assert shown["byQuestion"].count("data-row=") == 2
    assert "90.0%" in shown["byQuestion"]
    assert "d:urgent = &quot;yes&quot;" in shown["byQuestion"]
