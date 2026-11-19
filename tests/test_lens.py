# pyright: reportPrivateUsage=false
"""The record lens: what the page embeds and refuses, the command that writes it, and, under
node with the DOM stub in lens_stub.js, what the page's script draws from the outlines."""

import json
import os
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

import outline_showcase as showcase
from designs import full
from ledger_logs import LOGS, fresh, rehashed
from ommat import decide_sync, load, read_record
from ommat._documents import read_document
from ommat.lens import page
from ommat.lens.__main__ import main
from ommat.outline import design_outline, render_design, render_turn, turn_outline

TESTS = Path(__file__).parent
IDIA = TESTS / "idia_logs"
FORMAT_10 = TESTS / "format_10"
FORMAT_9 = TESTS / "format_9"
STUB = TESTS / "lens_stub.js"
DESIGN = showcase.loaded()
_SLOT = re.compile(r'<script id="lens-data" type="application/json">(.*?)</script>', re.S)
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not on the path")


def _record(subject: str = "Server down") -> bytes:
    """A turn the showcase design decides, its state's subject ``subject``."""
    state = {"subject": subject, "body": "Nothing loads since 9."}
    bindings = {"llm": showcase.llm(*showcase.DECIDED), "desk": showcase.desk}
    return decide_sync(DESIGN, state, bindings).record.document


def _ledger(name: str) -> bytes:
    """A format-8 ledger log as a record document, with no design to pair with."""
    log = json.loads(json.dumps(fresh(LOGS[name])))
    return read_record(rehashed({"log": log, "unhashed": {}})).document


def _data(html: str) -> dict[str, Any]:
    found = _SLOT.search(html)
    assert found is not None
    data: dict[str, Any] = json.loads(found.group(1))
    return data


def _docs(html: str) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = _data(html)["sets"][0]["docs"]
    return docs


def _paths(html: str, kind: str) -> list[str]:
    return [d["path"] for d in _docs(html) if d["outline"]["kind"] == kind]


def _with_data(html: str, data: dict[str, Any]) -> str:
    """The page with ``data`` embedded in place of its own, escaped as ``page`` escapes it."""
    found = _SLOT.search(html)
    assert found is not None
    embedded = json.dumps(data).replace("<", "\\u003c")
    return html[: found.start(1)] + embedded + html[found.end(1) :]


def test_the_page_embeds_each_document_s_outline_and_text_and_pairs_a_record_by_hash() -> None:
    record = _record()
    html = page({"turn.json": record}, [DESIGN.document])
    data = _data(html)
    assert data["read"] == 1
    assert data["sets"][0]["name"] == ""
    assert data["sets"][0]["skipped"] == []
    design, turn = data["sets"][0]["docs"]
    outline = design_outline(DESIGN.document)
    assert design == {
        "path": "design 1",
        "outline": json.loads(json.dumps(outline)),
        "text": render_design(outline),
    }
    paired = turn_outline(record, DESIGN.document)
    assert turn == {
        "path": "turn.json",
        "outline": json.loads(json.dumps(paired)),
        "text": render_turn(paired),
    }
    assert turn["outline"]["paired"] is True
    assert turn["outline"]["design_hash"] == design["outline"]["hash"]


def test_a_record_whose_design_is_not_given_is_outlined_unpaired() -> None:
    (turn,) = _docs(page([_record()]))
    assert turn["outline"]["paired"] is False
    assert turn["outline"]["nearest_bounds"] is None


def test_a_sequence_of_records_and_designs_is_named_by_position() -> None:
    other = (IDIA / "design.json").read_bytes()
    html = page([_record(), _record("Mail is slow")], [DESIGN.document, other])
    assert _paths(html, "design") == ["design 1", "design 2"]
    assert _paths(html, "turn") == ["record 1", "record 2"]


def test_a_mapping_of_designs_names_them_and_a_design_given_twice_is_kept_once() -> None:
    other = (IDIA / "design.json").read_bytes()
    html = page(
        {"turn.json": _record()},
        {"first.json": DESIGN.document, "again.json": DESIGN.document, "idia.json": other},
    )
    assert _paths(html, "design") == ["first.json", "idia.json"]
    assert _paths(html, "turn") == ["turn.json"]
    assert _docs(html)[-1]["outline"]["paired"] is True


def test_no_text_in_a_record_closes_the_page_s_script() -> None:
    html = page([_record("</script><script>alert(1)</script><!--")])
    template = files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
    assert html.count("</script>") == template.count("</script>")
    (turn,) = _docs(html)
    assert "alert(1)" in turn["outline"]["state"]["value"]["subject"]
    assert "alert(1)" in turn["text"]


def test_the_page_loads_nothing_from_the_network() -> None:
    template = files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
    assert "default-src 'none'" in template
    # The SVG namespace is a name, not an address the page fetches.
    named = template.replace("'http://www.w3.org/2000/svg'", "''")
    assert not re.search(
        r"https?://|//[a-z0-9.-]+\.[a-z]{2,}/|@import|\bsrc=|fetch\(|XMLHttpRequest", named, re.I
    )


@pytest.mark.parametrize(
    ("records", "designs", "message"),
    [
        (["{}"], (), "must be bytes"),
        ([b"not json"], (), "is not JSON"),
        ([b"[1]"], (), "is not a JSON object"),
        ([b'{"log": {}}'], (), "holds no 'log' list"),
        ([], [b"[]"], "design 1 is not a JSON object"),
        ([], {"d.json": b"[]"}, "d.json is not a JSON object"),
    ],
)
def test_page_refuses_what_is_not_a_record_or_design(
    records: list[object], designs: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        page(records, designs)  # type: ignore[arg-type]


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
    ("records", "designs", "skipped"),
    [
        (b"{}", (), ()),
        ("{}", (), ()),
        (None, (), ()),
        (5, (), ()),
        ([], b"{}", ()),
        ([], None, ()),
        ([], (), "abc"),
        ([], (), [1]),
    ],
)
def test_page_refuses_arguments_of_the_wrong_type(
    records: object, designs: object, skipped: object
) -> None:
    with pytest.raises(ValueError, match="must be"):
        page(records, designs, skipped=skipped)  # type: ignore[arg-type]


def test_the_data_is_ascii_and_keeps_a_lone_surrogate_as_an_escape() -> None:
    record = _record("PLACEHOLDER").replace(b"PLACEHOLDER", b"\\ud800 \\u2028 caf\\u00e9")
    html = page([record])
    html.encode("utf-8")
    found = _SLOT.search(html)
    assert found is not None
    found.group(1).encode("ascii")
    (turn,) = _docs(html)
    assert turn["outline"]["state"]["value"]["subject"] == "\ud800 \u2028 caf\u00e9"


def test_no_raw_angle_bracket_is_left_in_the_data() -> None:
    html = page([_record("<!-- <script> </script")], [DESIGN.document])
    found = _SLOT.search(html)
    assert found is not None
    assert "<" not in found.group(1)


def test_an_exact_large_integer_reads_as_ommat_reads_it() -> None:
    assert read_document(b'{"log": [], "n": 9007199254740992}', "x")["n"] == 2.0**53


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
    html = out.read_text(encoding="utf-8")
    assert _paths(html, "turn") == ["b.json", "older/a.jsonl · line 1", "older/a.jsonl · line 2"]
    assert _paths(html, "design") == ["design.json"]
    assert all(d["outline"]["paired"] for d in _docs(html) if d["outline"]["kind"] == "turn")
    assert _data(html)["sets"][0]["skipped"] == ["notes.json"]
    captured = capsys.readouterr()
    assert "wrote" in captured.out
    assert "skipped notes.json" in captured.err


def test_the_command_takes_one_record_file(tmp_path: Path) -> None:
    (tmp_path / "turn.json").write_bytes(_record())
    out = tmp_path / "lens.html"
    assert main([str(tmp_path / "turn.json"), "-o", str(out)]) == 0
    assert _paths(out.read_text(encoding="utf-8"), "turn") == ["turn.json"]


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
    html = out.read_text(encoding="utf-8")
    assert _paths(html, "turn") == []
    (design,) = _docs(html)
    assert design["outline"] == json.loads(json.dumps(design_outline(DESIGN.document)))
    assert main([str(tmp_path / "design.json"), "-o", str(out)]) == 0
    # With no record, a design matching none is the point, not a slip to note.
    assert "matches no record" not in capsys.readouterr().err


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
    assert sorted(_data(out.read_text(encoding="utf-8"))["sets"][0]["skipped"]) == [
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
    assert len(_paths(out.read_text(encoding="utf-8"), "turn")) == 1


def test_the_command_reads_a_hard_link_once_and_will_not_write_over_one(tmp_path: Path) -> None:
    stored = _record()
    (tmp_path / "r.json").write_bytes(stored)
    os.link(tmp_path / "r.json", tmp_path / "copy.json")
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert len(_paths(out.read_text(encoding="utf-8"), "turn")) == 1
    os.link(tmp_path / "r.json", tmp_path / "out.html")
    assert main([str(tmp_path), "-o", str(tmp_path / "out.html")]) == 1
    assert (tmp_path / "r.json").read_bytes() == stored


def test_the_command_notes_a_design_that_matches_no_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    pretty = json.dumps(json.loads(DESIGN.document), indent=2).encode()
    (tmp_path / "pretty.json").write_bytes(pretty)
    out = tmp_path / "lens.html"
    assert main([str(tmp_path), "-o", str(out)]) == 0
    assert "design pretty.json matches no record" in capsys.readouterr().err
    # Its bytes differ from the ones the record ran under, so the record is outlined unpaired.
    (turn,) = [d for d in _docs(out.read_text(encoding="utf-8")) if d["path"] == "r.json"]
    assert turn["outline"]["paired"] is False


def test_a_design_that_matches_its_record_is_not_noted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "r.json").write_bytes(_record())
    (tmp_path / "design.json").write_bytes(DESIGN.document)
    assert main([str(tmp_path), "-o", str(tmp_path / "lens.html")]) == 0
    assert "matches no record" not in capsys.readouterr().err


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
    assert len(_paths(out.read_text(encoding="utf-8"), "turn")) == 2


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
        assert _data(out.read_text(encoding="utf-8"))["sets"][0]["skipped"] == ["locked/"]
        assert "skipped locked/" in capsys.readouterr().err
    finally:
        locked.chmod(0o755)


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
    assert _paths(out.read_text(encoding="utf-8"), "turn") == ["r.json"]


@needs_node
def test_the_page_s_script_parses(tmp_path: Path) -> None:
    html = page([_record()], [DESIGN.document])
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    script = tmp_path / "page.js"
    script.write_text(scripts[0], encoding="utf-8")
    subprocess.run(["node", "--check", str(script)], check=True)


# ---------- the page's script under node ----------


_AT = "const at = (p) => cur().turns.findIndex((t) => t.path === p);"


def _run(html: str, tmp_path: Path, expression: str) -> Any:
    """``expression``'s JSON value, evaluated in the page's script under node after it ran,
    with no NaN, undefined or ``[object`` in anything the page drew on the way. The
    expression may call ``at(path)``, the index of the turn at ``path`` in the set."""
    (tmp_path / "page.html").write_text(html, encoding="utf-8")
    run = subprocess.run(
        [
            "node",
            str(STUB),
            str(tmp_path / "page.html"),
            f"(() => {{ {_AT} return {{value: ({expression}), "
            "problems: [...new Set(problems)]}; })()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    shown = json.loads(run.stdout)
    assert shown["problems"] == []
    return shown["value"]


@pytest.fixture(scope="module")
def turns() -> dict[str, bytes]:
    """Every kind of turn the page draws: the showcase's decided and undecided turns, idia's
    turn with no bound declared, format 10's, and three ledger turns no design pairs with."""
    return {
        "decided.json": showcase.turn(DESIGN, *showcase.DECIDED).document,
        "undecided.json": showcase.turn(DESIGN, *showcase.UNDECIDED).document,
        "idia.json": (IDIA / "turn.record.json").read_bytes(),
        "f10.json": (FORMAT_10 / "record.json").read_bytes(),
        "normal.json": _ledger("normal-8"),
        "broken.json": _ledger("broken-8"),
        "cancelled.json": _ledger("cancelled-8"),
    }


DESIGNS = {
    "showcase.json": DESIGN.document,
    "idia-design.json": (IDIA / "design.json").read_bytes(),
    "f10-design.json": (FORMAT_10 / "design.json").read_bytes(),
}


@needs_node
def test_the_showcase_design_alone_draws_its_rounds_and_the_policy(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          const c = document.getElementById("canvas"), L = LAST;
          const band = (b) => L.bands.find((B) => b.x >= B.x0 && b.x + b.w <= B.x1).round;
          return {
            nodes: [...L.nodes].map(([id, b]) => [id, b.n.round, b.n.wavefront, band(b)]),
            heads: byClass(c, "bandhead").map((e) => e.textContent),
            cols: byClass(c, "colhead").map((e) => e.textContent),
            pills: byClass(c, "node-g").filter((g) => byClass(g, "pill-t").length)
              .map((g) => [g.attrs["data-node"], byClass(g, "pill-t")[0].textContent,
                byClass(g, "policy").length]),
            badges: byClass(c, "badge").map((e) => e.textContent),
            summary: strip(document.getElementById("summary").innerHTML),
            notice: document.getElementById("notice").hidden };
        })()""",
    )
    assert shown["nodes"] == [
        ["state", 0, 0, 0],
        ["source:ask", 1, 0, 1],
        ["rule:urgency", 1, 1, 1],
        ["source:desk", 2, 0, 2],
        ["rule:act", 2, 1, 2],
    ]
    assert shown["heads"] == ["round 0", "round 1", "round 2"]
    assert shown["cols"] == ["state", "sources", "wavefront 1", "sources", "wavefront 1"]
    assert shown["pills"] == [["rule:act", "policy", 1]]  # the pill, and the policy's border
    assert "2 params · boundaries" in shown["badges"]
    assert "requirement" in shown["badges"]
    assert shown["summary"].startswith("design showcase")
    assert "policy d:act" in shown["summary"]
    assert shown["notice"] is True


@needs_node
def test_a_turn_s_composed_why_line_is_the_text_form_s(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page(turns, DESIGNS)
    composed = _run(
        html,
        tmp_path,
        """Object.fromEntries(cur().turns.map((t) => {
          const text = whyText(t), alone = whyText(Object.assign({}, t, {text: null}));
          return [t.path, [text, alone]]; }))""",
    )
    texts = {d["path"]: d["text"] for d in _docs(html) if d["outline"]["kind"] == "turn"}
    assert set(composed) == set(turns)
    for path, (from_text, alone) in composed.items():
        why = texts[path].splitlines()[1]
        assert why == "why  " + from_text
        assert alone == from_text, path
    assert "(tied with 2 others)" in composed["undecided.json"][1]
    assert "no bound declared" in composed["idia.json"][1]
    assert "(no design paired)" in composed["broken.json"][1]


@needs_node
def test_an_entry_is_marked_by_its_nearest_bound(tmp_path: Path, turns: dict[str, bytes]) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const refs = ["q:urgent", "q:tone", "t:state", "d:urgency", "d:act"];
          const c = document.getElementById("canvas"), out = {};
          for (const p of ["decided.json", "undecided.json", "idia.json", "normal.json"]) {
            const t = cur().turns[at(p)];
            selectTurn(at(p));
            out[p] = { marks: Object.fromEntries(refs.filter((r) => t.idx.byRef.has(r))
                .map((r) => [r, dispState(t.idx.byRef.get(r))])),
              drawn: Object.fromEntries(["bar-short", "bar-value", "nb-ring"]
                .map((k) => [k, byClass(c, k).length])) };
          }
          // The neutral mark is exactly a question's value whose paired design declares no bound.
          out.neutral = cur().turns.flatMap((t) => t.idx.all.filter((e) =>
            (dispState(e) === "nobounds") !== (e.holds === "value" && /^q:/.test(e.ref)
              && Array.isArray(e.bounds) && !e.bounds.length)).map((e) => t.path + " " + e.ref));
          out.unpaired = cur().turns.filter((t) => !t.design).flatMap((t) => t.idx.all
            .filter((e) => e.holds === "value" && dispState(e) !== "det").map((e) => e.ref));
          return out; })()""",
    )
    undecided = shown["undecided.json"]
    assert undecided["marks"] == {
        "q:urgent": "short",
        "q:tone": "short",
        "t:state": "det",
        "d:urgency": "Unsure",
        "d:act": "Unsure",
    }
    assert undecided["drawn"] == {"bar-short": 2, "bar-value": 0, "nb-ring": 0}
    decided = shown["decided.json"]
    assert decided["marks"] == {
        "q:urgent": "value",
        "q:tone": "value",
        "t:state": "det",
        "d:urgency": "det",
        "d:act": "det",
    }
    assert decided["drawn"] == {"bar-short": 0, "bar-value": 2, "nb-ring": 0}
    assert shown["idia.json"]["marks"]["t:state"] == "det"  # bounds [] but not a question
    assert shown["idia.json"]["drawn"]["nb-ring"] > 0
    assert shown["idia.json"]["drawn"]["bar-short"] == 0
    assert shown["normal.json"]["drawn"] == {"bar-short": 0, "bar-value": 0, "nb-ring": 0}
    assert shown["neutral"] == []
    assert shown["unpaired"] == []


@needs_node
def test_declared_hops_are_dashed_never_the_root_and_can_be_hidden(tmp_path: Path) -> None:
    html = page({"plain.json": showcase.plain_turn().document}, [load(showcase.PLAIN).document])
    shown = _run(
        html,
        tmp_path,
        """(() => {
          const out = {};
          const T = cur().turns[0];
          out.chain = mainChain(T.outline.lineage).map((h) => h.ref);
          out.precise = [...preciseRefs(T.outline.lineage)];
          selectTurn(0);
          out.drawn = LAST.wires.filter((w) => w.hop).map((w) => [w.hop.ref, w.hop.via, w.main]);
          S.declaredHops = false; render();
          out.hidden = LAST.wires.filter((w) => w.hop).length;
          S.declaredHops = true; render();
          out.shownAgain = LAST.wires.filter((w) => w.hop).length;
          out.branches = strip(branchesHtml(T.outline));
          return out; })()""",
    )
    # Neither rule names what it read: the root is the policy's own entry, and every hop drawn
    # is a declared one, not on the lineage's precise refs.
    assert shown["chain"] == ["d:verdict"]
    assert shown["precise"] == ["d:verdict"]
    assert [d[:2] for d in shown["drawn"]] == [
        ["d:flag", "declared"],
        ["q:urgent", "declared"],
        ["t:state", "declared"],
    ]
    assert not any(d[2] for d in shown["drawn"])
    assert shown["hidden"] == 0
    assert shown["shownAgain"] == 3
    assert "3 declared reads, not a record of what was read" in shown["branches"]


@needs_node
def test_a_declared_hop_has_its_own_class_and_wording(tmp_path: Path) -> None:
    html = page({"plain.json": showcase.plain_turn().document}, [load(showcase.PLAIN).document])
    (tmp_path / "page.html").write_text(html, encoding="utf-8")
    assert ".edge.lineage.declared, .edge.branch.declared { stroke-dasharray: 5 4; }" in html
    assert "'edge ' + cls + (w.hop.via === 'declared' ? ' declared' : '')" in html
    # the legend's toggle says how many declared hops there are and whether they are drawn
    assert "`declared reads (${n}): ${S.declaredHops ? 'shown' : 'hidden'}`" in html
    assert "Hidden is not unrelated." in html


@needs_node
def test_the_lineage_s_first_branch_is_the_outline_s(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page(turns, DESIGNS)
    shown = _run(
        html,
        tmp_path,
        """(() => {
          const out = {};
          for (const t of cur().turns) out[t.path] = mainChain(t.outline.lineage).map((h) => h.ref);
          selectTurn(at("undecided.json"));
          out.drawn = LAST.wires.filter((w) => w.hop).map((w) => [w.hop.ref, w.main]);
          out.branches = strip(document.getElementById("summary").innerHTML);
          return out; })()""",
    )
    for doc in _docs(html):
        outline = doc["outline"]
        if outline["kind"] != "turn":
            continue
        # The first branch: from the policy's entry, each next entry found from the one before.
        chain = outline["lineage"][:1]
        for item in outline["lineage"][1:]:
            if item["from"] != chain[-1]["ref"] or item["via"] == "declared":
                break
            chain.append(item)
        assert shown[doc["path"]] == [item["ref"] for item in chain]
    assert shown["undecided.json"] == ["d:act", "d:urgency", "q:urgent"]
    assert shown["drawn"] == [
        ["d:urgency", True],
        ["q:urgent", True],
        ["q:tone", False],
        ["t:ticket", False],
    ]
    assert "+ q:tone, t:ticket (cause, from d:act)" in shown["branches"]


@needs_node
def test_the_turns_table_sorts_filters_steps_and_tallies(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const s = cur(), path = (i) => s.turns[i].path, out = {};
          out.undecided = s.turns.filter((t) => t.sum.dkind !== "decided").map((t) => t.path);
          out.order = s.order.map(path);
          S.sort = "record"; S.dir = 1; out.byRecord = visibleOrder().map(path);
          S.dir = -1; out.byRecordDown = visibleOrder().map(path);
          S.sort = "nearest"; S.dir = 1;
          out.byNearest = visibleOrder().map((i) => s.turns[i].sum.nearAbs);
          S.sort = null; S.undecidedOnly = true; render();
          const vo = visibleOrder(); out.filtered = vo.map(path);
          out.bar = strip(document.getElementById("turns").innerHTML).slice(0, 60);
          selectTurn(vo[vo.length - 1]); step(1); out.wrapped = path(S.turn);
          step(-1); out.back = path(S.turn);
          S.turn = null; step(-1); out.fromNone = path(S.turn);
          S.undecidedOnly = false; render();
          const tb = document.getElementById("turns").innerHTML;
          out.foot = strip(tb.slice(tb.indexOf("<tfoot>")));
          return out; })()""",
    )
    assert shown["byRecord"] == sorted(turns)
    assert shown["byRecordDown"] == sorted(turns, reverse=True)
    measured = [x for x in shown["byNearest"] if x is not None]
    assert measured == sorted(measured)
    assert measured == [0.05, 0.25]
    assert shown["byNearest"][len(measured) :] == [None] * (len(turns) - len(measured))
    assert sorted(shown["undecided"]) == [
        "broken.json",
        "cancelled.json",
        "idia.json",
        "undecided.json",
    ]
    filtered = shown["filtered"]
    assert filtered == [p for p in shown["order"] if p in shown["undecided"]]
    assert shown["bar"].startswith(f"undecided only{len(filtered)} of {len(turns)} turns")
    assert shown["wrapped"] == filtered[0]
    assert shown["back"] == filtered[-1]
    assert shown["fromNone"] == filtered[-1]
    assert "short bounds by question: q:tone 2 · q:urgent 2" in shown["foot"]
    assert (
        "root of the undecided turns: q:triage/hazard 1 · q:urgent 1 · "
        "root not a question: t:state 1"
    ) in shown["foot"]


@needs_node
def test_the_nearest_chip_is_the_summary_s_nearest_column(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          render();
          const near = (p) => p && `${p.distance} ${p.side} ${p.param}@${p.e.ref}`;
          const picks = Object.fromEntries(cur().turns.map((t) =>
            [t.path, [t.outline.summary, near(t.sum.pick)]]));
          return { picks, table: strip(document.getElementById("turns").innerHTML) }; })()""",
    )
    for path, (summary, pick) in shown["picks"].items():
        nearest = summary.split("  ")[6]
        assert nearest.startswith("nearest "), path
        assert (pick or "-") == nearest.removeprefix("nearest "), path
    assert shown["picks"]["undecided.json"][1] == "-0.05 short urgent/bar@q:urgent"
    assert "(tied with 2 others)" in shown["table"]


@needs_node
def test_the_bounds_panel_leads_with_the_tie_line(tmp_path: Path, turns: dict[str, bytes]) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const strips = () => document.getElementById("strips").innerHTML, out = {};
          selectTurn(at("undecided.json")); out.undecided = strips();
          const rows = stripRows(curTurn()).rows;
          out.root = rows.filter((r) => r.isRoot).map((r) => [r.e.ref, r.param]);
          out.refs = rows.map((r) => r.e.ref);
          selectTurn(at("decided.json")); out.decided = strips();
          selectTurn(at("idia.json")); out.idia = strip(strips());
          return out; })()""",
    )
    assert '<div class="ties">3 bounds tie at \u22120.05, short</div>' in shown["undecided"]
    assert shown["root"] == [["q:urgent", "urgent/bar"]]
    refs = shown["refs"]
    assert [r for i, r in enumerate(refs) if i == 0 or refs[i - 1] != r] == sorted(
        set(refs), key=refs.index
    )
    assert 'class="ties"' not in shown["decided"]
    # With no bounds, it lists each question entry's reason.
    assert "No bounds derived for this turn." in shown["idia"]
    assert "q:triage/hazard: no boundaries field on rule hazard" in shown["idia"]


@needs_node
def test_a_record_with_no_paired_design_is_drawn_by_round_with_a_notice(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const c = document.getElementById("canvas"), out = {};
          for (const p of ["normal.json", "broken.json"]) {
            const t = cur().turns[at(p)];
            selectTurn(at(p));
            out[p] = { title: document.getElementById("skelTitle").textContent,
              notice: document.getElementById("notice").textContent,
              heads: byClass(c, "bandhead").map((e) => e.textContent),
              rounds: t.outline.rounds.filter((r) => r.entries.length)
                .map((r) => "round " + r.round),
              nodes: byClass(c, "node-g").map((g) => g.attrs["data-node"]),
              entries: t.outline.rounds.flatMap((r) => r.entries.slice()
                .sort((a, b) => a.index - b.index).map((e) => e.ref)),
              hops: LAST.wires.map((w) => [w.hop.ref, w.main]),
              strips: strip(document.getElementById("strips").innerHTML) };
          }
          return out; })()""",
    )
    normal = shown["normal.json"]
    assert normal["title"] == "Entries by round"
    assert normal["notice"].startswith("No design in the set has this record's design hash or name")
    assert normal["heads"] == normal["rounds"] == ["round 0", "round 1", "round 2"]
    assert normal["nodes"] == normal["entries"]
    assert normal["strips"] == "No design is paired, so no bounds."
    assert shown["broken.json"]["hops"] == [["t:state", True]]


@needs_node
def test_a_record_whose_design_hash_differs_is_drawn_over_the_design_of_its_name(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    data = _data(html)
    turn = data["sets"][0]["docs"][1]["outline"]
    turn["design_hash"] = "sha256:" + "0" * 64
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { selectTurn(0); return { by: curTurn().pairBy,
          notice: document.getElementById("notice").textContent,
          table: strip(document.getElementById("turns").innerHTML) }; })()""",
    )
    assert shown["by"] == "name"
    assert shown["notice"].startswith("Hash differs")
    assert "hash differs" in shown["table"]


@needs_node
def test_an_outline_of_another_version_is_counted_and_not_drawn(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page({"decided.json": turns["decided.json"]}, [DESIGN.document])
    data = _data(html)
    data["sets"][0]["docs"][1]["outline"]["outline_version"] = 3
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """({ other: cur().other, docs: cur().docs.map((d) => d.path),
          subtitle: document.getElementById("subtitle").textContent })""",
    )
    assert shown["other"] == 1
    assert shown["docs"] == ["design 1"]
    assert shown["subtitle"].endswith("skipped 1 outline of another version")


@needs_node
def test_the_page_opens_outline_json_in_every_form_it_takes(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const docs = cur().docs, line = (d) => JSON.stringify({path: d.path, outline: d.outline});
          const got = (xs) => xs.map((d) => [d.path, d.text === null, d.outline.kind]);
          const lines = parseOutlines(docs.map(line).join("\\n") + "\\n", "x.jsonl");
          const turn = docs.find((d) => d.outline.kind === "turn");
          const other = Object.assign({}, turn.outline, {outline_version: 3});
          return {
            lines: got(lines),
            paired: prepareSet({name: "dropped", docs: lines}).turns.filter((t) => t.design)
              .map((t) => t.path),
            one: got(parseOutlines(
              JSON.stringify({path: "p", outline: turn.outline, text: "t"}), "n")),
            bare: got(parseOutlines(JSON.stringify(turn.outline), "bare.json")),
            sets: parseOutlines(JSON.stringify(DATA), "lens.json").length,
            other: parseOutlines(JSON.stringify(other), "old.json").length,
            want: docs.map((d) => [d.path, true, d.outline.kind]),
            wantPaired: cur().turns.filter((t) => t.design).map((t) => t.path) }; })()""",
    )
    assert shown["lines"] == shown["want"]
    assert shown["paired"] == shown["wantPaired"]
    assert shown["one"] == [["p", False, "turn"]]
    assert shown["bare"] == [["bare.json", True, "turn"]]
    assert shown["sets"] == len(shown["want"])
    assert shown["other"] == 0


@needs_node
def test_no_view_draws_nan_or_undefined(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """Every design alone and with each node selected, every turn and each of its nodes or
    entries, the uses of ``t:state``, every sort and the filter, each tooltip rendered."""
    designs = {**DESIGNS, "full.json": load(full()).document}
    renders = _run(
        page(turns, designs),
        tmp_path,
        """(() => {
          let n = 0; const c = document.getElementById("canvas");
          for (const stateEdges of [false, true]) {
            S.stateEdges = stateEdges;
            cur().designs.forEach((d, di) => {
              S.design = di; S.turn = null; S.node = null; render(); tips(c); n++;
              for (const x of d.outline.nodes) {
                S.node = null; selectNode(x.id); tips(c); n++; } });
            cur().turns.forEach((t, ti) => {
              selectTurn(ti); tips(c); n++;
              const ids = t.design
                ? t.design.outline.nodes.map((x) => x.id) : t.idx.all.map((e) => e.ref);
              for (const id of ids) { S.node = null; selectNode(id); tips(c); n++; } });
          }
          for (const k of Object.keys(SORTS)) for (const dir of [1, -1]) {
            S.sort = k; S.dir = dir; render(); n++; step(1); step(-1); }
          S.sort = null; S.undecidedOnly = true; render(); step(1); n++;
          return n; })()""",
    )
    assert renders > 100


# ---------- the design views ----------


@needs_node
def test_every_design_view_draws_without_nan_or_undefined(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Turn, Path, Overview, Targets in both modes, Outcomes for every table, Explore from every
    node at every depth, in both directions with the details open, and Gravity in 2D and 3D with
    every body selected and described, each round of the turn, and every control, for each
    design alone and under each turn; a design with no policy node, targets, tables or
    requirements; and a format-9 policy, which provides no ref."""
    f9 = (FORMAT_9 / "design.json").read_bytes()
    html = page(turns, {**DESIGNS, "full.json": load(full()).document, "f9.json": f9})
    data = _data(html)
    bare = json.loads(json.dumps(data["sets"][0]["docs"][0]["outline"]))
    bare.update(policy_node=None, targets=[], requirements=[], path="bare")
    for n in bare["nodes"]:
        n.update(form="function" if n["kind"] == "rule" else n["form"], rows=None)
        n["steps_to_policy"] = None
    data["sets"][0]["docs"].append({"path": "bare.json", "outline": bare, "text": None})
    renders = _run(
        _with_data(html, data),
        tmp_path,
        """(() => {
          let n = 0;
          const all = () => {
            for (const v of ["turn", "path", "overview", "targets", "outcomes"]) {
              S.view = v; drawView(); n++; }
            S.tmode = "matrix"; S.view = "targets"; drawView(); S.tmode = null;
            const X = ixOf(curDesign()); if (!X) return;
            for (const t of X.tables) { S.table = t.id; S.view = "outcomes"; drawView(); n++; }
            S.view = "explore";
            for (const id of Object.keys(X.N)) { S.node = id;
              for (const d of [0, 1, 99]) for (const dir of ["v", "h"]) {
                Object.assign(EX, {up: d, down: d, dir, info: true}); drawView(); n++; } }
            EX.info = false; EX.mini = false; drawView(); EX.mini = true;
            S.view = "gravity";
            for (const three of [false, true]) {
              GV.mode3 = three; drawView(); n++;
              const L = gvL;
              for (const b of L.nodes) {
                gvSelect(L, b); gvTipAt(L, b, {clientX: 5, clientY: 5}); n++; }
              gvSelect(L, null); frames(80);
              for (let r = 0; r <= (L.ov ? L.ov.maxR : 0); r++) { gvSet("step", r); n++; }
              for (const [k, v] of [["hide", true], ["lin", true], ["light", false],
                ["light", true], ["cross", true], ["cross", false], ["hh", false], ["hh", true],
                ["byDomain", true], ["byDomain", false], ["flat", true], ["flat", false],
                ["halo", true], ["halo", false], ["find", "u"], ["find", ""], ["qPull", 0.9],
                ["hPull", 0.1], ["hide", false], ["lin", false]]) { gvSet(k, v); n++; }
              for (const a of ["fit", "fitsel", "home", "front", "top", "side", "relayout"]) {
                gvAct(a); }
              frames(80); gvLegend(L); }
            GV.mode3 = false; GV.qPull = 0.5; GV.hPull = 0.5; };
          cur().designs.forEach((d, di) => { S.design = di; S.turn = null; S.node = null; all(); });
          cur().turns.forEach((t, ti) => { S.turn = ti; S.node = null; all(); });
          return {n, designs: cur().designs.map((d) => d.path)}; })()""",
    )
    assert "bare.json" in renders["designs"]
    assert "f9.json" in renders["designs"]
    assert renders["n"] > 300


@needs_node
def test_explore_s_slice_keeps_the_layout_s_rows(tmp_path: Path) -> None:
    outline = design_outline(DESIGN.document)
    rows = {k: (v["col"], v["row"]) for k, v in outline["layout"]["nodes"].items()}
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { S.view = "explore"; S.node = "rule:act"; Object.assign(EX, {up: 1, down: 0});
          drawView(); return LAST_EX; })()""",
    )
    assert shown["focus"] == "rule:act"
    preds = {e["from"] for e in outline["edges"] if e["dependency"] and e["to"] == "rule:act"}
    assert set(shown["shown"]) == {"rule:act", *preds}
    for rank in shown["ranks"]:
        assert len({rows[i][0] for i in rank}) == 1
        assert [rows[i][1] for i in rank] == sorted(rows[i][1] for i in rank)


@needs_node
def test_the_design_views_draw_the_outline_s_members(tmp_path: Path) -> None:
    """Altered in the data, the steps to the policy, a requirement's lists, a target's links
    and path and a row's tests are what the views draw: the page computes none of them."""
    html = page([], [DESIGN.document])
    data = _data(html)
    o = data["sets"][0]["docs"][0]["outline"]
    by_id = {n["id"]: n for n in o["nodes"]}
    by_id["source:ask"]["steps_to_policy"] = 7
    o["requirements"][0]["behind"] = ["rule:urgency", "source:ask"]
    o["targets"][0].update(
        questions=[{"ref": "q:planted", "node": "source:ask", "role": "formative"}],
        estimated_by=[],
        readers=["rule:act"],
        path=["source:ask", "rule:urgency", "rule:act"],
    )
    table = next(n for n in o["nodes"] if n["form"] == "table")
    table["rows"][0]["tests"] = [
        {"input": 0, "read": "d:planted", "node": "rule:urgency", "cell": "seven"}
    ]
    shown = _run(
        _with_data(html, data),
        tmp_path,
        f"""(() => {{
          const html = (v, f) => {{ S.view = v; if (f) f(); drawView();
            const box = {{overview: "vOverview", targets: "vTargets", outcomes: "vOutcomes"}}[v];
            const el = (id) => document.getElementById(id).innerHTML;
            return strip(el(box || "vExplore")
              + (v === "explore" ? el("vcanvas") + el("sb") : "")); }};
          return {{
            card: html("explore", () => {{ S.node = "source:ask"; }}),
            sb: (S.node = "{o["requirements"][0]["source"]}", html("explore")),
            targets: html("targets"),
            outcomes: html("outcomes", () => {{ S.table = "{table["id"]}"; }}) }}; }})()""",
    )
    assert "7 steps to the policy" in shown["card"]
    assert "2 behind" in shown["sb"]
    assert "formative 1" in shown["targets"]
    assert "2 steps to the policy" in shown["targets"]
    assert "ask→urgency→act" in shown["targets"]
    assert "d:planted is seven" in shown["outcomes"]


@needs_node
def test_a_view_s_hash_opens_it_again(tmp_path: Path, turns: dict[str, bytes]) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const pushed = [];
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { pushed.push(h); location.hash = h; }};
          S.turn = 0; S.view = "explore"; Object.assign(EX, {up: 3, down: 0, dir: "h", info: true});
          S.table = "rule:act"; S.tmode = "matrix"; go("source:desk");
          const hash = location.hash, want = {view: S.view, node: S.node, turn: S.turn,
            up: EX.up, down: EX.down, dir: EX.dir, info: EX.info, table: S.table, tmode: S.tmode};
          Object.assign(EX, {up: 1, down: 1, dir: "v", info: false});
          S.view = null; S.node = null; S.table = null; S.tmode = null; S.turn = null;
          readHash();
          const got = {view: S.view, node: S.node, turn: S.turn, up: EX.up, down: EX.down,
            dir: EX.dir, info: EX.info, table: S.table, tmode: S.tmode};
          showView("gravity"); gvSet("mode3", true); gvSelect(gvL, gvL.byId.get("d:urgency"));
          const ghash = location.hash;
          GV.mode3 = false; GV.sel = null; S.view = null; readHash(); render();
          const gravity = {hash: ghash, view: S.view, mode3: GV.mode3, sel: GV.sel,
            drawn: LAST_GRAVITY.mode3, picked: LAST_GRAVITY.sel};
          S.view = "gravity"; S.turn = 0; GV.mode3 = false; GV.sel = null; GV.ov = true;
          render(); writeHash();
          const onHash = location.hash, onDrawn = LAST_GRAVITY.ov, shown = gvL.ovT === curTurn();
          gvSet("ov", false);
          const offHash = location.hash, offDrawn = LAST_GRAVITY.ov, kept = S.turn;
          GV.ov = true; S.view = null; S.turn = null; readHash(); render();
          const reread = {gov: GV.ov, turn: S.turn, drawn: LAST_GRAVITY.ov};
          gvSet("ov", true);
          const again = {hash: location.hash, drawn: LAST_GRAVITY.ov, turn: S.turn};
          const dh0 = location.hash; GV.drag = "move"; writeHash(); const dh1 = location.hash;
          GV.drag = "orbit"; readHash();
          const drag = {plain: dh0, moved: dh1, read: GV.drag}; GV.drag = "orbit"; writeHash();
          const overlay = {onHash, onDrawn, shown, offHash, offDrawn, kept, reread, again, drag};
          return {hash, pushed: pushed.length, want, got, gravity, overlay,
            plain: (location.hash = "", readHash(), S.view) };
        })()""",
    )
    assert shown["got"] == shown["want"]
    assert shown["pushed"] == 1
    assert "view=explore" in shown["hash"]
    assert shown["plain"] == "turn"  # a set with records opens on the Turn tab
    gravity = shown["gravity"]
    assert "view=gravity" in gravity["hash"]
    assert "g3=1" in gravity["hash"]
    assert gravity["view"] == "gravity"
    assert gravity["mode3"] is True
    assert gravity["drawn"] is True  # the view drawn again is the 3D layout
    assert gravity["sel"] == gravity["picked"] == "d:urgency"
    overlay = shown["overlay"]
    assert "gov" not in overlay["onHash"]  # on is the default and is not written
    assert overlay["onDrawn"] is True
    assert overlay["shown"] is True
    assert "gov=0" in overlay["offHash"]
    assert overlay["offDrawn"] is False
    assert overlay["kept"] == 0  # the selected turn stays
    assert overlay["reread"] == {"gov": False, "turn": 0, "drawn": False}
    assert "gov" not in overlay["again"]["hash"]
    assert overlay["again"]["drawn"] is True
    assert overlay["again"]["turn"] == 0
    assert "gdrag" not in overlay["drag"]["plain"]  # orbit is the default and is not written
    assert "gdrag=move" in overlay["drag"]["moved"]
    assert overlay["drag"]["read"] == "move"


@needs_node
def test_the_turn_view_says_what_took_place(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """A page given records opens on the Turn tab at its first turn; the tab names the
    decision, the row that gave it with this turn's values, what was asked and what was not and
    why; Outcomes marks the row that gave the value."""
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => {
          const view = S.view, first = S.turn, out = {view, first};
          const at = (name) => cur().turns.findIndex((t) => t.path.endsWith(name));
          for (const name of ["decided.json", "undecided.json"]) {
            S.turn = at(name); S.view = "turn"; drawView();
            out[name] = document.getElementById("vTurn").innerHTML;
            S.view = "outcomes"; S.table = "rule:act"; drawView();
            out[name + " outcomes"] = document.getElementById("vOutcomes").innerHTML;
          }
          return out; })()""",
    )
    assert shown["view"] == "turn"
    assert shown["first"] is not None
    decided, undecided = shown["decided.json"], shown["undecided.json"]
    assert "Decided" in decided
    assert 'The turn was given <span class="mono empty">t:state</span>' in decided
    assert '<span class="kk">subject</span><blockquote class="said">' in decided
    assert "Row 1 gave the value." in decided
    assert '<span class="mono">tg:urgent</span>' not in decided  # act estimates no target
    assert "no target declared" in decided
    assert "matched: its output is the value" in decided
    assert "Round by round" in decided
    assert '<span class="verb">asked</span>' in decided
    assert '<span class="verb">called</span>' in decided  # desk is a tool
    assert "ran because" in decided
    assert "(it runs when it is yes)" in decided
    assert "decides whether" in decided  # d:urgency lets desk run
    assert "Undecided" in undecided
    assert "unsure: no clear answer (it abstained)" in undecided
    assert "falls short of the bar" in undecided
    assert "Its source requirement did not hold" in undecided
    assert "Row 4 gave the value." in undecided
    assert 'class="ocard chosen"' in shown["decided.json outcomes"]
    assert "took row 1's output" in shown["decided.json outcomes"]


@needs_node
def test_the_path_view_folds_each_rule_s_answers_into_its_card(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Path opens with a card for the policy and one for each rule result it reads, the policy's
    selected until another rule is; a card's answers are its declared reads a source provides,
    and the ones read are those a helper of its derivation read or a matched table row tests or
    gives; a record no design pairs with draws no path."""
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const out = {}, el = () => document.getElementById("vPath").innerHTML;
          for (const p of ["decided.json", "undecided.json", "idia.json", "f10.json"]) {
            S.turn = at(p); S.node = null; S.view = "path"; drawView();
            out[p] = {path: LAST_PATH, html: el(), text: strip(el())}; }
          S.node = "rule:urgency"; S.turn = at("decided.json"); drawView();
          out.picked = {path: LAST_PATH, text: strip(el())};
          S.node = "source:ask"; drawView(); out.source = LAST_PATH.selected;
          S.turn = at("normal.json"); drawView(); out.unpaired = [LAST_PATH, strip(el())];
          S.turn = null; drawView(); out.none = strip(el());
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; }};
          S.turn = at("decided.json"); S.node = "rule:urgency"; writeHash();
          const hash = location.hash; S.view = null; S.node = null; readHash();
          out.hash = [hash, S.view, S.node];
          return out; })()""",
    )
    decided, undecided = shown["decided.json"], shown["undecided.json"]
    assert decided["path"]["selected"] == "rule:act"  # the policy, until a rule is picked
    assert decided["path"]["heads"] == ["rule:urgency"]
    assert decided["path"]["rules"]["rule:act"] == {
        "answers": ["q:tone", "t:ticket"],
        "read": ["t:ticket"],  # row 1 matched, and tests it; no matched row tests q:tone
        "results": ["d:urgency"],
        "given": [],
    }
    assert decided["path"]["rules"]["rule:urgency"]["read"] == ["q:urgent"]  # through its helper
    assert undecided["path"]["rules"]["rule:act"]["read"] == []
    assert "The policy and the 1 rule result it reads" in decided["text"]
    assert "The record shows 2 of 3 declared answers read, over 2 calls." in decided["text"]
    assert "1 of 2 answers read · 1 rule result" in decided["text"]
    assert "Answers it declares, not recorded as read · 1" in decided["text"]
    assert "Which tone is the mail written in?" in decided["text"]
    assert "Row 1 gave the value." in decided["text"]  # a table's rows, as Turn draws them
    assert decided["html"].count('<i class="read"') == 3  # the key's, q:urgent's and t:ticket's
    assert 'data-pnode="rule:urgency" aria-pressed="false"' in decided["html"]
    assert "0 of 2 answers read" in undecided["text"]
    assert 'class="mk" title="t:ticket: not asked' in undecided["html"]
    assert shown["picked"]["path"]["selected"] == "rule:urgency"
    assert "Answers it read · 1" in shown["picked"]["text"]
    assert shown["source"] == "rule:act"  # a node that is not a rule selects nothing here
    assert shown["unpaired"][0] is None
    assert shown["unpaired"][1].startswith("Path draws a design")
    assert shown["none"].startswith("Pick a turn")
    assert shown["hash"][1:] == ["path", "rule:urgency"]
    assert "view=path" in shown["hash"][0]
    # a design whose policy reads six results: a card for each, and the rules behind one alone
    idia = shown["idia.json"]
    assert len(idia["path"]["heads"]) == 6
    assert "The policy and the 6 rule results it reads" in idia["text"]
    assert idia["html"].count('class="pcard ') == 7


@needs_node
def test_the_path_map_holds_every_node_once_in_its_lane(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The map has a lane for each rule result the policy reads, holding the nodes whose
    dependencies first reach that result alone; a node that first reaches several is shared,
    and one that reaches the policy alone is the policy's. The selection lights what it reads,
    by whether the record shows the read, what the record shows those read, and what reads
    it."""
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const out = {}, el = () => document.getElementById("vPath").innerHTML;
          const draw = (p, node) => { S.turn = at(p); S.node = node; S.view = "path"; drawView();
            return {lanes: LAST_PATH.lanes, lane: LAST_PATH.lane, lit: LAST_PATH.lit,
              behind: LAST_PATH.behind, html: el(), text: strip(el()),
              nodes: cur().turns[at(p)].design.outline.nodes.map((n) => n.id)}; };
          out.decided = draw("decided.json", null);
          out.undecided = draw("undecided.json", null);
          out.urgency = draw("decided.json", "rule:urgency");
          out.idia = draw("idia.json", null);
          out.f10 = draw("f10.json", "rule:act");
          return out; })()""",
    )
    decided = shown["decided"]
    assert decided["lanes"] == {
        "shared": ["state"],
        "rule:urgency": ["source:ask", "rule:urgency"],  # ask feeds the policy too, through q:tone
        "policy": ["source:desk", "rule:act"],
    }
    for drawn in shown.values():
        assert sorted(i for lane in drawn["lanes"].values() for i in lane) == sorted(drawn["nodes"])
        assert drawn["html"].count('class="mc ') == len(drawn["nodes"]) + 6  # and the key's six
    assert decided["lit"] == {
        "rule:act": "sel",
        "rule:urgency": "read",
        "source:desk": "read",  # row 1 matched, and tests t:ticket
        "source:ask": "decl",  # q:tone, which no matched row tests
    }
    assert shown["undecided"]["lit"]["source:desk"] == "decl"
    assert shown["urgency"]["lit"] == {
        "rule:urgency": "sel",
        "source:ask": "read",
        "source:desk": "out",
        "rule:act": "out",
    }
    assert shown["urgency"]["lane"] == "rule:urgency"
    assert shown["idia"]["lanes"]["shared"] == ["state", "source:details", "source:triage"]
    assert shown["idia"]["lanes"]["rule:seen_at"] == ["source:timing", "rule:seen_at"]
    assert shown["f10"]["lit"] == {
        "rule:act": "sel",
        "rule:route": "read",
        "source:ask": "back",  # route's helper read it, the record shows
    }
    assert "Where it sits in the whole design" in decided["text"]
    assert '<button class="mc rule sel lin" data-pnode="rule:act"' in decided["html"]
    assert '<span class="mc other read"' in decided["html"]


# ---------- gravity ----------


def _gravity_edges(outline: dict[str, Any]) -> set[tuple[str, str, str]]:
    """The edges Gravity draws, from the outline's members alone: each dependency edge from its
    ref's base to the estimate of the rule it reaches, or to each ref of the source it reaches;
    each target's questions, asks or latent by their role; and the rules that estimate it."""
    nodes = {n["id"]: n for n in outline["nodes"]}

    def estimate(n: dict[str, Any]) -> str:
        return (n["refs"] or [n["id"]])[0]

    bodies = {t["id"] for t in outline["targets"]}
    if any(n["kind"] == "state" for n in outline["nodes"]):
        bodies.add("t:state")
    for n in outline["nodes"]:
        if n["kind"] == "rule":
            bodies.add(estimate(n))
        elif n["kind"] == "source":
            bodies |= {q["ref"] for q in n["questions"]} | {r["ref"] for r in n["tool_results"]}
    found: set[tuple[str, str, str]] = set()
    for e in outline["edges"]:
        base = re.match(r"^([qtd]):([^.\[\]]+)", e["ref"])
        if base is None:
            continue
        to = nodes[e["to"]]
        for r in [estimate(to)] if to["kind"] == "rule" else to["refs"]:
            found.add((base[0], r, e["role"]))
    for t in outline["targets"]:
        for q in t["questions"]:
            found.add((q["ref"], t["id"], "asks" if q["role"] == "asks" else "latent"))
        for i in t["estimated_by"]:
            found.add((estimate(nodes[i]), t["id"], "estimates"))
    return {e for e in found if e[0] != e[1] and e[0] in bodies and e[1] in bodies}


@needs_node
def test_the_gravity_view_draws_the_outline_s_edges(tmp_path: Path) -> None:
    """The bodies' edges are the outline's: a planted edge, target question and estimating rule
    are drawn, and a rule's declared reads, which the outline's edges already hold, add none."""
    designs = {**DESIGNS, "full.json": load(full()).document}
    designs["f9.json"] = (FORMAT_9 / "design.json").read_bytes()
    html = page([], designs)
    data = _data(html)
    o = data["sets"][0]["docs"][0]["outline"]
    o["edges"].append(
        {
            "from": "source:ask",
            "to": "rule:urgency",
            "role": "need",
            "ref": "q:tone.x",
            "dependency": True,
        }
    )
    o["edges"].append(
        {
            "from": "rule:urgency",
            "to": "source:ask",
            "role": "slot",
            "ref": "d:urgency",
            "dependency": True,
        }
    )
    o["targets"][0]["questions"].append(
        {"ref": "q:urgent", "node": "source:ask", "role": "formative"}
    )
    o["targets"][0]["estimated_by"].append("rule:act")
    act = next(n for n in o["nodes"] if n["id"] == "rule:act")
    act["reads"].append({"ref": "q:urgent", "via": "declared"})
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = null; const out = {};
          cur().designs.forEach((d, i) => { S.design = i; drawView();
            out[d.path] = {edges: LAST_GRAVITY.edges, drawn: LAST_GRAVITY.drawn,
              bodies: Object.keys(LAST_GRAVITY.pos)}; });
          return out; })()""",
    )
    outlines = {d["path"]: d["outline"] for d in data["sets"][0]["docs"]}
    for path, drawn in shown.items():
        want = _gravity_edges(outlines[path])
        assert {tuple(e) for e in drawn["edges"]} == want, path
        assert len(drawn["edges"]) == len(want), path  # each once
        assert {tuple(e.split("|")) for e in drawn["drawn"]} == want, path
    planted = {tuple(e) for e in shown["showcase.json"]["edges"]}
    assert ("q:tone", "d:urgency", "need") in planted
    assert ("d:urgency", "q:tone", "slot") in planted  # to a source: to each ref it provides
    assert ("d:urgency", "q:urgent", "slot") in planted
    assert ("q:urgent", "tg:urgent", "latent") in planted
    assert ("d:act", "tg:urgent", "estimates") in planted
    assert ("q:urgent", "d:act", "read") not in planted  # a declared read is not an edge here
    # the state is a body: each use of it (an edge that is not a dependency) is drawn from it
    uses = {(e[0], e[1]) for e in planted if e[0] == "t:state"}
    assert {("t:state", "q:tone"), ("t:state", "q:urgent")} <= uses
    assert "t:state" in shown["showcase.json"]["bodies"]
    # a format-9 policy provides no ref: its body is its node
    assert ("d:route", "rule:policy", "read") in {tuple(e) for e in shown["f9.json"]["edges"]}
    assert "rule:policy" in shown["f9.json"]["bodies"]


_GRAVITY_POS = """(() => { S.view = "gravity"; S.turn = null; const out = {};
  cur().designs.forEach((d, i) => { S.design = i; GV.mode3 = false; drawView();
    out[d.path] = {flat: LAST_GRAVITY.pos}; });
  cur().designs.forEach((d, i) => { S.design = i; GV.mode3 = true; drawView();
    out[d.path].deep = LAST_GRAVITY.pos; });
  S.design = 0; GV.mode3 = false; drawView(); const L = gvL;
  L.nodes[0].x += 40; gvAct("relayout");
  out.relaid = LAST_GRAVITY.pos;
  return out; })()"""


@needs_node
def test_the_gravity_layout_is_the_same_for_the_same_design(tmp_path: Path) -> None:
    """The layout is a function of the graph: two runs of the page place every body at the
    same point in 2D and in 3D, and a relayout after a body is moved returns it there."""
    designs = {**DESIGNS, "full.json": load(full()).document}
    html = page([], designs)
    first = _run(html, tmp_path, _GRAVITY_POS)
    second = _run(html, tmp_path, _GRAVITY_POS)
    assert first == second
    for path in designs:
        flat, deep = first[path]["flat"], first[path]["deep"]
        assert flat
        assert set(flat) == set(deep)
        assert all(p[2] == 0 for p in flat.values())
        assert any(p[2] != 0 for p in deep.values())
        assert len({(p[0], p[1]) for p in flat.values()}) == len(flat)  # no two bodies coincide
    assert first["relaid"] == first["showcase.json"]["flat"]


@needs_node
def test_the_gravity_overlay_marks_by_dispState(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """Under a turn each body takes the page's own mark, so a short bound rings amber; the
    lineage is lit, its first branch as the main hops, and declared hops are dashed and never
    the root; a round before an entry's leaves it in a later round; a target is not marked."""
    html = page(
        {"undecided.json": turns["undecided.json"], "plain.json": showcase.plain_turn().document},
        [DESIGN.document, load(showcase.PLAIN).document],
    )
    shown = _run(
        html,
        tmp_path,
        """(() => { const out = {};
          for (const p of ["undecided.json", "plain.json"]) {
            const t = cur().turns[at(p)]; S.turn = at(p); S.view = "gravity"; drawView();
            const G = LAST_GRAVITY, el = (id) => document.getElementById(id);
            const want = Object.fromEntries(Object.keys(G.pos).filter((r) => !r.startsWith("tg:"))
              .map((r) => [r, stateFor(t.idx, r) || "none"]));
            out[p] = {marks: G.marks, want, rings: G.rings, lit: G.lit, hops: G.hops, root: G.root,
              rootOf: rootOf(t.outline).ref, rounds: G.rounds,
              panel: ["gvov", "gvkey", "gvroundl"]
                .map((id) => strip(el(id).innerHTML || el(id).textContent)).join(" ")};
            gvSet("step", 0); out[p].step0 = LAST_GRAVITY.marks; out[p].lit0 = LAST_GRAVITY.lit;
            gvSet("step", null);
          }
          return out; })()""",
    )
    und = shown["undecided.json"]
    assert und["marks"] == und["want"]
    assert und["marks"]["q:urgent"] == "short"
    assert und["rings"]["q:urgent"] == "--ring-warning"  # amber, as every view marks a short bound
    assert und["rings"]["d:urgency"] == "--ring-warning"  # Unsure
    assert und["rings"]["t:ticket"] == "--neutral"  # Unasked, the page's word
    assert "tg:urgent" not in und["marks"]  # no target state in the outline: not shaded
    assert und["root"] == und["rootOf"] == "q:urgent"
    lit = [h[:4] for h in und["lit"]]
    assert sorted(lit) == sorted(
        [
            ["d:urgency", "d:act", "row", True],
            ["q:tone", "d:act", "cause", False],
            ["q:urgent", "d:urgency", "rested_on", True],
            ["t:ticket", "d:act", "cause", False],
        ]
    )
    mains = [h[3] for h in lit]
    assert mains == sorted(mains)  # the main chain is drawn after the other lit hops
    assert und["step0"]["q:urgent"] == "later"
    assert und["lit0"] == []
    assert "Unasked" in und["panel"]
    assert "round 2 of 2" in und["panel"]
    plain = shown["plain.json"]
    assert plain["marks"] == plain["want"]
    assert plain["root"] == plain["rootOf"] == "d:verdict"  # the policy's own entry
    assert {(h["s"], h["t"]) for h in plain["hops"] if h["declared"]} == {
        ("d:flag", "d:verdict"),
        ("q:urgent", "d:flag"),
        ("t:state", "d:verdict"),
    }
    assert not any(h["main"] for h in plain["hops"])
    assert plain["lit"]
    assert all(h[4] and not h[3] for h in plain["lit"])  # dashed, never main
    assert "3 declared reads, dashed" in plain["panel"]


@needs_node
def test_the_gravity_gestures_follow_the_pointer(tmp_path: Path) -> None:
    """Wheel zoom in 2D keeps the point under the cursor; an empty-space drag pans by the
    pointer's travel and is not a click; a body drag pins it under the pointer, reheats the
    layout and lets go; in 3D a drag turns by its travel, the wheel keeps the orbit point's
    spot, a shift-drag moves the orbit point; two fingers pinch the zoom."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView();
          const L = gvL, cv = document.getElementById("gvc"), out = {};
          const empty = () => { gvProject(L);
            for (let y = L.H - 5; y > 0; y -= 7) for (let x = 5; x < L.W; x += 7)
              if (!gvPick(L, {clientX: x, clientY: y})) return [x, y]; };
          const under = (x, y) => [(x - L.tf.x) / L.tf.k, (y - L.tf.y) / L.tf.k];
          const b = under(300, 400); fire(cv, "wheel", {clientX: 300, clientY: 400, deltaY: -300});
          out.wheel = {k: L.tf.k, before: b, after: under(300, 400)};
          const e = empty(), t0 = Object.assign({}, L.tf);
          fire(cv, "pointerdown", {clientX: e[0], clientY: e[1]});
          fire(cv, "pointermove", {clientX: e[0] + 40, clientY: e[1] - 25});
          fire(cv, "pointermove", {clientX: e[0] + 50, clientY: e[1] - 30});
          fire(cv, "pointerup", {});
          out.pan = [L.tf.x - t0.x, L.tf.y - t0.y, L.tf.k === t0.k, L.moved];
          fire(cv, "click", {clientX: e[0] + 50, clientY: e[1] - 30}); out.clickAfterPan = GV.sel;
          gvProject(L);
          const n = L.byId.get("d:act");
          const sx = L.tf.x + L.tf.k * (L.W / 2 + n.px), sy = L.tf.y + L.tf.k * (L.H / 2 + n.py);
          fire(cv, "pointerdown", {clientX: sx, clientY: sy});
          out.dragging = L.drag && L.drag.n.id; out.live = L.live;
          fire(cv, "pointermove", {clientX: sx + 120, clientY: sy + 60});
          out.pinned = [n.fx - n.x - 120 / L.tf.k, n.fy - n.y - 60 / L.tf.k];
          fire(cv, "pointerup", {});
          out.released = L.drag === null && L.nodes.every((m) => m.fx == null);
          frames(400); out.settled = L.live;
          gvSet("mode3", true); frames(200);
          const e3 = empty(), y0 = L.cam.yaw;
          fire(cv, "pointerdown", {clientX: e3[0], clientY: e3[1]});
          for (const d of [10, 20, 30]) {
            fire(cv, "pointermove", {clientX: e3[0] + d, clientY: e3[1]}); }
          out.turn = L.cam.yaw - y0; fire(cv, "pointerup", {}); L.vyaw = L.vpitch = 0;
          const spot = () => [L.tf.x + L.tf.k * L.W / 2, L.tf.y + L.tf.k * L.H / 2];
          const s0 = spot(); fire(cv, "wheel", {clientX: 100, clientY: 100, deltaY: -400});
          out.spot = [s0, spot()];
          const c0 = Object.assign({}, L.cam), tf0 = Object.assign({}, L.tf);
          fire(cv, "pointerdown", {clientX: e3[0], clientY: e3[1], shiftKey: true});
          fire(cv, "pointermove", {clientX: e3[0] + 30, clientY: e3[1] + 10});
          fire(cv, "pointerup", {});
          out.pan3 = [c0.tx !== L.cam.tx || c0.ty !== L.cam.ty || c0.tz !== L.cam.tz,
            tf0.x === L.tf.x && tf0.y === L.tf.y];
          gvSet("mode3", false); frames(200);
          const k0 = L.tf.k, touch = (t, x, id) => fire(cv, t, {clientX: x, clientY: 400,
            pointerId: id, pointerType: "touch"});
          touch("pointerdown", 400, 11); touch("pointerdown", 600, 12);
          touch("pointermove", 300, 11); touch("pointermove", 700, 12);
          out.pinch = L.tf.k / k0;
          touch("pointerup", 300, 11); out.pinchEnds = L.pinch === null;
          return out; })()""",
    )
    wheel = shown["wheel"]
    assert wheel["k"] > 1
    assert wheel["after"] == pytest.approx(wheel["before"])
    assert shown["pan"][:2] == pytest.approx([50, -30])
    assert shown["pan"][2:] == [True, True]  # a drag, so the click after it is swallowed
    assert shown["clickAfterPan"] is None
    assert shown["dragging"] == "d:act"
    assert shown["live"] is True
    assert shown["pinned"] == pytest.approx([0, 0], abs=1e-9)
    assert shown["released"] is True
    assert shown["settled"] is False
    assert shown["turn"] == pytest.approx(30 * 0.006)
    assert shown["spot"][0] == pytest.approx(shown["spot"][1])
    assert shown["pan3"] == [True, True]
    assert shown["pinch"] == pytest.approx(2)
    assert shown["pinchEnds"] is True


@needs_node
def test_the_gravity_keys_leave_the_page_s_keys_alone(tmp_path: Path) -> None:
    """In Gravity the arrows turn or pan only with the canvas focused, else they step turns as
    on every tab; Esc clears a selection first and is the page's with none; f, s, h, 3, + and -
    are the view's."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; drawView();
          const L = gvL, cv = document.getElementById("gvc");
          const key = (k, t) => gvKey({key: k, target: t || document.body});
          const out = {arrowOff: key("ArrowRight"), escNone: key("Escape")};
          const x0 = L.tf.x; out.arrowOn = key("ArrowRight", cv); out.panned = L.tf.x - x0;
          gvSelect(L, L.byId.get("d:act")); out.escSel = key("Escape"); out.cleared = GV.sel;
          const k0 = L.tf.k; out.plus = key("+"); out.zoom = L.tf.k / k0;
          out.three = key("3"); out.mode3 = GV.mode3; key("3");
          out.other = key("j");
          return out; })()""",
    )
    assert shown["arrowOff"] is False
    assert shown["escNone"] is False
    assert shown["arrowOn"] is True
    assert shown["panned"] == -60
    assert shown["escSel"] is True
    assert shown["cleared"] is None
    assert shown["plus"] is True
    assert shown["zoom"] == pytest.approx(1.25)
    assert shown["three"] is True
    assert shown["mode3"] is True
    assert shown["other"] is False


def test_the_page_is_the_same_bytes_for_the_same_documents(turns: dict[str, bytes]) -> None:
    """The page is a function of the documents: built twice it is the same bytes, and its
    script reads no clock and draws no random number."""
    first = page(turns, DESIGNS)
    assert page(turns, DESIGNS) == first
    script = first.split("<script>")[-1]
    for call in ("Math.random", "Date.now", "new Date", "performance.now", "crypto."):
        assert call not in script, call


@needs_node
def test_the_gravity_view_draws_the_state_and_colours_by_type(tmp_path: Path) -> None:
    """t:state is a body of its own kind with edges to what reads it; the default colour is by
    type, one hue per kind, and the domain colouring stays a control away."""
    html = page([], [DESIGN.document])
    o = _data(html)["sets"][0]["docs"][0]["outline"]
    uses = [e for e in o["edges"] if not e["dependency"]]
    assert uses
    shown = _run(
        html,
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = null; drawView(); const L = gvL, out = {};
          const st = L.byId.get("t:state");
          out.kind = st.kind; out.label = st.label; out.heavy = gvHeavy(st);
          out.edges = L.edges.filter((e) => e.s === "t:state").map((e) => e.t);
          out.byDomain = GV.byDomain; out.box = document.getElementById("gvkind").checked;
          out.typeCols = Object.fromEntries(L.nodes.map((n) => [n.id, gvCol(L, n)]));
          out.legendType = strip(document.getElementById("gvlegend").innerHTML);
          out.desc = strip(gvDescribe(L, st));
          gvSet("byDomain", true);
          out.legendDomain = strip(document.getElementById("gvlegend").innerHTML);
          gvSet("byDomain", false); out.stateCol = gvCol(L, st); out.kindCol = L.T.kind.state;
          gvSelect(L, st); out.sel = GV.sel;
          gvSet("find", "t:state"); out.hit = [...L.hits];
          return out; })()""",
    )
    assert shown["kind"] == "state"
    assert shown["label"] == "t:state"
    assert shown["heavy"]
    assert set(shown["edges"]) >= {"q:tone", "q:urgent"}
    assert shown["byDomain"] is False
    assert shown["box"] is False
    cols = shown["typeCols"]
    assert cols["q:tone"] == cols["q:urgent"] != cols["d:urgency"]  # same kind, same colour
    assert cols["t:state"] != cols["q:tone"]
    assert "state" in shown["legendType"]
    assert "rule result" in shown["legendType"]
    assert "the kind of body" in shown["legendType"]
    assert "first segment" in shown["legendDomain"]
    assert shown["sel"] == "t:state"
    assert shown["hit"] == ["t:state"]
    assert shown["stateCol"] == shown["kindCol"]
    assert "read by" in shown["desc"]


@needs_node
def test_the_gravity_overlay_shades_targets_and_can_be_turned_off(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Under a turn a target is shaded from the entries that estimate it, by the round shown,
    and says so in its description; the overlay control shows the model alone and back again
    without losing the turn; it is offered only when a turn is selected."""
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    shown = _run(
        html,
        tmp_path,
        """(() => { const out = {}; S.view = "gravity"; S.turn = 0; GV.ov = true; GV.step = null;
          drawView();
          const L = gvL, tg = L.byId.get("tg:urgent");
          const est = [...L.adj.get("tg:urgent")].map((i) => [i, L.byId.get(i).os]);
          out.est = est; out.ts = tg.ts; out.tstate = LAST_GRAVITY.tstate;
          out.tip = strip(gvDescribe(L, tg));
          out.key = strip(document.getElementById("gvkey").innerHTML);
          out.byRound = [];
          for (let r = 0; r <= L.ov.maxR; r++) { gvSet("step", r); out.byRound.push(tg.ts); }
          gvSet("step", null);
          out.offered = !document.getElementById("gvovl").hidden;
          gvSet("ov", false);
          out.off = {ov: !!L.ov, ts: tg.ts, marks: LAST_GRAVITY.marks, lit: L.lit.length,
            panel: document.getElementById("gvov").hidden, turn: S.turn,
            os: L.nodes.map((n) => n.os),
            offered: !document.getElementById("gvovl").hidden};
          gvSet("step", 0); gvSet("hide", true);
          out.offAfterControls = [L.nodes.every((n) => gvOvA(L, n) === 1), !!L.ov];
          gvSet("hide", false);
          gvSet("ov", true); out.on = {ov: !!L.ov, ts: tg.ts, turn: S.turn, step: GV.step};
          S.turn = null; drawView();
          out.noTurn = {hidden: document.getElementById("gvovl").hidden, ts: tg.ts};
          return out; })()""",
    )
    ts = ["answered", "mixed", "unresolved", "untested"]
    assert shown["ts"] in ts
    held = {"value", "short", "nobounds", "det"}
    opened = {"Unsure", "Unavailable", "Unasked"}
    marks = [m for _, m in shown["est"]]
    want = (
        "untested"
        if not (held | opened) & set(marks)
        else "unresolved"
        if not held & set(marks)
        else "mixed"
        if opened & set(marks)
        else "answered"
    )
    assert shown["ts"] == want
    assert shown["tstate"]["tg:urgent"] == shown["ts"]
    assert "derived" in shown["tip"]
    assert "target" in shown["key"]
    assert shown["byRound"][0] == "untested"  # round 0 reaches no estimating entry
    assert shown["byRound"][-1] == shown["ts"]
    assert shown["offered"] is True
    off = shown["off"]
    assert off["ov"] is False
    assert off["ts"] is None
    assert off["marks"] is None
    assert off["lit"] == 0
    assert off["panel"] is True
    assert off["turn"] == 0
    assert set(off["os"]) == {None}
    assert off["offered"] is True  # still offered, so it can be turned back on
    assert shown["offAfterControls"] == [True, False]
    on = shown["on"]
    assert on["ov"] is True
    assert on["ts"] == "untested"  # the round stepper kept its place across the switch
    assert on["step"] == 0
    assert on["turn"] == 0
    assert shown["noTurn"] == {"hidden": True, "ts": None}


@needs_node
def test_the_gravity_move_drag_slides_the_view_without_turning_it(tmp_path: Path) -> None:
    """In 3D a move drag (space held, the middle button, shift, or the drag mode) translates the
    camera's orbit point in the view plane: the view direction is unchanged, every body moves on
    screen by the pointer's travel at its depth, at any zoom, with no inertia; a plain drag
    still turns; a press on a body drags the body in either mode."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = true; drawView(); frames(200);
          const L = gvL, cv = document.getElementById("gvc"), out = {};
          L.cam.yaw = 0.7; L.cam.pitch = -0.3;
          const empty = () => { gvProject(L);
            for (let y = L.H - 5; y > 0; y -= 7) for (let x = 5; x < L.W; x += 7)
              if (!gvPick(L, {clientX: x, clientY: y})) return [x, y]; };
          const screen = () => { gvProject(L); return L.nodes.map((n) => [n.px, n.py, n.sc]); };
          const slide = (opt, dx, dy, pre, post) => {
            const e = empty(), c0 = Object.assign({}, L.cam), s0 = screen();
            if (pre) pre();
            fire(cv, "pointerdown", Object.assign({clientX: e[0], clientY: e[1]}, opt));
            fire(cv, "pointermove", Object.assign({clientX: e[0] + dx, clientY: e[1] + dy}, opt));
            fire(cv, "pointerup", {}); if (post) post();
            const s1 = screen(), k = L.tf.k;
            frames(120);
            return {yaw: L.cam.yaw - c0.yaw, pitch: L.cam.pitch - c0.pitch, k,
              moved: ["tx", "ty", "tz"].map((a) => L.cam[a] - c0[a]),
              err: Math.max(...s1.map((p, i) => Math.max(
                Math.abs(p[0] - s0[i][0] - dx / k * p[2]),
                Math.abs(p[1] - s0[i][1] - dy / k * p[2])))),
              after: {yaw: L.cam.yaw - c0.yaw, tx: L.cam.tx - c0.tx - 0, vy: L.vyaw}};
          };
          const space = () => gvKey({key: " ", target: cv});
          out.spaceSelf = space(); out.spaceFlag = GV.space;
          out.space = slide({}, 40, -25);
          gvKeyUp({key: " "}); out.spaceOff = GV.space;
          out.middle = slide({button: 1}, -30, 20);
          out.shift = slide({shiftKey: true}, 15, 15);
          L.tf = {x: L.tf.x, y: L.tf.y, k: 6};
          out.close = slide({button: 2}, 60, 30);
          L.tf = {x: L.tf.x, y: L.tf.y, k: 1};
          out.orbit = slide({}, 30, 0);
          gvAct("drag-move"); out.mode = GV.drag;
          out.plainMove = slide({}, 25, 10);
          const n = L.byId.get("d:act"); gvProject(L);
          const sx = L.tf.x + L.tf.k * (L.W / 2 + n.px), sy = L.tf.y + L.tf.k * (L.H / 2 + n.py);
          fire(cv, "pointerdown", {clientX: sx, clientY: sy});
          out.bodyDrag = L.drag && L.drag.n.id; out.bodyRot = L.rot === null;
          fire(cv, "pointerup", {}); frames(300);
          gvAct("drag-orbit"); out.back = GV.drag;
          return out; })()""",
    )
    for name in ("space", "middle", "shift", "close", "plainMove"):
        m = shown[name]
        assert m["yaw"] == 0, name
        assert m["pitch"] == 0, name
        assert any(abs(v) > 0 for v in m["moved"]), name
        assert m["err"] < 1e-6, name  # every body follows the pointer at its own depth
        assert m["after"]["vy"] == 0, name  # no inertia
    assert shown["close"]["k"] == 6
    assert shown["spaceSelf"] is True
    assert shown["spaceFlag"] is True
    assert shown["spaceOff"] is False
    assert shown["orbit"]["yaw"] > 0  # a plain drag turns
    assert shown["mode"] == "move"
    assert shown["bodyDrag"] == "d:act"
    assert shown["bodyRot"] is True
    assert shown["back"] == "orbit"


@needs_node
def test_the_gravity_round_keys_step_the_overlay(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """a and d step the round shown back and on, clamped at both ends with no wrap, only with
    the overlay on and not while typing or with a modifier; f, s, h, 3, / and the arrows stay."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.ov = true; GV.step = null; drawView();
          const L = gvL, key = (k, o) => gvKey(Object.assign({key: k, target: document.body}, o));
          const out = {max: L.ov.maxR, start: gvStepOf(L)};
          out.d = key("d"); out.atEnd = gvStepOf(L);
          const seq = [];
          for (let i = 0; i < L.ov.maxR + 2; i++) { key("a"); seq.push(gvStepOf(L)); }
          out.down = seq; out.back = key("D"); out.up = gvStepOf(L);
          out.ctrl = key("d", {ctrlKey: true}); out.meta = key("a", {metaKey: true});
          out.alt = key("d", {altKey: true}); out.still = gvStepOf(L);
          out.typing = key("d", {target: {tagName: "INPUT"}}); out.typed = gvStepOf(L);
          out.snap = LAST_GRAVITY.step;
          gvSet("ov", false); out.off = key("d"); out.offStep = gvStepOf(L);
          gvSet("ov", true);
          out.others = ["f", "s", "h", "3", "/"].map((k) => key(k)); key("3");
          return out; })()""",
    )
    top = shown["max"]
    assert top >= 2
    assert shown["start"] == top
    assert shown["d"] is True
    assert shown["atEnd"] == top  # clamped, not wrapped
    assert shown["down"] == [max(0, top - i - 1) for i in range(top + 2)]
    assert shown["back"] is True
    assert shown["up"] == 1
    assert shown["ctrl"] is False
    assert shown["meta"] is False
    assert shown["alt"] is False
    assert shown["still"] == 1
    assert shown["typing"] is False
    assert shown["typed"] == 1
    assert shown["snap"] == 1
    assert shown["off"] is False  # nothing to step with the overlay off
    assert all(shown["others"])


# ---------- gravity: the view's chrome (fit, hash, toolbar, keys, touch, frames) ----------


@needs_node
def test_the_gravity_fit_lands_in_the_free_rectangle(tmp_path: Path) -> None:
    """Fit frames the bodies in what the toolbar, panels and legend leave, measured from the
    boxes as they stand, not from fixed margins; a panel docked at the bottom shortens it."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(100);
          const L = gvL, el = (id) => document.getElementById(id), out = {};
          const at = (id, l, t, w, h) => { el(id).getBoundingClientRect = () => ({left: l, top: t,
            width: w, height: h}); };
          L.W = 1120; L.H = 600;
          el("gvov").hidden = false;
          at("gvbar", 8, 8, 1100, 60); at("gvov", 8, 96, 290, 300); at("gvside", 812, 96, 300, 300);
          at("gvlegend", 8, 520, 900, 70); at("gvgz", 1036, 516, 76, 76);
          out.free = gvFree(L);
          gvFit(L); frames(120);
          gvProject(L);
          const sx = (n) => L.tf.x + L.tf.k * (L.W / 2 + n.px);
          const sy = (n) => L.tf.y + L.tf.k * (L.H / 2 + n.py);
          out.box = [Math.min(...L.act.map(sx)), Math.max(...L.act.map(sx)),
            Math.min(...L.act.map(sy)), Math.max(...L.act.map(sy))];
          at("gvov", 8, 400, 1100, 150); out.bottom = gvFree(L);
          at("gvov", 8, 96, 600, 300); at("gvside", 700, 96, 300, 300); out.tall = gvFree(L);
          return out; })()""",
    )
    f = shown["free"]
    assert f == {"x0": 306, "y0": 76, "x1": 804, "y1": 512}
    x0, x1, y0, y1 = shown["box"]
    assert f["x0"] <= x0 <= x1 <= f["x1"]
    assert f["y0"] <= y0 <= y1 <= f["y1"]
    assert shown["bottom"]["y1"] == 392  # a panel docked low shortens the rectangle
    assert shown["tall"]["x0"] == 8  # too little left: the canvas under the toolbar


@needs_node
def test_the_gravity_edges_thin_with_the_hub_and_keep_the_long_ones(tmp_path: Path) -> None:
    """An edge's alpha falls with its busier end's degree and rises a little with its length; the
    edge colour is its own token, lighter in the dark theme."""
    html = page([], [DESIGN.document])
    assert "--gv-edge: #8c8a84" in html
    assert html.count("--gv-edge: #b4b2aa") == 2  # the system's dark and the picked dark
    shown = _run(
        html,
        tmp_path,
        """(() => ({hub: gvThin({deg: 40, px: 0, py: 0}, {deg: 1, px: 50, py: 0}),
          pair: gvThin({deg: 2, px: 0, py: 0}, {deg: 2, px: 50, py: 0}),
          far: gvThin({deg: 2, px: 0, py: 0}, {deg: 2, px: 500, py: 0}),
          tok: GV_TOK.edge}))()""",
    )
    assert shown["tok"] == "--gv-edge"
    assert shown["hub"] < shown["pair"] < shown["far"]
    assert shown["pair"] <= 1.45


@needs_node
def test_the_gravity_sprites_fall_back_to_flat_when_small(tmp_path: Path) -> None:
    """A body under 9px on screen is a flat fill, bigger ones a cached sprite; the cache is one
    for every design and survives a redraw, and is dropped when the theme's tokens change."""
    shown = _run(
        page([], [DESIGN.document, load(showcase.PLAIN).document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; GV.flat = false;
          drawView(); frames(100);
          const L = gvL, ctx = document.getElementById("gvc").getContext("2d"), out = {};
          const draws = (k) => { L.tf = {x: L.tf.x, y: L.tf.y, k}; ctx.counts.drawImage = 0;
            gvDraw(L); return ctx.counts.drawImage; };
          out.small = draws(0.2); out.big = draws(4);
          out.size = GV_SPR.size; gvTheme(L); gvDraw(L); out.kept = GV_SPR.size;
          S.design = 1; drawView(); out.shared = GV_SPR.size >= out.size;
          GV_SPR.sig = "other"; gvTheme(gvL); out.cleared = GV_SPR.size;
          out.noLocal = L.spr === undefined;
          return out; })()""",
    )
    assert shown["small"] == 0
    assert shown["big"] > 0
    assert shown["size"] > 0
    assert shown["kept"] == shown["size"]
    assert shown["shared"] is True
    assert shown["cleared"] == 0
    assert shown["noLocal"] is True


@needs_node
def test_the_gravity_link_carries_the_round_filters_and_colour(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The round, hide-not-reached, only-lineage, colour-by-domain and the two filters are each
    written only when not their default, read back, and a restored selection is framed once."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { location.hash = h; }};
          S.view = "gravity"; S.turn = 0; GV.step = null; GV.hide = false; GV.lin = false;
          GV.byDomain = false; GV.light = true; GV.cross = false; GV.sel = null; GV.ov = true;
          render(); writeHash(); const out = {plain: location.hash};
          gvSet("step", 1); gvSet("hide", true); gvSet("lin", true); gvSet("byDomain", true);
          gvSet("light", false); gvSet("cross", true); gvSelect(gvL, gvL.byId.get("d:act"));
          out.hash = location.hash;
          Object.assign(GV, {step: null, hide: false, lin: false, byDomain: false, light: true,
            cross: false, sel: null});
          let fits = 0; const fitSel = gvFitSel; gvFitSel = (L) => { fits++; return fitSel(L); };
          readHash(); render(); frames(50);
          out.back = {step: GV.step, shown: gvStepOf(gvL), hide: GV.hide, lin: GV.lin,
            dom: GV.byDomain, light: GV.light, cross: GV.cross, sel: GV.sel, fits};
          render(); out.fits = fits;
          gvSet("step", null); gvSet("hide", false); gvSet("lin", false); gvSet("byDomain", false);
          gvSet("light", true); gvSet("cross", false); gvSelect(gvL, null);
          out.reset = location.hash;
          return out; })()""",
    )
    assert "gr=" not in shown["plain"]
    for key in ("gr=1", "ghide=1", "glin=1", "gdom=1", "gq=0", "gx=1", "gsel=d%3Aact"):
        assert key in shown["hash"], key
    back = shown["back"]
    assert back["step"] == 1
    assert back["shown"] == 1
    assert (back["hide"], back["lin"], back["dom"]) == (True, True, True)
    assert (back["light"], back["cross"], back["sel"]) == (False, True, "d:act")
    assert back["fits"] == 1
    assert shown["fits"] == 1  # once, not on every draw
    for key in ("gr=", "ghide", "glin", "gdom", "gq=", "gx=", "gsel"):
        assert key not in shown["reset"], key


def test_the_gravity_toolbar_is_grouped_and_every_control_names_its_key() -> None:
    html = page([], [DESIGN.document])
    bar = html[html.index('id="gvbar"') : html.index('id="gvov"')]
    for label in ("Show", "Turn", "View"):
        assert f'<b class="gv-gl">{label}</b>' in bar
    assert '<details id="gvlayout">' in bar  # closed by default
    layout = bar[bar.index('<details id="gvlayout">') :]
    for control in ("gvqpull", "gvhpull", 'data-gvact="relayout"', "gvflat", "gvhalo", "gvspin"):
        assert control in layout
    before = bar[: bar.index('<details id="gvlayout">')]
    for control in ("gvqpull", "gvhpull", "relayout", "gvflat", "gvhalo", "gvspin"):
        assert control not in before
    assert 'id="gvfitsel" data-gvact="fitsel"' in bar
    assert "disabled>fit selection" in bar  # nothing selected yet
    for title in ("(f)", "(s)", "(h)", "(3)", "(/)", "(?)"):
        assert title in bar, title
    assert 'data-gvact="drag-orbit"' in bar
    assert 'data-gvact="drag-move"' in bar
    assert "drag: orbit</button>" not in bar


@needs_node
def test_the_header_keys_follow_the_view_and_esc_clears_one_layer_a_press(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The key strip names the open view's keys; the long gesture sentence is the help's, opened
    by ?; Esc clears the help, the search, the selection, the pinned round, one a press, and
    only then is the page's."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const out = {}, keys = () => document.getElementById("keys").innerHTML;
          S.turn = 0; S.view = "explore"; headKeys(); out.explore = keys();
          S.view = "gravity"; GV.step = null; GV.sel = null; GV.find = "";
          GV.help = false; drawView();
          headKeys(); out.gravity = keys();
          out.legend = document.getElementById("gvlegend").innerHTML;
          const L = gvL, key = (k) => gvKey({key: k, target: document.body});
          out.help = [key("?"), GV.help];
          gvSet("find", "u"); gvSelect(L, L.byId.get("d:act")); gvSet("step", 1);
          out.esc = [];
          for (let i = 0; i < 5; i++) {
            out.esc.push([key("Escape"), GV.help, GV.find, GV.sel, GV.step]); }
          return out; })()""",
    )
    assert "Explore:" in shown["explore"]
    assert "Gravity:" in shown["gravity"]
    assert "Explore:" not in shown["gravity"]
    assert "wheel zooms" not in shown["legend"]
    assert "? for gestures and keys" not in shown["legend"]  # the help holds the keys
    assert shown["help"] == [True, True]
    assert shown["esc"] == [
        [True, False, "u", "d:act", 1],  # the help first
        [True, False, "", "d:act", 1],  # then the search
        [True, False, "", None, 1],  # then the selection
        [True, False, "", None, None],  # then the round pinned
        [False, False, "", None, None],  # nothing left: the page's own back-out
    ]


def test_the_gravity_help_holds_the_gestures_with_touch_wording() -> None:
    html = page([], [DESIGN.document])
    helpbox = html[html.index('id="gvhelp"') : html.index('id="gvov"')]
    assert "Touch" in helpbox
    assert "Two fingers pan and pinch" in helpbox
    assert "shift-, right-, middle- or space-drag" in helpbox
    assert 'id="gvhelpb"' in html


@needs_node
def test_a_fingertip_picks_with_a_wider_slop_and_the_tooltip_is_short(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; GV.touch = false; GV.sel = null;
          drawView(); frames(60);
          const L = gvL, out = {}, n = L.byId.get("d:act");
          gvProject(L);
          const sx = L.tf.x + L.tf.k * (L.W / 2 + n.px), sy = L.tf.y + L.tf.k * (L.H / 2 + n.py);
          const off = n.r * n.sc * L.tf.k + 3 + 12;
          const at = {clientX: sx + off, clientY: sy};
          out.mouse = gvCands(L, {...at, pointerType: "mouse"}).includes(n);
          out.touch = gvCands(L, {...at, pointerType: "touch"}).includes(n);
          out.mouseTip = strip(gvTipHtml(L, n, 0));
          out.beneath = strip(gvTipHtml(L, n, 2));
          GV.touch = true; out.touchTip = strip(gvTipHtml(L, n, 0));
          out.touchBeneath = strip(gvTipHtml(L, n, 1));
          out.hint = strip(document.getElementById("gvside").innerHTML); gvPanel(L);
          out.touchHint = strip(document.getElementById("gvside").innerHTML);
          GV.touch = false; gvPanel(L);
          out.mouseHint = strip(document.getElementById("gvside").innerHTML);
          out.long = strip(gvDescribe(L, n)).length > out.mouseTip.length;
          return out; })()""",
    )
    assert shown["mouse"] is False
    assert shown["touch"] is True
    assert "d:act" in shown["mouseTip"]
    assert "click to select" in shown["mouseTip"]
    assert "+2 beneath, click again" in shown["beneath"]
    assert "tap to select" in shown["touchTip"]
    assert "+1 beneath, tap again" in shown["touchBeneath"]
    assert shown["touchHint"].startswith("Tap a body.")
    assert shown["mouseHint"].startswith("Hover or click a body.")


@needs_node
def test_the_gravity_pointer_handlers_ask_for_a_frame_and_sliders_reheat_live(
    tmp_path: Path,
) -> None:
    """A pointer move queues a frame instead of drawing; the pull sliders reheat the layout live
    instead of settling 200 ticks in the event; the tooltip is rebuilt only when the body under
    the pointer changes."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(100);
          const L = gvL, cv = document.getElementById("gvc"), out = {};
          let draws = 0, steps = 0, tips = 0;
          const d0 = gvDraw, s0 = gvStep, t0 = gvTipHtml;
          gvDraw = (l) => { draws++; return d0(l); };
          gvStep = (l, a) => { steps++; return s0(l, a); };
          gvTipHtml = (l, n, b) => { tips++; return t0(l, n, b); };
          fire(cv, "pointerdown", {clientX: 5, clientY: 5});
          fire(cv, "pointermove", {clientX: 40, clientY: 30});
          fire(cv, "pointerup", {});
          out.syncDraws = draws; frames(1); out.afterFrame = draws;
          const n = L.byId.get("d:act"); gvProject(L);
          const sx = L.tf.x + L.tf.k * (L.W / 2 + n.px), sy = L.tf.y + L.tf.k * (L.H / 2 + n.py);
          document.getElementById("gvtip").hidden = true;
          for (const dx of [0, 1, 0, 1]) fire(cv, "pointermove", {clientX: sx + dx, clientY: sy});
          out.tips = tips;
          const before = JSON.stringify(L.nodes.map((m) => [m.x, m.y]));
          steps = 0;
          document.getElementById("vGravity").listeners.input.forEach((f) => f({target:
            {id: "gvqpull", type: "range", value: "80"}}));
          out.inEvent = steps; out.live = L.live; out.pull = GV.qPull;
          frames(60); out.frames = steps;
          out.moved = JSON.stringify(L.nodes.map((m) => [m.x, m.y])) !== before;
          return out; })()""",
    )
    assert shown["syncDraws"] == 0
    assert shown["afterFrame"] >= 1
    assert shown["tips"] == 1
    assert shown["inEvent"] == 0
    assert shown["live"] is True
    assert shown["pull"] == pytest.approx(0.8)
    assert shown["frames"] > 0
    assert shown["moved"] is True


# ---------- gravity: legibility (target rings, lineage, labels, neutral tint) ----------

_RECORDER = """const rec = (() => { const ev = [];
  const st = {globalAlpha: 1, lineWidth: 1, strokeStyle: "", fillStyle: "",
    dash: []};
  const o = {ev, st, measureText: (t) => ({width: String(t).length * 6}),
    createRadialGradient: () => ({addColorStop() {}}),
    createLinearGradient: () => ({addColorStop() {}})};
  return new Proxy(o, {
    get: (t, k) => (k in t ? t[k] : k in t.st ? t.st[k] : typeof k !== "string" ? undefined :
      (...a) => {
      if (k === "setLineDash") t.st.dash = a[0];
      t.ev.push({k, a, al: t.st.globalAlpha, lw: t.st.lineWidth, ss: t.st.strokeStyle,
        fs: t.st.fillStyle, dash: t.st.dash}); }),
    set: (t, k, v) => { t.st[k] = v; return true; }}); })();
  document.getElementById("gvc").getContext = () => rec;"""


@needs_node
def test_the_target_ring_is_a_donut_split_by_the_held_to_the_open(tmp_path: Path) -> None:
    """Answered is one full ring, unresolved a double ring, mixed two arcs whose lengths are the
    held to the open; not evaluated stays the dashed ring; a value entry carries a tick."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          const T = {good: "G", warn: "W", bad: "B", neutral: "N"}, out = {};
          const ring = (ts, held, open) => { const arcs = [], cx = {
            beginPath() {}, arc: (x, y, r, a0, a1) => arcs.push([r, a0, a1]), stroke() {},
            set strokeStyle(v) { arcs.col = (arcs.col || []).concat(v); }, set lineWidth(v) {}};
            gvTsRing(cx, T, {ts, tc: {held, open}}, 0, 0, 10); return {arcs, col: arcs.col}; };
          out.answered = ring("answered", 3, 0); out.unresolved = ring("unresolved", 0, 2);
          out.mixed = ring("mixed", 1, 3); out.tick = GV_RING.value[2];
          out.shape = GV_TS_SHAPE;
          return out; })()""",
    )
    tau = 6.283185307179586
    assert shown["answered"]["arcs"] == [[13, 0, pytest.approx(tau)]]
    assert shown["answered"]["col"] == ["G"]
    radii = [a[0] for a in shown["unresolved"]["arcs"]]
    assert radii == [13, pytest.approx(16.6)]  # a double ring
    assert shown["unresolved"]["col"] == ["B", "B"]
    first, second = shown["mixed"]["arcs"]
    assert first[0] == second[0] == 13
    assert first[2] - first[1] == pytest.approx(tau * 0.25)  # held 1 of 4
    assert second[2] - second[1] == pytest.approx(tau * 0.75)
    assert second[1] == pytest.approx(first[2])
    assert shown["mixed"]["col"] == ["G", "W"]
    assert shown["tick"] == "✓"
    assert set(shown["shape"]) == {"answered", "mixed", "unresolved", "untested"}


@needs_node
def test_the_lineage_reads_as_a_direction_and_declared_hops_recede(tmp_path: Path) -> None:
    """Main-chain hops take the accent, are drawn after the other lit hops and carry an arrowhead
    toward the root; a declared hop is thinner and fainter than a read one."""
    shown = _run(
        page(
            {"plain.json": showcase.plain_turn().document},
            [DESIGN.document, load(showcase.PLAIN).document],
        ),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.design = 1; S.turn = 0; GV.mode3 = false; GV.ov = true;
          GV.step = null;
          drawView(); frames(100); const L = gvL, out = {};
          """
        + _RECORDER
        + """
          gvDraw(L);
          const k = L.tf.k, T = L.T;
          const strokes = rec.ev.filter((e) => e.k === "stroke");
          out.declared = strokes.filter((e) => e.dash.length === 2 && e.dash[0] > 4 / k)
            .map((e) => [e.al, e.lw * Math.sqrt(k)]);
          out.heads = rec.ev.filter((e) => e.k === "fill" && e.fs === T.branch).length;
          out.mainHeads = rec.ev.filter((e) => e.k === "fill" && e.fs === T.main).length;
          out.lit = L.lit.length; out.mains = L.lit.filter((h) => h[3]).length;
          out.token = [T.main, T.cause]; out.hasToken = L.T.main !== undefined;
          return out; })()""",
    )
    assert shown["lit"] > 0
    assert shown["heads"] + shown["mainHeads"] == shown["lit"]  # one arrowhead a lit hop
    assert shown["mainHeads"] == shown["mains"]
    assert shown["token"][0] != shown["token"][1]
    assert shown["declared"], "the plain turn has declared hops"
    for alpha, width in shown["declared"]:
        assert alpha <= 0.4 + 1e-9
        assert width == pytest.approx(1.2)


@needs_node
def test_a_gravity_label_never_covers_a_heavy_body_or_another_label(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(100);
          const L = gvL, out = {};
          """
        + _RECORDER
        + """
          gvDraw(L);
          const k = L.tf.k, texts = rec.ev.filter((e) => e.k === "fillText" && e.a.length === 3);
          const label = new Map(L.nodes.map((n) => [n.label, n]));
          const boxes = texts.map((e) => { const w = e.a[0].length * 6 + 4;
            return [e.a[1] - w / 2, e.a[2] - 11, e.a[1] + w / 2, e.a[2] + 3, e.a[0]]; });
          const body = L.nodes.filter((n) => gvHeavy(n) && n.va > .3).map((n) => {
            const rs = n.r * n.sc * n.va * k, sx = L.tf.x + k * (L.W / 2 + n.px),
              sy = L.tf.y + k * (L.H / 2 + n.py);
            return [sx - rs, sy - rs, sx + rs, sy + rs, n.id]; });
          const hit = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1];
          out.n = boxes.length; out.placed = L.labels.length;
          out.onBody = []; out.onLabel = [];
          boxes.forEach((b, i) => {
            body.forEach((d) => { if (hit(b, d)) out.onBody.push([b[4], d[4]]); });
            boxes.forEach((c, j) => {
              if (j > i && hit(b, c)) out.onLabel.push([b[4], c[4]]); }); });
          return out; })()""",
    )
    assert shown["n"] > 0
    assert shown["n"] == shown["placed"]
    assert shown["onBody"] == []
    assert shown["onLabel"] == []


@needs_node
def test_the_overlay_draws_bodies_in_one_neutral_tint(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """With a turn's overlay on, every body is one tint and flat (no sprite), so the state's
    rings are the only colour, and the legend says so; off, the type hues are back."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.ov = true; GV.mode3 = false; GV.flat = false;
          GV.halo = true; drawView(); frames(100);
          const L = gvL, out = {}, cv = document.getElementById("gvc").getContext("2d");
          out.on = new Set(L.nodes.map((n) => gvCol(L, n))).size;
          out.tint = L.T.tint; out.first = gvCol(L, L.nodes[0]);
          L.tf = {x: L.tf.x, y: L.tf.y, k: 4}; cv.counts.drawImage = 0; gvDraw(L);
          out.sprites = cv.counts.drawImage;
          out.legend = strip(document.getElementById("gvlegend").innerHTML);
          gvSet("ov", false); out.off = new Set(L.nodes.map((n) => gvCol(L, n))).size;
          out.legendOff = strip(document.getElementById("gvlegend").innerHTML);
          cv.counts.drawImage = 0; gvDraw(L); out.spritesOff = cv.counts.drawImage;
          GV.halo = false;
          return out; })()""",
    )
    assert shown["on"] == 1
    assert shown["first"] == shown["tint"]
    assert shown["sprites"] == 0
    assert "neutral tint" in shown["legend"]
    assert "the state is the colour" in shown["legend"]
    assert shown["off"] > 1
    assert "neutral tint" not in shown["legendOff"]
    assert "the kind of body" in shown["legendOff"]
    assert shown["spritesOff"] > 0


# ---------- gravity: navigation (links to the other tabs, search, the turn panel) ----------


@needs_node
def test_the_gravity_panel_links_to_path_targets_and_explore_and_back(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A selected body offers its tab links; each is a history step, so the hash returns to the
    graph as it stood; Explore and Path details offer 'show in Gravity' unless the node has no
    body."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const pushed = [], out = {}; globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { pushed.push(h); location.hash = h; }};
          const click = (d) => viewsClick({target: {closest: () => ({dataset: d})}});
          const panel = (id) => { const L = gvL; gvSelect(L, L.byId.get(id));
            return document.getElementById("gvside").innerHTML; };
          S.view = "gravity"; S.turn = 0; GV.sel = null; drawView(); frames(50);
          out.rule = panel("d:act"); out.question = panel("q:tone"); out.target = panel("tg:urgent");
          out.state = panel("t:state"); out.tool = panel("t:ticket");
          gvSelect(gvL, gvL.byId.get("d:act"));
          const before = location.hash; out.before = before;
          click({gvpath: "rule:act"});
          out.path = {view: S.view, node: S.node, pushed: pushed.length};
          location.hash = before; readHash(); render();
          out.back = {view: S.view, sel: GV.sel};
          gvSelect(gvL, gvL.byId.get("tg:urgent")); const tHash = location.hash;
          click({gvtgt: "tg:urgent"});
          out.tg = {view: S.view, tgt: S.tgt, pushed: pushed.length,
            html: document.getElementById("vTargets").innerHTML.includes('class="tfocus"')};
          location.hash = tHash; readHash(); render();
          out.back2 = {view: S.view, sel: GV.sel, tgt: S.tgt};
          S.view = "explore"; S.node = "rule:act"; drawView(); drawDetail(document.getElementById("exdetail"), "rule:act");
          out.explore = document.getElementById("exdetail").innerHTML;
          S.node = "source:ask"; drawDetail(document.getElementById("exdetail"), "source:ask");
          out.source = document.getElementById("exdetail").innerHTML;
          const D = curDesign(), bare = {kind: "source", id: "source:none", questions: [], tool_results: []};
          out.none = gvShowBtn(D, bare); out.nodeBody = gvBodyOf(D, {kind: "rule", refs: ["d:x"]});
          S.view = "path"; S.node = "rule:act"; drawView();
          out.pathView = document.getElementById("vPath").innerHTML;
          pushed.length = 0; let fits = 0; const fs = gvFitSel; gvFitSel = (L) => { fits++; return fs(L); };
          click({gvshow: "d:act"});
          out.show = {view: S.view, sel: GV.sel, pushed: pushed.length, fits};
          return out; })()""",  # noqa: E501
    )
    for name in ("rule", "question", "tool"):
        assert 'data-gvpath="rule:' in shown[name], name
    assert "in Explore" in shown["rule"]
    assert 'data-gvpath="rule:act"' in shown["rule"]
    assert "(the first rule that reads it)" in shown["question"]
    assert "data-gvpath" not in shown["target"]
    assert 'data-gvtgt="tg:urgent"' in shown["target"]
    assert "in Explore" not in shown["target"]
    assert "data-gvpath" not in shown["state"]
    assert shown["path"] == {"view": "path", "node": "rule:act", "pushed": 1}
    assert shown["back"] == {"view": "gravity", "sel": "d:act"}
    assert shown["tg"]["view"] == "targets"
    assert shown["tg"]["tgt"] == "tg:urgent"
    assert shown["tg"]["pushed"] == 2
    assert shown["tg"]["html"] is True
    assert shown["back2"] == {"view": "gravity", "sel": "tg:urgent", "tgt": None}
    assert 'data-gvshow="d:act"' in shown["explore"]
    assert "data-gvshow=" in shown["source"]
    assert shown["none"] == ""
    assert shown["nodeBody"] == "d:x"
    assert 'data-gvshow="d:act"' in shown["pathView"]
    assert shown["show"] == {"view": "gravity", "sel": "d:act", "pushed": 1, "fits": 1}


@needs_node
def test_the_gravity_panel_offers_no_path_link_without_a_record(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = null; drawView(); gvSelect(gvL, gvL.byId.get("d:act"));
          return document.getElementById("gvside").innerHTML; })()""",  # noqa: E501
    )
    assert "in Explore" in shown
    assert "data-gvpath" not in shown


@needs_node
def test_gravity_search_covers_hidden_bodies_cycles_and_shows_what_it_selects(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.sel = null; GV.find = ""; GV.light = true; GV.cross = false;
          drawView(); frames(50);
          const L = gvL, out = {}, el = (id) => document.getElementById(id);
          const enter = (shift) => fire(el("gvsearch"), "keydown", {key: "Enter", shiftKey: !!shift});
          gvSet("light", false); frames(50); gvSet("find", "tone");
          out.hid = {n: L.hitList.map((n) => n.id), want: L.byId.get("q:tone").want,
            stat: el("gvstat").textContent};
          enter(); out.sel = GV.sel; out.shown = L.byId.get("q:tone").want;
          out.statSel = el("gvstat").textContent; out.forced = [...L.force];
          gvSelect(L, L.byId.get("d:act")); out.released = [L.force.size, L.byId.get("q:tone").want];
          gvSet("light", true); frames(50);
          gvSelect(L, null); gvSet("find", "d:"); const n = L.hitList.length; out.n = n;
          const seq = []; for (let i = 0; i < n + 1; i++) { enter(); seq.push(L.hitList.indexOf(L.byId.get(GV.sel))); }
          out.seq = seq; out.counter = el("gvstat").textContent;
          enter(true); out.back = L.hitList.indexOf(L.byId.get(GV.sel));
          out.opts = (el("gvhits").innerHTML.match(/<option/g) || []).length;
          gvSet("find", "TONE"); out.label = L.hitList.some((m) => m.label === "tone" || m.id === "q:tone");
          const b = L.nodes.find((m) => m.os === "value" || m.os === "det");
          const e = entryAt(L.ov.t, b.id), v = String(valueOf(e)).toLowerCase();
          gvSet("find", v); out.value = L.hitList.some((m) => m.id === b.id);
          gvSet("find", "zzzz-none"); out.none = el("gvstat").textContent;
          gvSet("find", ""); out.cleared = el("gvstat").textContent;
          return out; })()""",  # noqa: E501
    )
    assert shown["hid"]["n"][0] == "q:tone"
    assert shown["hid"]["want"] is False
    assert "hidden by filters" in shown["hid"]["stat"]
    assert shown["sel"] == "q:tone"
    assert shown["shown"] is True  # chosen, it is shown while it is the selection
    assert "shown despite the filters" in shown["statSel"]
    assert "1 / " in shown["statSel"]
    assert shown["forced"] == ["q:tone"]
    assert shown["released"] == [0, False]
    n = shown["n"]
    assert n >= 3
    assert shown["seq"] == [*range(n), 0]  # wraps
    assert shown["counter"].startswith("1 / ")
    assert shown["back"] == n - 1  # Shift+Enter from the first goes to the last
    assert 0 < shown["opts"] <= 8
    assert shown["label"] is True
    assert shown["value"] is True  # under a turn, the entry's value text matches
    assert shown["none"] == "0 found"
    assert shown["cleared"] == ""


@needs_node
def test_the_gravity_turn_panel_is_compact_and_steps_rounds_turns_and_states(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => {
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; }, pushState() {}};
          const out = {}, el = (id) => document.getElementById(id);
          const fake = {outline: {decision: {holds: "value", ref: "d:turn", value:
            {route: "calendar", email: null, queue: null}}, why: "decided {...} by d:turn"},
            sum: {decision: "x"}};
          out.dec = gvDecision(fake);
          S.view = "gravity"; S.turn = at("undecided.json"); GV.step = null; GV.st = null;
          GV.whyOpen = false; GV.sel = null; drawView(); frames(30);
          const L = gvL, t = L.ov.t; out.maxR = L.ov.maxR;
          const box = () => el("gvov").innerHTML;
          out.head = box(); out.roundLine = el("gvroundd").textContent;
          const want = (r) => (t.idx.all.filter((e) => (e.round || 0) === r)).length;
          out.entries = [0, 1, 2].map(want);
          gvAct("prev-round"); out.prev = [GV.step, el("gvroundd").textContent];
          for (let i = 0; i < 5; i++) gvAct("prev-round"); out.floor = GV.step;
          for (let i = 0; i < 9; i++) gvAct("next-round"); out.ceil = GV.step;
          gvAct("why"); out.open = [GV.whyOpen, box().includes("gv-tx")]; gvAct("why");
          const click = (d) => $("vGravity").listeners.click.forEach((f) => f({target:
            {closest: () => ({dataset: d})}}));
          const key = (k) => gvKey({key: k, target: document.body});
          const states = L.nodes.filter((n) => n.os).map((n) => n.os);
          const st = ["Unsure", "Unavailable", "Unasked", "short", "value"].find((s) => states.includes(s));
          out.st = st;
          click({gvst: st}); out.pressed = [GV.st, el("gvkey").innerHTML.includes('aria-pressed="true"')];
          const inS = L.nodes.find((n) => n.os === st), other = L.nodes.find((n) => n.os && n.os !== st);
          out.dim = [gvOvA(L, inS), other ? gvOvA(L, other) : 0];
          writeHash(); out.hash = location.hash;
          click({gvst: st}); out.cleared = [GV.st, location.hash.includes("gst")];
          click({gvst: st}); out.esc = [key("Escape"), GV.st]; writeHash(); out.hash2 = location.hash;
          return out; })()""",  # noqa: E501
    )
    assert shown["dec"] == "route: calendar · by d:turn"
    assert '{"' not in shown["head"].split("gv-why")[1].split("</div>")[0]
    assert 'data-gvact="turn-prev"' in shown["head"]
    assert 'data-gvact="prev-round"' in shown["head"]
    assert shown["roundLine"].startswith(f"round {shown['maxR']}: +")
    assert shown["prev"][0] == shown["maxR"] - 1
    assert shown["prev"][1].startswith(f"round {shown['maxR'] - 1}: +")
    assert shown["floor"] == 0  # clamped
    assert shown["ceil"] == shown["maxR"]
    assert shown["open"] == [True, True]
    assert shown["pressed"] == [shown["st"], True]
    assert shown["dim"][0] == 1
    assert shown["dim"][1] < 0.2
    assert f"gst={shown['st']}" in shown["hash"]
    assert shown["cleared"] == [None, False]
    assert shown["esc"] == [True, None]  # Esc clears the state filter
    assert "gst" not in shown["hash2"]


@needs_node
def test_the_gravity_turn_buttons_keep_the_round_when_valid(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => {
          const out = {};
          S.view = "gravity"; S.turn = 0; GV.step = null; drawView(); frames(10);
          const maxes = cur().turns.map((t, i) => { S.turn = i; drawView(); return gvL.ov.maxR; });
          out.maxes = maxes;
          S.turn = 0; drawView(); gvSet("step", 1); out.at = [S.turn, GV.step];
          gvAct("turn-next"); out.next = [S.turn, GV.step, gvStepOf(gvL), gvL.ov.maxR];
          S.turn = 0; GV.step = null; drawView(); gvSet("step", 0); gvAct("turn-next");
          out.zero = [S.turn, GV.step];
          S.turn = 0; GV.step = null; drawView(); gvAct("turn-next"); out.last = [S.turn, GV.step];
          const none = gvKey && cur().turns.map((t, i) => { S.turn = i; drawView();
            const k = document.getElementById("gvkey").innerHTML, c = gvL.nodes.filter((n) => n.os === "none").length;
            return [k.includes("no entry"), c > 0]; });
          out.none = none;
          return out; })()""",  # noqa: E501
    )
    assert len(shown["maxes"]) == 2
    assert shown["at"] == [0, 1]
    other = 1
    kept = 1 if shown["maxes"][other] >= 1 else None
    assert shown["next"][0] == other
    assert shown["next"][1] == kept  # kept when the other turn has the round, else its last
    assert shown["zero"][1] == 0
    assert shown["last"][1] is None  # the last round stays the last
    for entry in shown["none"]:
        assert entry[0] == entry[1]  # '0 no entry' is not shown


# ---------- gravity: structure (keys, narrow layout, legend, the step buffers) ----------


def test_the_gravity_canvas_names_its_keys_and_the_panels_are_live_regions() -> None:
    html = page([], [DESIGN.document])
    canvas = html[html.index('<canvas id="gvc"') : html.index("</canvas>", html.index('id="gvc"'))]
    assert 'role="application"' in canvas
    for word in ("period and comma", "Enter", "arrows", "slash", "a and d", "Mouse:"):
        assert word in canvas, word
    assert 'id="gvside" aria-live="polite"' in html
    assert '<span id="gvroundl" aria-live="polite">' in html
    assert 'id="gvlive" class="gv-sr" role="status"' in html
    # the focus order: the toolbar, the canvas, then the panels
    order = [html.index(f'id="{i}"') for i in ("gvbar", "gvc", "gvov", "gvside", "gvlegend")]
    assert order == sorted(order)


@needs_node
def test_the_gravity_keys_walk_the_bodies_and_the_panel_keeps_the_focus(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; GV.sel = null; drawView(); frames(60);
          const L = gvL, cv = document.getElementById("gvc"), out = {};
          const key = (k, t) => gvKey({key: k, target: t || cv});
          const walk = gvWalk(L);
          out.heavyFirst = walk.findIndex((n) => !gvHeavy(n)) >= walk.filter(gvHeavy).length - 0;
          out.order = walk.slice(0, 12).map((n) => [gvHeavy(n), n.mass, n.id]);
          out.sorted = walk.every((n, i) => !i || gvHeavy(walk[i - 1]) >= gvHeavy(n)
            && (gvHeavy(walk[i - 1]) !== gvHeavy(n) || walk[i - 1].mass >= n.mass));
          out.next = [key("."), L.kfocus]; out.first = walk[0].id;
          key("."); out.second = L.kfocus;
          out.prev = [key(","), L.kfocus]; key(","); out.wrapBack = L.kfocus === walk[walk.length - 1].id;
          out.live = document.getElementById("gvlive").textContent;
          out.other = key(".", document.body); out.pageKeys = key(".", {tagName: "INPUT"});
          L.kfocus = walk[2].id; gvDraw(L);
          // the ring: a dashed stroke in the ink colour
          const ctx = document.getElementById("gvc").getContext("2d"); out.focusRing = true;
          out.enter = [key("Enter"), GV.sel]; out.enterWrongTarget = key("Enter", document.body);
          const btn = document.getElementById("gvside").innerHTML;
          out.buttons = (btn.match(/<button class="gv-nb" data-gvid=/g) || []).length;
          out.li = btn.includes('<li data-gvid');
          // a rebuilt panel gives the focus back to its header
          const side = document.getElementById("gvside"); let focused = 0;
          document.activeElement = {}; side.contains = () => true;
          side.querySelector = () => ({focus: () => { focused++; }});
          gvSelect(L, L.byId.get("d:act")); out.refocus = focused;
          document.activeElement = null; focused = 0; gvSelect(L, L.byId.get("q:tone"));
          out.noSteal = focused;
          L.tf = {x: L.tf.x, y: L.tf.y, k: 6}; gvMeasure(L); L.kfocus = null; gvKey({key: ".", target: cv});
          out.flew = L.tw !== null;
          fire(cv, "pointerdown", {clientX: 5, clientY: 5}); out.cleared = L.kfocus;
          return out; })()""",  # noqa: E501
    )
    walk = shown["order"]
    assert shown["sorted"] is True
    assert [w[0] for w in walk] == sorted((w[0] for w in walk), reverse=True)
    assert shown["next"] == [True, shown["first"]]
    assert shown["prev"][0] is True
    assert shown["wrapBack"] is True
    assert "of" in shown["live"]
    assert shown["other"] is True
    assert shown["pageKeys"] is False
    assert shown["enter"] == [True, shown["enter"][1]]
    assert shown["enter"][1] is not None
    assert shown["enterWrongTarget"] is False
    assert shown["buttons"] > 0
    assert shown["li"] is False  # neighbours are real buttons
    assert shown["refocus"] == 1
    assert shown["noSteal"] == 0
    assert shown["flew"] is True  # a body out of view is flown to
    assert shown["cleared"] is None  # a pointer press ends the key walk


@needs_node
def test_the_gravity_focus_ring_is_a_dashed_ink_ring_on_the_focused_body(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(60);
          const L = gvL, out = {};
          """
        + _RECORDER
        + """
          const rings = () => { rec.ev.length = 0; gvDraw(L); const k = L.tf.k;
            return rec.ev.filter((e) => e.k === "stroke" && e.dash.length === 2
              && Math.abs(e.dash[0] - 4 / k) < 1e-9 && e.ss === L.T.fg).length; };
          out.before = rings(); L.kfocus = L.nodes[0].id; out.after = rings();
          return out; })()""",
    )
    assert shown["before"] == 0
    assert shown["after"] == 1


@needs_node
def test_the_gravity_panels_fold_and_a_phone_shows_one_sheet(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    narrow = html[html.index("@media (max-width: 760px) {\n  #gvbar") :]
    narrow = narrow[: narrow.index("\n}\n")]
    assert "#gvlegend { display: none" not in narrow  # a one-row key, not nothing
    assert "#gvlegend { left: 8px; right: 8px; bottom: 8px" in narrow
    assert '#vGravity[data-sel="1"] #gvov { display: none; }' in narrow
    assert '#vGravity:not([data-sel="1"]) #gvside { display: none; }' in narrow
    assert "max-height: 42%; overflow: auto" in narrow
    assert "#gvov { max-height:" in html  # capped on a wide screen too
    shown = _run(
        html,
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.sel = null; drawView(); frames(20);
          const out = {}, el = (id) => document.getElementById(id), L = gvL;
          out.wide = ["bar", "ov", "side", "leg"].map((k) => gvFoldOpen(k));
          out.wideClass = el("gvov").className;
          globalThis.matchMedia = () => ({matches: true});
          GV.fold = {}; gvOvPanel(L); gvLegend(L); gvSelect(L, L.byId.get("d:act"));
          out.narrow = ["bar", "ov", "side", "leg"].map((k) => gvFoldOpen(k));
          out.classes = ["gvov", "gvside", "gvlegend"].map((id) => el(id).className);
          out.sheet = [el("vGravity").dataset.sel];
          out.head = el("gvside").innerHTML.slice(0, 300);
          out.keyrow = el("gvlegend").innerHTML.includes("gv-keyrow");
          const btn = {setAttribute(k, v) { this[k] = v; }};
          gvFoldToggle(L, "side", btn); out.open = [GV.fold.side, el("gvside").className, btn["aria-expanded"]];
          gvFoldToggle(L, "side", btn); out.shut = [GV.fold.side, el("gvside").className];
          gvSelect(L, null); out.noSel = el("vGravity").dataset.sel;
          out.freeKeep = L.free.y1 > L.free.y0;
          delete globalThis.matchMedia;
          return out; })()""",  # noqa: E501
    )
    assert shown["wide"] == [True, True, True, True]
    assert shown["wideClass"] == "gv-box"
    assert shown["narrow"] == [False, False, False, False]  # folded by default under 760px
    assert shown["classes"] == ["gv-box folded"] * 3
    assert shown["sheet"] == ["1"]  # a selection: it is the one sheet
    assert "gv-fh" in shown["head"]
    assert "d:act" in shown["head"]
    assert shown["keyrow"] is True
    assert shown["open"] == [True, "gv-box", "true"]
    assert shown["shut"] == [False, "gv-box folded"]
    assert shown["noSel"] == ""
    assert shown["freeKeep"] is True


@needs_node
def test_a_gravity_label_stays_clear_of_the_toolbar(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(60);
          const L = gvL, out = {};
          """
        + _RECORDER
        + """
          L.free = {x0: 8, y0: 300, x1: L.W - 8, y1: L.H - 8};
          gvDraw(L);
          const ys = rec.ev.filter((e) => e.k === "fillText" && e.a.length === 3).map((e) => e.a[2] - 11);
          out.n = ys.length; out.top = Math.min(...ys);
          return out; })()""",  # noqa: E501
    )
    assert shown["n"] > 0
    assert shown["top"] >= 296


@needs_node
def test_the_gravity_legend_draws_a_glyph_per_kind_and_swatches_each_count(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; GV.byDomain = false; GV.ov = true;
          drawView(); frames(20);
          const L = gvL, out = {}, el = (id) => document.getElementById(id);
          out.legend = el("gvlegend").innerHTML;
          out.latent = L.links.some((l) => l.r === "latent");
          S.turn = 0; drawView(); frames(20);
          out.key = el("gvkey").innerHTML; out.kind = el("gvlegend").innerHTML;
          return out; })()""",
    )
    legend = shown["legend"]
    for label in ("target", "rule result", "question", "tool result", "t:state"):
        assert f"</svg>{label}</span>" in legend, label
    assert "<polygon" in legend  # the state's diamond
    assert "<rect" in legend  # the tool result's square
    assert "stroke-dasharray" in legend  # a dotted latent target and edge
    assert ("latent edge" in legend) == shown["latent"]
    assert legend.count("<svg") >= 5
    for gone in ("wheel zooms", "shift-", "drag to move", "tinted, ringed"):
        assert gone not in legend
    chips = shown["key"].split('<button class="gv-chipb"')[1:]
    assert chips
    for chip in chips:
        assert 'class="gv-sw"' in chip.split("</button>")[0]  # a swatch beside every count


@needs_node
def test_the_step_on_float_buffers_is_the_step_on_objects(tmp_path: Path) -> None:
    """gvStep's repulsion and collision read Float64Array scratch buffers but keep the pairs,
    their order and the arithmetic, so a settle lands on exactly the same positions as the
    plain-object step: 2D and 3D, all bodies and heavy only, with a body pinned."""
    shown = _run(
        page([], [DESIGN.document, load(showcase.PLAIN).document]),
        tmp_path,
        """(() => {
          const ref = (L, a) => {
            const N = L.act, n3 = L.mode3, D2 = 300 * 300, rnd = () => (L.rng() - .5) * 1e-3;
            for (let i = 0; i < N.length; i++) { const p = N[i];
              for (let j = i + 1; j < N.length; j++) { const q = N[j];
                let dx = q.x - p.x, dy = q.y - p.y, dz = n3 ? q.z - p.z : 0, l = dx * dx + dy * dy + dz * dz;
                if (l >= D2) continue;
                if (l === 0) { dx = rnd(); dy = rnd(); l = dx * dx + dy * dy; }
                if (l < 1) l = Math.sqrt(l);
                const wp = q.cs * a / l, wq = p.cs * a / l;
                p.vx += dx * wp; p.vy += dy * wp; p.vz += dz * wp;
                q.vx -= dx * wq; q.vy -= dy * wq; q.vz -= dz * wq; } }
            for (const p of L.qh) { const Q = p.q, Hv = p.h; const dx = Hv.x - Q.x, dy = Hv.y - Q.y;
              const dz = n3 ? Hv.z - Q.z : 0, d = Math.hypot(dx, dy, dz) || 1;
              const e = gvClamp(d - (Q.r + Hv.r + 6), -120, 120) * p.w, ux = dx / d, uy = dy / d, uz = dz / d;
              const c = a * .6 * e / Q.qp;
              Q.vx += ux * c; Q.vy += uy * c; Q.vz += uz * c;
              const k = a * L.qs * 2 * e / Hv.qn; Hv.vx -= ux * k; Hv.vy -= uy * k; Hv.vz -= uz * k; }
            for (const p of L.hh) { const A = p.a, B = p.b; const dx = B.x - A.x, dy = B.y - A.y;
              const dz = n3 ? B.z - A.z : 0, d = Math.hypot(dx, dy, dz) || 1;
              const e = gvClamp(d - (A.r + B.r + 36), -150, 150);
              const k = a * L.hs * 2 * e / Math.min(A.hhd, B.hhd), ux = dx / d, uy = dy / d, uz = dz / d;
              const s = A.mass + B.mass;
              A.vx += ux * k * B.mass / s; A.vy += uy * k * B.mass / s; A.vz += uz * k * B.mass / s;
              B.vx -= ux * k * A.mass / s; B.vy -= uy * k * A.mass / s; B.vz -= uz * k * A.mass / s; }
            for (let it = 0; it < 2; it++) for (let i = 0; i < N.length; i++) { const p = N[i];
              for (let j = i + 1; j < N.length; j++) { const q = N[j], rr = p.cr + q.cr;
                let dx = p.x + p.vx - q.x - q.vx, dy = p.y + p.vy - q.y - q.vy;
                let dz = n3 ? p.z + p.vz - q.z - q.vz : 0, l = dx * dx + dy * dy + dz * dz;
                if (l >= rr * rr) continue;
                if (l === 0) { dx = rnd(); dy = rnd(); l = dx * dx + dy * dy; }
                const d = Math.sqrt(l), f = (rr - d) / d * .9, mp = q.cr2 / (p.cr2 + q.cr2);
                p.vx += dx * f * mp; p.vy += dy * f * mp; p.vz += dz * f * mp;
                q.vx -= dx * f * (1 - mp); q.vy -= dy * f * (1 - mp); q.vz -= dz * f * (1 - mp); } }
            for (const n of N) {
              n.vx -= n.x * .02 * a; n.vy -= n.y * .02 * a; if (n3) n.vz -= n.z * .02 * a;
              if (n.fx != null) { n.x = n.fx; n.vx = 0; } else { n.vx *= .6; n.x += n.vx; }
              if (n.fy != null) { n.y = n.fy; n.vy = 0; } else { n.vy *= .6; n.y += n.vy; }
              if (!n3) { n.z = 0; n.vz = 0; } else if (n.fz != null) { n.z = n.fz; n.vz = 0; }
              else { n.vz *= .6; n.z += n.vz; } } };
          const real = gvStep, out = {};
          cur().designs.forEach((d, i) => { S.design = i;
            for (const [three, light, pin] of [[false, true, false], [true, true, false],
              [false, false, false], [false, true, true], [true, true, true]]) {
              const run = (step) => { gvStep = step; GVL.delete(curDesign());
                Object.assign(GV, {mode3: three, light, cross: false}); const L = gvGraph(curDesign());
                gvLayout(L);
                if (pin) { const n = L.nodes[Math.min(3, L.nodes.length - 1)];
                  n.fx = 40; n.fy = -25; if (three) n.fz = 10; gvSettle(L, 120, .5); }
                return L.nodes.map((n) => [n.x, n.y, n.z]); };
              const a = run(ref), b = run(real);
              let max = 0; a.forEach((p, k) => p.forEach((v, c) => { max = Math.max(max, Math.abs(v - b[k][c])); }));
              out[`${d.path}|${three}|${light}|${pin}`] = [max, a.length]; } });
          gvStep = real; GV.mode3 = false; GV.light = true; return out; })()""",  # noqa: E501
    )
    assert len(shown) == 10
    for key, (diff, count) in shown.items():
        assert count > 0
        assert diff == 0, key
