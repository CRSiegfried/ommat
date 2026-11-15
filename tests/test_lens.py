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
    """Turn, Overview, Targets in both modes, Outcomes for every table, and Explore from every node
    at every depth, in both directions with the details open, for each design alone and under
    each turn; and a design with no policy node, targets, tables or requirements."""
    html = page(turns, {**DESIGNS, "full.json": load(full()).document})
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
            for (const v of ["turn", "overview", "targets", "outcomes"]) {
              S.view = v; drawView(); n++; }
            S.tmode = "matrix"; S.view = "targets"; drawView(); S.tmode = null;
            const X = ixOf(curDesign()); if (!X) return;
            for (const t of X.tables) { S.table = t.id; S.view = "outcomes"; drawView(); n++; }
            S.view = "explore";
            for (const id of Object.keys(X.N)) { S.node = id;
              for (const d of [0, 1, 99]) for (const dir of ["v", "h"]) {
                Object.assign(EX, {up: d, down: d, dir, info: true}); drawView(); n++; } }
            EX.info = false; EX.mini = false; drawView(); EX.mini = true; };
          cur().designs.forEach((d, di) => { S.design = di; S.turn = null; S.node = null; all(); });
          cur().turns.forEach((t, ti) => { S.turn = ti; S.node = null; all(); });
          return {n, designs: cur().designs.map((d) => d.path)}; })()""",
    )
    assert "bare.json" in renders["designs"]
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
          return {hash, pushed: pushed.length, want, got: {view: S.view, node: S.node,
            turn: S.turn, up: EX.up, down: EX.down, dir: EX.dir, info: EX.info,
            table: S.table, tmode: S.tmode}, plain: (location.hash = "", readHash(), S.view) };
        })()""",
    )
    assert shown["got"] == shown["want"]
    assert shown["pushed"] == 1
    assert "view=explore" in shown["hash"]
    assert shown["plain"] == "turn"  # a set with records opens on the Turn tab


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


def test_the_page_is_the_same_bytes_for_the_same_documents(turns: dict[str, bytes]) -> None:
    """The page is a function of the documents: built twice it is the same bytes, and its
    script reads no clock and draws no random number."""
    first = page(turns, DESIGNS)
    assert page(turns, DESIGNS) == first
    script = first.split("<script>")[-1]
    for call in ("Math.random", "Date.now", "new Date", "performance.now", "crypto."):
        assert call not in script, call
