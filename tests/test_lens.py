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
def test_a_design_alone_opens_on_gravity_with_its_summary(tmp_path: Path) -> None:
    """A page given a design alone opens on Gravity; with nothing selected the panel names the
    design, counts its bodies by kind, names its oracles and lists the policy and the targets,
    each a button that selects its body; the parameters and the design's text are folded."""
    html = page([], [DESIGN.document])
    shown = _run(
        html,
        tmp_path,
        """(() => {
          const side = () => document.getElementById("gvside").innerHTML, out = {};
          out.view = view(); out.tabs = [document.getElementById("tab-gravity").hidden,
            document.getElementById("tab-turn").hidden];
          out.panel = side(); out.text = strip(side());
          document.getElementById("vGravity").listeners.click.forEach((f) => f({target:
            {closest: () => ({dataset: {gvid: "tg:urgent"}})}}));
          out.sel = GV.sel; out.after = strip(side());
          return out; })()""",
    )
    assert shown["view"] == "gravity"
    assert shown["tabs"] == [False, True]  # no record: no Turn tab
    text = shown["text"]
    assert text.startswith("design showcase")
    assert "1 target · 2 rule results · 2 questions · 1 tool result · t:state" in text
    assert "oracles llm, desk" in text
    assert 'data-gvid="d:act"' in shown["panel"]  # the policy
    assert 'data-gvid="tg:urgent"' in shown["panel"]
    assert "The mail needs an answer today" in text
    assert "<summary>parameters · 4</summary>" in shown["panel"]
    assert "<summary>the design as text</summary>" in shown["panel"]
    assert "Hover or click a body." in text
    assert shown["sel"] == "tg:urgent"
    assert shown["after"].startswith("tg:urgent")


@needs_node
def test_an_entry_is_marked_by_its_nearest_bound(tmp_path: Path, turns: dict[str, bytes]) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const refs = ["q:urgent", "q:tone", "t:state", "d:urgency", "d:act"], out = {};
          for (const p of ["decided.json", "undecided.json", "idia.json"]) {
            const t = cur().turns[at(p)];
            selectTurn(at(p));
            out[p] = { marks: Object.fromEntries(refs.filter((r) => t.idx.byRef.has(r))
                .map((r) => [r, dispState(t.idx.byRef.get(r))])),
              rings: Object.values(LAST_GRAVITY.rings) };
          }
          // The neutral mark is exactly a question's value whose paired design declares no bound.
          out.neutral = cur().turns.flatMap((t) => t.idx.all.filter((e) =>
            (dispState(e) === "nobounds") !== (e.holds === "value" && /^q:/.test(e.ref)
              && Array.isArray(e.bounds) && !e.bounds.length)).map((e) => t.path + " " + e.ref));
          out.unpaired = cur().turns.filter((t) => !t.design).flatMap((t) => t.idx.all
            .filter((e) => e.holds === "value" && dispState(e) !== "det").map((e) => e.ref));
          return out; })()""",
    )
    assert shown["undecided.json"]["marks"] == {
        "q:urgent": "short",
        "q:tone": "short",
        "t:state": "det",
        "d:urgency": "Unsure",
        "d:act": "Unsure",
    }
    assert shown["decided.json"]["marks"] == {
        "q:urgent": "value",
        "q:tone": "value",
        "t:state": "det",
        "d:urgency": "det",
        "d:act": "det",
    }
    assert shown["idia.json"]["marks"]["t:state"] == "det"  # bounds [] but not a question
    assert "--neutral" in shown["idia.json"]["rings"]  # a question with no bound declared
    assert "--good" in shown["decided.json"]["rings"]
    assert shown["neutral"] == []
    assert shown["unpaired"] == []


@needs_node
def test_declared_hops_are_dashed_and_never_the_root(tmp_path: Path) -> None:
    html = page({"plain.json": showcase.plain_turn().document}, [load(showcase.PLAIN).document])
    shown = _run(
        html,
        tmp_path,
        """(() => {
          const T = cur().turns[0], out = {};
          out.chain = mainChain(T.outline.lineage).map((h) => h.ref);
          out.precise = [...preciseRefs(T.outline.lineage)];
          selectTurn(0); out.lit = LAST_GRAVITY.lit;
          S.view = "turn"; drawView(); out.turn = strip(document.getElementById("vTurn").innerHTML);
          return out; })()""",
    )
    # Neither rule names what it read: the root is the policy's own entry, and every hop lit is a
    # declared one, dashed, off the lineage's precise refs.
    assert shown["chain"] == ["d:verdict"]
    assert shown["precise"] == ["d:verdict"]
    assert sorted(h[0] for h in shown["lit"]) == ["d:flag", "q:urgent", "t:state"]
    assert all(h[4] and not h[3] for h in shown["lit"])
    assert "3 declared reads, not a record of what was read" in shown["turn"]


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
          out.main = LAST_GRAVITY.lit.filter((h) => h[3]).map((h) => h[0]);
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
    assert sorted(shown["main"]) == ["d:urgency", "q:urgent"]  # Gravity lights it as the main hops


@needs_node
def test_the_all_turns_table_lists_the_set_in_time_order_and_steps_walk_it(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Turn's folded All turns table has a row per turn in the set's turn order, each with its
    decision, root and nearest bound; a row selects its turn; prev and next walk that order,
    wrapping, and from no turn start at an end; the fold keeps its state across a step."""
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const s = cur(), path = (i) => s.turns[i].path, out = {};
          out.order = s.order.map(path);
          S.view = "turn"; selectTurn(s.order[0]);
          const html = document.getElementById("vTurn").innerHTML;
          out.folded = html.includes('<details class="fold" id="allturns">');
          out.rows = [...html.matchAll(/<tr data-turn="(\\d+)"/g)].map((m) => path(+m[1]));
          out.text = strip(allTurnsHTML(true));
          viewsClick({target: {closest: () => ({dataset: {turn: String(s.order[2])}})}});
          out.clicked = path(S.turn);
          S.allTurns = true; step(1); out.next = path(S.turn);
          out.open = document.getElementById("vTurn").innerHTML.includes('id="allturns" open');
          selectTurn(s.order[s.order.length - 1]); step(1); out.wrapped = path(S.turn);
          step(-1); out.back = path(S.turn);
          S.turn = null; step(-1); out.fromNone = path(S.turn);
          S.turn = null; render(); out.pick = strip(document.getElementById("vTurn").innerHTML);
          return out; })()""",
    )
    order = shown["order"]
    assert shown["folded"] is True
    assert shown["rows"] == order
    assert shown["clicked"] == order[2]
    assert shown["next"] == order[3]
    assert shown["open"] is True
    assert shown["wrapped"] == order[0]
    assert shown["back"] == order[-1]
    assert shown["fromNone"] == order[-1]
    assert "q:urgent bar -0.05 (tied with 2 others)" in shown["text"]
    assert "q:triage/hazard" in shown["text"]  # the root column
    assert "no design" in shown["text"]  # a ledger turn no design pairs with
    assert shown["pick"].startswith("Pick a turnAll turns")


@needs_node
def test_the_nearest_bound_is_the_summary_s_nearest_column(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const near = (p) => p && `${p.distance} ${p.side} ${p.param}@${p.e.ref}`;
          return Object.fromEntries(cur().turns.map((t) =>
            [t.path, [t.outline.summary, near(t.sum.pick)]])); })()""",
    )
    for path, (summary, pick) in shown.items():
        nearest = summary.split("  ")[6]
        assert nearest.startswith("nearest "), path
        assert (pick or "-") == nearest.removeprefix("nearest "), path
    assert shown["undecided.json"][1] == "-0.05 short urgent/bar@q:urgent"


@needs_node
def test_a_record_with_no_paired_design_opens_on_turn_and_says_so(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A record no design pairs with opens on Turn, which says so plainly and lists its entries
    round by round; Gravity, asked for, says it has nothing to draw and points to Turn."""
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => {
          const out = {}, el = (id) => document.getElementById(id);
          const t = cur().turns[at("normal.json")];
          selectTurn(at("normal.json"));
          out.view = view(); out.turn = strip(el("vTurn").innerHTML);
          out.entries = t.outline.rounds.flatMap((r) => r.entries.map((e) => e.ref));
          S.view = "gravity"; render();
          out.gravity = [el("gvui").hidden, el("gvmsg").hidden, el("gvmsg").innerHTML];
          viewsClick({target: {closest: () => ({dataset: {viewGo: "turn"}})}, preventDefault() {}});
          out.back = view();
          selectTurn(at("decided.json")); S.view = null; render(); out.paired = view();
          return out; })()""",
    )
    assert shown["view"] == "turn"
    assert "No design in this set pairs with this record" in shown["turn"]
    assert "Round by round" in shown["turn"]
    for ref in shown["entries"]:
        assert ref in shown["turn"]
    hidden, message, html = shown["gravity"]
    assert hidden is True
    assert message is False
    assert "nothing to draw" in html
    assert 'data-view-go="turn"' in html
    assert shown["back"] == "turn"
    assert shown["paired"] == "gravity"


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
        """(() => { selectTurn(0); return { by: curTurn().pairBy, view: view(),
          notice: document.getElementById("notice").textContent,
          table: strip(allTurnsHTML(true)) }; })()""",
    )
    assert shown["by"] == "name"
    assert shown["view"] == "gravity"
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
    """Every design alone and every turn, on Turn and on Gravity with every body selected and
    its tooltip drawn, with the All turns table open, stepping through the set."""
    designs = {**DESIGNS, "full.json": load(full()).document}
    renders = _run(
        page(turns, designs),
        tmp_path,
        """(() => {
          let n = 0; S.allTurns = true;
          const each = () => {
            for (const v of ["turn", "gravity"]) { S.view = v; render(); n++; }
            if (!gvL) return;
            for (const b of gvL.nodes) {
              gvSelect(gvL, b); gvTipAt(gvL, b, {clientX: 5, clientY: 5}); n++; }
            gvSelect(gvL, null); };
          cur().designs.forEach((d, di) => { S.design = di; S.turn = null; each(); });
          cur().turns.forEach((t, ti) => { S.turn = ti; each(); });
          for (const d of [1, 1, -1]) { step(d); n++; }
          return n; })()""",
    )
    assert renders > 80


# ---------- the design views ----------


@needs_node
def test_every_view_draws_without_nan_or_undefined(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """Turn, and Gravity in 2D and 3D with every body selected and described, each round of the
    turn, and every control, for each design alone and under each turn; a design with no policy
    node, targets, tables or requirements; and a format-9 policy, which provides no ref."""
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
            S.view = "turn"; drawView(); n++;
            S.view = "gravity";
            for (const three of [false, true]) {
              GV.mode3 = three; drawView(); n++;
              const L = gvL; if (!L) return;
              for (const b of L.nodes) {
                gvSelect(L, b); gvTipAt(L, b, {clientX: 5, clientY: 5}); n++; }
              gvSelect(L, null); frames(80);
              for (let r = 0; r <= (L.ov ? L.ov.maxR : 0); r++) { gvSet("step", r); n++; }
              for (const [k, v] of [["lin", true], ["light", false], ["light", true], ["find", "u"],
                ["find", ""], ["kind", "rule"], ["kind", null], ["lin", false]]) { gvSet(k, v); n++; }
              for (const a of ["fit", "help", "help"]) gvAct(a);
              for (const k of ["s", "h", "f"]) gvKey({key: k, target: document.body});
              frames(80); gvLegend(L); }
            GV.mode3 = false; };
          cur().designs.forEach((d, di) => { S.design = di; S.turn = null; all(); });
          cur().turns.forEach((t, ti) => { S.turn = ti; all(); });
          return {n, designs: cur().designs.map((d) => d.path)}; })()""",  # noqa: E501
    )
    assert "bare.json" in renders["designs"]
    assert "f9.json" in renders["designs"]
    assert renders["n"] > 300


@needs_node
def test_a_view_s_hash_opens_it_again(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """The hash names the view only when it is not the default, the turn, or the design shown
    alone; Gravity's 3D, selection and overlay round-trip; a link from before the two views
    loads without error: its removed keys are ignored, its view falls back to the default, and
    a design node it names opens as that node's body."""
    shown = _run(
        page(
            {"undecided.json": turns["undecided.json"]},
            [DESIGN.document, (IDIA / "design.json").read_bytes()],
        ),
        tmp_path,
        """(() => {
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { location.hash = h; }};
          const out = {};
          S.turn = 0; S.view = null; render(); out.plain = location.hash;
          showView("turn"); out.turn = [location.hash, view()];
          S.view = null; readHash(); render(); out.turnBack = view();
          showView("gravity"); gvSelect(gvL, gvL.byId.get("d:urgency"));
          out.ghash = location.hash; gvSet("mode3", false); out.flat = location.hash;
          gvSet("mode3", true);
          GV.mode3 = false; GV.sel = null; S.view = null; readHash(); render();
          out.gravity = {view: view(), mode3: GV.mode3, sel: GV.sel, drawn: LAST_GRAVITY.mode3,
            picked: LAST_GRAVITY.sel};
          gvSet("mode3", false);
          S.turn = null; S.design = 1; GV.sel = null; render(); out.design = location.hash;
          S.design = 0; readHash(); out.designBack = S.design;
          const old = "#view=explore&node=rule:act&up=3&down=0&dir=h&info=1&table=rule:act"
            + "&tmode=matrix&sort=-nearest&und=1&state=1&tgt=tg:urgent&ghide=1&gdom=1&gx=1"
            + "&gdrag=move&turn=0";
          location.hash = old; GV.sel = null; readHash(); render();
          out.old = {view: view(), sel: GV.sel, turn: S.turn, hash: location.hash};
          location.hash = "#view=path&turn=9&set=nope&design=7"; readHash(); render();
          out.bad = {view: view(), turn: S.turn, design: S.design};
          return out; })()""",
    )
    assert shown["plain"] == "#turn=0"  # a set with a design opens on Gravity, its default
    assert "view=turn" in shown["turn"][0]
    assert shown["turn"][1] == "turn"
    assert shown["turnBack"] == "turn"
    assert "view=gravity" not in shown["ghash"]
    assert "g3" not in shown["ghash"]  # 3D is the default
    assert "g3=0" in shown["flat"]
    assert "gsel=d%3Aurgency" in shown["ghash"]
    assert shown["gravity"] == {
        "view": "gravity",
        "mode3": True,
        "sel": "d:urgency",
        "drawn": True,
        "picked": "d:urgency",
    }
    assert shown["design"] == "#design=1&g3=0"
    assert shown["designBack"] == 1
    old = shown["old"]
    assert old["view"] == "gravity"
    assert old["sel"] == "d:act"  # node=rule:act, as its body
    assert old["turn"] == 0
    for gone in (
        "node=",
        "up=",
        "dir=",
        "table=",
        "tmode=",
        "sort=",
        "und=",
        "state=",
        "tgt=",
        "ghide",
        "gdom",
        "gx=",
        "gdrag",
        "view=",
    ):
        assert gone not in old["hash"], gone
    assert shown["bad"] == {"view": "gravity", "turn": None, "design": 0}


@needs_node
def test_the_turn_view_says_what_took_place(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """A page given records and their design opens on Gravity at its first turn; Turn names the
    decision, the row that gave it with this turn's values, what was asked and what was not and
    why, and keeps the record's hashes and text in its details; it names no other tab."""
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => {
          const view0 = view(), first = S.turn, out = {view: view0, first};
          const at = (name) => cur().turns.findIndex((t) => t.path.endsWith(name));
          for (const name of ["decided.json", "undecided.json"]) {
            S.turn = at(name); S.view = "turn"; drawView();
            out[name] = document.getElementById("vTurn").innerHTML;
          }
          return out; })()""",
    )
    assert shown["view"] == "gravity"
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
    assert decided.count('class="hchip"') == 2  # the record's and the design's, in its details
    assert "<summary>The record as text</summary>" in decided
    assert 'id="allturns"' in decided
    assert "Undecided" in undecided
    assert "unsure: no clear answer (it abstained)" in undecided
    assert "falls short of the bar" in undecided
    assert "Its source requirement did not hold" in undecided
    assert "Row 4 gave the value." in undecided
    for gone in ("Graph", "Explore", "Path", "Outcomes", "Overview"):
        assert f"The {gone} tab" not in decided + undecided, gone


@needs_node
def test_a_ref_chip_on_turn_opens_gravity_with_its_body_selected(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Every ref chip on Turn carries its node, and its ref where it names one; a click opens
    Gravity with that body selected, as a history step, so back returns to Turn; a chip for a
    source selects its first question or tool result; in Gravity a chip selects in place."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const pushed = [], out = {}; globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { pushed.push(h); location.hash = h; }};
          const click = (d) => viewsClick({target: {closest: () => ({dataset: d})}});
          S.turn = 0; showView("turn");
          const html = document.getElementById("vTurn").innerHTML;
          out.chips = [...html.matchAll(/data-go="([^"]+)"(?: data-ref="([^"]+)")?/g)]
            .map((m) => [m[1], m[2] || null]);
          const before = location.hash;
          click({go: "rule:urgency", ref: "d:urgency"});
          out.go = {view: view(), sel: GV.sel, pushed: pushed.length, shown: LAST_GRAVITY.sel};
          location.hash = before; readHash(); render(); out.back = view();
          click({go: "source:desk"}); out.source = GV.sel;
          click({go: "rule:act", ref: "d:act.field"}); out.inPlace = [GV.sel, pushed.length];
          return out; })()""",
    )
    chips = shown["chips"]
    assert chips
    assert ["rule:urgency", "d:urgency"] in chips
    assert all(
        node in {"state", "source:ask", "rule:urgency", "source:desk", "rule:act"}
        for node, _ in chips
    )
    # one push for the tab switch to Turn, one for the chip
    assert shown["go"] == {"view": "gravity", "sel": "d:urgency", "pushed": 2, "shown": "d:urgency"}
    assert shown["back"] == "turn"
    assert shown["source"] == "t:ticket"  # desk's only body
    assert shown["inPlace"] == ["d:act", 3]  # already in Gravity: no further history step


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
  return out; })()"""


@needs_node
def test_the_gravity_layout_is_the_same_for_the_same_design(tmp_path: Path) -> None:
    """The layout is a function of the graph: two runs of the page place every body at the
    same point in 2D and in 3D."""
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
    assert "3 declared, dashed" in plain["panel"]


@needs_node
def test_the_gravity_gestures_follow_the_pointer(tmp_path: Path) -> None:
    """Wheel zoom in 2D keeps the point under the cursor; an empty-space drag pans by the
    pointer's travel and is not a click; a body drag pins it under the pointer, reheats the
    layout and lets go; in 3D a drag turns by its travel, the wheel keeps the orbit point's
    pointer, a shift-drag moves the orbit point; two fingers pinch the zoom."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(300);
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
          out.dragging = L.drag && L.drag.n.id; out.still = L.live;  // a press reheats nothing
          fire(cv, "pointermove", {clientX: sx + 120, clientY: sy + 60}); out.live = L.live;
          out.pinned = [n.fx - n.x - 120 / L.tf.k / L.cam.sp, n.fy - n.y - 60 / L.tf.k / L.cam.sp];
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
          const s0 = under(100, 100); fire(cv, "wheel", {clientX: 100, clientY: 100, deltaY: -400});
          out.spot = [s0, under(100, 100)];  // in 3D too the wheel zooms about the pointer
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
    assert shown["still"] is False
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
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(300);
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
    """t:state is a body of its own kind with edges to what reads it; a body's colour is its
    kind's, one hue per kind."""
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
          out.typeCols = Object.fromEntries(L.nodes.map((n) => [n.id, gvCol(L, n)]));
          out.legend = strip(document.getElementById("gvlegend").innerHTML);
          out.desc = strip(gvDescribe(L, st));
          out.stateCol = gvCol(L, st); out.kindCol = L.T.kind.state;
          gvSelect(L, st); out.sel = GV.sel;
          gvSet("find", "t:state"); out.hit = [...L.hits];
          return out; })()""",
    )
    assert shown["kind"] == "state"
    assert shown["label"] == "t:state"
    assert shown["heavy"]
    assert set(shown["edges"]) >= {"q:tone", "q:urgent"}
    cols = shown["typeCols"]
    assert cols["q:tone"] == cols["q:urgent"] != cols["d:urgency"]  # same kind, same colour
    assert cols["t:state"] != cols["q:tone"]
    assert len(set(cols.values())) == 5
    assert "state" in shown["legend"]
    assert "rule result" in shown["legend"]
    assert shown["sel"] == "t:state"
    assert shown["hit"] == ["t:state"]
    assert shown["stateCol"] == shown["kindCol"]
    assert "read by" in shown["desc"]


@needs_node
def test_the_gravity_overlay_shades_targets_by_the_round(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Under a turn a target is shaded from the entries that estimate it, by the round shown,
    and says so in its description; with no turn there is no overlay."""
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    shown = _run(
        html,
        tmp_path,
        """(() => { const out = {}, el = (id) => document.getElementById(id);
          S.view = "gravity"; S.turn = 0; GV.step = null; drawView();
          const L = gvL, tg = L.byId.get("tg:urgent");
          out.est = [...L.adj.get("tg:urgent")].map((i) => [i, L.byId.get(i).os]);
          out.ts = tg.ts; out.tstate = LAST_GRAVITY.tstate;
          out.tip = strip(gvDescribe(L, tg));
          out.key = strip(el("gvtkey").innerHTML);
          out.byRound = [];
          for (let r = 0; r <= L.ov.maxR; r++) { gvSet("step", r); out.byRound.push(tg.ts); }
          gvSet("step", null);
          S.turn = null; drawView();
          out.noTurn = {ov: !!gvL.ov, panel: el("gvov").hidden, ts: gvL.byId.get("tg:urgent").ts};
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
    assert "mixed" in shown["key"]  # the target counts, in their own row
    assert shown["byRound"][0] == "untested"  # round 0 reaches no estimating entry
    assert shown["byRound"][-1] == shown["ts"]
    assert shown["noTurn"] == {"ov": False, "panel": True, "ts": None}


@needs_node
def test_the_gravity_move_drag_slides_the_view_without_turning_it(tmp_path: Path) -> None:
    """In 3D a move drag (space held, the middle or right button, or shift) translates the
    camera's orbit point in the view plane: the view direction is unchanged, every body moves on
    screen by the pointer's travel at its depth, at any zoom, with no inertia; a plain drag
    still turns; a press on a body drags the body."""
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
          const n = L.byId.get("d:act"); gvProject(L);
          const sx = L.tf.x + L.tf.k * (L.W / 2 + n.px), sy = L.tf.y + L.tf.k * (L.H / 2 + n.py);
          fire(cv, "pointerdown", {clientX: sx, clientY: sy});
          out.bodyDrag = L.drag && L.drag.n.id; out.bodyRot = L.rot === null;
          fire(cv, "pointerup", {}); frames(300);
          return out; })()""",
    )
    for name in ("space", "middle", "shift", "close"):
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
    assert shown["bodyDrag"] == "d:act"
    assert shown["bodyRot"] is True


@needs_node
def test_the_gravity_round_keys_step_the_overlay(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """a and d step the round shown back and on, clamped at both ends with no wrap, only under a
    turn and not while typing or with a modifier; f, s, h, 3, / and the arrows stay."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.step = null; drawView();
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
          out.others = ["f", "s", "h", "3", "/"].map((k) => key(k)); key("3");
          S.turn = null; drawView(); out.off = key("d");
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
    assert shown["off"] is False  # nothing to step with no turn
    assert all(shown["others"])


# ---------- gravity: the view's chrome (fit, hash, toolbar, keys, touch, frames) ----------


@needs_node
def test_the_gravity_fit_fills_the_canvas_above_the_legend(tmp_path: Path) -> None:
    """The panels sit beside the canvas, so fit frames the bodies in all of it but the legend's
    strip, measured as it stands; the bodies fill it in one direction, spreading out rather
    than growing past a readable size; the spread keeps a dragged body under the pointer."""
    shown = _run(
        page([], [DESIGN.document, (IDIA / "design.json").read_bytes()]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(100);
          const el = (id) => document.getElementById(id), out = {};
          const fill = (L) => { gvProject(L);
            const sx = (n) => L.tf.x + L.tf.k * (L.W / 2 + n.px);
            const sy = (n) => L.tf.y + L.tf.k * (L.H / 2 + n.py), r = (n) => n.r * n.sc * L.tf.k;
            const lo = (f) => Math.min(...L.act.map(f)), hi = (f) => Math.max(...L.act.map(f));
            return [lo((n) => sx(n) - r(n)), hi((n) => sx(n) + r(n)),
              lo((n) => sy(n) - r(n)), hi((n) => sy(n) + r(n))]; };
          for (const d of [0, 1]) { S.design = d; drawView(); frames(100); const L = gvL;
            L.W = 1120; L.H = 600;
            el("gvlegend").getBoundingClientRect = () => ({left: 8, top: 560, width: 500,
              height: 32});
            out["free" + d] = gvFree(L);
            gvFit(L); frames(120);
            out["box" + d] = fill(L); out["k" + d] = L.tf.k; out["sp" + d] = L.cam.sp; }
          el("gvlegend").getBoundingClientRect = () => ({width: 0, height: 0});
          out.noLegend = gvFree(gvL);
          return out; })()""",
    )
    assert shown["free0"] == {"x0": 8, "y0": 8, "x1": 1112, "y1": 556}
    assert shown["noLegend"]["y1"] == 592
    for d in (0, 1):
        x0, x1, y0, y1 = shown[f"box{d}"]
        f = shown[f"free{d}"]
        assert f["x0"] <= x0 <= x1 <= f["x1"], d
        assert f["y0"] <= y0 <= y1 <= f["y1"], d
        filled = max((x1 - x0) / (f["x1"] - f["x0"]), (y1 - y0) / (f["y1"] - f["y0"]))
        assert filled > 0.85, d  # the bodies fill the canvas one way
        assert shown[f"k{d}"] <= 2.6 + 1e-9
    assert shown["sp0"] > 1  # a small design spreads out instead of growing


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
def test_the_gravity_link_carries_the_round_and_the_filters(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The round, lineage only, the questions shown and the selection are each written only when
    not their default, read back, and a restored selection is framed once."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { location.hash = h; }};
          S.view = "gravity"; S.turn = 0; GV.step = null; GV.lin = false; GV.light = true;
          GV.sel = null;
          render(); writeHash(); const out = {plain: location.hash};
          gvSet("step", 1); gvSet("lin", true); gvSet("light", false);
          gvSelect(gvL, gvL.byId.get("d:act"));
          out.hash = location.hash;
          Object.assign(GV, {step: null, lin: false, light: true, sel: null});
          let fits = 0; const fitSel = gvFitSel; gvFitSel = (L) => { fits++; return fitSel(L); };
          readHash(); render(); frames(50);
          out.back = {step: GV.step, shown: gvStepOf(gvL), lin: GV.lin, light: GV.light,
            sel: GV.sel, fits, boxes: [document.getElementById("gvq").checked]};
          render(); out.fits = fits;
          gvSet("step", null); gvSet("lin", false); gvSet("light", true); gvSelect(gvL, null);
          out.reset = location.hash;
          return out; })()""",
    )
    assert "gr=" not in shown["plain"]
    for key in ("gr=1", "glin=1", "gq=0", "gsel=d%3Aact"):
        assert key in shown["hash"], key
    back = shown["back"]
    assert back["step"] == 1
    assert back["shown"] == 1
    assert (back["lin"], back["light"], back["sel"]) == (True, False, "d:act")
    assert back["boxes"] == [False]
    assert back["fits"] == 0  # a restored selection is panned into view, not framed
    assert shown["fits"] == 0
    for key in ("gr=", "glin", "gq=", "gsel"):
        assert key not in shown["reset"], key


def test_the_gravity_toolbar_is_one_short_row_and_every_control_says_what_it_does() -> None:
    html = page([], [DESIGN.document])
    bar = html[html.index('id="gvbar"') + 10 : html.index('<div class="gv-stage">')]
    ids = re.findall(r'id="(gv[a-z0-9]+)"', bar)
    assert ids == ["gv3", "gvsearch", "gvhits", "gvstat", "gvq", "gvhelpb"]
    assert 'data-gvact="fit"' in bar
    assert "turn overlay" not in bar
    assert "lineage only" not in bar  # the key l, in the help
    for control in ("gv3", "gvsearch", "gvq", "gvhelpb"):
        start = bar.index(f'id="{control}"')
        tag = bar[bar.rindex("<", 0, start) : bar.index(">", start)]
        label = bar[bar.rindex("<label", 0, start) : start] if "<label" in bar[:start] else ""
        assert "title=" in tag or ("title=" in label and "</label>" not in label), control
    for key in ("(3)", "(f)", "(/)", "(?)"):
        assert key in bar, key
    for gone in (
        "cross-domain",
        "colour by domain",
        "rule and target edges",
        "relayout",
        "halo",
        "spin",
        "front",
        "orbit",
        "fit selection",
        "options",
        "hide what",
    ):
        assert gone not in bar, gone


@needs_node
def test_esc_clears_one_layer_a_press_and_never_leaves_the_turn(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The header has no key strip: the keys are the help's, opened by ?; Esc, pressed on the page,
    clears the help, the search, the selection, the pinned round, one a press, then stops: the
    turn stays. The menu's "design alone" leaves it for its own design; the arrows and j and k step
    turns."""
    html = page(
        {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
        [(IDIA / "design.json").read_bytes(), DESIGN.document],
    )
    assert 'id="keys"' not in html
    shown = _run(
        html,
        tmp_path,
        """(() => {
          const out = {};
          S.turn = cur().order[0]; S.view = null; GV.step = null; GV.sel = null; GV.find = "";
          GV.help = false; render();
          const L = gvL, key = (k) => fire(document, "keydown", {key: k, target: document.body});
          key("?"); out.help = GV.help;
          gvSet("find", "u"); gvSelect(L, L.byId.get("d:act")); gvSet("step", 1);
          out.esc = [];
          for (let i = 0; i < 5; i++) {
            key("Escape"); out.esc.push([GV.help, GV.find, GV.sel, GV.step, S.turn !== null]); }
          out.turn = S.turn;
          fire(document.getElementById("turnPick"), "change", {target: {value: ""}});
          out.left = S.turn; out.design = curDesign().name;
          const t0 = cur().order[0]; S.turn = t0; render();
          key("ArrowRight"); out.right = S.turn; key("k"); out.k = S.turn;
          return out; })()""",
    )
    assert shown["help"] is True
    assert shown["esc"] == [
        [False, "u", "d:act", 1, True],  # the help first
        [False, "", "d:act", 1, True],  # then the search
        [False, "", None, 1, True],  # then the selection
        [False, "", None, None, True],  # then the round pinned
        [False, "", None, None, True],  # nothing left: the turn stays
    ]
    assert shown["turn"] is not None
    assert shown["left"] is None
    assert shown["design"] == "showcase"  # the turn's own design, not the first in the set
    assert shown["right"] != shown["k"]
    assert "Esc</kbd> clears one thing at a time" in html
    assert "It never leaves the turn" in html


def test_the_gravity_help_holds_the_gestures_with_touch_wording() -> None:
    html = page([], [DESIGN.document])
    helpbox = html[html.index('id="gvhelp"') : html.index('id="gvov"')]
    assert "Trackpad and touch" in helpbox
    assert "A two-finger scroll or a pinch zooms; drag to pan or turn." in helpbox
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
          gvPanel(L);
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
    assert "Tap a body." in shown["touchHint"]
    assert "Hover or click a body." in shown["mouseHint"]
    assert shown["long"] is True


@needs_node
def test_the_gravity_pointer_handlers_ask_for_a_frame(tmp_path: Path) -> None:
    """A pointer move queues a frame instead of drawing; the tooltip is rebuilt only when the
    body under the pointer changes."""
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = null; GV.mode3 = false; drawView(); frames(100);
          const L = gvL, cv = document.getElementById("gvc"), out = {};
          let draws = 0, tips = 0;
          const d0 = gvDraw, t0 = gvTipHtml;
          gvDraw = (l) => { draws++; return d0(l); };
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
          return out; })()""",
    )
    assert shown["syncDraws"] == 0
    assert shown["afterFrame"] >= 1
    assert shown["tips"] == 1


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
            gvTsRing(cx, T, {ts, tc: {held, open, hc, oc}}, 0, 0, 10); return {arcs, col: arcs.col}; };
          let hc = "good", oc = "bad";
          out.answered = ring("answered", 3, 0); out.unresolved = ring("unresolved", 0, 2);
          out.mixed = ring("mixed", 1, 3); out.tick = GV_RING.value[2];
          hc = "neutral"; oc = "neutral"; out.grey = [ring("answered", 2, 0).col, ring("unresolved", 0, 1).col];
          out.shape = GV_TS_SHAPE;
          return out; })()""",  # noqa: E501
    )
    tau = 6.283185307179586
    assert shown["answered"]["arcs"] == [[13, 0, pytest.approx(tau)]]
    assert shown["answered"]["col"] == ["G"]
    radii = [a[0] for a in shown["unresolved"]["arcs"]]
    assert radii == [13, pytest.approx(15.5)]  # a double ring
    assert shown["unresolved"]["col"] == ["B", "B"]
    first, second = shown["mixed"]["arcs"]
    assert first[0] == second[0] == 13
    assert first[2] - first[1] == pytest.approx(tau * 0.25)  # held 1 of 4
    assert second[2] - second[1] == pytest.approx(tau * 0.75)
    assert second[1] == pytest.approx(first[2])
    assert shown["mixed"]["col"] == ["G", "B"]
    assert shown["grey"] == [["N"], ["N", "N"]]  # determined with no bound; Unasked
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
          S.view = "gravity"; S.design = 1; S.turn = 0; GV.mode3 = false;
          GV.step = null;
          drawView(); frames(100); const L = gvL, out = {};
          """
        + _RECORDER
        + """
          gvDraw(L);
          const k = L.tf.k, T = L.T;
          const strokes = rec.ev.filter((e) => e.k === "stroke");
          out.declared = strokes.filter((e) => e.dash.length === 2 && e.dash[0] > 4 / k)
            .map((e) => [e.al, e.lw * k]);  // in screen pixels
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
def test_every_body_is_labelled_when_there_is_room_and_the_focus_wins(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """At the fit every shown body gets a label, targets and questions included, and no label
    overprints a body, a ring or another label; with a body selected its label and its
    neighbours' come first, the others after them, faint."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.mode3 = false; GV.sel = null; drawView(); frames(100);
          const L = gvL, out = {};
          out.shown = L.nodes.filter((n) => n.va > .3 && gvOvA(L, n) >= .3).map((n) => n.id).sort();
          out.labels = LAST_GRAVITY.labels.slice().sort();
          gvSelect(L, L.byId.get("tg:urgent")); frames(100); gvDraw(L);
          out.near = ["tg:urgent", ...L.adj.get("tg:urgent")];
          out.order = L.labels.filter((id) => !L.byId.get(id).policy).map((id) => out.near.includes(id));
          return out; })()""",  # noqa: E501
    )
    assert "tg:urgent" in shown["labels"]
    assert set(shown["shown"]) <= set(shown["labels"])
    order = shown["order"]
    assert order[0] is True  # the selection's own label first
    assert order == sorted(order, reverse=True)  # its neighbours' before every other


@needs_node
def test_the_overlay_keeps_each_kind_s_fill_and_draws_flat_with_inset_rims(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Under a turn's overlay a body keeps its kind's fill, flat (no sprite), with its rim just
    inside the body so the state ring stands clear of it; the state is a block with an inked
    left bar in both modes; the help says how a turn reads."""
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    shown = _run(
        html,
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; GV.mode3 = false; drawView(); frames(100);
          const L = gvL, out = {}, cv = document.getElementById("gvc").getContext("2d");
          """
        + _RECORDER
        + """
          const T = L.T;
          out.kinds = Object.fromEntries(["target", "rule", "question", "tool", "state"]
            .map((k) => [k, L.nodes.filter((n) => n.kind === k).every((n) => gvCol(L, n) === T.kind[k])]));
          out.distinct = new Set(L.nodes.map((n) => gvCol(L, n))).size;
          L.tf = {x: L.tf.x, y: L.tf.y, k: 4}; cv.counts.drawImage = 0;
          rec.ev.length = 0; gvDraw(L);
          out.sprites = rec.ev.filter((e) => e.k === "drawImage").length;
          const arcs = (col, lw) => { const r = [];
            rec.ev.forEach((e, i) => { if (e.k === "stroke" && e.ss === col && Math.abs(e.lw - lw) < 1e-9) {
              for (let j = i - 1; j >= 0; j--) if (rec.ev[j].k === "arc") { r.push(rec.ev[j].a[2]); break; } } });
            return r; };
          const radius = (kind, d) => L.nodes.filter((n) => n.kind === kind && n.va > .5)
            .map((n) => n.r * n.sc * n.va - d);
          out.ruleRims = arcs(T.kind.rule, 2.2); out.ruleWant = radius("rule", 1.1);
          out.qRims = arcs(T.kind.question, 1.8); out.qWant = radius("question", .9);
          out.bar = rec.ev.filter((e) => e.k === "fillRect" && e.fs === T.fg).length;
          out.diamond = rec.ev.filter((e) => e.k === "closePath" && e.fs === T.kind.state).length;
          gvSet("ov", false);
          rec.ev.length = 0; gvDraw(L); out.barOff = rec.ev.filter((e) => e.k === "fillRect" && e.fs === T.fg).length;
          return out; })()""",  # noqa: E501
    )
    assert all(shown["kinds"].values())
    assert shown["distinct"] == 5
    assert shown["sprites"] > 0  # the shaded spheres stay under a turn
    assert shown["ruleRims"]
    for got in shown["ruleRims"]:
        assert any(abs(got - w) < 1e-9 for w in shown["ruleWant"])  # centred at r - 1.1
    assert shown["qRims"]
    for got in shown["qRims"]:
        assert any(abs(got - w) < 1e-9 for w in shown["qWant"])  # at r - 0.9
    assert shown["bar"] >= 1  # the state's inked left bar
    assert shown["diamond"] == 0
    assert shown["barOff"] >= 1  # the block is the state's look off the overlay too
    helpbox = html[html.index('id="gvhelp"') : html.index('<aside id="gvright"')]
    assert "A body keeps its kind fill; its ring is its entry" in helpbox


# ---------- gravity: navigation (links to the other tabs, search, the turn panel) ----------


@needs_node
def test_the_gravity_panel_says_what_the_design_and_the_turn_say_of_each_kind(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """The detail panel is the one place to learn about a body. A table rule shows its rows: as
    the turn ran them, the row that gave the value lit and what each test held, else as the
    design declares them; a rule the parameters it uses with their values and, under a turn, the
    bounds it declares with how far each answer lies from them. A question shows its wording,
    when its source runs, what the turn did and the answer whole; a tool result its requirement
    and the not-asked line; a target its statement, values and estimators; the state, under a
    turn, what it was given. The neighbour lists are open; no button leads to another tab."""
    html = page(
        {"undecided.json": turns["undecided.json"], "decided.json": turns["decided.json"]},
        [DESIGN.document],
    )
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        if doc["path"] == "decided.json":  # planted: the panel draws the outline's tests
            doc["outline"]["review"]["tables"][0]["rows"][0]["tests"][0]["cell"] = "planted"
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => {
          const out = {}, side = () => document.getElementById("gvside").innerHTML;
          const panel = (id) => { gvSelect(gvL, gvL.byId.get(id)); return side(); };
          S.view = "gravity";
          for (const [name, turn] of [["none", null], ["undecided", at("undecided.json")],
            ["decided", at("decided.json")]]) {
            S.turn = turn; GV.sel = null; render(); frames(20);
            out[name] = Object.fromEntries(["d:act", "d:urgency", "q:tone", "q:urgent", "t:ticket",
              "tg:urgent", "t:state"].map((id) => [id, panel(id)])); }
          return out; })()""",
    )
    none, und, dec = shown["none"], shown["undecided"], shown["decided"]

    def text(h: str) -> str:
        return re.sub(r"<[^>]*>", "", h).replace("&gt;", ">")

    # a table rule: its rows as declared, then as the turn ran them
    assert '<div class="tblwrap"><table>' in none["d:act"]
    assert "its rowsThe first row that matches gives the value." in text(none["d:act"])
    assert "Row 4 gave the value." in text(und["d:act"])
    assert 'class="rr chosen open"' in und["d:act"]  # an open value is not painted green
    assert "row 4 → Unsure matched: its output is the value" in text(und["d:act"])
    assert und["d:act"].count('<details class="rr not-matched">') == 3  # the other rows, folded
    # a marker is said whole once, then in a word
    assert text(und["d:act"]).count("no clear answer (it abstained)") == 1
    assert "d:urgency is planted" in text(dec["d:act"])  # the outline's tests, not derived
    # its parameters, and under a turn the bounds it declares
    assert "tone/bar = 0.5" in text(none["d:act"])
    assert "q:tone top probability 0.45, short of bar 0.5 by 0.05" in text(und["d:act"])
    assert "urgent/bar = 0.6" in text(und["d:urgency"])
    # a question: its wording, when it runs, what the turn did, its answer whole
    for h in (none["q:tone"], und["q:tone"]):
        assert "Which tone is the mail written in?" in text(h)
        assert "no source requirement" in text(h)
    assert "asked in round 1" in text(und["q:tone"])
    assert '<span class="dist">' in und["q:tone"]  # the whole distribution, a visible list
    assert "warm 0.45" in text(und["q:tone"])
    assert "asked of llm · choice" in text(none["q:tone"])
    assert "act: top probability 0.45, short of bar 0.5 by 0.05" in text(und["q:tone"])
    assert "this turn" not in text(none["q:tone"])
    # a tool result: its requirement, and why it was not called
    assert "when d:urgency is yes" in text(none["t:ticket"])
    assert "not called: its source requirement did not hold." in text(und["t:ticket"])
    assert "desk runs when d:urgency is yes; this turn d:urgency was" in text(und["t:ticket"])
    assert "called in round 2, because d:urgency was" in text(dec["t:ticket"])
    # a target and the state
    assert "The mail needs an answer today" in text(none["tg:urgent"])
    assert "values: yes or no" in text(none["tg:urgent"])
    assert "estimated by 1 rule" in text(none["tg:urgent"])
    assert "derived" in text(und["tg:urgent"])
    assert "The turn was given" in text(und["t:state"])
    assert "The turn was given" not in text(none["t:state"])
    for panels in shown.values():
        for h in panels.values():
            assert "<details" not in h.split('class="gv-k"')[-1]  # the neighbours are not folded
            for gone in ("data-gvpath", "data-gvtgt", "in Explore", "in Path", "in Targets"):
                assert gone not in h, gone


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
def test_the_gravity_turn_panel_steps_rounds_turns_and_states(
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
          S.view = "gravity"; S.turn = at("undecided.json"); GV.step = null; GV.st = null;
          GV.sel = null; drawView(); frames(30);
          const L = gvL; out.maxR = L.ov.maxR;
          out.head = el("gvov").innerHTML; out.roundLine = el("gvroundd").textContent;
          const key = (k) => gvKey({key: k, target: document.body});
          key("a"); out.prev = [GV.step, el("gvroundd").textContent];
          for (let i = 0; i < 5; i++) key("a"); out.floor = GV.step;
          for (let i = 0; i < 9; i++) key("d"); out.ceil = GV.step;
          const click = (d) => $("vGravity").listeners.click.forEach((f) => f({target:
            {closest: () => ({dataset: d})}}));
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
    assert 'data-gvact="turn-prev"' not in shown["head"]  # the header and the arrows step turns
    assert 'data-gvact="prev-round"' not in shown["head"]  # a and d step rounds
    assert 'id="gvround"' in shown["head"]
    assert 'data-view-go="turn"' in shown["head"]
    assert "why → Turn" in shown["head"]
    assert 'class="hchip"' not in shown["head"]  # the hashes live in Turn's record details
    assert 'id="gvkey"' in shown["head"]  # the entry counts are its filter chips
    assert 'root <button class="ref' in shown["head"]
    assert shown["roundLine"] == "round 2: not asked desk"
    assert shown["prev"][0] == shown["maxR"] - 1
    assert shown["prev"][1].startswith("round 1: ")
    assert shown["floor"] == 0  # clamped
    assert shown["ceil"] == shown["maxR"]
    assert shown["pressed"] == [shown["st"], True]
    assert shown["dim"][0] == 1
    assert shown["dim"][1] < 0.2
    assert f"gst={shown['st']}" in shown["hash"]
    assert shown["cleared"] == [None, False]
    assert shown["esc"] == [True, None]  # Esc clears the state filter
    assert "gst" not in shown["hash2"]


@needs_node
def test_a_turn_step_keeps_the_round_when_valid(tmp_path: Path, turns: dict[str, bytes]) -> None:
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
          step(1); out.next = [S.turn, GV.step, gvStepOf(gvL), gvL.ov.maxR];
          S.turn = 0; GV.step = null; drawView(); gvSet("step", 0); step(1);
          out.zero = [S.turn, GV.step];
          S.turn = 0; GV.step = null; drawView(); step(1); out.last = [S.turn, GV.step];
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
    assert 'id="gvside"></div>' in html  # not a live region: one polite line speaks instead
    assert 'id="gvlive" class="gv-sr" role="status"' in html
    assert 'id="gvroundl"></span>' in html  # play's ticks are not announced
    assert html.count("aria-live=") == html.count('role="status"') == 1  # one live region
    assert "l shows the lineage only" in canvas
    # the focus order: the toolbar, the canvas, the legend, then the panels
    order = [html.index(f'id="{i}"') for i in ("gvbar", "gvc", "gvlegend", "gvov", "gvside")]
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
          gvSelect(L, L.byId.get("d:act")); const btn = document.getElementById("gvside").innerHTML;
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


def test_the_gravity_panels_sit_beside_the_canvas_and_under_it_when_narrow() -> None:
    """The turn and detail panels are docked beside the canvas, never over it, and stack under
    it on a narrow screen; nothing folds them; the view takes the height the header leaves."""
    html = page([], [DESIGN.document])
    assert "#gvui { display: grid; grid-template-columns: minmax(0, 1fr) 360px;" in html
    assert html.index('<aside id="gvright"') > html.index('<div class="gv-stage">')
    narrow = html[html.index("@media (max-width: 860px) {") :]
    narrow = narrow[: narrow.index("\n}\n")]
    assert "#gvui { grid-template-columns: minmax(0, 1fr); height: auto; }" in narrow
    assert ".gv-stage { flex: none; height: min(62vh, 100vw);" in narrow
    assert ".wrap.full { display: flex; flex-direction: column; height: 100vh;" in html
    assert "data-gvfold" not in html
    assert "gv-fh" not in html


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
          S.view = "gravity"; S.turn = null; GV.mode3 = false; GV.byDomain = false;
          drawView(); frames(20);
          const L = gvL, out = {}, el = (id) => document.getElementById(id);
          out.legend = el("gvlegend").innerHTML;
          out.latent = L.links.some((l) => l.r === "latent");
          S.turn = 0; drawView(); frames(20);
          out.key = el("gvkey").innerHTML; out.kind = el("gvlegend").innerHTML;
          return out; })()""",
    )
    legend = shown["legend"]
    for label in ("target", "rule result", "question", "tool result", "state"):
        assert f"</svg>{label}</button>" in legend, label
    assert "<polygon" not in legend  # the state is a block with an inked bar, not a diamond
    assert "latent target" not in legend  # no body is drawn that way
    assert "<rect" in legend  # the tool result's square
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
              const k = a * L.hs * 2 * e * p.w / Math.min(A.hhd, B.hhd), ux = dx / d, uy = dy / d;
              const uz = dz / d;
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
              n.vx -= (n.x - n.gx) * GV_LEAN * a; n.vy -= n.y * .02 * a; if (n3) n.vz -= n.z * .02 * a;
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


# ---------- hash chips (the record's and the design's ID, copied whole) ----------


@needs_node
def test_a_hash_chip_copies_the_full_hash_and_says_so(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const out = {}, el = (id) => document.getElementById(id);
          S.view = "gravity"; S.turn = 0; drawView(); frames(20);
          const T = curTurn().outline; out.rec = T.record_hash; out.des = T.design_hash;
          S.view = "turn"; drawView(); out.panel = el("vTurn").innerHTML;
          out.chip = hashChip(T.record_hash, "record", "record");
          out.none = [hashChip("", "x"), hashChip(null, "x"), hashChip(7, "x"), hashChip(undefined, "x")];
          const timers = []; globalThis.setTimeout = (f) => { timers.push(f); return timers.length; };
          globalThis.clearTimeout = () => {};
          const state = { dataset: {copy: T.record_hash, what: "record", state: ""}, _s: {textContent: ""},
            querySelector() { return this._s; } };
          // 1: the clipboard takes it
          const sent = []; globalThis.navigator = {clipboard: {writeText: (t) => { sent.push(t);
            return {then: (ok) => ok()}; }}};
          copyHash(state);
          out.clip = {sent, state: state.dataset.state, label: state._s.textContent,
            live: el("gvlive").textContent, timers: timers.length};
          timers[0](); out.reset = [state.dataset.state, state._s.textContent];
          // 2: the clipboard refuses, the textarea works
          timers.length = 0; const copied = [];
          globalThis.navigator = {clipboard: {writeText: () => ({then: (ok, no) => no()})}};
          document.execCommand = (c) => { copied.push(c); return true; };
          const ta = []; const ce = document.createElement;
          document.createElement = (t) => { const e = ce(t); if (t === "textarea") ta.push(e); return e; };
          document.body = document.body || {appendChild() {}}; document.body.appendChild = () => {};
          copyHash(state);
          out.fallback = {copied, state: state.dataset.state, area: ta.length ? ta[0].value : null,
            live: el("gvlive").textContent};
          // 3: nothing works: the full hash is left to select
          timers.length = 0; delete globalThis.navigator; document.execCommand = () => false;
          copyHash(state);
          out.failed = {state: state.dataset.state, label: state._s.textContent,
            live: el("gvlive").textContent};
          return out; })()""",  # noqa: E501
    )
    rec, des = shown["rec"], shown["des"]
    assert isinstance(rec, str)
    assert len(rec) >= 64
    assert f'data-copy="{rec}"' in shown["chip"]
    assert f"click to copy the full hash&#10;{rec}" in shown["chip"]
    assert 'class="hc-check"' in shown["chip"]
    assert "<button" in shown["chip"]
    assert shown["panel"].count('class="hchip"') == 2
    assert f'data-copy="{rec}"' in shown["panel"]
    assert f'data-copy="{des}"' in shown["panel"]
    assert shown["none"] == ["", "", "", ""]
    assert shown["clip"]["sent"] == [rec]  # the full hash, not its 12 characters
    assert shown["clip"]["state"] == "copied"
    assert shown["clip"]["label"] == "copied"
    assert shown["clip"]["live"] == "record copied"
    assert shown["clip"]["timers"] == 1
    assert shown["reset"] == ["", ""]
    assert shown["fallback"]["copied"] == ["copy"]
    assert shown["fallback"]["area"] == rec
    assert shown["fallback"]["state"] == "copied"
    assert shown["failed"]["state"] == "failed"
    assert shown["failed"]["label"] == "select to copy"
    assert "selected" in shown["failed"]["live"]


@needs_node
def test_a_record_without_a_hash_has_no_chip(tmp_path: Path, turns: dict[str, bytes]) -> None:
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        if "record_hash" in doc["outline"]:
            doc["outline"].pop("record_hash")
            doc["outline"].pop("design_hash", None)
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { S.view = "turn"; S.turn = 0; drawView();
          return document.getElementById("vTurn").innerHTML; })()""",
    )
    assert 'class="hchip"' not in shown
    assert "no hash" in shown


def test_the_hash_chip_is_styled_as_a_chip_with_hover_and_focus() -> None:
    html = page([], [DESIGN.document])
    assert ".hchip { display: inline-flex" in html
    assert "cursor: pointer" in html[html.index(".hchip {") : html.index(".hchip:hover")]
    assert ".hchip:hover, .hchip:focus-visible" in html
    assert '.hchip[data-state="copied"]' in html
    assert 'id="copylive"' not in html  # one live region speaks for the page


# ---------- gravity: the toolbar explains itself and dims what does nothing ----------

_TOOLS = ("gvcross", "gvkind", "gvflat", "gvhalo", "gvspin", "gvqpull", "gvfitsel")


# ---------- gravity: kind chips ----------


@needs_node
def test_the_legend_kinds_are_filters_that_combine_with_the_state_chips(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Pressing a kind dims every other kind, with or without a turn; with a state chip pressed
    too a body must match both; Esc clears the state, then the kind; each chip is a button with
    aria-pressed and a title; a kind hidden by the toolbar has no chip."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const out = {}, el = (id) => document.getElementById(id);
          const click = (d, ok) => el("vGravity").listeners.click.forEach((f) => f({target:
            {closest: () => ({dataset: d, getAttribute: (k) => (k === "aria-disabled" ? ok : null)})}}));
          const html = () => el("gvlegend").innerHTML;
          S.view = "gravity"; S.turn = null; GV.kind = null; GV.st = null; GV.sel = null; GV.step = null;
          GV.light = true; drawView(); frames(20);
          const L = gvL, a = (id) => gvOvA(L, L.byId.get(id));
          out.chips = (html().match(/data-gvkind="/g) || []).length;
          out.pressed0 = (html().match(/aria-pressed="true"/g) || []).length;
          out.tip = html().includes('title="Dim every body that is not rule results');
          // no turn: the kind alone
          click({gvkind: "rule"});
          out.noTurn = [GV.kind, a("d:act"), a("q:tone"), a("tg:urgent"), a("t:state")];
          out.pressed = html().match(/data-gvkind="rule" aria-pressed="(\\w+)"/)[1];
          out.otherPressed = html().match(/data-gvkind="question" aria-pressed="(\\w+)"/)[1];
          out.hash = (globalThis.location = {hash: ""}, globalThis.history = {replaceState: (x, y, h) =>
            { location.hash = h; }}, writeHash(), location.hash);
          click({gvkind: "rule"}); out.cleared = [GV.kind, a("q:tone")]; writeHash(); out.hash0 = location.hash;
          // a turn and a state chip: both must match
          S.turn = 0; drawView(); frames(20);
          const L2 = gvL, b = (id) => gvOvA(L2, L2.byId.get(id));
          const st = ["Unsure", "Unavailable", "Unasked", "short", "value"].find((s) => L2.nodes.some((n) => n.os === s));
          const rule = L2.nodes.find((n) => n.kind === "rule" && n.os === st);
          const ruleOther = L2.nodes.find((n) => n.kind === "rule" && n.os !== st && n.os !== "later" && n.os !== "none");
          const other = L2.nodes.find((n) => n.kind === "question" && n.os === st);
          const neither = L2.nodes.find((n) => n.kind === "question" && n.os !== st);
          out.st = st; gvSet("st", st); gvSet("kind", "rule");
          out.both = [rule && b(rule.id), ruleOther && b(ruleOther.id), other && b(other.id), neither && b(neither.id)];
          // Esc: the state first, then the kind
          const key = (k) => gvKey({key: k, target: document.body});
          out.esc = [key("Escape"), GV.st, GV.kind]; out.esc2 = [key("Escape"), GV.st, GV.kind];
          out.esc3 = key("Escape");
          // the questions hidden: their chips leave the legend
          gvSet("light", false);
          out.hidden = [html().includes('data-gvkind="question"'), html().includes('data-gvkind="tool"'),
            html().includes('data-gvkind="rule"')];
          gvSet("light", true);
          // from a link
          Object.assign(GV, {kind: null}); globalThis.location.hash = "#view=gravity&gk=tool"; readHash();
          out.read = GV.kind; globalThis.location.hash = "#view=gravity&gk=nonsense"; readHash(); out.bad = GV.kind;
          return out; })()""",  # noqa: E501
    )
    assert shown["chips"] == 5  # target, rule result, question, tool result and state
    assert shown["pressed0"] == 0
    assert shown["tip"] is True
    n = shown["noTurn"]
    assert n[0] == "rule"
    assert n[1] == 1  # a rule result shows in full
    assert n[2] == pytest.approx(0.12)  # a question is dimmed
    assert n[3] == pytest.approx(0.12)
    assert shown["pressed"] == "true"
    assert shown["otherPressed"] == "false"
    assert "gk=rule" in shown["hash"]
    assert shown["cleared"] == [None, 1]  # pressed again, cleared
    assert "gk" not in shown["hash0"]  # written only when set
    inside, outside, other, neither = shown["both"]
    assert inside == 1  # a rule result in the state
    if outside is not None:
        assert outside < 0.2
    if other is not None:
        assert other < 0.2
    if neither is not None:
        assert neither == pytest.approx(0.0144, abs=0.02)  # misses both
    assert shown["esc"] == [True, None, "rule"]  # the state goes first
    assert shown["esc2"] == [True, None, None]  # then the kind
    assert shown["esc3"] is False  # nothing left: the page's own back-out
    assert shown["hidden"] == [False, False, True]
    assert shown["read"] == "tool"
    assert shown["bad"] is None


@needs_node
def test_a_kind_with_no_body_has_no_chip_and_a_stale_link_is_dropped(tmp_path: Path) -> None:
    html = page([], [DESIGN.document])
    data = _data(html)
    outline = data["sets"][0]["docs"][0]["outline"]
    for node in outline["nodes"]:
        if node["kind"] == "source":
            node["tool_results"] = []
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => {
          globalThis.location = {hash: "#view=gravity&gk=tool"}; globalThis.history = {replaceState() {}};
          readHash(); const asked = GV.kind; drawView(); frames(10);
          return {asked, kept: GV.kind, legend: document.getElementById("gvlegend").innerHTML}; })()""",  # noqa: E501
    )
    assert shown["asked"] == "tool"
    assert shown["kept"] is None  # no body of that kind in this design
    assert 'data-gvkind="tool"' not in shown["legend"]
    assert 'data-gvkind="rule"' in shown["legend"]


def test_the_help_and_the_chips_are_keyboard_reachable() -> None:
    html = page([], [DESIGN.document])
    helpbox = html[html.index('id="gvhelp"') : html.index('<aside id="gvright"')]
    assert "legend's kinds" in helpbox
    assert "the state, target and kind filters" in helpbox
    assert ".gv-chipb[aria-pressed=" in html


@needs_node
def test_a_body_must_match_both_the_kind_and_the_state_chip(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => {
          const L = {ov: {linN: new Set()}}, body = (kind, os) => ({kind, os, id: kind + os});
          Object.assign(GV, {kind: "rule", st: "Unsure", hide: false, lin: false});
          const out = [gvOvA(L, body("rule", "Unsure")), gvOvA(L, body("rule", "value")),
            gvOvA(L, body("question", "Unsure")), gvOvA(L, body("question", "value"))];
          GV.st = null; out.push(gvOvA(L, body("rule", "value")), gvOvA(L, body("question", "value")));
          GV.kind = null; out.push(gvOvA(L, body("question", "value")));
          GV.kind = "rule"; const off = {ov: null};
          out.push(gvOvA(off, body("rule", null)), gvOvA(off, body("question", null)));
          GV.kind = null; GV.st = null; return out; })()""",  # noqa: E501
    )
    assert shown == pytest.approx([1, 0.12, 0.12, 0.0144, 1, 0.12, 1, 1, 0.12])


# ---------- gravity: the calmer turn panel and the stats window ----------


@needs_node
def test_the_turn_panel_says_the_verdict_as_turn_does(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Gravity's turn panel opens with the very sentences Turn opens with, from the same
    function: the decision, the marker in words, what it rests on and the bound on the
    lineage; the bound's caveat is said once, in Turn's Bounds fold, and the arrow chain of the
    why line is gone."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.view = "gravity"; S.turn = 0; drawView(); frames(10);
          const L = gvL, el = (id) => document.getElementById(id);
          const panel = el("gvov").innerHTML, verdict = verdictHTML(L.ov.t, ixOf(L.D));
          S.view = "turn"; drawView();
          return {panel, verdict, turn: el("vTurn").innerHTML, label: el("gvroundl").textContent,
            max: L.ov.maxR}; })()""",
    )
    panel, verdict = shown["panel"], shown["verdict"]
    assert verdict in panel
    assert verdict in shown["turn"]
    for words in (
        "Undecided",
        "is unsure: no clear answer (it abstained)",
        "It rests on",
        "falls short of the bar",
    ):
        assert words in panel, words
    caveat = "is not how far the decision lies from changing"
    assert caveat not in panel
    assert shown["turn"].count(caveat) == 1
    assert " ← " not in panel  # no arrow chain
    assert "ringed in ink" not in panel  # the explanation is the help's
    assert shown["label"] == f"round {shown['max']} of {shown['max']}"


def test_the_help_says_what_the_page_draws() -> None:
    """The help names the rings, marks and arrows as the canvas draws them, and the key l."""
    html = page([], [DESIGN.document])
    helpbox = html[html.index('id="gvhelp"') : html.index('<aside id="gvright"')]
    for words in (
        "Reading a turn",
        "The policy wears an ink ring",
        "a notched ring in the lineage colour",
        "its arrows pointing the way the information flowed, toward the policy",
        "a thin grey ring determined with no bound to clear",
        "dashed <i>Unasked</i> or absent",
        "<b>!</b> short",
        "A hop named by a marker is thin",
        "not a record of what was read",
        "<kbd>l</kbd> lineage only",
    ):
        assert words in helpbox, words


@needs_node
def test_lineage_round_trips_and_the_target_chips_filter(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          const out = {}, el = (id) => document.getElementById(id);
          globalThis.location = {hash: ""}; globalThis.history = {replaceState: (x, y, h) =>
            { location.hash = h; }};
          S.view = "gravity"; S.turn = 0; GV.lin = false; GV.tg = null; GV.st = null;
          GV.kind = null; GV.sel = null; GV.step = null; drawView(); frames(10);
          const L = gvL;
          gvSet("lin", true); out.hash = location.hash;
          const notLin = L.nodes.find((n) => n.os && n.kind !== "target" && !L.ov.linN.has(n.id) && gvReached(n));
          out.linDim = notLin ? gvOvA(L, notLin) : null;
          const tgOff = L.nodes.find((n) => n.kind === "target" && n.ts && !L.ov.linN.has(n.id));
          out.tgDim = tgOff ? [gvOvA(L, tgOff) < 1, tgOff.id] : null;
          GV.lin = false; readHash(); render();
          out.back = [GV.lin, location.hash.includes("glin=1")];
          gvKey({key: "l", target: document.body}); out.key = GV.lin;
          gvSet("lin", false);
          // the target chips are filters
          const tgt = L.nodes.filter((n) => n.kind === "target" && n.ts);
          const states = [...new Set(tgt.map((n) => n.ts))]; out.states = states;
          const ts = states[0], inT = tgt.find((n) => n.ts === ts), outT = tgt.find((n) => n.ts !== ts);
          const rule = L.nodes.find((n) => n.kind === "rule");
          const click = (d) => el("vGravity").listeners.click.forEach((f) => f({target:
            {closest: () => ({dataset: d, getAttribute: () => null})}}));
          click({gvtg: ts});
          out.pressed = [GV.tg, el("gvtkey").innerHTML.includes('aria-pressed="true"')];
          out.dim = [gvOvA(L, inT), outT ? gvOvA(L, outT) : null, gvOvA(L, rule)];
          writeHash(); out.hashTg = location.hash;
          click({gvtg: ts}); out.cleared = GV.tg; writeHash(); out.hash0 = location.hash;
          // Esc: the state, then the target state, then the kind
          gvSet("st", "Unsure"); gvSet("tg", ts); gvSet("kind", "rule");
          const key = (k) => gvKey({key: k, target: document.body});
          out.esc = [key("Escape"), GV.st, GV.tg, GV.kind]; out.esc2 = [key("Escape"), GV.tg, GV.kind];
          out.esc3 = [key("Escape"), GV.kind];
          location.hash = "#gt=" + ts; readHash(); out.read = GV.tg;
          location.hash = "#gt=nonsense"; readHash(); out.bad = GV.tg;
          return out; })()""",  # noqa: E501
    )
    assert "glin=1" in shown["hash"]
    if shown["linDim"] is not None:
        assert shown["linDim"] == pytest.approx(0.3)
    assert shown["back"] == [True, True]
    assert shown["key"] is False  # l turns it off again
    if shown["tgDim"] is not None:
        assert shown["tgDim"][0] is True  # a target off the lineage dims too
    ts = shown["states"][0]
    assert shown["pressed"] == [ts, True]
    assert shown["dim"][0] == 1
    if shown["dim"][1] is not None:
        assert shown["dim"][1] < 0.2  # another target state dims
    assert shown["dim"][2] == pytest.approx(0.12)  # a body that is not a target
    assert f"gt={ts}" in shown["hashTg"]
    assert shown["cleared"] is None
    assert "gt=" not in shown["hash0"]
    assert shown["esc"] == [True, None, ts, "rule"]
    assert shown["esc2"] == [True, None, "rule"]
    assert shown["esc3"] == [True, None]
    assert shown["read"] == ts
    assert shown["bad"] is None


# ---------- history, selection, the camera, labels, the panels, Turn's branches ----------

_HISTORY = """globalThis.location = {hash: ""}; const stack = [""]; let at_ = 0;
  globalThis.history = {replaceState: (a, b, h) => { stack[at_] = h; location.hash = h; },
    pushState: (a, b, h) => { stack.splice(at_ + 1); stack.push(h); at_++; location.hash = h; }};
  const back = () => { at_--; location.hash = stack[at_]; onHistory(); };
  const fwd = () => { at_++; location.hash = stack[at_]; onHistory(); };"""


@needs_node
def test_back_and_forward_replay_every_chip_jump(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """A chip from Turn and a tab switch are history steps; back, through the page's own history
    handler, returns to each step in turn, and forward replays them, however many there are."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { """
        + _HISTORY
        + """
          const out = [], snap = () => out.push([view(), GV.sel]);
          const click = (d) => viewsClick({target: {closest: () => ({dataset: d})}});
          S.turn = 0; render(); snap();
          showView("turn"); snap();
          click({go: "rule:urgency", ref: "d:urgency"}); snap();
          showView("turn"); snap();
          click({go: "rule:act", ref: "d:act"}); snap();
          back(); snap(); back(); snap(); back(); snap(); back(); snap(); fwd(); snap();
          return out; })()""",
    )
    assert shown == [
        ["gravity", None],
        ["turn", None],
        ["gravity", "d:urgency"],
        ["turn", "d:urgency"],
        ["gravity", "d:act"],
        ["turn", "d:urgency"],  # back
        ["gravity", "d:urgency"],  # back again: the second jump is not ignored
        ["turn", None],
        ["gravity", None],
        ["turn", None],  # forward
    ]


@needs_node
def test_a_chip_to_a_body_a_filter_hides_shows_and_selects_it(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => {
          S.turn = 0; GV.light = false; render(); showView("turn");
          viewsClick({target: {closest: () => ({dataset: {go: "source:ask", ref: "q:tone"}})}});
          frames(60); const n = gvL.byId.get("q:tone");
          return {sel: GV.sel, want: n.want, forced: [...gvL.force], light: GV.light,
            panel: document.getElementById("gvside").innerHTML.includes("Which tone")}; })()""",
    )
    assert shown == {
        "sel": "q:tone",
        "want": True,
        "forced": ["q:tone"],
        "light": False,
        "panel": True,
    }


@needs_node
def test_every_outline_value_is_escaped_in_the_panels(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        o = doc["outline"]
        if o["kind"] == "design":
            next(n for n in o["nodes"] if n["form"] == "table")["rows"][0]["n"] = "<i>planted</i>"
        else:
            o["review"]["tables"][0]["rows"][0]["n"] = "<i>planted</i>"
            for r in o["rounds"]:
                for e in r["entries"]:
                    if e["ref"] == "d:act":
                        e["round"] = "<i>planted</i>"
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { const side = () => document.getElementById("gvside").innerHTML;
          S.turn = null; render(); gvSelect(gvL, gvL.byId.get("d:act")); const a = side();
          S.turn = 0; render(); gvSelect(gvL, gvL.byId.get("d:act"));
          S.view = "turn"; drawView();
          return [a, side(), document.getElementById("vTurn").innerHTML]; })()""",
    )
    for h in shown:
        assert "<i>planted</i>" not in h
    assert any("&lt;i&gt;planted" in h for h in shown)


@needs_node
def test_a_theme_change_redraws_both_panels(tmp_path: Path, turns: dict[str, bytes]) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.turn = 0; render(); gvSelect(gvL, gvL.byId.get("d:act"));
          let side = 0, ov = 0; const p0 = gvPanel, o0 = gvOvPanel;
          gvPanel = (L) => { side++; return p0(L); }; gvOvPanel = (L) => { ov++; return o0(L); };
          fire(document.getElementById("theme"), "click", {}); return [side, ov, theme]; })()""",
    )
    assert shown == [1, 1, "light"]


@needs_node
def test_a_target_s_estimators_are_the_outline_s(tmp_path: Path) -> None:
    html = page([], [DESIGN.document])
    data = _data(html)
    data["sets"][0]["docs"][0]["outline"]["targets"][0]["estimated_by"].append("rule:ghost")
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { render(); gvSelect(gvL, gvL.byId.get("tg:urgent"));
          return strip(document.getElementById("gvside").innerHTML); })()""",
    )
    assert "estimated by 2 rules" in shown  # the outline's list, not the bodies drawn


@needs_node
def test_selecting_never_zooms_and_pans_only_to_bring_the_body_into_view(tmp_path: Path) -> None:
    shown = _run(
        page([], [(IDIA / "design.json").read_bytes()]),
        tmp_path,
        """(() => { render(); frames(100); const L = gvL, out = {};
          const cam = () => [L.tf.x, L.tf.y, L.tf.k];
          const c0 = cam(); gvSelect(L, L.byId.get("d:route")); frames(60);
          out.inView = [c0, cam()];
          L.tf = {x: L.tf.x + 900, y: L.tf.y, k: L.tf.k};
          const c1 = cam(); gvSelect(L, L.byId.get("d:hazard")); frames(60);
          out.outOfView = [c1, cam()];
          gvProject(L); const n = L.byId.get("d:hazard");
          out.at = L.tf.x + L.tf.k * (L.W / 2 + n.px);
          out.W = L.W;
          return out; })()""",
    )
    assert shown["inView"][0] == shown["inView"][1]  # in view: the camera does not move
    before, after = shown["outOfView"]
    assert after[2] == before[2]  # never a zoom
    assert after[0] != before[0]  # a pan
    assert 0 < shown["at"] < shown["W"]


@needs_node
def test_fit_leaves_room_for_rings_and_labels_and_follows_the_filters(tmp_path: Path) -> None:
    shown = _run(
        page([], [(IDIA / "design.json").read_bytes()]),
        tmp_path,
        """(() => { render(); frames(100); const L = gvL, out = {};
          gvProject(L);
          const top = Math.min(...L.act.map((n) =>
            L.tf.y + L.tf.k * (L.H / 2 + n.py - n.cr * n.sc)));
          out.top = [top, L.free.y0];
          const cam = () => JSON.stringify([L.tf, L.cam]);
          const c0 = cam(); gvSet("light", false); frames(200); out.refit = cam() !== c0;
          gvSet("light", true); frames(200); gvZoomBy(L, 1.5); const c1 = cam();
          gvSet("light", false); frames(200); out.kept = cam() === c1;
          gvAct("fit"); frames(100); out.userCam = L.userCam;
          return out; })()""",
    )
    top, y0 = shown["top"]
    assert top >= y0 + 20  # a label's height above the topmost ring
    assert shown["refit"] is True  # the questions hidden: the rest is framed again
    assert shown["kept"] is True  # but not once the user has moved the camera
    assert shown["userCam"] is False


@needs_node
def test_links_name_the_design_shown_alone_and_reset_what_they_leave_out(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"undecided.json": turns["undecided.json"]},
            [(IDIA / "design.json").read_bytes(), DESIGN.document],
        ),
        tmp_path,
        """(() => { globalThis.location = {hash: ""};
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; }, pushState() {}};
          const out = {}; S.turn = null; S.design = 0; render(); out.zero = location.hash;
          S.design = 1; render(); location.hash = "#turn=0"; readHash(); out.reset = S.design;
          location.hash = "#design=0"; readHash(); render();
          out.reopen = [S.turn, curDesign().name];
          return out; })()""",
    )
    assert shown["zero"] == "#design=0"  # design 0 is written, so the link does not reopen a turn
    assert shown["reset"] == 0
    assert shown["reopen"] == [None, "street-reports"]


@needs_node
def test_a_tab_switch_pushes_and_a_turn_step_or_selection_replaces(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => { globalThis.location = {hash: ""}; let pushes = 0;
          globalThis.history = {replaceState: (a, b, h) => { location.hash = h; },
            pushState: (a, b, h) => { pushes++; location.hash = h; }};
          S.turn = at("decided.json"); render();
          const out = {};
          showView("turn"); out.tab = pushes;
          step(1); out.step = pushes;
          showView("gravity"); gvSelect(gvL, gvL.nodes[0]); out.select = pushes;
          viewsClick({target: {closest: () => ({dataset: {viewGo: "turn"}})}, preventDefault() {}});
          out.why = pushes;
          return out; })()""",
    )
    assert shown == {"tab": 1, "step": 1, "select": 2, "why": 3}


def test_the_panel_s_table_scrolls_and_the_chrome_is_steady() -> None:
    html = page([], [DESIGN.document])
    assert ".tblwrap { overflow-x: auto; }" in html
    assert (
        "#gvov, #gvside { padding: 10px 12px; display: grid; grid-template-columns: minmax(0, 1fr);"
        in html
    )
    assert "#gvc:focus { outline: none; }" in html
    assert '[data-gvact="play"] { min-width: 4.6em; }' in html
    assert 'id="gvnone" class="gv-none" hidden' in html


@needs_node
def test_the_3d_perspective_does_not_grow_with_the_spread(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render(); const L = gvL; gvSet("mode3", true); frames(300);
          const scales = (sp) => { L.cam.sp = sp; gvProject(L); return L.nodes.map((n) => n.sc); };
          return [scales(1), scales(6)]; })()""",
    )
    one, six = shown
    assert six == pytest.approx(one)  # the spread spaces the bodies, the depth stays the world's


@needs_node
def test_a_filter_that_leaves_nothing_says_so_and_a_narrow_select_scrolls_to_the_panel(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.turn = 0; render(); const out = {}, el = (id) => document.getElementById(id);
          out.none0 = el("gvnone").hidden;
          gvSet("kind", "target"); gvSet("st", "Unsure"); out.none1 = el("gvnone").hidden;
          gvSet("st", null); out.none2 = el("gvnone").hidden;
          let scrolled = 0; el("gvside").scrollIntoView = () => { scrolled++; };
          globalThis.matchMedia = (q) => ({matches: q.includes("860")});
          gvSelect(gvL, gvL.byId.get("d:act")); out.scrolled = scrolled;
          delete globalThis.matchMedia; return out; })()""",
    )
    assert shown == {"none0": True, "none1": False, "none2": True, "scrolled": 1}


@needs_node
def test_a_body_s_label_is_its_full_ref(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render();
          return Object.fromEntries(gvL.nodes.map((n) => [n.id, n.lbl])); })()""",
    )
    assert shown["tg:urgent"] == "tg:urgent"  # its label, urgent, is q:urgent's too
    assert shown["q:urgent"] == "q:urgent"
    assert shown["q:tone"] == "q:tone"
    assert shown["d:act"] == "d:act"
    assert all(k == v for k, v in shown.items())


@needs_node
def test_the_layout_leans_from_the_sources_to_the_policy(tmp_path: Path) -> None:
    """A weak lean by the outline's steps to the policy: the policy to the left of what feeds it,
    and a target, with no steps, leans nowhere; the state's springs are weak."""
    shown = _run(
        page([], [(IDIA / "design.json").read_bytes()]),
        tmp_path,
        """(() => { render(); const L = gvL, x = (id) => L.byId.get(id).x;
          return {route: x("d:route"), rules: ["d:hazard", "d:asset", "d:noticed"].map(x),
            qs: ["q:triage/hazard", "q:triage/asset", "q:details/noticed"].map(x),
            gx: Object.fromEntries(L.nodes.map((n) => [n.id, n.gx])),
            w: L.hh.concat(L.qh).filter((p) => (p.a || p.q).kind === "state"
              || (p.b || p.h).kind === "state")
              .map((p) => p.w)}; })()""",
    )
    assert all(shown["route"] < r for r in shown["rules"])
    assert sum(shown["qs"]) / 3 > sum(shown["rules"]) / 3
    assert shown["gx"]["d:route"] < shown["gx"]["d:hazard"] < shown["gx"]["q:triage/hazard"]
    assert shown["w"]
    assert all(w <= 0.2 for w in shown["w"])


@needs_node
def test_the_panel_says_a_hit_policy_a_kind_and_an_unfilled_template_in_words(
    tmp_path: Path,
) -> None:
    html = page([], [DESIGN.document])
    data = _data(html)
    o = data["sets"][0]["docs"][0]["outline"]
    ask = next(n for n in o["nodes"] if n["id"] == "source:ask")
    ask["questions"][0]["wording"] = {
        "template": "Is {subject} urgent?",
        "slots": [{"name": "subject", "origin": {"ref": "t:state.subject"}}],
    }
    ask["questions"][0]["kind"] = "noul"
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { render(); const side = (id) => { gvSelect(gvL, gvL.byId.get(id));
            return strip(document.getElementById("gvside").innerHTML); };
          return [side("d:act"), side(gvL.nodes.find((n) => n.kind === "question"
            && n.node === "source:ask" && n.id === ask0()).id)]; })()""".replace(
            "ask0()", json.dumps(ask["questions"][0]["ref"])
        ),
    )
    act, q = shown
    assert "The first row that matches gives the value" in act
    assert "hit F" not in act
    assert "Is {subject} urgent? · unfilled template" in q
    assert "yes or no" in q
    assert "noul" not in q


@needs_node
def test_all_turns_leaves_out_a_column_that_never_changes(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => { const h = allTurnsHTML(true);
          return {heads: [...h.matchAll(/<th>([^<]*)<\\/th>/g)].map((m) => m[1]),
            roots: cur().turns.map((t) => t.sum.root.ref)}; })()""",
    )
    assert len(set(shown["roots"])) == 1
    assert "root" not in shown["heads"]  # the same root on every row
    assert "decision" in shown["heads"]


@needs_node
def test_turn_says_what_wrote_nothing_and_which_rules_hold_no_value(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """Turn's other branches: a source that wrote nothing; a rule with no value; a candidate set
    and what it is used as; a policy that is a function rule, with its derivation."""
    html = page(
        {"undecided.json": turns["undecided.json"], "idia.json": turns["idia.json"]},
        [DESIGN.document, (IDIA / "design.json").read_bytes()],
    )
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        o = doc["outline"]
        if o["kind"] == "design" and o["name"] == "showcase":
            next(n for n in o["nodes"] if n["id"] == "rule:urgency")["candidates"] = True
        if doc["path"] == "undecided.json":
            o["review"]["sources"][1]["outcome"] = "no entry"
            for r in o["rounds"]:
                for e in r["entries"]:
                    if e["ref"] == "d:urgency":
                        e.update(holds="value", value=["first", "second"], marker=None)
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { const out = {};
          for (const p of ["undecided.json", "idia.json"]) { S.turn = at(p); S.view = "turn";
            drawView(); out[p] = strip(document.getElementById("vTurn").innerHTML); }
          return out; })()""",
    )
    und, idia = shown["undecided.json"], shown["idia.json"]
    assert "Wrote nothing · 1 source" in und
    assert "The record holds no entry for these, and gives no reason." in und
    assert "candidates d:urgency 2 members" in und
    assert "Rules with no value · " in idia  # d:hazard and the others idia leaves unsure
    assert "d:route is a function rule; its body is not in the design document." in idia


# ---------- the turn's words, the rings, the folder, reach ----------


def _plant(html: str, path: str, change: Any) -> str:
    """The page with ``change`` applied to the outline of the document at ``path``."""
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        if doc["path"] == path:
            change(doc["outline"])
    return _with_data(html, data)


@needs_node
def test_turn_says_a_marker_bound_and_failed_call_as_the_outline_gives_them(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A required input that is not determined says so; the verdict quotes the summary's bound
    only when it lies on the lineage; a failed call says it failed and why; a tool's input that
    is a field of the state shows that field, not the whole state; a call's prompt shows when
    the outline gives one; a duration reads in sensible units."""
    html = page(
        {"undecided.json": turns["undecided.json"], "decided.json": turns["decided.json"]},
        [DESIGN.document],
    )

    def undecided(o: dict[str, Any]) -> None:
        o["summary_nearest"]["on_lineage"] = False
        o["decision"]["marker"]["cause"]["code"] = "required"
        call = o["rounds"][1]["calls"][0]
        call.update(
            outcome={"failed": {"code": "raised", "refs": ["q:tone"]}}, prompt="Asked: tone?"
        )
        call["unhashed"]["latency"] = 0.0123

    html = _plant(html, "undecided.json", undecided)
    shown = _run(
        html,
        tmp_path,
        """(() => { const out = {};
          for (const p of ["undecided.json", "decided.json"]) { S.turn = at(p); S.view = "turn";
            drawView(); out[p] = document.getElementById("vTurn").innerHTML; }
          return out; })()""",
    )
    und, dec = shown["undecided.json"], shown["decided.json"]
    assert "a required input was not determined" in und
    verdict = und[und.index("Undecided") : und.index("How it was decided")]
    assert "falls short of" not in verdict  # the bound is off the lineage
    assert "the call failed: it raised an error" in und
    assert "[object" not in und
    assert "Asked: tone?" in und
    assert "12 ms" in und
    with_line = dec[dec.index('class="with"') :][:400]
    assert "Server down" in with_line
    assert "Nothing loads" not in with_line  # the subject alone, not the whole state
    assert "Row 1 of" in dec and "gave it." in dec  # noqa: PT018
    assert "gave it:" not in dec  # Turn leaves the tests to the table below


@needs_node
def test_the_file_picker_reads_every_file_of_a_list_that_empties(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { const got = []; const real = loadFiles; loadFiles = (fs) => { got.push(fs.length); };
          const list = [{name: "a.json"}, {name: "b.json"}];
          const input = {get files() { return list; }, set value(v) { list.length = 0; }};
          fire(document.getElementById("fileIn"), "change", {target: input});
          loadFiles = real; return got; })()""",  # noqa: E501
    )
    assert shown == [2]


@needs_node
def test_a_resize_keeps_the_user_s_zoom_and_the_turn_page(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.turn = 0; render(); const L = gvL; let renders = 0; const r0 = render;
          render = () => { renders++; r0(); };
          gvZoomBy(L, 2); const k = L.tf.k; L.fitted = "other"; gvResize();
          const kept = L.tf.k === k; gvAct("fit"); L.fitted = "other"; gvResize();
          render = r0; return {kept, refit: L.tf.k !== k * 0, renders}; })()""",
    )
    assert shown == {"kept": True, "refit": True, "renders": 0}


@needs_node
def test_the_focus_comes_back_after_a_turn_step_a_chip_and_a_find(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => { S.turn = cur().order[0]; render(); const out = {}, cv = document.getElementById("gvc");
          cv.focus(); gvAct("play"); frames(5); out.play = document.activeElement === cv; gvAct("play");
          S.view = "turn"; render();
          viewsClick({target: {closest: () => ({dataset: {go: "rule:act", ref: "d:act"}})}});
          out.chip = document.activeElement === cv;
          document.activeElement = null; gvSet("find", "tone");
          fire(document.getElementById("gvsearch"), "keydown", {key: "Enter"}); out.find = document.activeElement === cv;
          return out; })()""",  # noqa: E501
    )
    assert shown == {"play": True, "chip": True, "find": True}


@needs_node
def test_rings_keep_a_screen_size_follow_their_body_and_number_three_at_most(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """At any zoom a ring is the same width on screen; a dimmed body's rings dim with it, the
    policy's never below .8; a determined body wears one thin neutral ring; the policy, when it
    is the root, wears at most three; a body too small for a ring takes its state's colour."""
    html = page({"plain.json": showcase.plain_turn().document}, [load(showcase.PLAIN).document])
    shown = _run(
        html,
        tmp_path,
        """(() => { S.turn = 0; GV.mode3 = false; render(); frames(300);
          const L = gvL, T = L.T, out = {};
          """
        + _RECORDER
        + """
          const arcsAround = (n, k) => { L.tf = {x: L.tf.x, y: L.tf.y, k}; rec.ev.length = 0; gvDraw(L); gvProject(L);
            const ring = (x) => x && x.k === "arc" && Math.abs(x.a[0] - n.px) < 1e-9
              && Math.abs(x.a[1] - n.py) < 1e-9 && x.a[2] > n.r * n.sc + 1e-9;
            const own = rec.ev.filter((e, i) => e.k === "stroke"
              && ring(rec.ev.slice(0, i).reverse().find((x) => x.k === "arc")));
            return own.map((e) => [+(e.lw * k).toFixed(3), +e.al.toFixed(3)]); };  // its rings
          const pol = L.nodes.find((n) => n.policy), det = L.nodes.find((n) => n.os === "det");
          out.pol1 = arcsAround(pol, 1); out.pol3 = arcsAround(pol, 3);
          out.radii = [...new Set(rec.ev.filter((e) => e.k === "arc" && Math.abs(e.a[0] - pol.px) < 1e-9
            && Math.abs(e.a[1] - pol.py) < 1e-9 && e.a[2] > pol.r * pol.sc + 1e-9).map((e) => +e.a[2].toFixed(6)))];
          out.root = L.ov.root === pol.id;
          out.ringDet = GV_RING.det; out.ringNb = GV_RING.nobounds;
          gvSelect(L, det); frames(60); out.dimPol = arcsAround(pol, 2);
          gvSelect(L, null); L.tf = {x: L.tf.x, y: L.tf.y, k: 0.05}; rec.ev.length = 0; gvDraw(L);
          out.tinyFill = rec.ev.some((e) => e.k === "fill" && e.fs === T.neutral);
          return out; })()""",  # noqa: E501
    )
    assert shown["root"] is True  # the plain turn rests on the policy's own entry
    assert [w for w, _ in shown["pol1"]] == [w for w, _ in shown["pol3"]]  # the same on screen
    assert len(shown["radii"]) <= 3  # never more than three concentric rings
    assert shown["ringDet"] == shown["ringNb"] == ["neutral", 0, "", "thin"]
    assert shown["dimPol"]
    assert all(a >= 0.8 * 0.6 for _, a in shown["dimPol"])  # not below .8; a thin ring is .6 of it
    assert shown["tinyFill"] is True


@needs_node
def test_a_lineage_hop_with_no_design_edge_is_drawn_as_a_curve(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])

    def hop(o: dict[str, Any]) -> None:
        o["lineage"].append(
            {"ref": "t:ticket", "from": "q:tone", "via": "cause", "index": None, "holds": "Unasked"}
        )

    shown = _run(
        _plant(html, "undecided.json", hop),
        tmp_path,
        """(() => { S.turn = 0; render(); frames(50); gvDraw(gvL);
          return {curve: gvL.lit.filter((h) => h[5] === "curve").map((h) => [h[0], h[1]]),
            linked: gvL.adj.get("t:ticket").has("q:tone")}; })()""",
    )
    assert shown["linked"] is False
    assert ["t:ticket", "q:tone"] in shown["curve"]


@needs_node
def test_determined_is_one_chip_and_the_root_has_a_legend_entry(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"idia.json": turns["idia.json"]}, {"idia-design.json": DESIGNS["idia-design.json"]}),
        tmp_path,
        """(() => { S.turn = 0; render(); const L = gvL, el = (id) => document.getElementById(id);
          const nb = L.nodes.find((n) => n.os === "nobounds");
          gvSet("st", "det"); const lit = gvOvA(L, nb); gvSet("st", null);
          return {key: el("gvkey").innerHTML, legend: el("gvlegend").innerHTML, lit}; })()""",
    )
    assert 'data-gvst="nobounds"' not in shown["key"]
    assert 'data-gvst="det"' in shown["key"]
    assert shown["lit"] == 1  # the determined chip takes in a body with no bound declared
    assert "root</span>" in shown["legend"]


@needs_node
def test_play_pulses_the_new_entries_and_keeps_its_pace_under_reduced_motion(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.turn = 0; render(); const L = gvL, out = {};
          gvSet("step", 1); out.pulse = L.pulse ? [...L.pulse.ids].sort() : null;
          frames(40); out.done = L.pulse;
          const ticks = []; const st = globalThis.setTimeout;
          globalThis.setTimeout = (f, ms) => { ticks.push(ms); return 1; };
          globalThis.matchMedia = (q) => ({matches: q.includes("reduce")});
          gvAct("play"); gvAct("play"); gvSet("step", 1); out.reduced = L.pulse;
          delete globalThis.matchMedia; globalThis.setTimeout = st; out.ticks = ticks; return out; })()""",  # noqa: E501
    )
    assert shown["pulse"] == ["q:tone", "q:urgent", "t:state"] or "q:urgent" in shown["pulse"]
    assert shown["done"] is None
    assert shown["ticks"] == [1200]
    assert shown["reduced"] is None


@needs_node
def test_a_chosen_row_with_an_open_value_is_not_green(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => { const out = {};
          for (const p of ["decided.json", "undecided.json"]) { S.turn = at(p); S.view = "turn"; drawView();
            out[p] = document.getElementById("vTurn").innerHTML; }
          return out; })()""",  # noqa: E501
    )
    assert 'class="rr chosen"' in shown["decided.json"]
    assert 'class="rr chosen open"' in shown["undecided.json"]


@needs_node
def test_the_panel_lists_neighbours_by_direction_and_says_what_the_policy_decides(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"decided.json": turns["decided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { const side = () => strip(document.getElementById("gvside").innerHTML);
          S.turn = null; render(); const out = {summary: side()};
          gvSelect(gvL, gvL.byId.get("d:urgency")); out.urgency = side();
          gvSelect(gvL, gvL.byId.get("d:act")); out.act = side();
          return out; })()""",
    )
    assert re.search(
        r"the policyd:act decides between (escalate|reply|file)( · |$)", shown["summary"]
    )
    assert "bodies · 7" in shown["summary"]
    assert "reads · " in shown["urgency"]
    assert "read by · " in shown["urgency"]
    assert "bears on · 1tg:urgent" in shown["urgency"]
    assert "decides between: " in shown["act"]
    assert "escalate" in shown["act"]
    assert "fired, of 1 loaded turns" in shown["act"]


@needs_node
def test_all_turns_counts_the_loaded_turns_and_a_count_filters_the_table(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(turns, DESIGNS),
        tmp_path,
        """(() => { S.view = "turn"; S.turn = at("decided.json"); render(); const out = {};
          const h = () => document.getElementById("vTurn").innerHTML;
          out.before = (h().match(/<tr data-turn=/g) || []).length;
          out.tally = strip(h().split('class="tallies">')[1].split("<table")[0]);
          viewsClick({target: {closest: () => ({dataset: {tally: "decided|escalate"}})}});
          out.after = (h().match(/<tr data-turn=/g) || []).length; out.pressed = h().includes('aria-pressed="true"');
          out.menu = document.getElementById("turnPick").innerHTML;
          out.buttons = (h().match(/<button class="rowbtn mono" data-turn=/g) || []).length;
          out.tabindex = h().includes('<tr data-turn="0" tabindex');
          return out; })()""",  # noqa: E501
    )
    assert shown["before"] == 7
    assert shown["tally"].startswith("Counts over the loaded turns:")
    assert "decided: " in shown["tally"] and "undecided: " in shown["tally"]  # noqa: PT018
    assert shown["after"] < shown["before"]
    assert shown["pressed"] is True
    assert "decided.json" not in shown["menu"]  # the record's name without its file ending
    assert " · escalate</option>" in shown["menu"]
    assert shown["buttons"] == shown["after"]  # each row opens through a named button
    assert shown["tabindex"] is False


@needs_node
def test_every_wheel_zooms_and_a_pinch_zooms_faster(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render(); frames(100); const L = gvL, cv = document.getElementById("gvc"), out = {};
          const k = () => L.tf.k;
          const w = (o) => fire(cv, "wheel", Object.assign({clientX: 200, clientY: 200}, o));
          let c = k(); w({deltaX: 3, deltaY: -7}); out.smallPixels = k() / c; // a trackpad, or a mouse on macOS
          c = k(); w({deltaY: 100}); out.notch = k() / c;
          c = k(); w({deltaY: 3, deltaMode: 1}); out.lines = k() < c;
          c = k(); w({deltaY: -50, ctrlKey: true}); out.pinch = k() / c;
          fire(cv, "gesturestart", {}); c = k(); w({deltaY: -50, ctrlKey: true});
          out.gesture = k() === c; fire(cv, "gestureend", {}); // Safari's pinch is not zoomed twice
          gvSet("mode3", true); frames(300); const y0 = L.cam.yaw; c = k();
          w({deltaX: 20, deltaY: -40}); out.turn3 = L.cam.yaw - y0; out.zoom3 = k() > c;
          return out; })()""",  # noqa: E501
    )
    assert shown["smallPixels"] == pytest.approx(2 ** (7 * 0.002))
    assert shown["notch"] == pytest.approx(2**-0.2)
    assert shown["lines"] is True
    assert shown["pinch"] == pytest.approx(2.718281828**0.5)
    assert shown["gesture"] is True
    assert shown["turn3"] == 0
    assert shown["zoom3"] is True


@needs_node
def test_single_keys_leave_a_focused_control_alone_and_a_selection_is_announced(
    tmp_path: Path,
) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render(); const out = {};
          out.onButton = gvKey({key: "f", target: {tagName: "BUTTON"}});
          out.onSummary = gvKey({key: "3", target: {tagName: "SUMMARY"}});
          out.escOnButton = (gvHelp(true), gvKey({key: "Escape", target: {tagName: "BUTTON"}}));
          out.onCanvas = gvKey({key: "f", target: document.getElementById("gvc")});
          gvSelect(gvL, gvL.byId.get("d:act"));
          out.said = document.getElementById("gvlive").textContent;
          return out; })()""",
    )
    assert shown == {
        "onButton": False,
        "onSummary": False,
        "escOnButton": True,
        "onCanvas": True,
        "said": "d:act, rule result",
    }


@needs_node
def test_hostile_text_is_isolated_and_kept_out_of_the_canvas(tmp_path: Path) -> None:
    html = page([], [DESIGN.document])
    assert (
        ".mono, .ref, .tv, .gv-t, .gv-nb, .hchip, blockquote, td { unicode-bidi: isolate; }" in html
    )
    data = _data(html)
    o = data["sets"][0]["docs"][0]["outline"]
    o["short"]["q:tone"] = "‮tone" + "x" * 200
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { render(); return gvL.byId.get("q:tone").lbl; })()""",
    )
    assert "‮" not in shown
    assert len(shown) <= 40


@needs_node
def test_copy_link_falls_back_and_the_drop_overlay_is_for_files(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { globalThis.location = {hash: "#design=0", href: "file:///x.html#design=0"};
          globalThis.history = {replaceState() {}};
          globalThis.navigator = {clipboard: {writeText: () => ({then: (ok, no) => no()})}};
          document.execCommand = () => false; const b = document.getElementById("copy");
          b.parentNode = {insertBefore: (el) => { b.after = el; }};
          fire(b, "click", {}); const out = {state: b.dataset.state, shown: b.after && b.after.textContent};
          return out; })()""",  # noqa: E501
    )
    assert shown == {"state": "failed", "shown": "file:///x.html#design=0"}
    html = page([], [DESIGN.document])
    assert ":focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }" in html
    assert (
        "if (m.addEventListener) m.addEventListener('change', gvRetheme); else if (m.addListener)"
        in html
    )


# ---------- honesty: absent reads as absent; colour says only what the outline says ----------

_WORDS = ("null", "undefined", "NaN", "[object")


@needs_node
def test_a_null_value_and_missing_members_read_as_no_value(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A null value, a null parameter and a record missing its time, format and engine say "no
    value" or leave the line out; no "null", "undefined", "NaN" or "[object" reaches the page
    outside the record's own text form, which is quoted as ommat writes it."""
    html = page(
        {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
        [DESIGN.document],
    )
    data = _data(html)
    for doc in data["sets"][0]["docs"]:
        o = doc["outline"]
        if o["kind"] == "design":
            o["parameters"][0]["value"] = None
            continue
        for key in ("turn", "record_format", "engine", "settings", "status"):
            o.pop(key, None)
        for r in o["rounds"]:
            for e in r["entries"]:
                if e["holds"] == "value" and e["ref"] != "t:state":
                    e["value"] = None
        if o["decision"]["holds"] == "value":
            o["decision"]["value"] = None
    shown = _run(
        _with_data(html, data),
        tmp_path,
        """(() => { const out = [], el = (id) => document.getElementById(id);
          const look = (h) => strip(h.replace(/<pre class="(val txt|gv-text)">[\\s\\S]*?<\\/pre>/g, "").replace(/p_null/g, "").replace(/ title="[^"]*"/g, ""));
          for (let i = 0; i < cur().turns.length; i++) {
            S.view = "gravity"; S.turn = i; render();
            for (const n of gvL.nodes) { gvSelect(gvL, n); out.push(look(el("gvside").innerHTML)); out.push(look(gvTipHtml(gvL, n, 0))); }
            gvSelect(gvL, null); out.push(look(el("gvov").innerHTML)); out.push(el("gvroundd").textContent);
            S.view = "turn"; render(); out.push(look(el("vTurn").innerHTML)); out.push(look(el("turnPick").innerHTML)); }
          return out.join("\\n"); })()""",  # noqa: E501
    )
    for word in _WORDS:
        assert word not in shown, (
            word,
            shown[max(0, shown.find(word) - 80) : shown.find(word) + 20],
        )
    assert "Decided none" in shown  # the policy's entry holds null: the value none
    assert "no value" in shown  # the null parameter, a member the outline gives as null


@needs_node
def test_an_entry_holding_null_is_none_and_a_ref_with_no_entry_is_no_value(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """An entry that exists and holds JSON null, bare or as an extraction's value, renders as
    the value none in a value chip; "no value" is only for a ref with no entry."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => ({bare: valueCell({holds: "value", value: null}),
          extraction: briefFull({holds: "value", value: {kind: "extraction", value: null}}),
          missing: briefFull(null), cell: valueCell(null)}))()""",
    )
    assert ">none<" in shown["bare"] and 'class="tv' in shown["bare"]  # noqa: PT018
    assert shown["extraction"] == "none"
    assert shown["missing"] == "no value"
    assert "no value" not in shown["bare"]


@needs_node
def test_green_is_only_for_a_clear_value_and_red_only_for_unavailable(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A determined entry with no bound is a neutral mark, not a green tick; a target is green only
    when its estimates are clear, amber when one is short, neutral when they are only determined
    or Unasked, red only for an Unavailable answer; the panel's mark is the canvas's."""
    html = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    shown = _run(
        html,
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = 0; render(); const L = gvL, tg = L.byId.get("tg:urgent");
          const est = [...L.adj.get("tg:urgent")].map((i) => L.byId.get(i)).filter((x) => x.kind !== "state");
          const out = {det: markChip("det"), nob: markChip("nobounds"), val: markChip("value"),
            marks: Object.fromEntries(Object.entries(GV_RING).map(([k, v]) => [k, v[2] || ""]))};
          out.tc = tg.tc; out.ts = tg.ts;
          return out; })()""",  # noqa: E501
    )
    for mark in (shown["det"], shown["nob"]):
        assert 'class="st det"' in mark
        assert "✓" not in mark
    assert "✓" in shown["val"]
    assert shown["marks"]["short"] == "!"  # one glyph a state, here and on the canvas
    tc = shown["tc"]
    assert tc["hc"] in ("good", "warn", "neutral")
    assert tc["oc"] in ("bad", "warn", "neutral")


@needs_node
def test_a_target_s_ring_colour_follows_what_its_estimates_hold(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    def plant(states: list[tuple[str, dict[str, Any]]]) -> Any:
        def change(o: dict[str, Any]) -> None:
            q = [
                e
                for r in o["rounds"]
                for e in r["entries"]
                if e["ref"] in ("q:urgent", "q:tone", "d:urgency")
            ]
            for e, (holds, marker) in zip(q, states, strict=False):
                e["holds"], e["marker"] = holds, marker
                if holds != "value":
                    e.pop("value", None)
                    e["bounds"] = []

        return change

    unasked: dict[str, Any] = {
        "kind": "Unasked",
        "cause": {"code": "need-undetermined", "refs": []},
    }
    gone: dict[str, Any] = {"kind": "Unavailable", "cause": {"code": "raised", "refs": []}}
    base = page({"undecided.json": turns["undecided.json"]}, [DESIGN.document])
    out = {}
    for name, states in (
        ("unasked", [("Unasked", unasked)] * 3),
        ("unavailable", [("Unavailable", gone), ("Unasked", unasked), ("Unasked", unasked)]),
    ):
        out[name] = _run(
            _plant(base, "undecided.json", plant(states)),
            tmp_path,
            """(() => { S.view = "gravity"; S.turn = 0; render(); const tg = gvL.byId.get("tg:urgent");
              return [tg.ts, tg.tc.oc]; })()""",  # noqa: E501
        )
    assert out["unasked"] == ["unresolved", "neutral"]  # Unasked is grey, not red
    assert out["unavailable"] == ["unresolved", "bad"]


@needs_node
def test_a_tiny_body_keeps_a_hairline_state_ring(tmp_path: Path, turns: dict[str, bytes]) -> None:
    """Zoomed far out a body is too small for its ring; it keeps a hairline one in the state's
    colour, dashed for Unasked or absent, so the state is not told by its fill alone."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = 0; render(); frames(60); const L = gvL;
          """
        + _RECORDER
        + """
          L.tf = {x: L.tf.x, y: L.tf.y, k: .05}; rec.ev.length = 0; gvDraw(L);
          const px = 1 / L.tf.k, hair = rec.ev.filter((e) => e.k === "stroke" && Math.abs(e.lw - px) < 1e-9);
          return {hair: hair.length, dashed: hair.filter((e) => e.dash.length).length,
            unasked: L.nodes.filter((n) => n.os === "Unasked" || n.os === "absent").length}; })()""",  # noqa: E501
    )
    assert shown["hair"] > 0
    if shown["unasked"]:
        assert shown["dashed"] > 0


@needs_node
def test_a_hop_s_arrow_points_toward_the_policy_and_a_marker_s_hop_is_thin(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    """A lit hop's arrowhead points from the entry read to the entry that read it, the way the
    information flowed, toward the policy; a hop the outline gives as a marker's cause is drawn
    thin, as a declared one is, never with the main chain's weight."""
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = 0; render(); frames(80); const L = gvL;
          """
        + _RECORDER
        + """
          rec.ev.length = 0; gvDraw(L); const k = L.tf.k, ev = rec.ev, out = {checked: 0, wrong: 0};
          const tris = []; ev.forEach((e, i) => { if (e.k === "moveTo" && ev[i + 1] && ev[i + 1].k === "lineTo"
            && ev[i + 2] && ev[i + 2].k === "lineTo" && ev[i + 3] && ev[i + 3].k === "closePath") tris.push(e.a); });
          for (const h of L.ov.hops) { const a = L.byId.get(h.s), b = L.byId.get(h.t);
            if (!a || !b || !L.adj.get(a.id).has(b.id)) continue;
            const mx = (a.px + b.px) / 2, my = (a.py + b.py) / 2;
            const tip = tris.find(([x, y]) => Math.hypot(x - mx, y - my) < 9 / k); if (!tip) continue;
            out.checked++;
            if (Math.hypot(tip[0] - b.px, tip[1] - b.py) >= Math.hypot(mx - b.px, my - b.py)) out.wrong++; }
          const policy = L.nodes.find((n) => n.policy).id;
          out.toPolicy = L.ov.hops.some((h) => h.t === policy);
          out.noneFromPolicy = !L.ov.hops.some((h) => h.s === policy);
          L.ov.hops.forEach((h) => { h.via = "cause"; h.main = true; });
          rec.ev.length = 0; gvDraw(L);
          out.heavy = rec.ev.filter((e) => e.k === "stroke" && Math.abs(e.lw * k - 2.8) < 1e-6).length;
          return out; })()""",  # noqa: E501
    )
    assert shown["checked"] > 0
    assert shown["wrong"] == 0
    assert shown["toPolicy"] is True
    assert shown["noneFromPolicy"] is True
    assert shown["heavy"] == 0


# ---------- rows, fired, wording ----------


@needs_node
def test_fired_counts_only_the_turns_paired_by_hash(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    html = page(
        {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
        [DESIGN.document],
    )

    def rename(o: dict[str, Any]) -> None:
        o["design_hash"] = "0" * 64  # paired by name only

    shown = _run(
        _plant(html, "undecided.json", rename),
        tmp_path,
        """(() => { S.turn = null; render(); gvSelect(gvL, gvL.byId.get("d:act"));
          return {side: document.getElementById("gvside").innerHTML,
            pair: cur().turns.map((t) => t.pairBy)}; })()""",
    )
    assert sorted(shown["pair"], key=str) == ["hash", "name"]
    assert "fired, of 1 loaded turns" in shown["side"]
    assert "a count over the turns loaded here" in shown["side"].lower()


@needs_node
def test_rows_say_also_matched_no_output_and_an_untested_row_in_one_wording(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { const out = {};
          out.how = tableHow({rows: [{n: 1, result: "matched", chosen: true}, {n: 2, result: "matched"}], hit: "F"});
          out.out = designRowsHtml({rows: [{n: 1, output: null, cells: []}], inputs: []}, null);
          out.untested = RESULT_TEXT.untested;
          out.bound = boundSentence; out.cell = cellText("Known");
          out.brief = briefFull({holds: "value", value: {probabilities: {moderate: .4, low: .1}}});
          return out; })()""",  # noqa: E501
    )
    assert "1 other row also matched." in shown["how"]
    assert "later" not in shown["how"]
    assert "no output" in shown["out"] and "C#" not in shown["out"]  # noqa: PT018
    assert (
        shown["untested"] == "not tested: an input it tests was unavailable"
    )  # the outline's untested
    assert shown["cell"] == "any known value"
    assert shown["brief"] == "moderate (p 0.4)"


@needs_node
def test_the_design_buttons_show_only_for_more_than_one_design(tmp_path: Path) -> None:
    one = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render();
      return document.getElementById("designPick").children.length; })()""",
    )
    two = _run(
        page([], [DESIGN.document, (IDIA / "design.json").read_bytes()]),
        tmp_path,
        """(() => { render(); return document.getElementById("designPick").children.length; })()""",
    )
    assert one == 0
    assert two == 2


# ---------- state that resets, keys that keep off controls, print, drop, camera ----------


@needs_node
def test_the_tally_resets_with_the_set_and_keys_keep_off_a_focused_control(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page(
            {"decided.json": turns["decided.json"], "undecided.json": turns["undecided.json"]},
            [DESIGN.document],
        ),
        tmp_path,
        """(() => { const out = {}; S.view = "turn"; S.turn = cur().order[0]; render();
          viewsClick({target: {closest: () => ({dataset: {tally: "decided|escalate"}})}});
          out.set = S.tally;
          fire(document.getElementById("setPick"), "change", {target: {value: "0"}}); out.reset = S.tally;
          S.turn = cur().order[0]; render(); const t0 = S.turn;
          for (const k of ["j", "k", "ArrowRight", "ArrowLeft"])
            fire(document, "keydown", {key: k, target: {tagName: "BUTTON"}, preventDefault() {}});
          out.kept = S.turn === t0;
          fire(document, "keydown", {key: "j", target: document.body, preventDefault() {}});
          out.moved = S.turn !== t0;
          return out; })()""",  # noqa: E501
    )
    assert shown == {"set": "decided|escalate", "reset": None, "kept": True, "moved": True}


@needs_node
def test_print_draws_turn_opens_its_folds_and_closes_them_after(
    tmp_path: Path, turns: dict[str, bytes]
) -> None:
    shown = _run(
        page({"undecided.json": turns["undecided.json"]}, [DESIGN.document]),
        tmp_path,
        """(() => { S.view = "gravity"; S.turn = 0; render(); const out = {};
          const folds = [{open: false}, {open: false}]; let asked = "";
          document.querySelectorAll = (q) => { asked = q; return q.includes("#vTurn details") ? folds : []; };
          const fire_ = (t) => fire(globalThis, t, {});
          let drawn = 0; const d0 = drawTurnView; drawTurnView = () => { drawn++; d0(); };
          fire_("beforeprint"); out.during = folds.map((d) => d.open); out.drawn = drawn;
          fire_("afterprint"); out.after = folds.map((d) => d.open);
          out.turnHtml = document.getElementById("vTurn").innerHTML.includes("Undecided");
          return out; })()""",  # noqa: E501
    )
    assert shown["during"] == [True, True]
    assert shown["after"] == [False, False]
    assert shown["drawn"] == 1
    assert shown["turnHtml"] is True
    html = page([], [DESIGN.document])
    pr = html[html.index("@media print {") :]
    assert ':root, :root[data-theme="dark"], :root:not([data-theme="light"])' in pr[:200]


@needs_node
def test_a_drag_that_is_not_files_is_let_through_and_the_theme_is_checked(tmp_path: Path) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { const out = {}; let prevented = 0;
          const ev = (types) => ({dataTransfer: {types, files: []}, preventDefault() { prevented++; }});
          const fire_ = (t, e) => fire(globalThis, t, e);
          fire_("dragover", ev(["text/plain"])); fire_("drop", ev(["text/plain"])); out.text = prevented;
          fire_("dragover", ev(["Files"])); out.files = prevented;
          globalThis.location = {hash: "#theme=<x>"}; readHash(); out.bad = theme;
          location.hash = "#theme=dark"; readHash(); out.dark = theme;
          return out; })()""",  # noqa: E501
    )
    assert shown["text"] == 0
    assert shown["files"] == 1
    assert shown["bad"] != "<x>"
    assert shown["dark"] == "dark"


@needs_node
def test_the_camera_is_kept_across_views_and_a_copy_fallback_does_not_pile_up(
    tmp_path: Path,
) -> None:
    shown = _run(
        page([], [DESIGN.document]),
        tmp_path,
        """(() => { render(); frames(60); const L = gvL; gvZoomBy(L, 2); const k = L.tf.k;
          S.view = "turn"; render(); S.view = "gravity"; render(); L.fitted = "changed"; gvResize();
          const out = {kept: gvL.tf.k === k};
          globalThis.location = {hash: "", href: "file:///x.html"}; globalThis.history = {replaceState() {}};
          globalThis.navigator = {clipboard: {writeText: () => ({then: (ok, no) => no()})}};
          document.execCommand = () => false; const b = document.getElementById("copy"); let n = 0;
          b.parentNode = {insertBefore: (el) => { n++; b.nextSibling = el; el.remove = () => { n--; b.nextSibling = null; }; }};
          fire(b, "click", {}); fire(b, "click", {}); fire(b, "click", {}); out.fields = n;
          return out; })()""",  # noqa: E501
    )
    assert shown == {"kept": True, "fields": 1}
