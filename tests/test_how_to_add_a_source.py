"""The how-to's example runs as the guide says: docs/how-to/add_a_source. Each Python block in
the guide is a passage of one of the example's files, and the driver prints the guide's
output."""

# pyright: reportPrivateUsage=false

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import chat_run
import reports_run
from chat_binding import ChatBinding
from chat_endpoint import StubEndpoint
from ommat import (
    AnswerKind,
    Cause,
    RenderedQuestion,
    SourceCall,
    Unavailable,
    ValueType,
)
from ommat.bindings.conformance import check
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


def _output(marker: str) -> str:
    """The guide's one ``text`` block holding ``marker``, a line only that driver prints."""
    (block,) = [b for b in _blocks("text") if marker in b]
    return block


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
    assert capsys.readouterr().out == _output("own batches: []")
    assert [decision for decision, _ in results] == [
        {"log": ["Quay Street"]},
        {"log": ["Elm Row", "Mill Lane"]},
        {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["d:streets"]}},
    ]
    assert [calls for _, calls in results] == [0, 0, 0]


def test_the_idia_driver_prints_what_the_guide_shows(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("idia")
    import idia_run

    assert idia_run.main() == [
        {"log": ["Elm Row", "Mill Lane"]},
        {"kind": "Unsure", "cause": {"code": "abstained", "refs": ["d:streets"]}},
    ]
    assert capsys.readouterr().out == _output("leaf calls: 0")


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
    model = reports_run.Counted(StreetSpotter())
    binding = SpanBinding(model)
    with binding._lock:
        result = asyncio.run(binding(_batch("street"), seconds=0.01))
    assert result.values["q:places/street"] == Unavailable(Cause("timed-out"))
    assert model.calls == 0
    assert result.revision is None


_WITHOUT_IDIA = """
import importlib.abc
import sys


class _NoIdia(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "idia" or name.startswith("idia."):
            raise ModuleNotFoundError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, _NoIdia())
sys.path.insert(0, sys.argv[1])
import chat_run

chat_run.main()
loaded = sorted(m for m in sys.modules if m.split(".")[0] == "idia" or m == "ommat.bindings.idia")
assert not loaded, loaded
"""


def test_the_chat_driver_prints_what_the_guide_shows_with_idia_blocked(tmp_path: Path) -> None:
    script = tmp_path / "without_idia.py"
    script.write_text(_WITHOUT_IDIA, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", script, EXAMPLE], capture_output=True, text=True, check=True
    )
    assert result.stdout == _output("endpoint calls: 0")


def test_the_chat_driver_decides_and_replays_with_no_endpoint_call() -> None:
    assert chat_run.main() == [("emergency crew", 0), ("roads team", 0)]


class _Scripted(StubEndpoint):
    """The stand-in, but for questions whose wording names what the request does instead."""

    async def __call__(self, body: Mapping[str, Any], *, seconds: float) -> dict[str, Any]:
        reply = await super().__call__(body, seconds=seconds)
        wording = json.loads(body["messages"][-1]["content"])["question"]
        message = reply["choices"][0]["message"]
        if wording == "refuse":
            message.update(content=None, refusal="I can't help with that.")
        elif wording == "garble":
            message["content"] = "Probably yes."
        elif wording == "three places":
            message["content"] = json.dumps({"probabilities": {"yes": 0.333, "no": 0.667}})
        elif wording == "one label":
            message["content"] = json.dumps({"probabilities": {"yes": 0.9}})
        elif wording == "stall":
            raise TimeoutError("no reply within the request's limit")
        elif wording == "break":
            raise ConnectionError("connection reset")
        return reply


def _nouls(*wordings: str) -> tuple[SourceCall, ...]:
    questions = {
        f"q:faults/{n}": RenderedQuestion(AnswerKind.noul, wording, ("yes", "no"))
        for n, wording in enumerate(wordings)
    }
    return (
        SourceCall("faults", None, {"t:state.text": "A wire is exposed."}, (), questions),
        SourceCall(
            "other",
            None,
            {"t:state.text": "A lamp is out."},
            (),
            {"q:other/ok": RenderedQuestion(AnswerKind.noul, "answer", ("yes", "no"))},
        ),
    )


_CAUSES = ("refuse", "garble", "three places", "one label", "stall", "break")


def test_a_chat_request_that_fails_marks_its_own_ref_and_the_rest_stand() -> None:
    binding = ChatBinding(_Scripted(), provider="example-chat", model="chat-small")
    result = asyncio.run(binding(_nouls(*_CAUSES), seconds=5))
    codes = ["refused", "invalid-answer", "invalid-answer", "invalid-answer", "timed-out", "raised"]
    for n, code in enumerate(codes):
        assert result.values[f"q:faults/{n}"] == Unavailable(Cause(code))
        assert any(line.startswith(f"q:faults/{n}: {code}: ") for line in result.warnings)
    assert result.values["q:other/ok"] == {"yes": 0.1, "no": 0.9}
    assert "q:faults/4: timed-out: no reply within the request's limit" in result.warnings
    assert "q:faults/5: raised: ConnectionError: connection reset" in result.warnings
    assert result.attempts == 7
    assert result.precision == 2
    assert result.model_reported == "chat-small-2026-06"
    assert result.claimed == ("model_reported",)
    assert result.extras[4] == result.extras[5] == {}
    assert len(result.extras) == 7
    assert check(binding, batches=[_nouls(*_CAUSES)], deterministic=True) == []


def test_a_chat_call_whose_every_request_fails_gives_each_detail_a_reason() -> None:
    binding = ChatBinding(_Scripted(), provider="example-chat", model="chat-small")
    result = asyncio.run(binding(_nouls("stall", "break")[:1], seconds=5))
    assert result.model_reported is None
    assert result.usage is None
    assert result.missing == {
        "model_reported": "not-sent",
        "request_id": "not-sent",
        "revision": "not-pinned",
        "runtime": "not-applicable",
        "usage.input_tokens": "not-sent",
        "usage.output_tokens": "not-sent",
    }
    assert check(binding, batches=[_nouls("stall", "break")[:1]]) == []


def test_a_chat_reply_that_leaves_out_a_detail_gives_its_reason() -> None:
    class _Bare(StubEndpoint):
        async def __call__(self, body: Mapping[str, Any], *, seconds: float) -> dict[str, Any]:
            reply = await super().__call__(body, seconds=seconds)
            return {"choices": reply["choices"]}

    result = asyncio.run(ChatBinding(_Bare(), provider="p", model="m")(_nouls()[1:], seconds=5))
    assert result.values["q:other/ok"] == {"yes": 0.1, "no": 0.9}
    assert result.claimed == ()
    assert result.extras == ()
    assert result.missing is not None
    assert {k: result.missing[k] for k in ("model_reported", "request_id")} == {
        "model_reported": "not-reported",
        "request_id": "not-reported",
    }
    assert result.missing["usage.input_tokens"] == "not-reported"
