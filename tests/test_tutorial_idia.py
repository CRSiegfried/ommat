"""The idia tutorial runs as it says: docs/tutorials/decide_with_idia, replayed from the
logs its live run recorded, with placeholder credentials and no network."""

import asyncio
import contextlib
import io
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("idia")
pytest.importorskip("openai")
pytest.importorskip("httpx2")

import decide_live
import idia_e2e as e2e
from ommat.bindings.idia import oracle_binding

TUTORIAL = Path(__file__).parent.parent / "docs" / "tutorials" / "decide-with-idia.md"


def replayed() -> tuple[str, list[Any]]:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        results = asyncio.run(decide_live.run(e2e.recorded_providers(e2e.PLACEHOLDER, "none")))
    return printed.getvalue(), results


def test_the_run_replays_from_its_logs_to_what_the_live_run_printed() -> None:
    printed, _ = replayed()
    assert printed == (e2e.TUTORIAL / "output.txt").read_text(encoding="utf-8")


def test_each_turn_decides_and_replays_with_no_finding() -> None:
    _, results = replayed()
    first, second, again = results
    decided, findings, replays = first
    assert decided["marker"]["kind"] == "Unsure"
    assert decided["marker"]["cause"]["refs"] == ["q:triage/hazard"]
    assert second[0]["value"] == "send_crew"
    for _, findings, replays in (first, second):
        assert findings == []
        assert replays
    decided, record = again
    assert decided["marker"] == first[0]["marker"]
    registry = next(i for i in record.log if i.get("ref") == "t:open_orders")
    assert registry["marker"]["cause"]["code"] == "unsupported"


def test_the_tutorial_shows_what_the_run_printed() -> None:
    text = TUTORIAL.read_text(encoding="utf-8")
    for line in (e2e.TUTORIAL / "output.txt").read_text(encoding="utf-8").splitlines():
        assert line.strip() in text, line


def test_the_live_providers_are_dispatchers_the_binding_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable, value in (
        ("OPENAI_API_KEY", "placeholder"),
        ("CLOUDFLARE_ACCOUNT_ID", "0" * 32),
        ("CLOUDFLARE_API_TOKEN", "placeholder"),
    ):
        monkeypatch.setenv(variable, value)

    async def opened() -> list[object]:
        async with decide_live.live_providers() as providers:
            return [oracle_binding(p) for p in providers]

    assert len(asyncio.run(opened())) == 2
