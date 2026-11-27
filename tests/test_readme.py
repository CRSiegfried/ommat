"""The README's demo runs and prints what the README shows, and the demo its recording
shows decides and replays as recorded."""

import os
import re
import subprocess
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def _demo() -> tuple[str, str]:
    text = README.read_text(encoding="utf-8")
    start = text.index("<!-- demo -->")
    end = text.index("<!-- /demo -->", start)
    blocks = re.findall(r"^```(\w+)\n(.*?)^```$", text[start:end], re.MULTILINE | re.DOTALL)
    code = "\n".join(body for lang, body in blocks if lang == "python")
    (output,) = [body for lang, body in blocks if lang == "text"]
    return code, output


def test_the_demo_prints_what_the_readme_shows(tmp_path: Path) -> None:
    code, output = _demo()
    path = tmp_path / "demo.py"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-I", path], capture_output=True, text=True, check=True
    )
    assert result.stdout == output


DEMO = README.parent / "docs" / "assets" / "demo"
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _assistant(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, DEMO / "assistant.py", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        cwd=cwd,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    return ANSI.sub("", result.stdout)


def test_the_recorded_demo_decides_and_replays(tmp_path: Path) -> None:
    found = _assistant(tmp_path, "Find my recent emails from Neo or Smith")
    assert "asked 5 of 9 questions in 2 model calls" in found
    assert '→ find_email(senders=["Neo", "Smith"])' in found
    vague = _assistant(tmp_path, "Can you deal with that thing from yesterday")
    assert "asked 3 of 9 questions in 1 model call" in vague
    assert "→ undecided: Unsure, no clear answer at q:asks/find_email" in vague
    replayed = _assistant(tmp_path, "--replay")
    assert "replayed 2 recorded turns" in replayed
    assert "0 model calls, 0 differences" in replayed
