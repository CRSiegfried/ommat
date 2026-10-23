"""The README's demo runs and prints what the README shows."""

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
