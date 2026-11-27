"""The agent skill parses, its code compiles, and its links point into the tagged repo."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "ommat" / "SKILL.md"
TEXT = SKILL.read_text(encoding="utf-8")
TAG = "v0.5.0"


def test_the_frontmatter_names_the_skill() -> None:
    match = re.match(r"---\n(.*?)\n---\n", TEXT, re.DOTALL)
    assert match is not None
    fields = dict(line.split(": ", 1) for line in match.group(1).splitlines())
    assert fields["name"] == "ommat"
    assert fields["description"].strip()


def test_every_python_block_compiles() -> None:
    blocks = re.findall(r"^```python\n(.*?)^```$", TEXT, re.MULTILINE | re.DOTALL)
    assert blocks
    for block in blocks:
        compile(block, str(SKILL), "exec")


def test_every_ommat_link_is_pinned_to_the_tag_and_names_a_file() -> None:
    links = re.findall(r"https://github\.com/CRSiegfried/ommat/[^\s)]+", TEXT)
    assert links
    for link in links:
        prefix = f"https://github.com/CRSiegfried/ommat/blob/{TAG}/"
        assert link.startswith(prefix), link
        path = link.removeprefix(prefix).partition("#")[0]
        assert (ROOT / path).is_file(), link
