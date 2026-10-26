"""Every ``# spec: <section>`` comment in the package names a heading of the spec, and the
sites that write a source requirement's markers cite its section."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "docs" / "spec" / "README.md"
CITES = re.compile(r"# spec: ([^:]+?)(?::|$)")


def _headings() -> set[str]:
    return set(re.findall(r"^## (.+?)\s*$", SPEC.read_text(encoding="utf-8"), re.MULTILINE))


def test_every_spec_citation_in_the_package_names_a_spec_heading() -> None:
    headings = _headings()
    cited = 0
    for module in sorted((ROOT / "src" / "ommat").rglob("*.py")):
        for n, line in enumerate(module.read_text(encoding="utf-8").splitlines(), 1):
            for section in CITES.findall(line):
                cited += 1
                assert section in headings, f"{module.name}:{n}: {section!r}"
    assert cited


SITES = re.compile(
    r"Cause\(_C\.(when_false, \[requirement\.ref\]"
    r"|need_(unavailable|undetermined), code_point_sorted)"
)


def test_the_source_requirement_markers_cite_their_section() -> None:
    lines = (ROOT / "src" / "ommat" / "_interpret.py").read_text(encoding="utf-8").splitlines()
    sites = [n for n, line in enumerate(lines) if SITES.search(line)]
    assert len(sites) == 3
    for n in sites:
        assert CITES.findall(lines[n - 1].strip()) == ["Source requirements"], lines[n]
