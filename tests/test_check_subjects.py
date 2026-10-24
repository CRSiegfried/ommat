"""``scripts/check_subjects.py``, which CI runs over the commit subjects a push to main adds:
it passes subjects that describe library changes and fails one that names how a change was
made."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "check_subjects.py"


def run(subjects: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)], input=subjects, capture_output=True, text=True, check=False
    )


def test_plain_subjects_pass() -> None:
    subjects = (
        "Format 4: each binding call gets its seconds\n\nRound 3 of the fixpoint\n"
        "The call item records its deadline\n"
    )
    result = run(subjects)
    assert result.returncode == 0
    assert result.stdout == "check_subjects: 3 subject(s) pass\n"


def test_a_subject_in_build_vocabulary_fails_naming_its_terms() -> None:
    result = run("Plain subject\nOffice review: round 2 point 3 closed\n")
    assert result.returncode == 1
    assert result.stdout == "'Office review: round 2 point 3 closed': build round, office\n"
