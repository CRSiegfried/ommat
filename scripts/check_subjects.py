"""Check that commit subjects describe library changes.

CI pipes in the subjects of the commits a push to main adds, one per line::

    git log --format=%s <base>..HEAD | python scripts/check_subjects.py

It prints one line per subject that names how a change was made rather than what changed, and
exits 1, or exits 0 when none does. Standard library only.

Each pattern is a whole word or a phrase. A numbered round is a build round only when a lap,
point, item or step follows it, so "round 3 of the fixpoint" passes while "round 5 lap 2" does
not, and "a ruled line" passes because only "ruled on" and "ruled by" are a ruling.

Before reading any subject the script checks itself: every sentence in ``PLANTED`` must be
caught and every sentence in ``ORDINARY`` must pass, so a pattern that stops matching, or starts
matching ordinary prose, fails the run.
"""

import re
import sys

TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "build round",
        re.compile(r"\bround ?\d+\b[\s,:;-]*(?:laps?|points?|items?|steps?)\b", re.IGNORECASE),
    ),
    ("build item", re.compile(r"\bitems? ?\d+\b", re.IGNORECASE)),
    ("orchestration", re.compile(r"\borchestrat\w*", re.IGNORECASE)),
    ("delegation", re.compile(r"\bdelegation\b|\buser'?s delegate\b", re.IGNORECASE)),
    ("office", re.compile(r"\boffices?\b", re.IGNORECASE)),
    ("charter", re.compile(r"\bcharters?\b", re.IGNORECASE)),
    ("ruling", re.compile(r"\brulings?\b|\bruled (?:on|by)\b", re.IGNORECASE)),
    ("section sign", re.compile(r"§")),
)

PLANTED: tuple[str, ...] = (
    "Closed in round 2 point 3",
    "Round 5 lap 2: the cassettes",
    "Item 4 closed",
    "The items 3 and 4 land",
    "Orchestrator review fixes",
    "Made on a reviewer's delegation",
    "As the user's delegate asked",
    "The office reviewed it",
    "Charters for the new code",
    "Rulings applied",
    "Ruled on by the reviewer",
    "As §some-slug says",
)

ORDINARY: tuple[str, ...] = (
    "Format 4: each binding call gets its seconds",
    "Round 0 is a rule pass; round 3 of the fixpoint asks the held sources",
    "The round_bound is 8, and a source is asked again in round 2",
    "An item of the tuple is checked; the items are typed",
    "The checker ruled out the bad value on a ruled line",
    "A wrapper that delegates to the inner binding",
    "Officially supported Python versions",
    "Step 1.5 of the plan binds an earlier step's result",
    "Record each binding's deadline in its call item",
)


def hits(subject: str) -> list[str]:
    """The names of the terms ``subject`` holds."""
    return [name for name, pattern in TERMS if pattern.search(subject)]


def self_check() -> list[str]:
    """One problem per planted sentence the patterns miss and per ordinary one they catch."""
    problems = [f"self-check: missed {s!r}" for s in PLANTED if not hits(s)]
    problems += [f"self-check: caught {s!r} as {hits(s)}" for s in ORDINARY if hits(s)]
    return problems


def main(subjects: list[str]) -> int:
    problems = self_check()
    if not problems:
        for subject in subjects:
            found = hits(subject)
            if found:
                problems.append(f"{subject!r}: {', '.join(found)}")
    for problem in problems:
        print(problem)
    if problems:
        print(f"check_subjects: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print(f"check_subjects: {len(subjects)} subject(s) pass")
    return 0


if __name__ == "__main__":
    sys.exit(main([line for line in sys.stdin.read().splitlines() if line.strip()]))
