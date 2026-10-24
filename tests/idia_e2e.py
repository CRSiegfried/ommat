"""The street-reports design decided end to end through the idia binding: OpenAI's Decisions
answers the classifier oracle, Cloudflare's Clairic the extractor, each leaf wrapped in idia's
``Recorded`` leaf over a log in ``tests/idia_logs``.

``scripts/record_idia_e2e.py`` records one turn live, with the credentials it reads from the
environment, and keeps the turn's record document beside the logs;
``scripts/regenerate_idia_fixtures.py`` rebuilds that document offline from the logs when only
ommat's record changes. ``tests/test_idia_e2e.py``
replays the turn from the logs, with placeholder credentials and no network.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from idia import Dispatcher, Leaf, RetryPolicy
from idia.providers.cloudflare import Clairic
from idia.providers.openai_decisions import OpenAIDecisions
from idia.providers.recorded import Recorded

from ommat import CallResult, Decision, LoadedDesign, SourceCall, decide_sync, load
from ommat.bindings.idia import oracle_binding

LOGS = Path(__file__).parent / "idia_logs"
CLASSIFIER_LOG = LOGS / "classifier.json"
EXTRACTOR_LOG = LOGS / "extractor.json"
RECORD = LOGS / "turn.record.json"
DESIGN = LOGS / "design.json"
"""The design document the turn ran under, kept beside its record so a replay loads it from
bytes. A change to the street-reports design leaves this document and the logs as they were
recorded; loading it then fails, since its rules' source no longer matches, and the turn must be
recorded again with ``scripts/record_idia_e2e.py``."""

REPORT = "The street light outside 14 Alder Lane has flickered since 3 March and is dark by 9 pm."
STATE = {"report": REPORT, "filed_on": "2026-03-10"}

PLACEHOLDER = {
    "openai_key": "replay",
    "account_id": "00000000000000000000000000000000",
    "api_token": "replay",
}
"""What a replay builds the leaves with: never sent, since a replay calls no wire."""


async def registry(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """The registry's tool value: no place has an open work order."""
    return CallResult({"t:open_orders": []})


def leaves(credentials: dict[str, str]) -> tuple[Leaf, Leaf]:
    """The classifier's leaf and the extractor's, built with ``credentials``."""
    classifier = OpenAIDecisions(api_key=credentials["openai_key"])
    extractor = Clairic(account_id=credentials["account_id"], api_token=credentials["api_token"])
    return classifier, extractor


def decide(
    credentials: dict[str, str],
    mode: Literal["none", "once"],
    wrap: Callable[[Leaf], Leaf] = lambda leaf: leaf,
) -> Decision:
    """The turn, each leaf recorded to (``once``) or replayed from (``none``) its log, under the
    dispatcher's own retry policy when recording and with no delay when replaying. ``wrap`` is
    applied to each recorded leaf, as a test reads what it returns."""
    classifier, extractor = leaves(credentials)

    def provider(leaf: Leaf, log: Path) -> Dispatcher:
        recorded = wrap(Recorded(leaf, log, mode))
        if mode == "once":
            return Dispatcher(recorded)
        return Dispatcher(recorded, retry=RetryPolicy.without_delay())

    bindings = {
        "classifier": oracle_binding(provider(classifier, CLASSIFIER_LOG)),
        "extractor": oracle_binding(provider(extractor, EXTRACTOR_LOG)),
        "registry": registry,
    }
    return decide_sync(loaded(), STATE, bindings)


def loaded() -> LoadedDesign:
    """The design the turn ran under, from its stored document."""
    return load(DESIGN.read_bytes())


TUTORIAL = LOGS / "tutorial"
"""The logs the idia tutorial's run recorded, one per leaf."""


def recorded_providers(
    credentials: dict[str, str], mode: Literal["none", "once"], logs: Path = TUTORIAL
) -> Callable[[], Any]:
    """A function opening the classifier's and the extractor's dispatchers over their leaves
    recorded to (``once``) or replayed from (``none``) the logs under ``logs``, as the tutorial's
    ``run`` takes it."""

    @asynccontextmanager
    async def providers() -> AsyncGenerator[tuple[Dispatcher, ...]]:
        classifier, extractor = leaves(credentials)
        retry = RetryPolicy() if mode == "once" else RetryPolicy.without_delay()
        async with (
            Dispatcher(Recorded(classifier, logs / "classifier.json", mode), retry=retry) as one,
            Dispatcher(Recorded(extractor, logs / "extractor.json", mode), retry=retry) as two,
        ):
            yield one, two

    return providers
