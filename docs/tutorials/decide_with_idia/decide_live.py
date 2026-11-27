"""Decide two street reports with real models through the idia binding, read each record,
replay it, and answer a turn from a stored record.

Run it from this directory, with ``street_reports_design.py`` importable and the credentials in
the environment (``OPENAI_API_KEY``, ``CLOUDFLARE_ACCOUNT_ID`` and ``CLOUDFLARE_API_TOKEN``)::

    PYTHONPATH=../street_reports python decide_live.py
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import idia

from ommat import (
    CallResult,
    Decision,
    LoadedDesign,
    SourceCall,
    decide,
    from_record,
    load,
    read_record,
    replay,
)
from ommat.bindings.idia import oracle_binding
from street_reports_design import DESIGN

REPORTS = [
    "The street light outside 14 Alder Lane has flickered since 3 March and is dark by 9 pm.",
    "There is a deep pothole on Station Road near the library; I first saw it on 2 March.",
]
FILED_ON = "2026-03-10"

Providers = Callable[[], AbstractAsyncContextManager[tuple[idia.Dispatcher, ...]]]


@asynccontextmanager
async def live_providers() -> AsyncGenerator[tuple[idia.Dispatcher, ...]]:
    """OpenAI's Decisions for the classifier and Cloudflare's Clairic for the extractor, each
    closed when the block ends."""
    async with (
        idia.provider("openai-decisions") as classifier,
        idia.provider("cloudflare-clairic") as extractor,
    ):
        yield classifier, extractor


async def registry(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    """The city's registry, a tool: Station Road already has an open work order."""
    return CallResult({"t:open_orders": ["Station Road"]})


ROUTE = "d:route"
"""The ref the caller acts on: the design's route rule."""


def said(entry: dict[str, Any]) -> str:
    """A value entry as its value, a marker entry as its kind and its cause's code and refs."""
    if "value" in entry:
        return str(entry["value"])
    marker = entry["marker"]
    cause = marker["cause"]
    return f"{marker['kind']} ({cause['code']} on {', '.join(cause['refs'])})"


def show(decision: Decision) -> None:
    """The decision, then each call: its sources, precision, attempts and warnings lines."""
    print("decision:", said(decision.record.entries[ROUTE]))
    unhashed = decision.record.unhashed["calls"]
    for index, item in enumerate(decision.record.log):
        if item["item"] == "call":
            call = unhashed[str(index)]
            print(
                f"  call {index}: {item['sources']}, precision {item['precision']}, "
                f"attempts {call['attempts']}"
            )
            for line in call.get("warnings", []):
                print(f"    {line}")


async def run(providers: Providers = live_providers) -> list[Any]:
    """Decide each report, replay its record, then answer the first turn from its record with
    the registry bound to the idia binding; return what each turn gave."""
    loaded: LoadedDesign = load(DESIGN)
    results: list[Any] = []
    stored: list[bytes] = []
    async with providers() as (classifier, extractor):
        bindings = {
            "classifier": oracle_binding(classifier),
            "extractor": oracle_binding(extractor),
            "registry": registry,
        }
        for report in REPORTS:
            decision = await decide(loaded, {"report": report, "filed_on": FILED_ON}, bindings)
            show(decision)
            stored.append(decision.record.document)
            rerun, findings = await replay(read_record(stored[-1]), loaded)
            print("  replay findings:", findings)
            results.append((decision.record.entries[ROUTE], findings, rerun is not None))

        # The first turn again, every model answer from its record and no provider called; the
        # registry is bound to the idia binding, which asks q: refs only.
        first = read_record(stored[0])
        answered = {
            "classifier": from_record(first, loaded),
            "extractor": from_record(first, loaded),
            "registry": oracle_binding(classifier),
        }
        state = {"report": REPORTS[0], "filed_on": FILED_ON}
        again = await decide(loaded, state, answered)
        show(again)
        results.append((again.record.entries[ROUTE], again.record))
    return results


def main() -> list[Any]:
    return asyncio.run(run())


if __name__ == "__main__":
    main()
