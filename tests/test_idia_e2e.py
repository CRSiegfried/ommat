"""The idia binding end to end: the street-reports turn recorded live against OpenAI's
Decisions and Cloudflare's Clairic replays from its logs to the same record, and every answer
the leaves returned reaches the record as returned."""

from typing import Any

import pytest

pytest.importorskip("idia")
pytest.importorskip("openai")
pytest.importorskip("httpx2")

from idia import (
    Capabilities,
    Deadline,
    ExtractionAnswer,
    LabelAnswer,
    Leaf,
    NoulAnswer,
    TokenEstimator,
    WireRequest,
    WireResponse,
    noul_complement,
)

import idia_e2e as e2e
from ommat import read_record, replay_sync


class Spy:
    """A leaf that passes each call to ``leaf`` and keeps what it returned."""

    def __init__(self, leaf: Leaf) -> None:
        self.leaf = leaf
        self.responses: list[WireResponse] = []

    @property
    def name(self) -> str:
        return self.leaf.name

    @property
    def model(self) -> str:
        return self.leaf.model

    @property
    def capabilities(self) -> Capabilities:
        return self.leaf.capabilities

    @property
    def estimator(self) -> TokenEstimator:
        return self.leaf.estimator

    async def call(self, wire_request: WireRequest, deadline: Deadline) -> WireResponse:
        response = await self.leaf.call(wire_request, deadline)
        self.responses.append(response)
        return response


def stored() -> Any:
    return read_record(e2e.RECORD.read_bytes())


def test_the_turn_replays_from_the_logs_to_the_recorded_record() -> None:
    live = stored()
    again = e2e.decide(e2e.PLACEHOLDER, "none").record
    assert again.record_hash == live.record_hash
    assert again.log == live.log


def test_the_recorded_record_replays_with_no_finding() -> None:
    rerun, findings = replay_sync(stored(), e2e.loaded())
    assert findings == []
    assert rerun is not None


def test_each_answer_reaches_the_record_as_the_leaf_returned_it() -> None:
    spies: list[Spy] = []

    def wrap(leaf: Leaf) -> Leaf:
        spies.append(Spy(leaf))
        return spies[-1]

    record = e2e.decide(e2e.PLACEHOLDER, "none", wrap).record
    entries = {i["ref"]: i for i in record.log if i["item"] == "entry"}
    answered: dict[str, Any] = {}
    for spy in spies:
        for response in spy.responses:
            answered.update(response.answers)
    assert sorted(answered) == sorted(r for r in entries if r.startswith("q:"))
    for ref, answer in answered.items():
        value = entries[ref]["value"]
        if isinstance(answer, NoulAnswer):
            yes = answer.probability
            assert value["probabilities"] == {"yes": yes, "no": noul_complement(yes)}
        elif isinstance(answer, LabelAnswer):
            assert value["probabilities"] == dict(answer.probabilities)
        else:
            assert isinstance(answer, ExtractionAnswer)
            assert (value["p_null"], value["confidence"]) == (answer.p_null, answer.confidence)
            assert value["value"] == (
                list(answer.value) if isinstance(answer.value, tuple) else str(answer.value)
            )


def test_the_record_holds_each_calls_provenance() -> None:
    record = stored()
    calls = [i for i in record.log if i["item"] == "call"]
    assert [c["sources"] for c in calls] == [["details", "timing"], ["registry"], ["triage"]]
    assert all(c["outcome"] == "returned" for c in calls)
    by_source = {c["sources"][0]: c for c in calls}
    assert by_source["triage"]["precision"] == 2
    assert by_source["details"]["precision"] is None
    assert by_source["triage"]["prompt"] is not None
