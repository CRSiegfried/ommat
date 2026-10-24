"""Path A run offline: the design's oracle answered through idia, by idia's scripted leaf.

The scripted leaf stands in for a model leaf: it declares capabilities like a text span leaf's,
with an input window of 256 tokens, and answers each question by the street finder
``street_spotter.py`` holds. Run it from this
directory with ``python idia_run.py``; it needs ``ommat[idia]``.
"""

import asyncio
from typing import Any

from idia import (
    UNBOUNDED,
    Capabilities,
    Deadline,
    Dispatcher,
    ExtractionAnswer,
    Rounding,
    Span,
    TextPart,
    WireRequest,
)
from idia.providers.scripted import Scripted

from idia_path import idia_bindings
from ommat import PlanKind, decide_sync, load, read_record, replay_sync
from ommat.bindings.conformance import check
from reports_design import DESIGN
from reports_run import answers
from street_spotter import StreetSpotter

REPORTS = [
    "Two lamps are out, one on Elm Row and one on Mill Lane.",
    "The lamp on Quay Street is out. " * 40,
]

CAPABILITIES = Capabilities(
    kinds=frozenset({"extraction"}),
    modalities=frozenset({"text"}),
    image_carriers=frozenset(),
    max_input_tokens=256,
    max_question_tokens=256,
    max_body_bytes=UNBOUNDED,
    max_questions=UNBOUNDED,
    max_images=0,
    max_images_per_question=0,
    max_image_bytes=0,
    max_total_image_bytes=0,
    max_labels=0,
    max_levels=0,
    max_total_options=0,
    min_levels=0,
    min_image_side=0,
    image_media_types=None,
    key_patterns=(),
    recounts_state_per_question=False,
    returns_refusal=False,
    calibrated=False,
    snapshot_pinnable=True,
    carries_noul_descriptions=False,
    carries_structured_instructions=False,
    truncates_silently=False,
    confidence={},
    rounding=Rounding(probability_decimals=None, score_decimals=None),
    opt_in_features=frozenset(),
    enabled_features=frozenset(),
    feature_kinds={},
    feature_modalities={},
)
"""Capabilities like a text span leaf's: extractions of text, with an input window of 256
tokens so that a long report overflows it."""


class Spans:
    """The leaf's script: each question's answer, as a span leaf maps one, the top span or every
    span; ``calls`` counts the requests it answered."""

    def __init__(self) -> None:
        self.spotter = StreetSpotter()
        self.calls = 0

    def __call__(self, request: WireRequest, deadline: Deadline) -> dict[str, Any]:
        self.calls += 1
        part = request.state.parts[0]
        assert isinstance(part, TextPart)
        out: dict[str, Any] = {}
        for q in request.questions:
            assert isinstance(q.instruction, str)  # the record carries no structured one
            found = self.spotter.predict(part.text, [q.instruction], 0.5)
            if q.is_list:
                out[q.key] = ExtractionAnswer(
                    value=tuple(f["text"] for f in found),
                    span=tuple(Span(part=0, start=f["start"], end=f["end"]) for f in found),
                )
            elif found:
                top = max(found, key=lambda f: f["score"])
                span = Span(part=0, start=top["start"], end=top["end"])
                out[q.key] = ExtractionAnswer(value=top["text"], confidence=top["score"], span=span)
            else:
                out[q.key] = ExtractionAnswer(value=None)
        return out


async def provider(leaf: Any) -> Dispatcher:
    """Load the leaf when it has a load step, then wrap it in a dispatcher."""
    load_step = getattr(leaf, "load", None)
    if load_step is not None:
        await load_step()
    return Dispatcher(leaf)


def main() -> list[PlanKind]:
    """Decide each report through idia and replay it; return each plan's kind."""
    script = Spans()
    leaf = Scripted(capabilities=CAPABILITIES, script=script)
    dispatcher = asyncio.run(provider(leaf))  # loaded on one loop, then decided on others
    print("conformance:", check(idia_bindings(dispatcher)["spans"], deterministic=True))
    loaded = load(DESIGN)
    kinds: list[PlanKind] = []
    for report in REPORTS:
        decision = decide_sync(loaded, {"text": report}, idia_bindings(dispatcher))
        plan = decision.plan
        print(plan.kind, "|", plan.reply or plan.reason, "|", decision.verdict.kind)
        print("  answers:", answers(decision))
        for call in decision.record.unhashed["calls"].values():
            for line in call.get("warnings", []):
                print("  warning:", line)
        before = script.calls
        _, findings = replay_sync(read_record(decision.record.document), loaded)
        print("  replay findings:", findings, "| leaf calls:", script.calls - before)
        kinds.append(plan.kind)
    asyncio.run(dispatcher.aclose())  # release the leaf: a loaded model's worker, a client
    return kinds


if __name__ == "__main__":
    main()
