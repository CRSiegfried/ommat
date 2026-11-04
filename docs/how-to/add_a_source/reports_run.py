"""Load the model once, hold the binding to the protocol, decide three reports and replay each.

Run it from this directory with ``python reports_run.py``.
"""

from typing import Any

from ommat import (
    AnswerKind,
    Decision,
    RenderedQuestion,
    SourceCall,
    ValueType,
    decide_sync,
    load,
    read_record,
    replay_sync,
)
from ommat.bindings.conformance import check
from reports_design import DESIGN
from span_binding import SpanBinding, SpanModel
from street_spotter import StreetSpotter

REPORTS = [
    "The lamp on Quay Street has been dark since Monday.",
    "Two lamps are out, one on Elm Row and one on Mill Lane.",
    "The light by the park flickers all night.",
]

TEXT = "The lamps on Elm Row and Quay Street are out."
OWN = (
    SourceCall(
        "places",
        None,
        {"t:state.text": TEXT},
        (),
        {
            "q:places/street": RenderedQuestion(
                AnswerKind.extraction, "street", (), value_type=ValueType.text, text="t:state.text"
            ),
            "q:places/streets": RenderedQuestion(
                AnswerKind.extraction,
                "street",
                (),
                value_type=ValueType.text,
                is_list=True,
                text="t:state.text",
            ),
        },
    ),
)


class Counted:
    """A model that counts its calls, so a run can show that replay makes none."""

    def __init__(self, model: SpanModel) -> None:
        self.model = model
        self.provider, self.model_id = model.provider, model.model_id
        self.revision, self.runtime = model.revision, model.runtime
        self.calls = 0

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        self.calls += 1
        return self.model.predict(text, labels, threshold)


def answers(decision: Decision) -> dict[str, object]:
    """Each extraction question's value, or its marker's cause, from the record's entries."""
    return {
        ref: item["value"]["value"] if "value" in item else item["marker"]["cause"]["code"]
        for ref, item in decision.record.entries.items()
        if ref.startswith("q:")
    }


def main() -> list[tuple[object, int]]:
    """Decide each report and replay its record; return each decision with the model calls replay
    made."""
    model = Counted(StreetSpotter())  # load once, before decide
    binding = SpanBinding(model)
    print("conformance:", check(binding, deterministic=True))
    print("own batches:", check(binding, batches=[OWN], deterministic=True))

    loaded = load(DESIGN)
    results: list[tuple[object, int]] = []
    for report in REPORTS:
        decision = decide_sync(loaded, {"text": report}, {"spans": binding})
        print("decision:", decision.estimate)
        print("  answers:", answers(decision))

        record = read_record(decision.record.document)
        before = model.calls
        _, findings = replay_sync(record, loaded)
        print("  replay findings:", findings, "| model calls:", model.calls - before)
        results.append((decision.estimate, model.calls - before))
    return results


if __name__ == "__main__":
    main()
