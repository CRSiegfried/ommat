"""Load the model once, hold the binding to the protocol, decide three reports and replay each.

Run it from this directory with ``python reports_run.py``.
"""

from ommat import (
    AnswerKind,
    Plan,
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
from span_binding import SpanBinding
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
            "q:places/streets": RenderedQuestion(
                AnswerKind.extraction,
                "street",
                (),
                value_type=ValueType.text,
                is_list=True,
                text="t:state.text",
            )
        },
    ),
)


def main() -> list[tuple[Plan, int]]:
    """Decide each report and replay its record; return each plan with the model calls replay
    made."""
    model = StreetSpotter()  # load once, before decide
    binding = SpanBinding(model)
    print("conformance:", check(binding, deterministic=True))
    print("own batches:", check(binding, batches=[OWN], deterministic=True))

    loaded = load(DESIGN)
    results: list[tuple[Plan, int]] = []
    for report in REPORTS:
        decision = decide_sync(loaded, {"text": report}, {"spans": binding})
        plan = decision.plan
        print(plan.kind, "|", plan.reply or plan.reason, "|", decision.verdict.kind)

        record = read_record(decision.record.document)
        before = model.calls
        _, findings = replay_sync(record, loaded)
        print("  replay findings:", findings, "| model calls:", model.calls - before)
        results.append((plan, model.calls - before))
    return results


if __name__ == "__main__":
    main()
