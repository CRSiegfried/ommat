"""Hold Path C's binding to the protocol, decide two reports through it and replay each.

Run it from this directory with ``python chat_run.py``. The endpoint is a stand-in, so the
script needs no key, no network and no idia.
"""

from chat_binding import ChatBinding
from chat_endpoint import StubEndpoint
from ommat import Decision, decide_sync, load, read_record, replay_sync
from ommat.bindings.conformance import check
from triage_design import DESIGN

REPORTS = [
    "The lamp on Quay Street is down and a wire is exposed at its base.",
    "There is a deep pothole on Mill Lane.",
]


def provenance(decision: Decision) -> dict[str, object]:
    """The call's provenance: its hashed details from the call item, and from the call's
    ``unhashed`` entry what lies outside every hash."""
    record = read_record(decision.record.document)
    ((index, item),) = [(i, x) for i, x in enumerate(record.log) if x["item"] == "call"]
    entry = record.unhashed["calls"][str(index)]
    hashed = ("provider", "model_requested", "model_reported", "precision", "claimed")
    kept = ("usage", "attempts", "missing", "extras")
    return {key: item[key] for key in hashed} | {key: entry[key] for key in kept}


def main() -> list[tuple[object, int]]:
    """Decide each report and replay its record; return each decision with the endpoint calls
    replay made."""
    endpoint = StubEndpoint()
    binding = ChatBinding(endpoint, provider="example-chat", model="chat-small")
    print("conformance:", check(binding, deterministic=True))

    loaded = load(DESIGN)
    results: list[tuple[object, int]] = []
    for report in REPORTS:
        decision = decide_sync(loaded, {"text": report}, {"chat": binding})
        print("decision:", decision.estimate)
        entries = decision.record.entries
        print(
            "  answers:",
            {r: e["value"]["probabilities"] for r, e in entries.items() if r[0] == "q"},
        )
        for key, value in provenance(decision).items():
            print(f"  {key}: {value}")

        before = endpoint.calls
        _, findings = replay_sync(read_record(decision.record.document), loaded)
        print("  replay findings:", findings, "| endpoint calls:", endpoint.calls - before)
        results.append((decision.estimate, endpoint.calls - before))
    return results


if __name__ == "__main__":
    main()
