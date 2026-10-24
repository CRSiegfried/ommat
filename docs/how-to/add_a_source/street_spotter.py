"""A stand-in for a local span model, such as a named-entity model run in-process on torch.

It has the shape ``span_binding.SpanModel`` asks for: it loads once, then ``predict`` takes a
text, the entity labels to look for and a score threshold, and returns each span it finds with
its offsets and score. A real model is wrapped in an adapter to that shape. It finds only
streets, a capitalised word or two before Street, Road, Lane or Row, so the example runs with no
model to download.
"""

import re
from collections.abc import Mapping
from typing import Any

_STREET = re.compile(r"\b(?:[A-Z][a-z]+ ){1,2}(Street|Road|Lane|Row)\b")
_SCORES = {"Street": 0.95, "Road": 0.95, "Lane": 0.7, "Row": 0.7}


class StreetSpotter:
    """The model: ``model_id`` and ``revision`` name the weights a real model loads, and
    ``runtime`` the code that runs them."""

    provider = "street_spotter"
    model_id = "example/street-spotter"
    revision = "1"
    runtime: Mapping[str, str] = {"python": "3.11", "street_spotter": "1"}

    def __init__(self) -> None:
        pass  # a real model loads its weights here, which takes seconds

    def predict(self, text: str, labels: list[str], threshold: float) -> list[dict[str, Any]]:
        """Each span of a label in ``labels`` scored at or above ``threshold``, in text order."""
        if "street" not in labels:
            return []
        found = [
            {"label": "street", "text": m[0], "start": m.start(), "end": m.end(), "score": score}
            for m in _STREET.finditer(text)
            if (score := _SCORES[m[1]]) >= threshold
        ]
        return found
