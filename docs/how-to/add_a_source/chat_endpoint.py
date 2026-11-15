"""A stand-in for a chat-completions endpoint, so Path C runs with no key and no network.

``StubEndpoint()`` is the transport the binding takes: ``await endpoint(body, seconds=s)``
reads the request body a binding would post and returns the reply's JSON as a chat-completions
service shapes it, ``id``, ``model``, ``choices``, ``usage``, and one figure of this provider's
own, ``processing_ms``. It answers by keywords in the needs, with two decimal places, the same
way each time.
"""

import json
from collections.abc import Mapping
from typing import Any

KEYWORDS = {
    "yes": ("wire", "exposed", "fallen"),
    "lighting": ("lamp", "light"),
    "road": ("pothole", "road"),
}
"""The words that make the stand-in favour a label."""


class StubEndpoint:
    """``endpoint(body, seconds=s)``: a chat-completions reply to ``body``; counts its calls."""

    model = "chat-small-2026-06"  # the dated model the alias resolves to, as a reply names it

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, body: Mapping[str, Any], *, seconds: float) -> dict[str, Any]:
        self.calls += 1
        content = body["messages"][-1]["content"]
        asked = json.loads(content)
        text = json.dumps(asked["needs"]).lower()
        names = [label if isinstance(label, str) else label["name"] for label in asked["labels"]]
        favoured = [n for n in names if any(word in text for word in KEYWORDS.get(n, ()))]
        top = favoured[0] if favoured else ("no" if "no" in names else names[-1])
        rest = round(0.1 / (len(names) - 1), 2) if len(names) > 1 else 0.0
        answer = {name: 0.9 if name == top else rest for name in names}
        reply = json.dumps({"probabilities": answer})
        return {
            "id": f"chatcmpl-{self.calls}",
            "model": self.model,
            "choices": [{"message": {"role": "assistant", "content": reply, "refusal": None}}],
            "usage": {"prompt_tokens": len(content.split()), "completion_tokens": len(names) * 4},
            "processing_ms": 30 + len(content) % 50,
        }
