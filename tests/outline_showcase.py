"""A small design for the outline tests that exercises what the stored fixtures leave out: a
function rule with a ``boundaries`` field, a table with a helper input, a source requirement,
wording slots from the state and from a parameter, and targets; and stub bindings that answer
it, so a test decides a turn under it. The documents the outline and lens tests read are stored
under tests/showcase/ (``stored``), and ``policy_free`` gives them as a design that names no
policy and its records are written."""

import json
from pathlib import Path
from typing import Any

from ledger_logs import rehashed
from ommat import (
    CallResult,
    Design,
    Helper,
    LoadedDesign,
    Parameter,
    Question,
    ReadView,
    Record,
    Rule,
    Source,
    SourceCall,
    SourceRequirement,
    Table,
    Target,
    Text,
    Unsure,
    decide_sync,
    load,
)
from ommat import types as t
from ommat._canonical import encode
from ommat._documents import design_key
from ommat.bindings import OracleBinding
from ommat.helpers import pick_with_margin

SHOWCASE = Path(__file__).parent / "showcase"


def stored(name: str) -> bytes:
    """The stored document ``tests/showcase/<name>.json``: ``design``, ``plain.design`` or
    ``full.design``, a format-11 design that names a policy; or ``<turn>.record``, a format-10
    record decided under one, for the turns ``decided``, ``undecided``, ``level``, ``broken``
    and ``plain``."""
    return (SHOWCASE / f"{name}.json").read_bytes()


def policy_free(design: str = "design") -> bytes:
    """The stored design ``design`` as a design that names no policy is written: no
    ``policy`` member, no source's ``feedback`` member, design format 12."""
    doc: dict[str, Any] = json.loads(stored(design))
    del doc["policy"]
    for source in doc["sources"].values():
        del source["feedback"]
    doc["format"] = 12
    return encode(doc)


def policy_free_record(turn: str, design: str = "design") -> bytes:
    """The stored record ``turn`` as a turn under ``policy_free(design)`` is written: record
    format 11, its ``design`` item holding that design's hash, its ``settings`` item no
    ``policy`` and its run settings no ``certificate_cap``, and a broken entry's log ending at
    its ``t:state`` entry."""
    doc: dict[str, Any] = json.loads(stored(f"{turn}.record"))
    log: list[dict[str, Any]] = doc["log"]
    log[0].update(format=11, hash=design_key(policy_free(design)))
    del log[1]["policy"], log[1]["settings"]["certificate_cap"]
    state = next(i for i, x in enumerate(log) if x.get("ref") == "t:state")
    if "marker" in log[state]:
        del log[state + 1 :]
    return encode(rehashed(doc))


def urgency(r: ReadView) -> object:
    return pick_with_margin(r, "q:urgent", r.param("urgent/bar"), r.param("urgent/margin"))


DESIGN = Design(
    name="showcase",
    state=t.record(subject=t.string, body=t.string),
    targets={"tg:urgent": Target("The mail needs an answer today", "yes or no")},
    sources={
        "ask": Source(
            {
                "q:urgent": Question(
                    "noul",
                    Text(
                        "Does the mail titled {subject} need an answer today? (bar {bar})",
                        [
                            {"name": "subject", "origin": {"ref": "t:state.subject"}},
                            {"name": "bar", "origin": {"param": "urgent/bar"}},
                        ],
                    ),
                    target="tg:urgent",
                ),
                "q:tone": Question(
                    "choice",
                    Text("Which tone is the mail written in?"),
                    labels=("warm", "cold", "neutral"),
                    latent_targets=[{"target": "tg:urgent", "role": "unknown"}],
                ),
            },
            needs=["t:state.subject", "t:state.body"],
            oracle="llm",
            cost=1,
        ),
        "desk": Source(
            {"t:ticket": t.string},
            needs=["t:state.subject"],
            requirement=SourceRequirement("d:urgency", ["yes"]),
            oracle="desk",
        ),
    },
    rules={
        "urgency": Rule(
            "d:urgency",
            t.label("no", "yes"),
            urgency,
            version="1",
            reads=["q:urgent"],
            params=["urgent/bar", "urgent/margin"],
            boundaries={
                "q:urgent": {
                    "bar": ["urgent/bar"],
                    "thresholds": [],
                    "margin": ["urgent/margin"],
                    "mass": [],
                }
            },
            target="tg:urgent",
        ),
        "act": Table(
            "d:act",
            t.label("escalate", "reply", "file"),
            "F",
            inputs=[
                "d:urgency",
                Helper("pick_with_margin", "q:tone", bar="tone/bar", margin="tone/margin"),
                "t:ticket",
            ],
            rows=[
                ["yes", "-", "Known", "escalate"],
                ["no", "{warm, neutral}", "-", "reply"],
                ["no", "cold", "-", "file"],
                ["Undetermined", "-", "-", Unsure],
            ],
        ),
    },
    parameters={
        "urgent/bar": Parameter(t.number(0, 1), 0.6),
        "urgent/margin": Parameter(t.number(0, 1), 0.2),
        "tone/bar": Parameter(t.number(0, 1), 0.5),
        "tone/margin": Parameter(t.number(0, 1), 0.1),
    },
)

STATE = {"subject": "Server down", "body": "Nothing loads since 9."}
DECIDED = (0.85, {"warm": 0.1, "cold": 0.8, "neutral": 0.1})
"""The answers of a turn that decides: urgency ``yes``, so ``escalate``, and a tone that clears
its bar and margin."""
UNDECIDED = (0.55, {"warm": 0.45, "cold": 0.4, "neutral": 0.15})
"""The answers of a turn whose urgency abstains, short of both its bar and its margin."""


def llm(p_yes: float, tone: dict[str, float]) -> OracleBinding:
    """A binding answering ``q:urgent`` with ``p_yes`` and ``q:tone`` with ``tone``."""

    async def binding(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult(
            {"q:urgent": {"yes": p_yes, "no": round(1 - p_yes, 10)}, "q:tone": tone},
            provider="example",
            model_requested="m-1",
            model_reported="m-1-2026",
            precision=2,
            elicitation="logprobs",
            prompt="Does the mail titled " + "x" * 200,
            usage={"input_tokens": 120, "output_tokens": 3},
            attempts=1,
        )

    return binding


async def desk(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
    return CallResult({"t:ticket": "T-1001"})


def loaded() -> LoadedDesign:
    return load(DESIGN)


def turn(design: LoadedDesign, p_yes: float, tone: dict[str, float]) -> Record:
    """The record of one turn under the showcase design with the answers given."""
    return decide_sync(design, STATE, {"llm": llm(p_yes, tone), "desk": desk}).record


def flag(r: ReadView) -> object:
    return True


def verdict(r: ReadView) -> object:
    return "yes"


PLAIN = Design(
    name="plain",
    state=t.record(subject=t.string),
    sources={
        "ask": Source(
            {
                "q:urgent": Question(
                    "noul",
                    Text(
                        "Does the mail titled {subject} need an answer today?",
                        [{"name": "subject", "origin": {"ref": "t:state.subject"}}],
                    ),
                )
            },
            needs=["t:state.subject"],
            oracle="llm",
        )
    },
    rules={
        "flag": Rule("d:flag", t.boolean, flag, version="1", reads=["q:urgent"]),
        "verdict": Rule(
            "d:verdict",
            t.label("no", "yes"),
            verdict,
            version="1",
            reads=["d:flag", "q:urgent", "t:state.subject"],
        ),
    },
)
"""Two plain function rules, no derivation and no table: the policy reads ``d:flag``, which
reads the one question, so the record names nothing either rests on."""


def plain_turn() -> Record:
    """The record of one decided turn under ``PLAIN``."""

    async def ask(batch: tuple[SourceCall, ...], *, seconds: float) -> CallResult:
        return CallResult({"q:urgent": {"yes": 0.9, "no": 0.1}}, provider="example")

    return decide_sync(load(PLAIN), {"subject": "Server down"}, {"llm": ask}).record
