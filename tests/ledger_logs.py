"""Synthetic record logs for the ledger, record and view tests: a format-1 normal turn, broken
entry, cancelled turn and turn whose codec raised a cancellation, a format-2 extraction turn and
a format-3 turn with a described label, a call's precision and warnings and an ``unsupported``
marker; the format-8 normal turn, broken entry and cancelled turn, a finished one ending
with interpret's last item; and the format-9 normal turn and broken entry, whose settings item
holds shape trees in place of class trees. Each is built item by item so a test can take any
prefix or swap one item."""

import copy
from typing import Any, cast

from ommat._canonical import hash_value, order_of, with_order
from ommat._values import HASH_FORMAT, KEY_ORDER_FORMAT, MAX_DOCUMENT_DEPTH

HASH = "sha256:" + "ab" * 32
IDENTITY = "sha256:" + "cd" * 32

DESIGN: dict[str, Any] = {"item": "design", "hash": HASH, "format": 1, "helpers": 1}

SETTINGS: dict[str, Any] = {
    "item": "settings",
    "settings": {
        "round_bound": 8,
        "timeout": 60,
        "budget": None,
        "deadline": None,
        "certificate_cap": None,
    },
    "providers": {
        "t:mail/body": "fetch_mail",
        "t:mail/subject": "fetch_mail",
        "q:spam": "ask_spam",
        "d:route": "route",
        "d:tags": "tagger",
    },
    "classes": {
        "t:state": {"subject": "content", "to": "identifier"},
        "t:mail/body": "content",
        "t:mail/subject": "content",
        "q:spam": "open",
        "d:route": "open",
        "d:tags": ["open"],
    },
}

STATE: dict[str, Any] = {
    "item": "entry",
    "ref": "t:state",
    "round": 0,
    "call": None,
    "value": {"subject": "hello", "to": "dana@example.com"},
}


def call(round_: int, source: str, **members: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "item": "call",
        "round": round_,
        "sources": [source],
        "outcome": "returned",
        "snapshot": None,
        "elicitation": None,
        "prompt": None,
        "questions": {},
    }
    item.update(members)
    return item


def entry(
    ref: str, round_: int, call_: int | None, value: Any = None, **members: Any
) -> dict[str, Any]:
    item: dict[str, Any] = {"item": "entry", "ref": ref, "round": round_, "call": call_}
    item.update(members if "marker" in members else {"value": value, **members})
    return item


def round_item(n: int, cost: float = 0, deadline: bool = False) -> dict[str, Any]:
    return {"item": "round", "round": n, "cost": cost, "deadline": deadline}


ANSWER = {"kind": "noul", "labels": ["yes", "no"], "probabilities": {"yes": 0.9, "no": 0.1}}
QUESTIONS = {"q:spam": {"identity": IDENTITY, "kind": "noul", "labels": None}}

PLAN: dict[str, Any] = {
    "item": "plan",
    "kind": "execute",
    "steps": [
        {
            "tool": "archive",
            "arguments": {"folder": {"literal": "spam"}, "what": {"ref": "d:route"}},
            "requirements": [{"ref": "d:route", "value": "archive"}],
        }
    ],
    "reason": None,
    "reply": None,
    "cause": None,
    "derivation": [],
}
VERDICT: dict[str, Any] = {"item": "verdict", "kind": "passed", "failures": []}

# Indices: 0 design, 1 settings, 2 state, 3 call, 4-5 mail entries, 6 round 1, 7 call,
# 8 q:spam, 9 d:route, 10 d:tags, 11 round 2, 12 plan, 13 verdict.
NORMAL: list[dict[str, Any]] = [
    DESIGN,
    SETTINGS,
    STATE,
    call(1, "fetch_mail"),
    entry("t:mail/body", 1, 3, "text of the mail"),
    entry("t:mail/subject", 1, 3, "hello"),
    round_item(1, 2.5),
    call(2, "ask_spam", questions=QUESTIONS, prompt="Is this spam?"),
    entry("q:spam", 2, 7, ANSWER),
    entry("d:route", 2, None, "archive", derivation=[]),
    entry("d:tags", 2, None, [{"a": "x"}, None]),
    round_item(2, 1),
    PLAN,
    VERDICT,
]

BROKEN_PLAN: dict[str, Any] = {
    "item": "plan",
    "kind": "broken",
    "steps": [],
    "reason": None,
    "reply": None,
    "cause": {"code": "invalid-return", "refs": ["t:state"]},
}
BROKEN: list[dict[str, Any]] = [DESIGN, SETTINGS, BROKEN_PLAN, VERDICT]

CANCEL: dict[str, Any] = {"item": "cause", "cause": {"code": "cancelled", "refs": []}}
CANCELLED: list[dict[str, Any]] = [*NORMAL[:7], CANCEL]

# A cancellation the codec raised: delivered after the settings item, before any state entry.
CODEC_CANCELLED: list[dict[str, Any]] = [DESIGN, SETTINGS, CANCEL]

EXTRACTED: dict[str, Any] = {
    "kind": "extraction",
    "value_type": "number",
    "is_list": False,
    "value": 120,
    "guess": 120,
    "p_null": 0.1,
    "confidence": 0.9,
    "span": [4, 7],
}
EXTRACTION_TREE = {
    "kind": "open",
    "value_type": "open",
    "is_list": "open",
    "value": "content",
    "guess": "content",
    "p_null": "open",
    "confidence": "open",
    "span": "content",
}
# A format-2 turn asking one extraction question, its call's truncated in UNHASHED. Indices:
# 0 design, 1 settings, 2 state, 3 call, 4 q:amount, 5 round 1, 6 plan, 7 verdict.
EXTRACTION: list[dict[str, Any]] = [
    {**DESIGN, "format": 2, "helpers": 2},
    {
        "item": "settings",
        "settings": SETTINGS["settings"],
        "providers": {"q:amount": "read"},
        "classes": {"t:state": {"body": "content"}, "q:amount": EXTRACTION_TREE},
    },
    {**STATE, "value": {"body": "Owe 120"}},
    call(
        1,
        "read",
        questions={
            "q:amount": {
                "identity": IDENTITY,
                "kind": "extraction",
                "labels": None,
                "value_type": "number",
                "is_list": False,
            }
        },
    ),
    entry("q:amount", 1, 3, EXTRACTED),
    round_item(1),
    {**PLAN, "kind": "inform", "steps": [], "reply": "big"},
    VERDICT,
]

TONE = {"kind": "choice", "labels": ["warm", "cold"], "probabilities": {"warm": 0.7, "cold": 0.3}}
UNSUPPORTED = {"kind": "Unavailable", "cause": {"code": "unsupported", "refs": ["q:other"]}}
# A format-3 turn: one call asking a choice question with a described label, answered, and a
# noul question its binding refused with unsupported, its reason under the call's warnings in
# UNHASHED. Indices: 0 design, 1 settings, 2 state, 3 call, 4 q:other, 5 q:tone, 6 round 1,
# 7 plan, 8 verdict.
DESCRIBED: list[dict[str, Any]] = [
    {**DESIGN, "format": 3, "helpers": 2},
    {
        "item": "settings",
        "settings": SETTINGS["settings"],
        "providers": {"q:other": "ask", "q:tone": "ask"},
        "classes": {"t:state": {"body": "content"}, "q:other": "open", "q:tone": "open"},
    },
    {**STATE, "value": {"body": "Thanks so much"}},
    call(
        1,
        "ask",
        precision=2,
        questions={
            "q:other": {"identity": IDENTITY, "kind": "noul", "labels": None},
            "q:tone": {
                "identity": IDENTITY,
                "kind": "choice",
                "labels": [{"name": "warm", "description": "Friendly or grateful"}, "cold"],
            },
        },
    ),
    entry("q:other", 1, 3, marker=UNSUPPORTED),
    entry("q:tone", 1, 3, TONE),
    round_item(1, 1),
    {**PLAN, "kind": "inform", "steps": [], "reply": "warm"},
    VERDICT,
]

# Format 8: the settings item names the policy, ``d:route``.
DESIGN_8: dict[str, Any] = {**DESIGN, "format": 8, "helpers": 3}
SETTINGS_8: dict[str, Any] = {**SETTINGS, "grounded": {}, "sets": {}, "policy": "d:route"}
_PROFILE: dict[str, Any] = {
    "precision": None,
    "provider": None,
    "model_requested": None,
    "model_reported": None,
    "revision": None,
    "runtime": None,
    "claimed": [],
}


def call_8(round_: int, source: str, **members: Any) -> dict[str, Any]:
    """A format-8 call item: the profile's members in place of ``snapshot``."""
    item = {k: v for k, v in call(round_, source, **members).items() if k != "snapshot"}
    return {**item, **{k: v for k, v in _PROFILE.items() if k not in members}}


# Indices as NORMAL's.
NORMAL_8: list[dict[str, Any]] = [
    DESIGN_8,
    SETTINGS_8,
    STATE,
    call_8(1, "fetch_mail"),
    entry("t:mail/body", 1, 3, "text of the mail"),
    entry("t:mail/subject", 1, 3, "hello"),
    round_item(1, 2.5),
    call_8(2, "ask_spam", questions=QUESTIONS, prompt="Is this spam?"),
    entry("q:spam", 2, 7, ANSWER),
    entry("d:route", 2, None, "archive", derivation=[]),
    entry("d:tags", 2, None, [{"a": "x"}, None]),
    round_item(2, 1),
]

ENTRY_CAUSE = {"kind": "Unavailable", "cause": {"code": "invalid-return", "refs": ["t:state"]}}
# A broken entry: t:state and the policy's ref Unavailable in round 0, and nothing after.
BROKEN_8: list[dict[str, Any]] = [
    DESIGN_8,
    SETTINGS_8,
    entry("t:state", 0, None, marker=ENTRY_CAUSE),
    entry("d:route", 0, None, marker=copy.deepcopy(ENTRY_CAUSE)),
]
CANCELLED_8: list[dict[str, Any]] = [*NORMAL_8[:7], CANCEL]


def shape_of(tree: Any) -> Any:
    """A class tree's shape tree: its structure, ``"leaf"`` at every leaf."""
    if isinstance(tree, dict):
        return {k: shape_of(v) for k, v in cast("dict[str, Any]", tree).items()}
    if isinstance(tree, list):
        return [shape_of(cast("list[Any]", tree)[0])]
    return "leaf"


# Format 9: the settings item holds ``shapes``, never ``classes``.
DESIGN_9: dict[str, Any] = {**DESIGN, "format": 9, "helpers": 4}
SETTINGS_9: dict[str, Any] = {
    **{k: v for k, v in SETTINGS_8.items() if k != "classes"},
    "shapes": {ref: shape_of(tree) for ref, tree in SETTINGS["classes"].items()},
}
# Indices as NORMAL_8's.
NORMAL_9: list[dict[str, Any]] = [DESIGN_9, SETTINGS_9, *NORMAL_8[2:]]
BROKEN_9: list[dict[str, Any]] = [DESIGN_9, SETTINGS_9, *BROKEN_8[2:]]

# Format 10: a ``calls`` entry may hold ``extras``. Indices as NORMAL_8's.
DESIGN_10: dict[str, Any] = {**DESIGN_9, "format": 10}
NORMAL_10: list[dict[str, Any]] = [DESIGN_10, SETTINGS_9, *NORMAL_8[2:]]
BROKEN_10: list[dict[str, Any]] = [DESIGN_10, SETTINGS_9, *BROKEN_8[2:]]

LOGS = {
    "normal": NORMAL,
    "broken": BROKEN,
    "cancelled": CANCELLED,
    "codec-cancelled": CODEC_CANCELLED,
    "extraction": EXTRACTION,
    "described": DESCRIBED,
    "normal-8": NORMAL_8,
    "broken-8": BROKEN_8,
    "cancelled-8": CANCELLED_8,
    "normal-9": NORMAL_9,
    "broken-9": BROKEN_9,
    "normal-10": NORMAL_10,
    "broken-10": BROKEN_10,
}

# Each log's unhashed members by log-index key, written once the keyed item is in a prefix.
UNHASHED: dict[str, dict[str, dict[str, Any]]] = {
    "extraction": {
        "calls": {
            "3": {
                "start": "2026-10-20T00:00:00Z",
                "latency": 0.25,
                "usage": None,
                "attempts": None,
                "truncated": ["q:amount"],
            }
        }
    },
    "described": {
        "calls": {
            "3": {
                "start": "2026-10-20T00:00:00Z",
                "latency": 0.5,
                "usage": None,
                "attempts": None,
                "warnings": ["q:other: this provider takes no noul question"],
            }
        }
    },
    "normal-10": {
        "calls": {
            "3": {
                "start": "2026-10-20T00:00:00Z",
                "latency": 0.25,
                "usage": {"input_tokens": 3},
                "attempts": 2,
                "extras": [{"passes": 1, "note": "Dear Ann, your account 4417 is overdue"}, {}],
            },
            "7": {
                "start": "2026-10-20T00:00:01Z",
                "latency": 0.5,
                "usage": None,
                "attempts": 1,
                "extras": [{"customer": "user-8812"}],
            },
        }
    },
}


def unhashed_of(name: str, n: int) -> dict[str, Any]:
    """``name``'s unhashed members keyed by an item of its first ``n``."""
    members = {
        member: {k: v for k, v in keyed.items() if int(k) < n}
        for member, keyed in UNHASHED.get(name, {}).items()
    }
    return copy.deepcopy({m: keyed for m, keyed in members.items() if keyed})


def fresh(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return copy.deepcopy(items)


_ORDERED = ("entry", "plan", "verdict")


def reordered(item: dict[str, Any]) -> None:
    """An entry, plan or verdict item of a format-6 or later log given its ``order`` afresh, in
    place, as the ledger derives it at the append: the stale ``order`` dropped, the item's own
    members put in canonical order, then ``order_of`` the item, each object read in the order
    it holds."""
    rest = {k: v for k, v in item.items() if k != "order"}
    item.clear()
    item.update({k: rest[k] for k in sorted(rest)})  # an item's own members are ASCII
    item["order"] = order_of(item)


def _kept(item: dict[str, Any]) -> bool:
    """Whether an item keeps its ``order``: it holds one that fits it, and every object of it
    stands in canonical order, as the record's document holds it."""
    if "order" not in item:
        return False
    rest = {k: v for k, v in item.items() if k != "order"}
    try:
        with_order(rest, item["order"])
    except ValueError:
        return False
    return order_of(rest) == {}


def rehashed(doc: dict[str, Any], *, reorder: bool = True) -> dict[str, Any]:
    """A record document whose log a test edited, its ``hash`` recomputed from the log when its
    format has one, as a job that edits a record's bytes recomputes it. From format 6 each
    entry, plan and verdict item is given its ``order`` afresh (``reordered``) from the order
    its objects hold, as ``Record.log`` gives them, except an item read off the record's
    document: all in canonical order, whose ``order`` still fits, keeps it. With ``reorder``
    false none is, so a test can hold a stale one."""
    log = doc["log"]
    fmt = log[0].get("format", 0) if log and type(log[0]) is dict else 0
    if reorder and type(fmt) is int and fmt >= KEY_ORDER_FORMAT:
        for found in log:
            if type(found) is not dict:
                continue
            item = cast("dict[str, Any]", found)
            if item.get("item") in _ORDERED and not _kept(item):
                reordered(item)
    if log and type(log[0]) is dict and log[0].get("format", 0) >= HASH_FORMAT:
        doc["hash"] = hash_value(log, MAX_DOCUMENT_DEPTH)
    else:
        doc.pop("hash", None)
    return doc


def classes_of(shape: Any, name: str = "content") -> Any:
    """A shape tree's class tree with the class ``name`` at every leaf."""
    if isinstance(shape, dict):
        return {k: classes_of(v, name) for k, v in cast("dict[str, Any]", shape).items()}
    if isinstance(shape, list):
        return [classes_of(cast("list[Any]", shape)[0], name)]
    return name


def before_shapes(doc: dict[str, Any], fmt: int = 8, name: str = "content") -> dict[str, Any]:
    """A current record document made format ``fmt``, before format 9: the format number set
    and the settings item's ``shapes`` written as ``classes``, ``name`` at every leaf. Its
    ``hash`` is left to recompute."""
    doc["log"][0]["format"] = fmt
    for item in doc["log"]:
        if item["item"] == "settings" and "shapes" in item:
            item["classes"] = classes_of(item.pop("shapes"), name)
    return doc


def before_key_order(doc: dict[str, Any], fmt: int = KEY_ORDER_FORMAT - 1) -> dict[str, Any]:
    """A current record document made format ``fmt``, before format 6: ``before_shapes``, then
    each item's ``order`` and the settings item's ``grounded``, ``sets`` and ``policy`` dropped,
    its objects left in the order they hold. Its ``hash`` is left to recompute."""
    before_shapes(doc, fmt)
    for item in doc["log"]:
        item.pop("order", None)
        if item["item"] == "settings":
            item.pop("grounded", None)
            item.pop("sets", None)
            item.pop("policy", None)
    return doc


_PROFILE_ONLY = ("provider", "model_requested", "model_reported", "runtime", "claimed")
_CALLS_PROFILE = ("request_id", "calibrated", "missing", "clients")


def before_profile(doc: dict[str, Any], fmt: int) -> dict[str, Any]:
    """A current record document made format ``fmt``: ``before_key_order``, then each call item's
    ``revision`` renamed back to ``snapshot``, the profile's other members and its ``calls``
    entries' dropped, and each plan item's ``context``, as a record before format 5 holds them.
    A caller below format 3 also drops what format 3 added. Its ``hash`` is left to recompute."""
    before_key_order(doc, fmt)
    for item in doc["log"]:
        if item["item"] == "plan":
            item.pop("context", None)  # a record before format 5 holds no block
        if item["item"] == "call":
            item["snapshot"] = item.pop("revision")
            for name in _PROFILE_ONLY:
                item.pop(name, None)
            if type(item.get("precision")) is not int:
                item["precision"] = None
    for entry in doc["unhashed"].get("calls", {}).values():
        for name in _CALLS_PROFILE:
            entry.pop(name, None)
        invalid = entry.get("invalid", {})
        for name in (*_PROFILE_ONLY, "revision", *_CALLS_PROFILE):
            invalid.pop(name, None)
        if "invalid" in entry and not invalid:
            del entry["invalid"]
    return doc
