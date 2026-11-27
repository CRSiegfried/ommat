"""The design behind the README's terminal recording: an assistant that turns a spoken request
into one tool call.

Round one asks three yes-or-no questions in one call: which action the request is for. Each
action then has a source of its own, asked only when ``d:action`` names it, that reads the
action's arguments out of the request. ``d:call`` is the entry the caller acts on.
"""

import json

from ommat import (
    Cause,
    Design,
    Question,
    ReadView,
    Rule,
    Source,
    SourceRequirement,
    Text,
    Undetermined,
    Unsure,
)
from ommat import types as t
from ommat.helpers import extract, pick_with_margin

ACTIONS = ("find_email", "send_email", "set_reminder")
ASKS = {
    "find_email": "Does the user ask to find emails?",
    "send_email": "Does the user ask to send an email?",
    "set_reminder": "Does the user ask to be reminded of something?",
}
ARGUMENTS = {
    "find_email": {"senders": "Each person the emails are from", "about": "What they are about"},
    "send_email": {"to": "Each person the email goes to", "subject": "A subject for the email"},
    "set_reminder": {"what": "What to be reminded of", "when": "When the reminder is due"},
}
LISTS = ("senders", "to")


def action(r: ReadView) -> object:
    """The one action whose question came back yes, or abstain."""
    refs = [f"q:asks/{name}" for name in ACTIONS]
    picked = [pick_with_margin(r, ref, 0.7, 0.4) for ref in refs]
    for pick in picked:
        if isinstance(pick, Undetermined):
            return pick
    yes = [name for name, pick in zip(ACTIONS, picked, strict=True) if pick == "yes"]
    return yes[0] if len(yes) == 1 else Unsure(Cause("abstained", refs))


def call(r: ReadView) -> object:
    """The tool call as text: the action and the arguments the request gives."""
    name = r.read("d:action")
    if isinstance(name, Undetermined):
        return name
    arguments: list[str] = []
    for argument in ARGUMENTS[name]:
        value = extract(r, f"q:{name}/{argument}", 0.5)
        if not isinstance(value, Undetermined):  # an argument the request does not give
            arguments.append(f"{argument}={json.dumps(value)}")
    return f"{name}({', '.join(arguments)})"


def reader(name: str) -> Source:
    """The source that reads ``name``'s arguments, asked only when the action is ``name``."""
    return Source(
        {
            f"q:{name}/{argument}": Question(
                "extraction",
                Text(wording),
                None,
                value_type="text",
                is_list=argument in LISTS,
                text={"ref": "t:state.request"},
            )
            for argument, wording in ARGUMENTS[name].items()
        },
        needs=[],
        requirement=SourceRequirement("d:action", [name]),
    )


SOURCES = {
    "asks": Source(
        {f"q:asks/{name}": Question("noul", Text(wording)) for name, wording in ASKS.items()},
        needs=["t:state.request"],
    ),
    **{name: reader(name) for name in ACTIONS},
}

RULES = {
    "action": Rule(
        "d:action",
        t.label(*ACTIONS),
        action,
        version="1",
        reads=[f"q:asks/{name}" for name in ACTIONS],
    ),
    "call": Rule(
        "d:call",
        t.string,
        call,
        version="1",
        reads=["d:action", *(f"q:{n}/{a}" for n in ACTIONS for a in ARGUMENTS[n])],
    ),
}

DESIGN = Design(name="assistant", state=t.record(request=t.string), sources=SOURCES, rules=RULES)
