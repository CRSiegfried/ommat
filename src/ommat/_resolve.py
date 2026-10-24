"""A step argument's value read from a closed record, as the reference runner reads it: the
board ref's entry value, its field path walked over it, copied afresh at each read. The runner,
the handoff view and the conversation renderers share it."""

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ommat._refs import parse_ref


def board_entries(log: list[dict[str, Any]]) -> dict[str, object]:
    """Each board ref's entry value, first by log position; a marker entry holds none, and
    plan rule 4 keeps a step from naming one."""
    out: dict[str, object] = {}
    for i in log:
        if i["item"] == "entry" and "value" in i:
            out.setdefault(i["ref"], i["value"])
    return out


def board_value(entries: Mapping[str, object], ref: str) -> object:
    """A board ref's value from the record's entries, copied afresh at each read so no two
    arguments share an object: the base ref's entry value, its field path walked over it, a
    null on the path giving null."""
    # The entry, and a record at every field on the path but a null, hold by plan rule 4 and
    # conformance in a record decide returned.
    fields = parse_ref(ref).fields
    base = ref[: len(ref) - len(".".join(fields)) - 1] if fields else ref
    value: Any = entries[base]
    for field in fields:
        if value is None:
            return None
        value = value[field]
    return deepcopy(value)
