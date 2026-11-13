"""Basics no other file checks: the read view's attributes and a bytes field's default
redaction."""

from designs import minimal
from ommat import ReadView, decide_sync, load, redact
from ommat import types as t
from ommat._canonical import decode
from ommat._read_view import ComponentKind, Role


def test_a_read_view_has_no_public_attribute_beyond_its_declared_ones() -> None:
    view = ReadView(Role(ComponentKind.rule, "route"), [], [], {}, {})
    assert not hasattr(view, "__dict__")
    public = {name for name in dir(ReadView) if not name.startswith("_")}
    assert public == {
        "cause_refs",
        "close",
        "deciding_stop",
        "helper_log",
        "param",
        "read",
        "require",
        "role",
        "stops",
    }


def test_a_bytes_field_defaults_to_the_content_class_and_is_redacted_to_a_digest() -> None:
    loaded = load(minimal(state=t.record(blob=t.bytes_, size=t.integer())))
    record = decide_sync(loaded, {"blob": b"\x00\x01", "size": 3}, {}).record
    root = bytes(range(32))
    view = decode(redact(record, root))
    settings = next(item for item in view["log"] if item["item"] == "settings")
    assert settings["classes"]["t:state"] == {"blob": "content", "size": "open"}
    state = next(item for item in view["log"] if item.get("ref") == "t:state")
    assert state["value"]["blob"].startswith("hmac-sha256:")
    assert state["value"]["size"] == 3
