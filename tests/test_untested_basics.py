"""Basics no other file checks: the read view's attributes."""

from ommat import ReadView
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
