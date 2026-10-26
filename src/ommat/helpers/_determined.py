"""``determined``: several reads' values, or the marker of highest precedence among them."""

from ommat._markers import Undetermined
from ommat._read_view import ReadView
from ommat.helpers._call import called
from ommat.helpers._combinator import _PRECEDENCE, _refs  # pyright: ignore[reportPrivateUsage]


def determined(r: ReadView, *refs: str) -> tuple[object, ...] | Undetermined:
    """Each ref read through ``r`` in argument order: a new ``tuple`` of their values, for one
    ref as for several, when every read is a value; else the marker of highest precedence,
    ``Unavailable``, then ``Unsure``, then ``Unasked``, the first of its kind in argument order,
    unchanged. It looks inside no value, so a null and a per-candidate read's ``dict`` are
    values, and it stops nothing: return the marker to pass it on. ``refs`` are one or more,
    no two equal, else it raises."""
    with called(r, "determined", refs) as call:
        given = _refs("determined", refs)
        reads = list(zip(given, call.values, strict=True))
        for kind in _PRECEDENCE:
            found = next(((ref, v) for ref, v in reads if type(v) is kind), None)
            if found is not None:
                call.rested_on([found[0]])
                return found[1]
        call.rested_on(given)
        return tuple(call.values)
