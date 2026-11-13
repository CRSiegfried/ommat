"""The outline: one stored design or turn as a plain dict of fixed shape, and as text a person
or a program reads line by line, outside the compatibility promise while it is validated.

``design_outline(document)`` gives a design document's header, parameters, nodes by round and
wavefront, the dependency graph's edges and the state's, groups and
targets. ``turn_outline(record_document, design_document=None)`` gives a record document round
by round, each call's provenance and each entry's value or marker, the decision and the
escalation; with the design the record ran under, each entry also holds what its component
declares, marked as declared since the record holds no read set, and each bound its design
declares for the entry, with the distance to it. A member the documents give nothing for is
``None``, never guessed. A design outline also computes a layered ``layout``, a ``fingerprint``
with the names erased and a ``short`` label per ref; a turn outline the ``lineage`` from the
decision back to what it rests on, the ``nearest_bounds``, every bound nearest first, and a
one-line ``summary``. ``render_design(outline)`` and ``render_turn(outline, detail=False)`` give
the same text for the same outline. ``python -m ommat.outline PATH...`` prints the text, with
``--json`` the outlines or with ``--oneline`` each record's summary, for record and design
files or folders, each record beside the design whose bytes' SHA-256 its ``design`` item holds.

A document is ``bytes`` or a parsed object. The text holds the records' values, user text
included, so keep it as you keep them. ``import ommat`` does not import this package, and it
takes nothing beyond the standard library.
"""

from ommat.outline._design import design_outline
from ommat.outline._text import render_design, render_turn
from ommat.outline._turn import turn_outline

__all__ = ["design_outline", "render_design", "render_turn", "turn_outline"]
