"""The record lens: stored records and designs drawn as one HTML page a person reads, from
their outlines, outside the compatibility promise while it is validated.

``page(records, designs=())`` returns a self-contained page. It takes each document's outline
(``ommat.outline``) and text form, and the page draws from the outline alone: a turn's path,
one card per rule with a dependency path to the policy, in the layout's columns, each folding in
the answers it declares, the ones the record shows it read filled, over a map of every node
and dependency with the selected rule's reads lit; a design's
overview; a slice of its dependency graph around one node, by round and in the layout's rows,
with the selected turn's entries marked on it; its targets and its tables' rows; and the design
as its whole dependency graph, one band per round, a sub-column per wavefront, the sources,
rules and the policy with the refs each provides and the edges between them, each declared
read, helper, need, slot or source requirement by its role; a turn over that graph, each entry
washed by what it holds and marked by its nearest bound, the lineage from the decision to its
root as a tree, the round costs, and a headline that says what was decided, or what the
undecided turn rests on and how far its root fell short; a panel of every bound by entry; a
table of the turns, sorted and filtered, with a tally; and the text form beside them. A record
pairs with the design whose bytes' SHA-256 its ``design`` item holds, else with a design of the
same name, with a notice, else it is drawn alone, its entries by round. ``python -m ommat.lens
PATH`` writes the page for a record or design file or a folder.

The page embeds the outlines and text whole, user text included, so keep it as you keep the
records, and loads nothing from the network. ``import ommat`` does not import this package, and
it takes nothing beyond the standard library.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from importlib.resources import files
from typing import Any, cast

from ommat._documents import design_hash, design_key, read_document
from ommat.outline import design_outline, render_design, render_turn, turn_outline

__all__ = ["page"]

_SLOT = "__OMMAT_LENS_DATA__"


def page(
    records: Sequence[bytes] | Mapping[str, bytes],
    designs: Sequence[bytes] | Mapping[str, bytes] = (),
    *,
    skipped: Sequence[str] = (),
) -> str:
    """The page for ``records``, record documents as ``bytes`` (``decision.record.document``),
    named by a mapping's keys or else ``record 1``, ``record 2`` and so on, with ``designs``,
    design documents as ``bytes`` (``loaded.document``), named alike or else ``design 1`` and so
    on, each paired with the records whose ``design`` item holds the SHA-256 of its exact
    bytes. ``skipped`` names files a caller passed
    over, which the page counts. Raises ``ValueError`` for an argument of the wrong type, a
    document that is not strict JSON, a record that is not a JSON object holding a ``log`` list,
    or a design that is not a JSON object."""
    named = (
        [(str(name), document) for name, document in records.items()]
        if isinstance(records, Mapping)
        else [(f"record {i}", r) for i, r in enumerate(_listed(records, "records"), 1)]
    )
    designs_named = (
        [(str(name), document) for name, document in designs.items()]
        if isinstance(designs, Mapping)
        else [(f"design {i}", d) for i, d in enumerate(_listed(designs, "designs"), 1)]
    )
    keyed: dict[str, tuple[str, bytes]] = {}
    for name, document in designs_named:
        read_document(document, name)
        keyed.setdefault(design_key(cast(bytes, document)), (name, document))
    docs: list[dict[str, Any]] = []
    for name, document in keyed.values():
        outline = design_outline(document)
        docs.append({"path": name, "outline": outline, "text": render_design(outline)})
    for name, document in named:
        value = read_document(document, name)
        if not isinstance(value.get("log"), list):
            raise ValueError(f"{name} is not a record document: it holds no 'log' list")
        paired = keyed.get(str(design_hash(value)))
        outline = turn_outline(document, None if paired is None else paired[1])
        docs.append({"path": name, "outline": outline, "text": render_turn(outline)})
    names = _listed(skipped, "skipped")
    if not all(type(name) is str for name in names):
        raise ValueError("skipped must be a sequence of str")
    data = {
        "sets": [{"name": "", "skipped": names, "docs": docs}],
        "read": len(named),
    }
    # ASCII only, and every "<" escaped, so no text in a record can close the script element
    # and the page reads the same under any charset.
    embedded = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    return _template().replace(_SLOT, embedded, 1)


def _listed(values: object, what: str) -> list[Any]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{what} must be a sequence, not {type(values).__name__}")
    return list(cast("Iterable[Any]", values))


def _template() -> str:
    return files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
