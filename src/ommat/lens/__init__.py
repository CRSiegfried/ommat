"""The record lens: stored records rendered as one HTML page a person reads, outside the
compatibility promise while it is validated.

``page(records, designs=())`` returns a self-contained page that shows each record round by
round: its calls and entries, each choice's probabilities and an extraction's confidence as
percentages, each marker by kind and cause, then the plan and the verdict. Selecting an entry, a
call or the plan shows its lineage: the links the record holds, and, with the design the record
ran under, each rule's and the policy's declared reads, marked as declared. A design pairs with a
record by its hash. ``python -m ommat.lens PATH`` writes the page for a record file or a folder.

The page embeds the records whole, user text included, so keep it as you keep them, and loads
nothing from the network. ``import ommat`` does not import this package, and it takes nothing
beyond the standard library.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from typing import Any, cast

__all__ = ["page"]

_SLOT = "__OMMAT_LENS_DATA__"


def page(
    records: Sequence[bytes] | Mapping[str, bytes],
    designs: Sequence[bytes] = (),
    *,
    skipped: Sequence[str] = (),
) -> str:
    """The page for ``records``, record documents as ``bytes`` (``decision.record.document``),
    named by a mapping's keys or else ``record 1``, ``record 2`` and so on, with ``designs``,
    design documents as ``bytes`` (``loaded.document``), each paired with the records whose
    ``design`` item holds its hash. ``skipped`` names files a caller passed over, which the page
    counts. Raises ``ValueError`` for an argument that is not ``bytes``, a record that is not a
    JSON object holding a ``log`` list, or a design that is not a JSON object."""
    named = (
        list(records.items())
        if isinstance(records, Mapping)
        else [(f"record {i}", r) for i, r in enumerate(records, 1)]
    )
    shown: list[dict[str, Any]] = []
    for name, document in named:
        value = _document(document, f"record {name!r}")
        if not isinstance(value.get("log"), list):
            raise ValueError(f"record {name!r} is not a record document: it holds no 'log' list")
        shown.append({"name": str(name), "record": value})
    keyed: dict[str, Any] = {}
    for i, document in enumerate(designs, 1):
        keyed["sha256:" + hashlib.sha256(_bytes(document, f"design {i}")).hexdigest()] = _document(
            document, f"design {i}"
        )
    data = {"records": shown, "designs": keyed, "skipped": list(skipped), "read": len(shown)}
    # Every "<" is escaped, so no text in a record can close the script element.
    embedded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return _template().replace(_SLOT, embedded, 1)


def _bytes(document: object, what: str) -> bytes:
    if type(document) is not bytes:
        raise ValueError(f"{what} must be bytes, not {type(document).__name__}")
    return document


def _document(document: object, what: str) -> dict[str, Any]:
    try:
        value = json.loads(_bytes(document, what))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{what} is not JSON: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{what} is not a JSON object")
    return cast("dict[str, Any]", value)


def _template() -> str:
    return files("ommat.lens").joinpath("page.html").read_text(encoding="utf-8")
