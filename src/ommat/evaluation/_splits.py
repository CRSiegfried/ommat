# pyright: reportPrivateUsage=false
"""Splits, read by the correlated-group confirmation job: a split passed as
``(id, records)``, its records read once each by record hash, and its id, SHA-256 over the
canonical array of its records' hashes in code-point order (§evaluation-store).

``split_id(records)``, public, gives that id, so a caller names a split as the job checks it.
"""

from collections.abc import Iterable, Mapping
from typing import cast

from ommat._canonical import hash_value
from ommat._ledger import Record
from ommat._refs import code_point_sorted
from ommat._values import type_name
from ommat.evaluation._diff import _read
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason


def split_id(records: Iterable[object]) -> str:
    """The id of a split of ``records``, each a stored document or a ``Record``, read as the
    jobs read a split's; ``invalid-record`` for one that fails the read, ``TypeError`` for
    ``records`` that are not an iterable other than a string, bytes or a mapping."""
    _refuse_records("records", records)
    return _id(read_split("given", records))


def given_split(name: str, pair: object) -> tuple[str, Iterable[object]]:
    """The split ``pair`` as ``(id, records)``; ``TypeError`` for any other shape, a split id
    that is not a string, or records that are not an iterable other than a string, bytes or a
    mapping."""
    if not isinstance(pair, list | tuple) or len(cast("tuple[object, ...]", pair)) != 2:
        raise TypeError(
            f"the {name} split is a {type_name(cast(object, pair))}, not an (id, records) pair"
        )
    given, records = cast("tuple[object, object]", tuple(cast("tuple[object, ...]", pair)))
    if type(given) is not str:
        raise TypeError(f"the {name} split's id is a {type_name(given)}, not a string")
    _refuse_records(f"the {name} split's records", records)
    return given, cast("Iterable[object]", records)


def _refuse_records(what: str, records: object) -> None:
    if not isinstance(records, Iterable) or isinstance(records, str | bytes | Mapping):
        raise TypeError(f"{what} are a {type_name(cast(object, records))}, not records")


def read_split(name: str, records: Iterable[object]) -> dict[str, Record]:
    """A split's records by record hash, each once, in the order given; ``invalid-record`` for
    a record that fails the read."""
    read: dict[str, Record] = {}
    for i, document in enumerate(records):
        record = _read(document, f"the {name} split's record {i}")
        read.setdefault(record.record_hash, record)
    return read


def refuse_split_id(name: str, given: str, read: dict[str, Record]) -> None:
    """``split-id`` when the id recomputed from a split's records is not the one given."""
    found = _id(read)
    if found != given:
        raise EvaluationRefusal(
            f"the {name} split's id is {given}, but its records' is {found}",
            reason=RefusalReason.split_id,
            split=(given, found),
        )


def _id(read: dict[str, Record]) -> str:
    return hash_value(code_point_sorted(read))
