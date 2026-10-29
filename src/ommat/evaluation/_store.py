# pyright: reportPrivateUsage=false
"""The design store and the evaluation store: the reference stores, each a directory holding
one file per hash, at ``sha256/<hex>`` under the design store's root and
``<kind>/sha256/<hex>`` under the evaluation store's (§one-hash-function).

Each file is written whole by exclusive create: the bytes go to a temporary file beside it,
synced and made readable to all, which is then hard-linked under the file's name, so a reader
never sees part of a file and two writers cannot both claim one name. A store therefore needs a
filesystem with hard links. Neither store replaces a file it holds. An I/O failure other than a
missing file propagates as ``OSError``: an unreadable store is not a refusal.
"""

import contextlib
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from ommat._canonical import (
    _SHA256_REF,
    _CanonicalError,
    _DecodeError,
    decode,
    encode,
    sha256_ref,
)
from ommat._design import LoadedDesign
from ommat._load import load
from ommat._values import MAX_DOCUMENT_DEPTH, type_name
from ommat.evaluation._refusal import EvaluationRefusal, RefusalReason

_R = RefusalReason


def _create(path: Path, data: bytes) -> bool:
    """Write ``data`` to ``path`` whole, by exclusive create; ``False`` when ``path`` is held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.fchmod(f.fileno(), 0o644)  # mkstemp's 0600 would shut a shared store
        os.link(tmp, path)  # fails on a held name, where open(path, "xb") would too
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)
    return True


class DesignStore:
    """The design store (§store-integrity): each loaded design's ``document`` kept under the
    design hash it carries, at ``<root>/sha256/<hex>``, and given back through ``load``.

    Its root needs a filesystem with hard links; an I/O failure other than a missing file is an
    ``OSError``, not a refusal."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        return self._root / "sha256" / key.removeprefix("sha256:")

    def keep(self, loaded: LoadedDesign) -> None:
        """Keep ``loaded.document`` under ``loaded.design_hash``; ``held-mismatch`` when the
        hash is held with other bytes, and nothing when with the same."""
        if type(loaded) is not LoadedDesign:
            raise TypeError(f"loaded is a {type_name(loaded)}, not exactly a LoadedDesign")
        key, data = loaded.design_hash, loaded.document
        path = self._path(key)
        if not _create(path, data) and path.read_bytes() != data:
            raise EvaluationRefusal(
                f"the store holds {key} with other bytes", reason=_R.held_mismatch, key=[key]
            )

    def fetch(
        self, key: str, *, functions: Mapping[str, Callable[..., Any]] | None = None
    ) -> LoadedDesign:
        """The design held under ``key``, loaded with ``functions``; refused in order with
        ``bad-key``, ``not-held`` and ``hash-mismatch``. ``load``'s ``InvalidDesign`` passes
        through, a design whose code the process lacks failing check 11."""
        if not isinstance(key, str) or _SHA256_REF.fullmatch(key) is None:  # pyright: ignore[reportUnnecessaryIsInstance]
            raise EvaluationRefusal(
                f"{key!r} is not sha256: and 64 lowercase hex characters",
                reason=_R.bad_key,
                key=[key],
            )
        try:
            data = self._path(key).read_bytes()
        except FileNotFoundError:
            raise EvaluationRefusal(
                f"the store holds no {key}", reason=_R.not_held, key=[key]
            ) from None
        found = sha256_ref(data)
        if found != key:
            raise EvaluationRefusal(
                f"the bytes held under {key} hash to {found}", reason=_R.hash_mismatch, key=[key]
            )
        return load(data, functions=functions)  # check 11 makes its hash the bytes'


# --- the evaluation store (§evaluation-store) --------------------------------------------------

_KEYS: dict[str, tuple[str, ...]] = {
    "gold-answer": ("gold_set", "question", "turn", "candidate"),
    "switching-value": ("record", "design", "parameter"),
    "certificate": ("design", "sweep"),
    "group-counts": ("design", "group", "split", "gold_set"),
}
_ENVELOPE = ("kind", "format", "key", "value")

_Fault = tuple[RefusalReason, str]


def _refuse(fault: _Fault | None) -> None:
    if fault is not None:
        raise EvaluationRefusal(fault[1], reason=fault[0])


def _kind_fault(kind: object) -> _Fault | None:
    if isinstance(kind, str) and kind in _KEYS:
        return None
    return _R.unknown_kind, f"kind {kind!r} is not one of {', '.join(_KEYS)}"


def _key_fault(kind: str, key: object) -> _Fault | None:
    members = _KEYS[kind]
    if not isinstance(key, dict):
        return _R.invalid_entry, f"key is a {type_name(key)}, not an object"
    fields = cast("dict[Any, Any]", key)
    if set(fields) != set(members):
        return _R.invalid_entry, f"a {kind} key's members are exactly {', '.join(members)}"
    for name in members:
        if not isinstance(fields[name], str):
            return (
                _R.invalid_entry,
                f"key member {name} is a {type_name(fields[name])}, not a string",
            )
    return None


def _entry_fault(entry: object) -> _Fault | None:
    """The first fault of an envelope, in §evaluation-store's order."""
    shape = "an entry is an object of exactly kind, format, key and value"
    if not isinstance(entry, dict):
        return _R.invalid_entry, shape
    fields = cast("dict[Any, Any]", entry)
    if set(fields) != set(_ENVELOPE):
        return _R.invalid_entry, shape
    kind, fmt = fields["kind"], fields["format"]
    if (fault := _kind_fault(kind)) is not None:
        return fault
    if not isinstance(fmt, int) or isinstance(fmt, bool):
        return _R.invalid_entry, f"format is a {type_name(fmt)}, not an integer"
    return _key_fault(kind, fields["key"])


def _encoded(value: object, max_depth: int | None = None) -> bytes:
    """``value``'s canonical bytes, or ``invalid-entry`` with the encoder's reason."""
    try:
        return encode(value, max_depth)
    except _CanonicalError as exc:
        raise EvaluationRefusal(str(exc), reason=_R.invalid_entry) from exc


class EvaluationStore:
    """The evaluation store (§evaluation-store): write-once entries, each the canonical
    encoding of ``{"kind", "format", "key", "value"}``, at ``<root>/<kind>/sha256/<hex>``,
    ``<hex>`` the hash of the key's canonical bytes. It reads no entry's ``value``.

    Its root needs a filesystem with hard links; an I/O failure other than a missing file is an
    ``OSError``, not a refusal."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root)

    def _path(self, kind: str, key: Mapping[str, str]) -> Path:
        return self._root / kind / "sha256" / sha256_ref(_encoded(key)).removeprefix("sha256:")

    def write(self, entry: dict[str, Any]) -> None:
        """Keep ``entry``; refused with ``invalid-entry``, ``unknown-kind`` and ``held-key``,
        each before anything is written."""
        _refuse(_entry_fault(entry))
        kind, key = entry["kind"], entry["key"]
        path = self._path(kind, key)
        if not _create(path, _encoded(entry, MAX_DOCUMENT_DEPTH)):
            raise EvaluationRefusal(
                f"the store holds a {kind} entry under {key!r}", reason=_R.held_key, key=[key]
            )

    def lookup(self, kind: str, key: dict[str, str]) -> dict[str, Any] | None:
        """The entry held under ``kind`` and ``key``, or ``None``; ``kind`` and ``key`` refused
        as ``write`` refuses them, and a held file that is not this entry with
        ``misfiled-entry``."""
        _refuse(_kind_fault(kind) or _key_fault(kind, key))
        path = self._path(kind, key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        entry: Any = None
        with contextlib.suppress(_DecodeError):
            entry = decode(data)
        # A valid envelope holding this key is of this kind, since no two kinds share members.
        if _entry_fault(entry) is not None or entry["key"] != key:
            raise EvaluationRefusal(
                f"{path} does not hold a {kind} entry under its key", reason=_R.misfiled_entry
            )
        return entry
