# pyright: reportPrivateUsage=false
"""The design store and the evaluation store: round trips, the files each writes, write-once
under exclusive create, and every refusal in its order."""

import shutil
import threading
from pathlib import Path
from typing import Any

import pytest

from designs import full, minimal, rule, with_rule
from ommat import InvalidDesign, load
from ommat._canonical import encode, hash_value, sha256_ref
from ommat._values import MAX_DOCUMENT_DEPTH
from ommat.evaluation import (
    DesignStore,
    EvaluationRefusal,
    EvaluationStore,
    RefusalReason,
    _store,
)

R = RefusalReason
OTHERS = ("turn", "ref", "source", "path", "group", "split", "error")
HEX = "0" * 64


def refused(call: Any, *args: Any, **kwargs: Any) -> EvaluationRefusal:
    with pytest.raises(EvaluationRefusal) as info:
        call(*args, **kwargs)
    exc = info.value
    assert all(getattr(exc, name) is None for name in OTHERS)
    return exc


def files(root: Path) -> list[Path]:
    """Every file under ``root``, temporary ones included."""
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------------------------
# The design store


def test_a_kept_design_is_fetched_back_and_filed_under_its_hash(tmp_path: Path) -> None:
    loaded = load(full())
    store = DesignStore(tmp_path)
    store.keep(loaded)
    hex_ = loaded.design_hash.removeprefix("sha256:")
    assert files(tmp_path) == [Path("sha256", hex_)]
    assert (tmp_path / "sha256" / hex_).read_bytes() == loaded.document
    again = store.fetch(loaded.design_hash)
    assert (again.document, again.design_hash) == (loaded.document, loaded.design_hash)


def test_keeping_a_held_design_again_accepts_equal_bytes(tmp_path: Path) -> None:
    loaded = load(minimal())
    store = DesignStore(tmp_path)
    store.keep(loaded)
    store.keep(load(loaded.document))
    assert len(files(tmp_path)) == 1


def test_keeping_a_hash_held_with_other_bytes_is_held_mismatch(tmp_path: Path) -> None:
    loaded = load(minimal())
    path = tmp_path / "sha256" / loaded.design_hash.removeprefix("sha256:")
    path.parent.mkdir()
    path.write_bytes(b"other")
    exc = refused(DesignStore(tmp_path).keep, loaded)
    assert (exc.reason, exc.key) == (R.held_mismatch, [loaded.design_hash])
    assert path.read_bytes() == b"other"
    assert files(tmp_path) == [path.relative_to(tmp_path)]


def test_keep_takes_exactly_a_loaded_design(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exactly a LoadedDesign"):
        DesignStore(tmp_path).keep(minimal())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "key",
    [
        "sha256:" + "0" * 63,
        "sha256:" + "0" * 65,
        "sha256:" + "A" * 64,
        "sha512:" + HEX,
        HEX,
        "sha256:../../" + "0" * 58,
        "sha256:" + HEX + "\n",
        5,
        None,
    ],
)
def test_fetching_a_malformed_key_is_bad_key_before_any_path(tmp_path: Path, key: Any) -> None:
    exc = refused(DesignStore(tmp_path / "store").fetch, key)
    assert (exc.reason, exc.key) == (R.bad_key, [key])
    assert not (tmp_path / "store").exists()


def test_fetching_a_key_not_held_is_not_held(tmp_path: Path) -> None:
    exc = refused(DesignStore(tmp_path).fetch, "sha256:" + HEX)
    assert (exc.reason, exc.key) == (R.not_held, ["sha256:" + HEX])


def test_fetching_bytes_whose_hash_is_not_the_key_is_hash_mismatch(tmp_path: Path) -> None:
    loaded = load(minimal())
    key = "sha256:" + HEX
    (tmp_path / "sha256").mkdir()
    (tmp_path / "sha256" / HEX).write_bytes(loaded.document)
    exc = refused(DesignStore(tmp_path).fetch, key)
    assert (exc.reason, exc.key) == (R.hash_mismatch, [key])
    assert loaded.design_hash in str(exc)


def test_fetch_passes_functions_through_to_load(tmp_path: Path) -> None:
    def inner(r: Any) -> bool:
        return True

    loaded = load(with_rule(minimal(), "r", rule("r", function=inner)))
    store = DesignStore(tmp_path)
    store.keep(loaded)
    with pytest.raises(InvalidDesign) as info:
        store.fetch(loaded.design_hash)
    assert {f.check for f in info.value.failures} == {11}
    again = store.fetch(loaded.design_hash, functions={"r": inner})
    assert again.design_hash == loaded.design_hash


# ---------------------------------------------------------------------------------------------
# The evaluation store

KEYS: dict[str, dict[str, str]] = {
    "gold-answer": {
        "gold_set": "g1",
        "question": "sha256:" + HEX,
        "turn": "sha256:1",
        "candidate": "",
    },
    "switching-value": {"record": "sha256:1", "design": "sha256:2", "parameter": "intent/bar"},
    "group-counts": {"design": "sha256:2", "group": "g", "split": "sha256:3", "gold_set": "g1"},
}


def entry(kind: str = "switching-value", **changes: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {"kind": kind, "format": 1, "key": dict(KEYS[kind]), "value": None}
    fields.update(changes)
    return fields


def path_of(root: Path, kind: str, key: dict[str, str]) -> Path:
    return root / kind / "sha256" / hash_value(key).removeprefix("sha256:")


@pytest.mark.parametrize("kind", list(KEYS))
def test_an_entry_of_each_kind_round_trips_as_its_canonical_envelope(
    tmp_path: Path, kind: str
) -> None:
    written = entry(kind, format=2, value={"value": [1, "a"], "b": None})
    store = EvaluationStore(tmp_path)
    assert store.write(written) is None
    path = path_of(tmp_path, kind, KEYS[kind])
    assert files(tmp_path) == [path.relative_to(tmp_path)]
    assert path.read_bytes() == encode(written, MAX_DOCUMENT_DEPTH)
    assert store.lookup(kind, dict(KEYS[kind])) == written


def test_a_key_not_held_looks_up_as_none(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    assert store.lookup("switching-value", KEYS["switching-value"]) is None
    store.write(entry())
    assert (
        store.lookup("switching-value", {**KEYS["switching-value"], "parameter": "intent/baz"})
        is None
    )


def test_a_grounded_gold_answer_key_is_its_own_entry(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    plain = entry("gold-answer", format=2, value="yes")
    dana = entry("gold-answer", format=2, value="no")
    dana["key"]["candidate"] = '"Dana"'
    store.write(plain)
    store.write(dana)
    assert len(files(tmp_path)) == 2
    assert store.lookup("gold-answer", KEYS["gold-answer"]) == plain
    assert store.lookup("gold-answer", dana["key"]) == dana


def test_the_store_keeps_format_and_value_as_given(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    store.write(entry(format=-7, value=["anything", {"at": 1.5}]))
    assert store.lookup("switching-value", KEYS["switching-value"]) == entry(
        format=-7, value=["anything", {"at": 1.5}]
    )


def test_a_held_key_is_held_key_and_the_first_entry_stands(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    store.write(entry(value="first"))
    for again in (entry(value="second"), entry(value="first")):
        exc = refused(store.write, again)
        assert (exc.reason, exc.key) == (R.held_key, [KEYS["switching-value"]])
    assert store.lookup("switching-value", KEYS["switching-value"]) == entry(value="first")
    assert len(files(tmp_path)) == 1


def test_racing_writers_under_one_key_leave_one_whole_entry(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    won: list[int] = []
    refusals: list[R] = []
    start = threading.Barrier(8)

    def write(n: int) -> None:
        start.wait()
        try:
            store.write(entry(value=[n] * 1000))
            won.append(n)
        except EvaluationRefusal as exc:
            refusals.append(exc.reason)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert (len(won), refusals) == (1, [R.held_key] * 7)
    assert store.lookup("switching-value", KEYS["switching-value"]) == entry(value=[won[0]] * 1000)
    assert len(files(tmp_path)) == 1  # no temporary file left behind


def test_create_claims_a_name_once_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b"
    assert _store._create(path, b"one")
    assert not _store._create(path, b"two")
    assert path.read_bytes() == b"one"
    assert files(tmp_path) == [Path("a", "b")]


def test_each_store_writes_its_files_readable_to_all(tmp_path: Path) -> None:
    loaded = load(minimal())
    DesignStore(tmp_path / "designs").keep(loaded)
    EvaluationStore(tmp_path / "results").write(entry())
    paths = [
        tmp_path / "designs" / "sha256" / loaded.design_hash.removeprefix("sha256:"),
        path_of(tmp_path / "results", "switching-value", KEYS["switching-value"]),
    ]
    assert [p.stat().st_mode & 0o777 for p in paths] == [0o644, 0o644]


def _nested(depth: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(depth - 1):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("given", "reason", "says"),
    [
        # the envelope comes first, then the kind, then the format, then the key
        (["kind", "format", "key", "value"], R.invalid_entry, "exactly kind"),
        ({"kind": "nope", "format": "x", "key": 1}, R.invalid_entry, "exactly kind"),
        ({**entry(), "extra": 1}, R.invalid_entry, "exactly kind"),
        ({**entry(), "kind": "calibration"}, R.unknown_kind, "'calibration'"),
        ({**entry(), "kind": 5, "format": "x"}, R.unknown_kind, "5"),
        ({**entry(), "kind": ["switching-value"]}, R.unknown_kind, "['switching-value']"),
        ({**entry(), "kind": "calibration", "key": None}, R.unknown_kind, "'calibration'"),
        ({**entry(), "format": "1", "key": 1}, R.invalid_entry, "format is a str"),
        (entry(format=True), R.invalid_entry, "format is a bool"),
        (entry(format=1.0), R.invalid_entry, "format is a float"),
        (entry(key=[]), R.invalid_entry, "key is a list"),
        (entry(key={"design": "d"}), R.invalid_entry, "exactly record, design, parameter"),
        (entry(key={**KEYS["switching-value"], "extra": "x"}), R.invalid_entry, "exactly record"),
        (
            entry(key={"record": 1, "design": 2, "parameter": 3}),
            R.invalid_entry,
            "member record is a int",
        ),
        (
            entry(key={"record": "r", "design": "d", "parameter": None}),
            R.invalid_entry,
            "member parameter",
        ),
        (entry(value=float("nan")), R.invalid_entry, ""),
        (entry(value={1, 2}), R.invalid_entry, ""),
        (entry(key={"record": "\ud800", "design": "d", "parameter": "p"}), R.invalid_entry, ""),
        (entry(value=["\udc80"]), R.invalid_entry, ""),
        (entry(value=_nested(MAX_DOCUMENT_DEPTH)), R.invalid_entry, "nests more than"),
    ],
)
def test_write_refuses_in_order_and_writes_nothing(
    tmp_path: Path, given: Any, reason: R, says: str
) -> None:
    exc = refused(EvaluationStore(tmp_path).write, given)
    assert (exc.reason, exc.key) == (reason, None)
    assert says in str(exc)
    assert files(tmp_path) == []


@pytest.mark.parametrize(
    ("kind", "key", "reason"),
    [
        ("calibration", KEYS["switching-value"], R.unknown_kind),
        ("calibration", None, R.unknown_kind),  # the kind before the key
        (None, KEYS["switching-value"], R.unknown_kind),
        ("switching-value", None, R.invalid_entry),
        ("switching-value", KEYS["group-counts"], R.invalid_entry),
        ("switching-value", {"record": "r", "design": "d", "parameter": 1}, R.invalid_entry),
        ("switching-value", {"record": "r", "design": "\udc80", "parameter": "p"}, R.invalid_entry),
    ],
)
def test_lookup_refuses_its_kind_and_key_as_write_does(
    tmp_path: Path, kind: Any, key: Any, reason: R
) -> None:
    exc = refused(EvaluationStore(tmp_path).lookup, kind, key)
    assert (exc.reason, exc.key) == (reason, None)


def _misfile(root: Path, data: bytes) -> Path:
    path = path_of(root, "switching-value", KEYS["switching-value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b'{"kind": "switching-value", "kind": "switching-value"}',
        encode(["switching-value"]),
        encode({k: v for k, v in entry().items() if k != "value"}),
        encode(entry(format=True)),
        encode(entry(key={"design": "sha256:2"})),
        b'{"format":1,"key":{"design":"sha256:2","parameter":"\\ud800","record":"sha256:1"},"kind":"switching-value","value":null}',
    ],
)
def test_a_file_that_is_not_an_envelope_is_misfiled_entry(tmp_path: Path, data: bytes) -> None:
    path = _misfile(tmp_path, data)
    exc = refused(EvaluationStore(tmp_path).lookup, "switching-value", KEYS["switching-value"])
    assert (exc.reason, exc.key) == (R.misfiled_entry, None)
    assert str(path) in str(exc)


def test_an_envelope_filed_under_another_key_is_misfiled_entry(tmp_path: Path) -> None:
    store = EvaluationStore(tmp_path)
    other = entry(key={**KEYS["switching-value"], "parameter": "intent/baz"})
    store.write(other)
    shutil.copy(path_of(tmp_path, "switching-value", other["key"]), _misfile(tmp_path, b""))
    exc = refused(store.lookup, "switching-value", KEYS["switching-value"])
    assert exc.reason is R.misfiled_entry
    assert store.lookup("switching-value", other["key"]) == other


def test_the_file_name_is_the_hash_of_the_keys_canonical_bytes(tmp_path: Path) -> None:
    key = {"parameter": "p1", "record": "sha256:1", "design": "sha256:2"}  # members in any order
    EvaluationStore(tmp_path).write(entry(key=key))
    hex_ = sha256_ref(b'{"design":"sha256:2","parameter":"p1","record":"sha256:1"}').removeprefix(
        "sha256:"
    )
    assert files(tmp_path) == [Path("switching-value", "sha256", hex_)]
