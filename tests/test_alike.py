"""``_alike`` against the canonical bytes: a probe's entry, built by ``_logged``, and the same
value read back from its encoding."""

from __future__ import annotations

import unicodedata
from typing import Any

from ommat._canonical import decode, encode
from ommat._ledger import _logged  # pyright: ignore[reportPrivateUsage]
from ommat.evaluation._switching import _alike  # pyright: ignore[reportPrivateUsage]

_VALUES: list[Any] = [
    0,
    -0.0,
    0.0,
    1,
    1.0,
    2**53 - 1,
    float(2**53 - 1),
    2.0**60,
    "é",
    unicodedata.normalize("NFC", "é"),
    unicodedata.normalize("NFD", "é"),
    "\x00",
    '"',
    "\\",
    {"": 1, "\U0001f600": 2},
    {"\U0001f600": 2, "": 1},
    {"￿": 1, "\U0001f600": 2},
    [-0.0],
    [0],
    [1.0],
    [1],
    [2.0**60],
    {"a": 1.0},
    {"a": 1},
]


def test_alike_values_share_their_canonical_bytes_by_both_paths() -> None:
    for x in _VALUES:
        for y in _VALUES:
            a = _logged({"item": "x", "value": x})["value"]  # a probe's entry
            b = decode(encode({"item": "x", "value": y}))["value"]  # a record's, read back
            same = encode(a) == encode(b)
            assert _alike(a, b) == same, (x, y)
