# pyright: reportPrivateUsage=false
"""A design's groups: each names refs to be considered together and gives its reasons, from a
closed set, which check 9 holds; format 9 writes a group as ``{refs, reason}``, and an earlier
format's group, its refs alone, reads with the reason ``convergent``."""

from typing import Any

import pytest

from designs import full, minimal
from ommat import Group, InvalidDesign, load
from ommat._canonical import decode, encode, sha256_ref


def failures(design: Any) -> list[tuple[int, tuple[str, ...], str]]:
    with pytest.raises(InvalidDesign) as info:
        load(design)
    return [(f.check, f.names, f.message) for f in info.value.failures]


def grouped(*reason: str) -> Any:
    return full(groups={"g": Group(["q:intent/tone", "q:intent/email"], list(reason))})


def test_every_reason_loads_and_is_held_in_code_point_order() -> None:
    reasons = ["exhaustive", "convergent", "consistent", "discriminant", "exclusive"]
    loaded = load(grouped(*reasons))
    assert loaded._group_reasons == {"g": tuple(sorted(reasons))}
    document = decode(loaded.document)
    assert document["groups"]["g"] == {
        "refs": ["q:intent/email", "q:intent/tone"],
        "reason": sorted(reasons),
    }


def test_the_order_reasons_are_given_in_does_not_change_the_design() -> None:
    a = load(grouped("convergent", "exclusive")).design_hash
    assert a == load(grouped("exclusive", "convergent")).design_hash
    assert a != load(grouped("convergent")).design_hash


@pytest.mark.parametrize(
    ("reason", "says"),
    [
        ((), 'group "g" gives no reason'),
        (("convergnet",), 'group "g"\'s reason "convergnet" is not one of convergent'),
        (("Convergent",), "is not one of"),
        (("exclusive", "exclusive"), 'group "g" gives the reason "exclusive" more than once'),
    ],
    ids=["none", "misspelt", "capitalised", "twice"],
)
def test_a_reason_outside_the_five_or_none_is_check_9(reason: tuple[str, ...], says: str) -> None:
    found = failures(grouped(*reason))
    assert [(c, n) for c, n, _ in found] == [(9, ("g",))]
    assert says in found[0][2]


def test_a_reason_that_is_not_a_string_is_check_10() -> None:
    found = failures(minimal(groups={"g": Group([], [1])}))  # type: ignore[list-item]
    assert {c for c, _, _ in found} == {10}


def format_8(groups: object) -> dict[str, Any]:
    doc = decode(load(full()).document)
    doc["format"] = 8
    doc["groups"] = groups
    return doc


def test_an_earlier_formats_group_reads_as_convergent_and_keeps_its_hash() -> None:
    doc = format_8({"intents": ["q:intent/email", "q:intent/tone"]})
    data = encode(doc, 128)
    loaded = load(data)
    assert loaded.design_hash == sha256_ref(data)
    assert loaded._group_reasons == {"intents": ("convergent",)}
    assert loaded._groups == {"intents": ("q:intent/email", "q:intent/tone")}


def test_an_earlier_formats_group_as_an_object_is_check_10() -> None:
    doc = format_8({"intents": {"refs": ["q:intent/email"], "reason": ["convergent"]}})
    assert {c for c, _, _ in failures(encode(doc, 128))} == {10}


@pytest.mark.parametrize(
    ("group", "check"),
    [
        ({"refs": ["q:intent/email", "q:intent/tone"]}, 11),
        ({"refs": ["q:intent/email", "q:intent/tone"], "reason": ["convergent"], "x": 1}, 11),
        (["q:intent/email", "q:intent/tone"], 10),
    ],
    ids=["no reason member", "an extra member", "a list"],
)
def test_a_format_9_group_of_another_shape_is_refused(group: object, check: int) -> None:
    doc = decode(load(full()).document)
    doc["groups"] = {"intents": group}
    assert {c for c, _, _ in failures(encode(doc, 128))} == {check}
