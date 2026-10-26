"""``determined``: its reads' values as a tuple, or the marker of highest precedence among them,
unchanged, and its log item."""

from typing import Any

import pytest

from ommat import Answer, Cause, CauseCode, Unasked, Unavailable, Unsure
from ommat._read_view import ComponentKind, ReadView, Role
from ommat._types import RecordOf
from ommat.helpers import determined

INTENT = Answer("noul", ["yes", "no"], {"yes": 0.7, "no": 0.3})
UNSURE = Unsure(Cause(CauseCode.abstained, ["q:intent"]))
UNASKED = Unasked(Cause(CauseCode.when_false, ["lamps"]))
GONE = Unavailable(Cause(CauseCode.raised, ["route"]))
PICKS = {"Elm Row": "yes", "Quay Street": UNSURE}
BOARD: dict[str, object] = {
    "t:state": {},
    "q:intent": INTENT,
    "d:none": None,
    "d:picks": PICKS,
    "d:unsure": UNSURE,
    "d:unasked": UNASKED,
    "d:gone": GONE,
}


def view(kind: ComponentKind = ComponentKind.rule) -> ReadView:
    name = {"rule": "route", "policy": "policy", "when": "mail"}[kind.value]
    state = RecordOf(()) if kind is ComponentKind.when else None
    return ReadView(Role(kind, name), tuple(BOARD), (), BOARD, {}, state=state)


def logged(r: ReadView) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    return [(i.helper, i.refs, i.rested_on) for i in r.helper_log()]


def test_every_value_gives_a_tuple_in_argument_order_for_one_ref_as_for_several() -> None:
    r = view()
    assert determined(r, "q:intent") == (INTENT,)
    assert determined(r, "d:none", "q:intent") == (None, INTENT)
    assert logged(r) == [
        ("determined", ("q:intent",), ("q:intent",)),
        ("determined", ("d:none", "q:intent"), ("d:none", "q:intent")),
    ]


def test_a_null_and_a_per_candidate_dict_holding_a_marker_are_values() -> None:
    (picks,) = determined(view(), "d:picks")  # type: ignore[misc]
    assert picks == PICKS


@pytest.mark.parametrize(
    ("refs", "expected", "rested"),
    [
        (("q:intent", "d:unasked", "d:unsure"), UNSURE, "d:unsure"),
        (("d:unasked", "d:unsure", "d:gone"), GONE, "d:gone"),
        (("d:unasked", "q:intent"), UNASKED, "d:unasked"),
        (("d:unsure", "d:gone", "d:unsure_again"), GONE, "d:gone"),
    ],
    ids=["unsure over unasked", "unavailable first", "unasked alone", "kind before order"],
)
def test_the_marker_of_highest_precedence_comes_back_unchanged(
    refs: tuple[str, ...], expected: Any, rested: str
) -> None:
    board = {**BOARD, "d:unsure_again": Unsure(Cause(CauseCode.abstained, ["q:other"]))}
    r = ReadView(Role(ComponentKind.rule, "route"), tuple(board), (), board, {})
    got = determined(r, *refs)
    assert got is expected  # the read marker itself, its cause unchanged
    assert logged(r) == [("determined", refs, (rested,))]


def test_the_first_marker_of_the_winning_kind_in_argument_order_wins() -> None:
    later = Unsure(Cause(CauseCode.abstained, ["q:later"]))
    board = {**BOARD, "d:later": later}
    r = ReadView(Role(ComponentKind.rule, "route"), tuple(board), (), board, {})
    assert determined(r, "d:later", "d:unsure") is later


@pytest.mark.parametrize(
    ("refs", "message"),
    [((), "was given no ref"), (("q:intent", "q:intent"), "was given a ref twice")],
)
def test_no_ref_or_a_repeated_ref_raises(refs: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        determined(view(), *refs)


def test_a_repeated_ref_raises_even_over_a_marker() -> None:
    with pytest.raises(ValueError, match="was given a ref twice"):
        determined(view(), "d:gone", "d:gone")


def test_it_raises_on_a_when_s_view_before_any_read() -> None:
    from ommat.helpers._call import HelperOnWhen

    r = view(ComponentKind.when)
    with pytest.raises(HelperOnWhen):
        determined(r, "q:intent")
    assert r.helper_log() == ()
