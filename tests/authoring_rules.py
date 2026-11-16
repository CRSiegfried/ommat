"""The rule functions the authoring fixtures name: `authoring_slice.toml` and its Python twin."""

from typing import Any


def folder_kind(r: Any) -> str:
    return "move"


def folder_op(r: Any) -> str:
    return "keep"


def folder_plan(r: Any) -> float:
    return 0.5


def folder_act(r: Any) -> str:
    return "keep"


def act(r: Any) -> str:
    return "keep"


def streets(r: Any) -> list[str]:
    return ["a", "b"]


def pick(r: Any) -> str:
    return "x"


class Kit:
    """A class whose method a rule may name by a dotted qualname."""

    @staticmethod
    def pick(r: Any) -> str:
        return "x"
