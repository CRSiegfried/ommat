"""Shared fixtures."""

import pytest

import ommat._load
import ommat._values
from counting import SmallBounds


@pytest.fixture
def small_bounds(monkeypatch: pytest.MonkeyPatch) -> SmallBounds:
    """Set the size bounds the walks read at each count, so a test builds small values: by
    default 2^6 values and 2^8 code points. Called again, it sets others, and load's type-size
    bound too when ``type_size`` is given."""

    def set_bounds(
        values: int = 2**6, code_points: int = 2**8, type_size: int | None = None
    ) -> None:
        monkeypatch.setattr(ommat._values, "MAX_VALUES", values)
        monkeypatch.setattr(ommat._values, "MAX_CODE_POINTS", code_points)
        if type_size is not None:
            monkeypatch.setattr(ommat._load, "_TYPE_SIZE", type_size)

    set_bounds()
    return set_bounds
