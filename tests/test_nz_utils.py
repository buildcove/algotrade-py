from algotrade.core import nz
from algotrade.treasury import (
    DECIMAL_TO_TREASURY_PART,
    TREASURY_PART_TO_DECIMAL,
)


def test_conversion_treasury_to_decimal():
    # PASSED
    for i, v in TREASURY_PART_TO_DECIMAL.items():
        whole = f"109.{i}"
        decimal = nz.treasury_to_decimal(whole)
        _, part = str(decimal).split(".")
        assert part == v


def test_conversion_decimal_to_treasury():
    # PASSED
    for i, v in DECIMAL_TO_TREASURY_PART.items():
        decimal = f"109.{i}"
        treasury = nz.decimal_to_treasury(decimal)
        _, part = str(treasury).split(".")

        part = f"{part:03}"
        assert part == v


def test_treasury_distance_between_two_points():
    # PASSED
    a = nz.treasury_to_decimal(109.0)
    b = nz.treasury_to_decimal(108.315)
    assert nz.decimal_ticks_diff(a, b) == 1

    a = nz.treasury_to_decimal(109.0)
    b = nz.treasury_to_decimal(109.010)
    assert nz.decimal_ticks_diff(a, b) == 2

    a = nz.treasury_to_decimal(109.0)
    b = nz.treasury_to_decimal(109.015)
    assert nz.decimal_ticks_diff(a, b) == 3

    a = nz.treasury_to_decimal(109.0)
    b = nz.treasury_to_decimal(107.315)
    assert nz.decimal_ticks_diff(a, b) == 65
