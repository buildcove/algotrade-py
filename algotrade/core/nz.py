import pandas as pd
from algotrade.treasury import DECIMAL_TO_TREASURY_PART


def naive_decimal_to_treasury(decimal: float) -> float:
    if not decimal:
        return decimal

    # Convert the decimal to shorthand i.e. 110.6875 to 110'22 or 110.22
    whole, dec = str(decimal).split(".")
    if dec == "000":
        return 0.0

    value = DECIMAL_TO_TREASURY_PART[dec]
    return float(f"{whole}.{value}")


def decimal_to_treasury(decimal, size=64, gcf=5):
    """
    WARNING: DO NOT EDIT THIS FUNCTION. It has been tested and works as expected.

    $ZN: 1 / 64 = 0.015625
    """
    if decimal is None or pd.isnull(decimal):
        return decimal

    size = 1 / size
    whole, part = str(decimal).split(".")
    part = float(f"0.{part}") / size
    part *= gcf
    part = f"{int(part):03}"
    return float(f"{whole}.{part}")


def naive_treasury_to_decimal(price):
    """
    WARNING: DO NOT EDIT THIS FUNCTION. It has been tested and works as expected.

    Converts a treasury priced in 32nds into a decimal. This works for
    treasurys priced up to 3dp i.e. "99-126" or "99.126"
    """
    whole, part = str(price).split(".")
    part = int(part) / 32
    part = str(part).replace(".", "")
    return float(f"{whole}.{part}")


def treasury_to_decimal(price, size=64, gcf=5):
    """
    WARNING: DO NOT EDIT THIS FUNCTION. It has been tested and works as expected.

    To get the GCF
    - Create a list of 1/64 upto 63/64
    - Find the GCF based in the list
    """
    if price is None or pd.isnull(price):
        return price

    whole, part = str(price).split(".")
    part = float(f"0.{part}")
    part = f"{part:.3f}"
    part = part.split(".")[1]
    part = int(part)
    part = part / gcf
    part = str(part / size).split(".")[1]
    return float(f"{whole}.{part}")


def naive_treasury_ticks_diff(a, b, step_size=5, multiplier=64):
    # param is decimals
    # beware: If you use minus, convert the result to positive

    a = decimal_to_treasury(a)
    b = decimal_to_treasury(b)

    a_whole, a_part = str(a).split(".")
    a_whole, a_part = int(a_whole), int(f"{a_part:03}")

    b_whole, b_part = str(b).split(".")
    b_whole, b_part = int(b_whole), int(f"{b_part:03}")

    whole_diff = abs(a_whole - b_whole)

    a_part_index = a_part / step_size
    b_part_index = b_part / step_size

    part_diff = abs(a_part_index - b_part_index)
    diff = abs(part_diff - (whole_diff * multiplier))

    return int(diff)


def decimal_ticks_diff(a, b, tick_size=1 / 64):
    return abs((a - b) / tick_size)


def decimal_addition_by_ticks(a, b, tick_size=1 / 64):
    return a + (b * tick_size)


def decimal_subtraction_by_ticks(a, b, tick_size=1 / 64):
    return a - (b * tick_size)


def to_readable(df):
    df = df.copy()
    # Select columns that end with open, high, low, and close
    target_cols = df.filter(regex="(open|high|low|close|price|stoploss|price_17_00)$").columns

    # Apply nz.decimal_to_treasury to the selected columns
    df[target_cols] = df[target_cols].applymap(decimal_to_treasury)
    return df
