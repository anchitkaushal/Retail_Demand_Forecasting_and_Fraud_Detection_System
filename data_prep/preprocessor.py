import pandas as pd


# HELPER FUNCTIONS

def check_required_columns(df, required_columns, dataset_name):
    """
    Check whether all required columns are present.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def remove_duplicates(df, dataset_name):
    """
    Remove completely duplicated rows.
    """

    before = len(df)

    df = (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    removed = before - len(df)

    if removed > 0:
        print(
            f"{dataset_name}: "
            f"removed {removed:,} duplicate rows."
        )

    return df


# 1. CALENDAR


def clean_calendar(calendar):
    """
    Clean M5 calendar dataset.

    Operations:
    - Validate required columns
    - Remove duplicate rows
    - Convert date to datetime
    - Handle missing event information
    - Create is_event
    - Sort chronologically

    Parameters
    ----------
    calendar : pandas.DataFrame
        Raw M5 calendar dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned calendar dataset.
    """

    print("\nCleaning calendar dataset...")
    print("-" * 50)

    # Work on a copy so original dataframe is not modified
    calendar = calendar.copy()

 
    # Required columns

    check_required_columns(
        calendar,
        ["date", "d"],
        "Calendar dataset"
    )

    # Remove duplicates

    calendar = remove_duplicates(
        calendar,
        "Calendar dataset"
    )

    # Convert date

    calendar["date"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    )

    invalid_dates = calendar["date"].isna().sum()

    if invalid_dates > 0:

        print(
            f"Warning: "
            f"{invalid_dates:,} invalid dates found."
        )

        calendar = (
            calendar
            .dropna(subset=["date"])
            .reset_index(drop=True)
        )

    # Event columns

    event_columns = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"
    ]

    for column in event_columns:

        if column in calendar.columns:

            calendar[column] = (
                calendar[column]
                .fillna("No_Event")
                .astype(str)
                .str.strip()
            )

    # Create event indicator
    # 

    calendar["is_event"] = (
        (calendar["event_name_1"] != "No_Event")
        |
        (calendar["event_name_2"] != "No_Event")
    ).astype(int)


    # Sort chronologically

    calendar = (
        calendar
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Summary

    print(
        f"Rows after cleaning : "
        f"{len(calendar):,}"
    )

    print(
        f"Event days          : "
        f"{calendar['is_event'].sum():,}"
    )

    print("Calendar cleaning completed.")

    return calendar


# 2. SALES

def clean_sales(sales):
    """
    Clean M5 sales dataset.

    Operations:
    - Remove duplicate rows
    - Identify daily sales columns
    - Convert sales to numeric
    - Handle missing values
    - Handle negative demand
    - Convert demand to integer

    Parameters
    ----------
    sales : pandas.DataFrame
        Raw M5 sales dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned sales dataset.
    """

    print("\nCleaning sales dataset...")
    print("-" * 50)

    # Work on a copy
    sales = sales.copy()

    # Remove duplicates

    sales = remove_duplicates(
        sales,
        "Sales dataset"
    )

    # Find daily sales columns

    sales_columns = [
        column
        for column in sales.columns
        if column.startswith("d_")
    ]

    if not sales_columns:

        raise ValueError(
            "No daily sales columns (d_*) were found."
        )

    print(
        f"Daily sales columns : "
        f"{len(sales_columns):,}"
    )

    # Convert sales to numeric

    for column in sales_columns:

        sales[column] = pd.to_numeric(
            sales[column],
            errors="coerce"
        )

    # Missing values

    missing_values = (
        sales[sales_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:

        print(
            f"Missing sales values : "
            f"{missing_values:,}"
        )

        # M5 demand is treated as zero
        sales[sales_columns] = (
            sales[sales_columns]
            .fillna(0)
        )

    else:

        print("Missing sales values : 0")

    # Negative values

    negative_values = (
        sales[sales_columns] < 0
    ).sum().sum()

    if negative_values > 0:

        print(
            f"Negative sales values : "
            f"{negative_values:,}"
        )

        # Demand cannot be negative
        sales[sales_columns] = (
            sales[sales_columns]
            .clip(lower=0)
        )

    else:

        print("Negative sales values : 0")

    # Convert demand to integer

    sales[sales_columns] = (
        sales[sales_columns]
        .round()
        .astype(int)
    )

    # Summary

    print(
        f"Rows after cleaning : "
        f"{len(sales):,}"
    )

    print("Sales cleaning completed.")

    return sales


# 3. SELL PRICES

def clean_prices(prices):
    """
    Clean M5 sell_prices dataset.

    Operations:
    - Validate required columns
    - Remove duplicate rows
    - Convert sell_price to numeric
    - Detect missing prices
    - Detect invalid/non-positive prices
    - Sort data

    Parameters
    ----------
    prices : pandas.DataFrame
        Raw M5 sell_prices dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned sell_prices dataset.
    """

    print("\nCleaning sell price dataset...")
    print("-" * 50)

    # Work on a copy
    prices = prices.copy()

    # Required columns

    check_required_columns(
        prices,
        [
            "store_id",
            "item_id",
            "wm_yr_wk",
            "sell_price"
        ],
        "Sell price dataset"
    )

    # Remove duplicates

    prices = remove_duplicates(
        prices,
        "Sell price dataset"
    )

    # Convert price to numeric

    prices["sell_price"] = pd.to_numeric(
        prices["sell_price"],
        errors="coerce"
    )

    # Missing prices

    missing_prices = (
        prices["sell_price"]
        .isna()
        .sum()
    )

    print(
        f"Missing price values : "
        f"{missing_prices:,}"
    )

    # Invalid prices

    invalid_prices = (
        prices["sell_price"].notna()
        &
        (prices["sell_price"] <= 0)
    ).sum()

    if invalid_prices > 0:

        print(
            f"Invalid/non-positive prices : "
            f"{invalid_prices:,}"
        )

        # Do not replace invalid prices with zero.
        # Mark them as missing.
        prices.loc[
            prices["sell_price"] <= 0,
            "sell_price"
        ] = pd.NA

    else:

        print(
            "Invalid/non-positive prices : 0"
        )

    # Sort

    prices = (
        prices
        .sort_values(
            [
                "store_id",
                "item_id",
                "wm_yr_wk"
            ]
        )
        .reset_index(drop=True)
    )

    # Summary

    print(
        f"Rows after cleaning : "
        f"{len(prices):,}"
    )

    print("Sell price cleaning completed.")

    return prices