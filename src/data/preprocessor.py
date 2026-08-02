
import pandas as pd



# CALENDAR DATA CLEANING

def clean_calendar(calendar):
    """
    Clean the M5 calendar dataset.

    Handles:
    - Duplicate rows
    - Date conversion
    - Missing event information
    """

    calendar = calendar.copy()

  
    # Convert date column to datetime
  
    calendar["date"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    )

    # Check if date conversion created missing values
    if calendar["date"].isna().any():
        print("Warning: Invalid dates found in calendar dataset.")

    # Handle missing event information

    event_columns = [
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2"
    ]

    for column in event_columns:
        if column in calendar.columns:
            calendar[column] = calendar[column].fillna("No_Event")

    # Create a general event indicator
   
    calendar["is_event"] = (
        (calendar["event_name_1"] != "No_Event") |
        (calendar["event_name_2"] != "No_Event")
    ).astype(int)

    return calendar



# SALES DATA CLEANING


def clean_sales(sales):
   

    sales = sales.copy()

    
    
    # Identify sales/day columns
    
    sales_columns = [
        column for column in sales.columns
        if column.startswith("d_")
    ]

    # Make sure sales values are numeric
   
    sales[sales_columns] = sales[sales_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # If conversion created NaN, treat them as zero
    sales[sales_columns] = sales[sales_columns].fillna(0)

    return sales


# SELL PRICE DATA CLEANING


def clean_prices(prices):
    """
    Clean the M5 sell_prices dataset.

    Handles:
    - Duplicate rows
    - Numeric conversion
    - Missing prices
    - Invalid/non-positive prices
    """

    prices = prices.copy()

   
    # Convert price to numeric
    
    prices["sell_price"] = pd.to_numeric(
        prices["sell_price"],
        errors="coerce"
    )

    
    # Check missing prices
    
    missing_prices = prices["sell_price"].isna().sum()

    if missing_prices > 0:
        print(
            f"Warning: {missing_prices} missing price values found."
        )

    
    # 4. Check invalid prices
    
    invalid_prices = (
        prices["sell_price"].notna() &
        (prices["sell_price"] <= 0)
    ).sum()

    if invalid_prices > 0:
        print(
            f"Warning: {invalid_prices} non-positive price values found."
        )

    return prices