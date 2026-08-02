import pandas as pd


def load_calendar(url: str) -> pd.DataFrame:
    """
    Loads the calendar dataset from the given link or file path.

    Parameters:
        url (str): URL link or file path to the calendar CSV dataset.

    Returns:
        pd.DataFrame: Dataframe containing the calendar dataset.
    """
    return pd.read_csv(url)


def load_sales(url: str) -> pd.DataFrame:
    """
    Loads the sales dataset from the given link or file path.

    Parameters:
        url (str): URL link or file path to the sales CSV dataset.

    Returns:
        pd.DataFrame: Dataframe containing the sales dataset.
    """
    return pd.read_csv(url)


def load_prices(url: str) -> pd.DataFrame:
    """
    Loads the prices dataset from the given link or file path.

    Parameters:
        url (str): URL link or file path to the prices CSV dataset.

    Returns:
        pd.DataFrame: Dataframe containing the prices dataset.
    """
    return pd.read_csv(url)
