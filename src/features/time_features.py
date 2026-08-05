# src/features/time_features.py

import numpy as np
import pandas as pd


def add_time_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["date"].dt.dayofweek >= 5
    ).astype(int)

    # Weekly cyclic features
    df["weekday_sin"] = np.sin(
        2 * np.pi * df["date"].dt.dayofweek / 7
    )

    df["weekday_cos"] = np.cos(
        2 * np.pi * df["date"].dt.dayofweek / 7
    )

    # Yearly cyclic features
    df["day_of_year_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    df["day_of_year_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    )

    return df