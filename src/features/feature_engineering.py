# src/features/feature_engineering.py

from pathlib import Path

import pandas as pd

from src.features.time_features import add_time_features


def create_lag_features(df):
    """
    Create historical demand features.
    """

    df = df.copy()

    # Lag features
    lag_periods = [1, 7, 14, 21, 28]

    for lag in lag_periods:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    return df


def create_rolling_features(df):
    """
    Create rolling statistics using only previous observations.

    shift(1) is extremely important to prevent data leakage.
    """

    df = df.copy()

    df["rolling_mean_7"] = (
        df["sales"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["rolling_mean_14"] = (
        df["sales"]
        .shift(1)
        .rolling(window=14)
        .mean()
    )

    df["rolling_mean_28"] = (
        df["sales"]
        .shift(1)
        .rolling(window=28)
        .mean()
    )

    df["rolling_std_7"] = (
        df["sales"]
        .shift(1)
        .rolling(window=7)
        .std()
    )

    return df


def create_event_features(df):
    """
    Convert event-related information into model-friendly features.
    """

    df = df.copy()

    if "is_event" in df.columns:
        df["is_event"] = df["is_event"].fillna(0).astype(int)

    return df


def create_features(input_path, output_path):
    """
    Complete feature engineering pipeline.
    """

    print("Loading cleaned dataset...")

    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    print("Creating time features...")
    df = add_time_features(df)

    print("Creating lag features...")
    df = create_lag_features(df)

    print("Creating rolling features...")
    df = create_rolling_features(df)

    print("Creating event features...")
    df = create_event_features(df)

    # Remove rows where lag/rolling features are unavailable
    df = df.dropna().reset_index(drop=True)

    # Remove unnecessary columns
    columns_to_drop = [
        "Unnamed: 0",
        "event_name_1",
        "event_name_2",
    ]

    df = df.drop(
        columns=[
            col for col in columns_to_drop
            if col in df.columns
        ]
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_path, index=False)

    print(f"Feature engineered dataset saved to:")
    print(output_path)

    print(f"Final shape: {df.shape}")

    return df