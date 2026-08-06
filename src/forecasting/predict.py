# src/forecasting/predict.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    create_event_features,
    create_lag_features,
    create_rolling_features,
)
from src.features.time_features import add_time_features


def prepare_features(df, feature_names):
    """
    Prepare prediction features in exactly the same
    structure used during training.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    drop_columns = [
        "sales",
        "date",
        "event_type_1",
        "event_type_2",
    ]

    X = df.drop(
        columns=[
            col for col in drop_columns
            if col in df.columns
        ]
    )

    X = pd.get_dummies(
        X,
        columns=[
            col for col in [
                "weekday",
                "month",
                "year"
            ]
            if col in X.columns
        ],
        drop_first=True
    )

    # Make sure columns match training
    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    X = X.astype(float).fillna(0)

    return X


def forecast_future(
    df,
    model_path,
    feature_path,
    horizon=30
):
    """
    Generate future demand predictions recursively using trained model.

    Parameters:
        df: pd.DataFrame or str/Path (historical dataset with 'date' and 'sales')
        model_path: str/Path or loaded model
        feature_path: str/Path or list of feature names
        horizon: int (number of future days to forecast, 1 to 90+)

    Returns:
        pd.DataFrame containing ['date', 'forecast']
    """
    # 1. Load data if path
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)
    else:
        df = df.copy()

    # 2. Load model & feature list
    if isinstance(model_path, (str, Path)):
        model = joblib.load(model_path)
    else:
        model = model_path

    if isinstance(feature_path, (str, Path)):
        feature_names = joblib.load(feature_path)
    else:
        feature_names = feature_path

    # 3. Validate required columns
    required_cols = ["date", "sales"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe missing required column(s): {missing_cols}")

    # 4. Clean & sort data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df = df.dropna(subset=["date", "sales"]).sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Input dataframe contains no valid historical date and sales data.")

    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")

    # 5. Extract sales history and max date
    last_date = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    history_sales = list(df["sales"].astype(float).values)

    predictions = []

    # Check for event history mapping if present
    event_map = {}
    if "is_event" in df.columns:
        event_map = dict(zip(df["date"], df["is_event"].fillna(0).astype(int)))

    # 6. Recursive forecasting
    for step_date in future_dates:
        row_dict = {
            "date": step_date,
            "sales": np.nan
        }

        row_df = pd.DataFrame([row_dict])
        row_df = add_time_features(row_df)

        # Categorical columns used in get_dummies
        row_df["weekday"] = step_date.day_name()
        row_df["month"] = step_date.month
        row_df["year"] = step_date.year

        # Event feature
        row_df["is_event"] = event_map.get(step_date, 0)
        row_df = create_event_features(row_df)

        # Lag features
        for lag in [1, 7, 14, 21, 28]:
            row_df[f"lag_{lag}"] = (
                history_sales[-lag] if len(history_sales) >= lag else 0.0
            )

        # Rolling features
        if len(history_sales) >= 7:
            r7 = history_sales[-7:]
            row_df["rolling_mean_7"] = np.mean(r7)
            row_df["rolling_std_7"] = np.std(r7, ddof=1) if len(r7) > 1 else 0.0
            row_df["rolling_7"] = np.mean(r7)
        else:
            row_df["rolling_mean_7"] = 0.0
            row_df["rolling_std_7"] = 0.0
            row_df["rolling_7"] = 0.0

        if len(history_sales) >= 14:
            r14 = history_sales[-14:]
            row_df["rolling_mean_14"] = np.mean(r14)
        else:
            row_df["rolling_mean_14"] = 0.0

        if len(history_sales) >= 28:
            r28 = history_sales[-28:]
            row_df["rolling_mean_28"] = np.mean(r28)
            row_df["rolling_28"] = np.mean(r28)
        else:
            row_df["rolling_mean_28"] = 0.0
            row_df["rolling_28"] = 0.0

        # One-hot encoding & align features
        drop_cols = ["sales", "d", "date", "event_type_1", "event_type_2"]
        X_row = row_df.drop(columns=[col for col in drop_cols if col in row_df.columns])

        cat_cols = [c for c in ["weekday", "month", "year"] if c in X_row.columns]
        if cat_cols:
            X_row = pd.get_dummies(X_row, columns=cat_cols, drop_first=True)

        X_row = X_row.reindex(columns=feature_names, fill_value=0).astype(float).fillna(0)

        # Predict next step
        pred = model.predict(X_row)[0]
        pred = max(0.0, float(pred))  # Prevent negative demand

        predictions.append(pred)
        history_sales.append(pred)  # Use prediction recursively

    # 7. Build output dataframe
    result_df = pd.DataFrame({
        "date": future_dates,
        "forecast": predictions
    })

    return result_df


def generate_predictions(
    df,
    model_path,
    feature_path,
    output_path
):
    """
    Generate predictions for existing dataset (used by main.py training/eval pipeline).
    """
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)

    X = prepare_features(df, feature_names)
    predictions = model.predict(X)

    result = df[["date", "sales"]].copy()
    result["predicted_sales"] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    return result