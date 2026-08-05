# src/forecasting/train.py

# src/forecasting/train.py

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


TARGET = "sales"


def prepare_data(df):
    """
    Prepare features and target for forecasting.

    Important:
    - Data is sorted chronologically.
    - Target variable 'sales' is removed from X.
    - Date-related categorical variables are encoded.
    - Missing feature values are filled with 0.
    """

    df = df.copy()

    # ------------------------------------------------
    # Convert date
    # ------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # ------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------
    # Remove columns that should NOT directly
    # enter the forecasting model
    # ------------------------------------------------

    drop_columns = [
        "sales",
        "d",
        "date",
        "event_type_1",
        "event_type_2",
    ]

    X = df.drop(
        columns=[
            col
            for col in drop_columns
            if col in df.columns
        ]
    )

    y = df[TARGET]

    # ------------------------------------------------
    # Convert categorical columns
    # ------------------------------------------------

    categorical_columns = [
        col
        for col in [
            "weekday",
            "month",
            "year"
        ]
        if col in X.columns
    ]

    if categorical_columns:

        X = pd.get_dummies(
            X,
            columns=categorical_columns,
            drop_first=True
        )

    # ------------------------------------------------
    # Convert everything to numeric
    # ------------------------------------------------

    X = X.astype(float)

    # ------------------------------------------------
    # Handle missing values
    # ------------------------------------------------

    X = X.fillna(0)

    return X, y


def time_split(
    X,
    y,
    df,
    train_ratio=0.8
):
    # Make sure the original dataframe is chronological
    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(df) * train_ratio
    )

    # ------------------------------------------------
    # Split X and y using the same chronological index
    # ------------------------------------------------

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # ------------------------------------------------
    # Print split information
    # ------------------------------------------------

    print("\nTime-Series Train/Test Split")
    print("=" * 60)

    print(
        f"Training rows : {len(X_train)}"
    )

    print(
        f"Testing rows  : {len(X_test)}"
    )

    print(
        f"Training period: "
        f"{df['date'].iloc[0].date()} → "
        f"{df['date'].iloc[split_index - 1].date()}"
    )

    print(
        f"Testing period : "
        f"{df['date'].iloc[split_index].date()} → "
        f"{df['date'].iloc[-1].date()}"
    )

    print("=" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def train_models(
    df,
    model_dir="models"
):
    """
    Train forecasting models.

    Models:
        1. Ridge Regression
        2. Gradient Boosting
        3. XGBoost
    """

    model_dir = Path(
        model_dir
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------
    # Prepare data
    # ------------------------------------------------

    X, y = prepare_data(
        df
    )

    # ------------------------------------------------
    # Time-based split
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = time_split(
        X,
        y,
        df
    )

    # ------------------------------------------------
    # Diagnostic information
    # ------------------------------------------------

    print("\nForecasting Dataset")
    print("=" * 60)

    print(
        f"Total rows   : {len(X)}"
    )

    print(
        f"Total features: {X.shape[1]}"
    )

    print(
        "\nTarget variable:"
    )

    print(
        y.describe()
    )

    print(
        "\nTraining target:"
    )

    print(
        y_train.describe()
    )

    print(
        "\nTesting target:"
    )

    print(
        y_test.describe()
    )

    # ------------------------------------------------
    # Check potential sales leakage
    # ------------------------------------------------

    sales_related_features = [
        col
        for col in X.columns
        if (
        col.startswith("lag_")
        or
        col.startswith("rolling_")
    )
    ]

    print(
        "\nSales-related features:"
    )

    for feature in sales_related_features:
        print(
            f"  - {feature}"
        )

    # ------------------------------------------------
    # Models dictionary
    # ------------------------------------------------

    models = {}

    # =================================================
    # Ridge Regression
    # =================================================

    print(
        "\nTraining Ridge Regression..."
    )

    ridge = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(
                alpha=1.0
            )
        )
    ])

    ridge.fit(
        X_train,
        y_train
    )

    models["ridge"] = ridge

    # =================================================
    # Gradient Boosting
    # =================================================

    print(
        "Training Gradient Boosting..."
    )

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    gradient_boosting.fit(
        X_train,
        y_train
    )

    models[
        "gradient_boosting"
    ] = gradient_boosting

    # =================================================
    # XGBoost
    # =================================================

    if XGBOOST_AVAILABLE:

        print(
            "Training XGBoost..."
        )

        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        )

        xgb.fit(
            X_train,
            y_train
        )

        models[
            "xgboost"
        ] = xgb

    else:

        print(
            "XGBoost is not installed. Skipping XGBoost."
        )

    # =================================================
    # Save trained models
    # =================================================

    print(
        "\nSaving models..."
    )

    for name, model in models.items():

        model_path = (
            model_dir /
            f"{name}_forecasting_model.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"Saved: {model_path}"
        )

    # =================================================
    # Save feature names
    # =================================================

    feature_path = (
        model_dir /
        "forecasting_features.pkl"
    )

    joblib.dump(
        list(X.columns),
        feature_path
    )

    print(
        f"Saved: {feature_path}"
    )

    return (
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )