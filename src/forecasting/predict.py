# src/forecasting/predict.py

from pathlib import Path

import joblib
import pandas as pd


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

    X = X.astype(float)

    return X


def generate_predictions(
    df,
    model_path,
    feature_path,
    output_path
):

    model = joblib.load(model_path)

    feature_names = joblib.load(
        feature_path
    )

    X = prepare_features(
        df,
        feature_names
    )

    predictions = model.predict(X)

    result = df[
        ["date", "sales"]
    ].copy()

    result["predicted_sales"] = predictions

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Predictions saved to {output_path}"
    )

    return result