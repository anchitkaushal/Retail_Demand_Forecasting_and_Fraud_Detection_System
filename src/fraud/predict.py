# src/fraud/predict.py

from pathlib import Path

import joblib
import pandas as pd


def predict_fraud(
    input_path="data/simulated/fraud_transactions.csv",
    model_path="models/fraud_ann.pkl",
    output_path="reports/tables/fraud_predictions.csv"
):
    """
    Generate fraud probabilities and predictions.
    """

    # ------------------------------------------------
    # Load model
    # ------------------------------------------------

    model = joblib.load(
        model_path
    )

    feature_names = joblib.load(
        "models/fraud_features.pkl"
    )

    # ------------------------------------------------
    # Load transactions
    # ------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    # ------------------------------------------------
    # Validate features
    # ------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing fraud features: "
            + str(missing_features)
        )

    X = df[
        feature_names
    ]

    # ------------------------------------------------
    # Predict probability
    # ------------------------------------------------

    df["fraud_probability"] = (
        model.predict_proba(
            X
        )[:, 1]
    )

    # ------------------------------------------------
    # Predict class
    # ------------------------------------------------

    df["fraud_prediction"] = (
        model.predict(
            X
        )
    )

    # ------------------------------------------------
    # Save predictions
    # ------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Fraud predictions saved to {output_path}"
    )

    return df