# src/fraud/predict.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def predict_fraud(
    df,
    model_path="models/fraud_ann.pkl",
    feature_path="models/fraud_features.pkl",
    output_path=None
):
    """
    Generating fraud probabilities, predictions, and risk levels.

    Parameters:
        df: pd.DataFrame or str/Path (input transaction data)
        model_path: str/Path or loaded model
        feature_path: str/Path, list of features, or output CSV path (for backward compatibility)
        output_path: str/Path or None (optional CSV output destination)

    Returns:
        pd.DataFrame containing original columns plus 'fraud_probability', 'fraud_prediction', 'risk_level'
    """
    # Backward compatibility check if 3rd positional argument is output_path
    if isinstance(feature_path, (str, Path)) and (
        str(feature_path).endswith(".csv") or "reports" in str(feature_path) or "predictions" in str(feature_path)
    ):
        output_path = feature_path
        feature_path = "models/fraud_features.pkl"

    # 1. Loading dataframe if path is given
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)
    else:
        df = df.copy()

    # 2. Load model
    if isinstance(model_path, (str, Path)):
        model = joblib.load(model_path)
    else:
        model = model_path

    # 3. Load features
    if isinstance(feature_path, (str, Path)):
        feature_names = joblib.load(feature_path)
    else:
        feature_names = feature_path

    # 4. Validate features
    missing_features = [
        feature for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required fraud features: {missing_features}"
        )

    # 5. Prepare X matrix
    X = df[feature_names].astype(float).fillna(0)

    # 6. Generate predictions and probabilities
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["fraud_prediction"] = model.predict(X)

    # 7. Add risk level (HIGH >= 0.75, MEDIUM >= 0.40, LOW < 0.40)
    df["risk_level"] = np.where(
        df["fraud_probability"] >= 0.75,
        "HIGH",
        np.where(
            df["fraud_probability"] >= 0.40,
            "MEDIUM",
            "LOW"
        )
    )

    # 8. Save output if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        df.to_csv(output_path, index=False)
        print(f"Fraud predictions saved to {output_path}")

    return df