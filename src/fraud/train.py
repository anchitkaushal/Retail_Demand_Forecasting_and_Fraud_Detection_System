# src/fraud/train.py

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURES = [
    "quantity",
    "transaction_amount",
    "return_days",
    "customer_frequency",
    "previous_returns"
]

TARGET = "is_fraud"


def train_fraud_model(
    input_path="data/simulated/fraud_transactions.csv",
    model_path="models/fraud_ann.pkl"
):
    """
    Train ANN-based fraud detection model.
    """

    # ------------------------------------------------
    # Load data
    # ------------------------------------------------

    df = pd.read_csv(
        input_path
    )

    # ------------------------------------------------
    # Features and target
    # ------------------------------------------------

    X = df[FEATURES]

    y = df[TARGET]

    # ------------------------------------------------
    # Display class distribution
    # ------------------------------------------------

    print(
        "\nFraud Training Dataset"
    )

    print(
        "=" * 50
    )

    print(
        f"Total transactions: {len(df)}"
    )

    print(
        f"Fraud transactions : {y.sum()}"
    )

    print(
        f"Normal transactions: {(y == 0).sum()}"
    )

    # ------------------------------------------------
    # Train/test split
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # ------------------------------------------------
    # ANN model
    # ------------------------------------------------

    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ann",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20
            )
        )
    ])

    # ------------------------------------------------
    # Train
    # ------------------------------------------------

    print(
        "\nTraining ANN..."
    )

    model.fit(
        X_train,
        y_train
    )

    # ------------------------------------------------
    # Save model
    # ------------------------------------------------

    model_path = Path(
        model_path
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    # ------------------------------------------------
    # Save feature names
    # ------------------------------------------------

    feature_path = (
        model_path.parent /
        "fraud_features.pkl"
    )

    joblib.dump(
        FEATURES,
        feature_path
    )

    print(
        f"Fraud ANN saved to {model_path}"
    )

    print(
        f"Fraud features saved to {feature_path}"
    )

    return (
        model,
        X_test,
        y_test
    )