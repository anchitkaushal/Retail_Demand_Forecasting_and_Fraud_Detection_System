# src/fraud/evaluate.py

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_fraud_model(
    model,
    X_test,
    y_test,
    output_path="reports/tables/fraud_results.csv"
):
    """
    Evaluate ANN fraud detection model.
    """

    
    # Predictions
    
    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # Metrics

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    
    # Print results

    print(
        "\nFraud Detection Results"
    )

    print(
        "=" * 60
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
            digits=4
        )
    )

    print(
        "Confusion Matrix:"
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        cm
    )

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    
    # Results dataframe

    results = pd.DataFrame([{

        "Model": "ANN",

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc
    }])

    
    # Save
    
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        output_path,
        index=False
    )

    return results