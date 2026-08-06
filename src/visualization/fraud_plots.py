from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_fraud_probabilities(
    predictions_path="reports/predictions/fraud_predictions.csv",
    output_dir="reports/figures"
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        predictions_path
    )

    plt.figure(
        figsize=(10, 6)
    )

    normal = df[
        df["is_fraud"] == 0
    ]["fraud_probability"]

    fraud = df[
        df["is_fraud"] == 1
    ]["fraud_probability"]

    plt.hist(
        normal,
        bins=30,
        alpha=0.6,
        label="Normal"
    )

    plt.hist(
        fraud,
        bins=30,
        alpha=0.6,
        label="Fraud"
    )

    plt.xlabel(
        "Fraud Probability"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.title(
        "Fraud Probability Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "fraud_probability_distribution.png",
        dpi=150
    )

    plt.close()

    print(
        "Fraud probability plot saved."
    )