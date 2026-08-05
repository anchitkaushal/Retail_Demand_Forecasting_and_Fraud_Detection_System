from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_residuals(
    predictions_path="reports/forecast_predictions.csv",
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

    df["date"] = pd.to_datetime(
        df["date"]
    )

    prediction_columns = [
        col
        for col in df.columns
        if col.endswith("_prediction")
    ]

    for column in prediction_columns:

        model_name = column.replace(
            "_prediction",
            ""
        )

        residuals = (
            df["sales"] -
            df[column]
        )

        plt.figure(
            figsize=(12, 5)
        )

        plt.scatter(
            df["date"],
            residuals,
            s=10
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel("Date")
        plt.ylabel("Residual")

        plt.title(
            f"Forecast Residuals - {model_name}"
        )

        plt.tight_layout()

        plt.savefig(
            output_dir /
            f"{model_name}_residuals.png",
            dpi=150
        )

        plt.close()

    print(
        "Residual plots generated."
    )