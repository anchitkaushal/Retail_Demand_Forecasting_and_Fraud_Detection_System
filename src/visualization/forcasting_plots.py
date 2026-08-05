from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
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

    # ------------------------------------------------
    # Detect prediction columns
    # ------------------------------------------------

    prediction_columns = [
        col
        for col in df.columns
        if col.endswith("_prediction")
    ]

    # ------------------------------------------------
    # Plot every model
    # ------------------------------------------------

    for column in prediction_columns:

        model_name = column.replace(
            "_prediction",
            ""
        )

        plt.figure(
            figsize=(14, 6)
        )

        plt.plot(
            df["date"],
            df["sales"],
            label="Actual"
        )

        plt.plot(
            df["date"],
            df[column],
            label=model_name
        )

        plt.xlabel("Date")
        plt.ylabel("Sales")

        plt.title(
            f"Actual vs Predicted Sales - {model_name}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_dir /
            f"{model_name}_actual_vs_predicted.png",
            dpi=150
        )

        plt.close()

    print(
        f"Forecast plots saved to {output_dir}"
    )