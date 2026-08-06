from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_actual_vs_predicted(
    predictions_path="reports/predictions/forecast_predictions.csv",
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

    
    # Detect prediction columns

    prediction_columns = [
        col
        for col in df.columns
        if col.endswith("_prediction") or col.startswith("predicted_") or col == "predicted_sales"
    ]

    
    # Plot every model

    for column in prediction_columns:

        if column.endswith("_prediction"):
            model_name = column.replace(
                "_prediction",
                ""
            )
        elif column.startswith("predicted_"):
            model_name = column.replace(
                "predicted_",
                ""
            )
        else:
            model_name = column

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