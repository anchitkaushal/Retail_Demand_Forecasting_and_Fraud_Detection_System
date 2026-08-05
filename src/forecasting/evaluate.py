from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    # ------------------------------------------------
    # Safe MAPE
    # Ignore extremely small actual values
    # ------------------------------------------------

    mask = np.abs(y_true) >= 100

    if mask.sum() > 0:

        mape = np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        ) * 100

    else:
        mape = np.nan

    # ------------------------------------------------
    # WMAPE
    # ------------------------------------------------

    wmape = (
        np.sum(
            np.abs(y_true - y_pred)
        )
        /
        np.sum(
            np.abs(y_true)
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WMAPE": wmape
    }


def evaluate_models(
    models,
    X_test,
    y_test,
    output_path="reports/tables/forecasting_results.csv"
):

    results = []

    for name, model in models.items():

        predictions = model.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            predictions
        )

        metrics["Model"] = name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "MAE",
            "RMSE",
            "MAPE",
            "WMAPE"
        ]
    ]

    # Lower RMSE is better
    results_df = results_df.sort_values(
        "RMSE"
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nForecasting Results")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df