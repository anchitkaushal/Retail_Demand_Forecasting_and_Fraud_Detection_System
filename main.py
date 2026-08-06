# main.py

from pathlib import Path

from src.features.feature_engineering import (
    create_features
)

from src.forecasting.train import (
    train_models
)

from src.forecasting.evaluate import (
    evaluate_models
)

from src.forecasting.predict import (
    generate_predictions
)

from src.fraud.simulate import (
    simulate_fraud_data
)

from src.fraud.train import (
    train_fraud_model
)

from src.fraud.evaluate import (
    evaluate_fraud_model
)

from src.fraud.predict import (
    predict_fraud
)

from src.visualization.forcasting_plots import(
    plot_actual_vs_predicted
)

from src.visualization.fraud_plots import(
    plot_fraud_probabilities
)

from src.visualization.residuals_plot import(
    plot_residuals
)


# PROJECT PATHS


ROOT = Path(__file__).resolve().parent

INTERIM_DATA = (
    ROOT /
    "data" /
    "interim"
)

PROCESSED_DATA = (
    ROOT /
    "data" /
    "processed"
)

MODELS = (
    ROOT /
    "models"
)

REPORTS = (
    ROOT /
    "reports"
)


# FORECASTING PIPELINE


def run_forecasting_pipeline():

    print("\n")
    print("=" * 70)
    print("FORECASTING PIPELINE")
    print("=" * 70)

    input_file = (
        INTERIM_DATA /
        "daily_total_sales.csv"
    )

    feature_file = (
        PROCESSED_DATA /
        "forecasting_features.csv"
    )

  
    # Feature Engineering


    df = create_features(
        input_file,
        feature_file
    )

  
    # Train models


    (
        models,
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_models(
        df,
        MODELS
    )

    # Evaluate
  

    results = evaluate_models(
        models,
        X_test,
        y_test,
        REPORTS /
        'tables'/
        "forecasting_results.csv"
    )

    print("\nBest forecasting model:")

    print(
        results.iloc[0]
    )

    
    # Generate predictions


    best_model_name = (
        results.iloc[0]["Model"]
    )

    best_model_path = (
        MODELS /
        f"{best_model_name}_forecasting_model.pkl"
    )

    prediction_file = (
        REPORTS /
        'predictions'/
        "forecast_predictions.csv"
    )

    generate_predictions(
        df,
        best_model_path,
        MODELS /
        "forecasting_features.pkl",
        prediction_file
    )

    return results



# FRAUD PIPELINE


def run_fraud_pipeline():

    print("\n")
    print("=" * 70)
    print("FRAUD DETECTION PIPELINE")
    print("=" * 70)

    fraud_file = (
        ROOT /
        "data" /
        "simulated" /
        "fraud_transactions.csv"
    )

  
    # Simulate data
  

    simulate_fraud_data(
        n_transactions=10000,
        fraud_rate=0.05,
        output_path=fraud_file
    )

   
    # Train ANN
  

    (
        model,
        X_test,
        y_test
    ) = train_fraud_model(
        fraud_file,
        MODELS /
        "fraud_ann.pkl"
    )


    # Evaluate


    evaluate_fraud_model(
        model,
        X_test,
        y_test,
        REPORTS /
        'tables'/
        "fraud_results.csv"
    )

    # Prediction


    predict_fraud(
        fraud_file,
        MODELS /
        "fraud_ann.pkl",
        REPORTS /
        'predictions'/
        "fraud_predictions.csv"
    )


#Final Plots
def plot_figures():
    plot_actual_vs_predicted()
    plot_fraud_probabilities()
    plot_residuals()
    return 

# MAIN


def main():

    print("=" * 70)
    print("RETAIL DEMAND FORECASTING + FRAUD DETECTION")
    print("=" * 70)

    # Run forecasting
    run_forecasting_pipeline()

    # Run fraud detection
    run_fraud_pipeline()

    print("\n")
    print("=" * 70)
    print("COMPLETE PIPELINE FINISHED")
    print("=" * 70)

    plot_figures()
    print("\n plots of actul vs prediction forcasting , fraud probabilities and residuals are saved in reports/figures ")

if __name__ == "__main__":
    main()
