# Retail Demand Forecasting & Fraud Simulation System

A machine learning pipeline for time-indexed retail demand forecasting, store clustering, statistical hypothesis testing, and return fraud anomaly detection using Artificial Neural Networks (ANN).

## 📌 Project Overview

This project addresses key retail analytics and operations challenges:
1. **Time-Aware EDA & Seasonality**: Analysis of sales patterns across time, holidays, and promotional periods.
2. **Hypothesis Testing**: Statistical evaluation of promotion significance on sales volume.
3. **Demand Forecasting**: Comparative forecasting using Linear Regression, XGBoost, and Gradient Boosting.
4. **Store Segmentation**: KMeans clustering to segment stores based on operational and purchasing behaviors.
5. **Fraud & Anomaly Flagging**: Neural network (ANN / Autoencoder) to detect fraud-like return patterns.

---

## 📁 Directory Structure

```
Retail_Demand_Forecasting_and_Fraud_Detection_System/
├── config/
│   └── config.yaml                 # Configuration parameters & paths
├── data/
│   ├── external/                   # External datasets (holidays, weather, promo data)
│   ├── interim/                    # Intermediate transformed & cleaned files
│   ├── processed/                  # Final feature tables ready for modeling
│   ├── raw/                        # Original immutable datasets
│   ├── simulated/                  # Synthetic return & fraud transaction data
│   └── README.md                   # Data directory layout documentation
├── models/                         # Serialized model artifacts and model outputs
├── notebooks/
│   ├── eda/                        # Exploratory Data Analysis notebooks
│   ├── experiments/                # Experimental notebooks & prototypes
│   ├── 01_eda_and_seasonality.ipynb        # Time-series EDA & holiday/promo analysis
│   ├── 02_hypothesis_testing.ipynb        # Statistical significance tests for promos
│   ├── 03_demand_forecasting_models.ipynb  # Regression, XGBoost & Gradient Boosting
│   ├── 04_store_clustering_kmeans.ipynb    # Store segmentation with KMeans
│   ├── 05_fraud_anomaly_detection_ann.ipynb # ANN for return fraud detection
│   └── README.md                   # Notebooks directory documentation
├── reports/
│   ├── figures/                    # Generated plots, graphics, and figures
│   └── tables/                     # Generated data tables and metrics summaries
├── src/
│   ├── data/
│   │   ├── loader.py              # Data simulation and loading routines
│   │   └── preprocessor.py        # Preprocessing and scaling functions
│   ├── features/
│   │   ├── feature_engineering.py # Feature engineering pipeline
│   │   └── time_features.py       # Time-series & lag/rolling feature extraction
│   ├── forcasting/
│   │   ├── train.py               # Demand forecasting model training
│   │   ├── evaluate.py            # Forecasting metrics and evaluation
│   │   └── predict.py             # Inference / forecasting predictions
│   ├── fraud/
│   │   ├── simulate.py            # Synthetic return & fraud generation
│   │   ├── predict.py             # ANN return fraud anomaly detection inference
│   │   └── evaluate.py            # Fraud model evaluation metrics
│   └── visualization/
│       └── plots.py               # Plotting and visualization helpers
├── .gitignore                      # Git ignore file
├── main.py                         # End-to-end execution script
└── requirements.txt                # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites & Setup

1. **Activate Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Full Pipeline**:
   ```bash
   python main.py
   ```

