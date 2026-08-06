# 🛒 Retail Demand Forecasting and Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-red)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![Git](https://img.shields.io/badge/Version_Control-Git-F05032?logo=git)](https://git-scm.com/)

An end-to-end Machine Learning project that combines **retail demand forecasting** and **fraud detection** into a single data-driven system.

The project focuses on two practical business problems:

1. **Retail Demand Forecasting** — predicting future sales demand using historical sales, calendar, event, and pricing information.
2. **Fraud Detection** — identifying potentially fraudulent transactions using machine learning classification techniques.

---

### 📌 Project Overview

#### 🛒 Retail Demand Forecasting

The demand forecasting pipeline follows:

```text
Raw M5 Dataset
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Time-Series Analysis
      ↓
Hypothesis Testing
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Demand Forecasting
```

#### 🚨 Fraud Detection

The fraud detection pipeline follows:

```text
Transaction Data
      ↓
Data Simulation / Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Analysis
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Fraud Classification
      ↓
Model Evaluation
      ↓
Fraud Prediction
```

---

### 🎯 Objectives

#### 🛒 Demand Forecasting

- Analyze historical retail sales data.
- Understand trends and seasonal patterns.
- Identify weekly and monthly demand behavior.
- Analyze the effect of events on sales.
- Study price variation and its relationship with demand.
- Perform time-series analysis.
- Engineer meaningful forecasting features.
- Train and compare machine learning regression models.
- Evaluate forecasting performance.
- Build reusable forecasting components.

#### 🚨 Fraud Detection

- Generate or process transaction-level data.
- Simulate realistic transaction behavior when real data is unavailable.
- Identify patterns associated with fraudulent transactions.
- Engineer useful transaction-level features.
- Train classification models.
- Evaluate fraud detection performance.
- Focus on Precision, Recall and F1-score rather than accuracy alone.
- Build reusable fraud prediction components.

---

## 📊 Dataset

## M5 Retail Dataset

The demand forecasting component uses the **M5 Forecasting dataset**, which contains Walmart retail sales information.

The main datasets are:

### 1. Calendar Dataset

Contains:

- Date information
- Weekday
- Month
- Year
- M5 day identifier
- Events
- Event types
- SNAP indicators

Important columns include:

```text
date
wm_yr_wk
weekday
wday
month
year
d
event_name_1
event_type_1
event_name_2
event_type_2
snap_CA
snap_TX
snap_WI
```

### 2. Sales Dataset

Contains historical daily sales for products across stores.

The dataset provides information about:

- Item
- Department
- Category
- Store
- State
- Daily sales

### 3. Sell Prices Dataset

Contains historical selling prices for products across stores and weeks.

Important information includes:

- Store
- Item
- Week
- Selling price

---

## 🧹 Data Preprocessing

The project uses separate preprocessing functions for the M5 datasets.

The preprocessing workflow includes:

- Loading raw datasets.
- Validating required columns.
- Checking data types.
- Handling missing values.
- Converting date columns to datetime.
- Handling missing event information.
- Creating event indicators.
- Removing unnecessary columns.
- Validating cleaned datasets.
- Saving processed datasets.

The preprocessing logic is separated by dataset so that each M5 dataset can be cleaned independently.

Example functions:

```python
clean_calendar()
clean_sales()
clean_sell_prices()
```

This makes the preprocessing code reusable from notebooks and application code.

---

## 🔎 Exploratory Data Analysis

The EDA phase is used to understand the structure and behavior of the retail data before machine learning.

The analysis includes:

- Dataset structure.
- Data types.
- Missing values.
- Duplicate values.
- Descriptive statistics.
- Sales distribution.
- Daily demand trends.
- Store-level sales.
- Item-level sales.
- Category-level sales.
- Event-based sales analysis.
- Price distribution.
- Price variation across items.
- Price variation across stores.

---

## 📈 Time-Series Analysis

Time-series analysis is an important part of the demand forecasting component.

## Trend Analysis

Historical demand was analyzed to identify long-term movement in sales.

The moving averages showed an overall upward demand trend.

## Weekly Seasonality

The analysis identified recurring weekly patterns.

The ACF analysis showed noticeable correlation around:

```text
Lag 7
Lag 14
Lag 21
Lag 28
```

This indicates a strong weekly seasonal component.

## Monthly Patterns

Monthly aggregation was also used to investigate longer-term seasonal behavior.

## Moving Averages

The project uses:

```text
7-day rolling average
28-day rolling average
```

These rolling averages help smooth daily fluctuations and reveal the underlying demand trend.

---

## 📊 Hypothesis Testing

Hypothesis testing was performed to investigate relationships within the retail data.

One analysis focused on whether event days are associated with different sales behavior.

The analysis included event categories such as:

- Sports events
- Religious events
- Cultural events
- National events

The analysis showed that organized event categories such as **Super Bowl and Religious events** were among the prominent event categories in the dataset.

Statistical testing is used to determine whether observed differences are meaningful rather than relying only on visual patterns.

---

## 💰 Price Analysis

The sell-price dataset was analyzed to understand product pricing behavior.

The analysis includes:

- Minimum price.
- Maximum price.
- Mean price.
- Standard deviation.
- Price variation across products.
- Price variation across stores.
- Price changes over time.

Example:

```text
Item
 ├── Minimum Price
 ├── Maximum Price
 ├── Mean Price
 └── Standard Deviation
```

Price information can be incorporated into forecasting features to help models learn relationships between price and demand.

---

## ⚙️ Feature Engineering

Feature engineering transforms raw data into useful machine-learning features.

Potential demand forecasting features include:

## Date Features

```text
year
month
week
day
day_of_week
day_of_month
quarter
```

## Lag Features

```text
lag_1
lag_7
lag_14
lag_28
```

## Rolling Features

```text
rolling_7
rolling_14
rolling_28
```

## Event Features

```text
is_event
event_type
event_name
```

## Price Features

```text
sell_price
price_change
price_difference
```

Feature engineering is performed after preprocessing and exploratory analysis so that the model receives validated and meaningful inputs.

---

## 🤖 Machine Learning Models

Several machine learning algorithms are explored for demand forecasting.

## Ridge Regression

Ridge Regression is used as a regularized linear baseline model.

It provides a simple benchmark while reducing the risk of overfitting from highly correlated features.

## Gradient Boosting

Gradient Boosting is used to capture nonlinear relationships between engineered features and demand.

## XGBoost

XGBoost is used as a powerful tree-based boosting algorithm capable of learning complex relationships between:

- Historical demand
- Lag features
- Rolling statistics
- Calendar features
- Events
- Prices

---

## 🚨 Fraud Detection

The fraud detection component focuses on identifying suspicious transactions.

When real transaction data is unavailable, synthetic transaction data can be generated to simulate realistic fraud detection scenarios.

The simulator can generate transaction characteristics such as:

- Transaction amount.
- Transaction frequency.
- Transaction timing.
- Customer behavior.
- Fraud labels.

NumPy probability distributions can be used to create realistic synthetic variables.

Examples include:

```python
numpy.random.normal()
numpy.random.lognormal()
numpy.random.poisson()
```

---

## 🚨 Fraud Detection Models

The fraud detection problem is treated as a binary classification task.

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

Potential models include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- Artificial Neural Network

The models are evaluated using fraud-specific classification metrics.

---

## 📏 Model Evaluation

### 🛒 Demand Forecasting Metrics

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted demand.

```text
MAE = average(|actual - predicted|)
```

Lower MAE is better.

### RMSE

Root Mean Squared Error penalizes larger prediction errors more strongly.

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

Lower RMSE is better.

### R² Score

R² measures the proportion of variance explained by the model.

A higher R² generally indicates better model fit.

---

### 🚨 Fraud Detection Metrics

Accuracy alone is not sufficient for fraud detection because fraud is normally a minority class.

The project therefore focuses on:

### Precision

Measures how many transactions predicted as fraudulent were actually fraudulent.

### Recall

Measures how many actual fraudulent transactions were successfully detected.

### F1 Score

Combines Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### ROC-AUC

Measures the model's ability to distinguish between legitimate and fraudulent transactions across different classification thresholds.

---

## 📁 Project Structure

```text
Retail_Demand_Forecasting_and_Fraud_Detection_System/
│
├── data/
│   ├── raw/
│   │   ├── calendar.csv
│   │   ├── sales_train_validation.csv
│   │   └── sell_prices.csv
│   │
│   ├── interim/
│   │   ├── clean_calendar.csv
│   │   └── daily_total_sales.csv
│   │
│   └── processed/
│
├── notebooks/
│   ├── 01_eda_and_seasonality.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_demand_forecasting_models.ipynb
│
├── src/
│   ├── forecasting/
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── models.py
│   │   └── predict.py
│   │
│   ├── fraud/
│   │   ├── simulate.py
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── models.py
│   │   └── predict.py
│   │
│   └── utils/
│
├── models/
│
├── reports/
│   └── figures/
│
├── app/
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

The exact structure may evolve as the project develops. The main principle is to keep raw data, processed data, experimentation, reusable source code, trained models, and application code separated.

---

## 🛠️ Technologies Used

### 🐍 Programming

- Python

### 🧹 Data Processing

- Pandas
- NumPy

### 📊 Data Visualization

- Matplotlib
- Seaborn

### 🤖 Machine Learning

- Scikit-learn
- XGBoost

### 🧠 Deep Learning

- PyTorch

### 💾 Model Persistence

- Joblib

### 🌐 Application

- Streamlit

### 💻 Development

- VS Code
- Jupyter Notebook
- WSL / Ubuntu

### 🔀 Version Control

- Git
- GitHub

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/anchitkaushal/Retail_Demand_Forecasting_and_Fraud_Detection_System.git
```

Move into the project directory:

```bash
cd Retail_Demand_Forecasting_and_Fraud_Detection_System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Linux / WSL

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main Python application:

```bash
python main.py
```

If the application uses Streamlit, run:

```bash
streamlit run main.py
```

or, if the Streamlit entry point is a separate file:

```bash
streamlit run app.py
```

---

## 📓 Running the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Recommended execution order:

```text
01_eda_and_seasonality.ipynb
        ↓
02_hypothesis_testing.ipynb
        ↓
03_demand_forecasting_models.ipynb
```

The workflow follows:

```text
EDA
 ↓
Time-Series Understanding
 ↓
Hypothesis Testing
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Evaluation
```

---

## 🔄 Data Processing Workflow

The overall data workflow is:

```text
Raw Data
   ↓
Dataset Validation
   ↓
Preprocessing
   ↓
Clean Data
   ↓
EDA
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction
```

Cleaned and intermediate datasets are stored separately so that expensive preprocessing does not need to be repeated every time the application is executed.

---

## 🏗️ Project Design

The project separates experimentation from reusable production logic.

Jupyter notebooks are used for:

- EDA
- Statistical analysis
- Visualization
- Experimentation
- Model comparison

Reusable Python modules are kept inside `src/`.

The design is:

```text
Notebook
    ↓
Experimentation
    ↓
Validated Logic
    ↓
src/
    ↓
Reusable Code
    ↓
Application
```

This prevents the final application from depending entirely on notebook execution.

---

## 🧪 Reproducibility

The project aims to make experiments reproducible by using:

- Fixed random seeds where appropriate.
- Consistent preprocessing.
- Saved preprocessing objects.
- Saved trained models.
- Clearly separated raw and processed data.
- Version-controlled source code.

Example:

```python
random_state=42
```

---

## 💼 Real-World Applications

### 🛒 Demand Forecasting

A retail company could use this system to:

- Predict upcoming product demand.
- Improve inventory planning.
- Reduce stockouts.
- Reduce overstocking.
- Optimize purchasing decisions.
- Improve warehouse planning.
- Understand seasonal demand.
- Analyze the effect of events and pricing.

### 🚨 Fraud Detection

A retail or financial organization could use the fraud detection system to:

- Identify suspicious transactions.
- Reduce financial losses.
- Prioritize transactions for manual review.
- Detect unusual customer behavior.
- Automate fraud screening.
- Improve transaction security.

---

## ⚠️ Limitations

## M5 Dataset

The original M5 dataset is large, and working with complete item-store level data can require significant memory and computational resources.

For experimentation, aggregated datasets may be used to make analysis and model development more manageable.

## Synthetic Fraud Data

If synthetic transaction data is used, it cannot perfectly represent real-world fraud.

Real fraud datasets typically contain:

- Severe class imbalance.
- Complex behavioral patterns.
- Concept drift.
- Changing fraud strategies.
- False positives.
- False negatives.

Therefore, a model trained on synthetic data should not be directly used in production without validation against real-world data.

---

## 🚀 Future Improvements

### 🛒 Demand Forecasting

- Forecast demand at item-store level.
- Implement Prophet.
- Experiment with LightGBM.
- Perform hyperparameter optimization.
- Implement walk-forward validation.
- Implement multi-step forecasting.
- Add confidence intervals.
- Add automated retraining.
- Add model monitoring.
- Improve forecasting at different aggregation levels.

### 🚨 Fraud Detection

- Use real-world transaction datasets.
- Handle severe class imbalance.
- Experiment with SMOTE and other resampling methods.
- Add anomaly detection.
- Add transaction velocity features.
- Add customer behavioral features.
- Optimize fraud classification thresholds.
- Add SHAP-based explainability.
- Add model monitoring and drift detection.

### 🌐 Application

- Build an interactive Streamlit dashboard.
- Add demand forecast visualizations.
- Add fraud prediction interface.
- Add model performance dashboards.
- Add downloadable prediction results.
- Add automated model retraining.
- Deploy the application to a cloud platform.

---

## 📚 Learning Outcomes

This project provides practical experience with:

- Python programming.
- Pandas and NumPy.
- Data preprocessing.
- Exploratory Data Analysis.
- Data visualization.
- Time-series analysis.
- Trend and seasonality analysis.
- Autocorrelation.
- Hypothesis testing.
- Feature engineering.
- Regression.
- Classification.
- Gradient Boosting.
- XGBoost.
- Neural Networks.
- Model evaluation.
- Model persistence.
- Streamlit.
- Git and GitHub.
- Machine learning project structure.
- End-to-end ML workflow.

---

## 🔐 Git and Data Management

Large datasets should generally not be committed directly to GitHub when they exceed repository limits.

The project therefore separates:

```text
Source Code
    ↓
GitHub

Large Raw Data
    ↓
Local / External Storage
```

The `.gitignore` file should prevent:

- Virtual environments.
- Large raw datasets.
- Generated files.
- Temporary files.
- Model artifacts when appropriate.

from being unnecessarily committed to the repository.

---

## 👨‍💻 Author

**Anchit Kaushal**

B.Tech Computer Science Engineering  
Artificial Intelligence & Machine Learning

GitHub:

https://github.com/anchitkaushal

---

## ⭐ Project Purpose

This project was developed as a practical Machine Learning project to demonstrate the complete workflow of transforming real-world data into predictive systems.

The main objective is not only to train machine learning models, but to demonstrate an end-to-end process:

```text
Problem Understanding
        ↓
Data Collection
        ↓
Data Validation
        ↓
Preprocessing
        ↓
EDA
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Model Development
        ↓
Model Evaluation
        ↓
Prediction
        ↓
Application
```

The project combines **data science, machine learning, time-series analysis, demand forecasting, fraud detection, and application development** into one integrated system.

---

## 📄 License

This project is intended for educational and portfolio purposes.

If an open-source license is added to the repository, this section should be updated accordingly.