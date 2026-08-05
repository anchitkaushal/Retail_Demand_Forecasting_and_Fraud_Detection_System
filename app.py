# app.py

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Demand Forecasting & Fraud Detection",
    page_icon="🛒",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

PROCESSED_DIR = DATA_DIR / "processed"
SIMULATED_DIR = DATA_DIR / "simulated"

TABLE_DIR = REPORT_DIR / "tables"
PREDICTION_DIR = REPORT_DIR / "predictions"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_forecasting_data():

    path = PROCESSED_DIR / "forecasting_features.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_forecast_predictions():

    path = PREDICTION_DIR / "forecast_predictions.csv"

    if not path.exists():

        # Backward compatibility with your old location
        path = REPORT_DIR / "forecast_predictions.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_forecasting_results():

    path = TABLE_DIR / "forecasting_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_fraud_data():

    path = SIMULATED_DIR / "fraud_transactions.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_fraud_predictions():

    path = PREDICTION_DIR / "fraud_predictions.csv"

    if not path.exists():

        # Backward compatibility
        path = REPORT_DIR / "fraud_predictions.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def load_fraud_results():

    path = TABLE_DIR / "fraud_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


def format_number(value):

    if value is None:
        return "N/A"

    if isinstance(value, (int, float, np.integer, np.floating)):

        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"{value:,.0f}"

        return f"{value:.2f}"

    return str(value)


# ============================================================
# LOAD DATA
# ============================================================

forecast_data = load_forecasting_data()
forecast_predictions = load_forecast_predictions()
forecast_results = load_forecasting_results()

fraud_data = load_fraud_data()
fraud_predictions = load_fraud_predictions()
fraud_results = load_fraud_results()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🛒 Retail Analytics")

st.sidebar.markdown(
    """
    ### Retail Demand Forecasting
    & Fraud Detection System

    Use the dashboard to explore:

    - Demand trends
    - Forecasting performance
    - Model predictions
    - Fraud transactions
    - Fraud detection performance
    """
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📈 Demand Forecasting",
        "🚨 Fraud Detection",
        "🔍 Data Exploration"
    ]
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛒 Retail Demand Forecasting & Fraud Detection System"
)

st.caption(
    "Machine Learning system for retail demand forecasting "
    "and return-fraud anomaly detection"
)

st.markdown("---")


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header("System Overview")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    if forecast_data is not None:

        total_sales = forecast_data["sales"].sum()

        average_sales = forecast_data["sales"].mean()

        max_sales = forecast_data["sales"].max()

        total_days = forecast_data["date"].nunique()

    else:

        total_sales = 0
        average_sales = 0
        max_sales = 0
        total_days = 0

    if fraud_data is not None:

        total_transactions = len(fraud_data)

        fraud_count = fraud_data["is_fraud"].sum()

        fraud_rate = (
            fraud_count /
            total_transactions *
            100
        )

    else:

        total_transactions = 0
        fraud_count = 0
        fraud_rate = 0

    col1.metric(
        "Total Sales",
        format_number(total_sales)
    )

    col2.metric(
        "Average Daily Demand",
        format_number(average_sales)
    )

    col3.metric(
        "Maximum Daily Demand",
        format_number(max_sales)
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Sales trend
    # --------------------------------------------------------

    if forecast_data is not None:

        st.subheader("📈 Retail Demand Trend")

        daily_sales = (
            forecast_data
            .groupby("date", as_index=False)["sales"]
            .sum()
        )

        fig = px.line(
            daily_sales,
            x="date",
            y="sales",
            title="Daily Retail Demand"
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sales"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # System components
    # --------------------------------------------------------

    st.subheader("System Components")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            """
            ### 📊 EDA & TSA

            - Demand trends
            - Weekly seasonality
            - ACF analysis
            - Event analysis
            - Price analysis
            """
        )

    with c2:

        st.markdown(
            """
            ### 📈 Forecasting

            Models:

            - Ridge Regression
            - Gradient Boosting
            - XGBoost

            Time-aware train/test split.
            """
        )

    with c3:

        st.markdown(
            """
            ### 🚨 Fraud Detection

            Synthetic return transactions with:

            - ANN classifier
            - Fraud probability
            - Precision
            - Recall
            - F1-score
            - ROC-AUC
            """
        )


# ============================================================
# DEMAND FORECASTING
# ============================================================

elif page == "📈 Demand Forecasting":

    st.header("📈 Demand Forecasting")

    if forecast_data is None:

        st.error(
            "forecasting_features.csv was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # Forecasting metrics
    # --------------------------------------------------------

    if forecast_results is not None:

        st.subheader("Model Performance")

        display_results = forecast_results.copy()

        # Convert values for display only
        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if "RMSE" in forecast_results.columns:

            best_model = (
                forecast_results
                .sort_values("RMSE")
                .iloc[0]
            )

            st.success(
                f"Best model based on RMSE: "
                f"**{best_model['Model']}**"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Best Model",
                str(best_model["Model"])
            )

            c2.metric(
                "MAE",
                f"{best_model['MAE']:.2f}"
            )

            c3.metric(
                "RMSE",
                f"{best_model['RMSE']:.2f}"
            )

            if "WMAPE" in best_model:

                c4.metric(
                    "WMAPE",
                    f"{best_model['WMAPE']:.2f}%"
                )

    st.markdown("---")

    # --------------------------------------------------------
    # Actual vs predicted
    # --------------------------------------------------------

    st.subheader("Actual vs Forecast")

    if forecast_predictions is not None:

        prediction_df = forecast_predictions.copy()

        # Try to detect actual/prediction columns
        actual_col = None
        predicted_col = None

        for col in prediction_df.columns:

            col_lower = col.lower()

            if (
                "actual" in col_lower
                or col_lower == "sales"
            ):
                actual_col = col

            if (
                "prediction" in col_lower
                or "predicted" in col_lower
                or col_lower == "forecast"
            ):
                predicted_col = col

        if (
            actual_col is not None
            and predicted_col is not None
        ):

            if "date" in prediction_df.columns:

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=prediction_df["date"],
                        y=prediction_df[actual_col],
                        mode="lines",
                        name="Actual"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=prediction_df["date"],
                        y=prediction_df[predicted_col],
                        mode="lines",
                        name="Forecast"
                    )
                )

                fig.update_layout(
                    title="Actual vs Forecasted Demand",
                    xaxis_title="Date",
                    yaxis_title="Sales"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.warning(
                    "Prediction file does not contain a date column."
                )

        else:

            st.warning(
                "Could not automatically identify actual "
                "and prediction columns."
            )

        with st.expander("View Forecast Predictions"):

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

    else:

        st.warning(
            "Forecast prediction file not found."
        )

    # --------------------------------------------------------
    # Demand trend
    # --------------------------------------------------------

    st.subheader("Historical Demand")

    daily_sales = (
        forecast_data
        .groupby("date", as_index=False)["sales"]
        .sum()
    )

    fig = px.line(
        daily_sales,
        x="date",
        y="sales",
        title="Historical Daily Demand"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Weekly seasonality
    # --------------------------------------------------------

    if "weekday" in forecast_data.columns:

        st.subheader("Weekly Seasonality")

        weekday_sales = (
            forecast_data
            .groupby("weekday", as_index=False)["sales"]
            .mean()
        )

        fig = px.bar(
            weekday_sales,
            x="weekday",
            y="sales",
            title="Average Demand by Weekday"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# FRAUD DETECTION
# ============================================================

elif page == "🚨 Fraud Detection":

    st.header("🚨 Fraud Detection")

    if fraud_data is None:

        st.error(
            "fraud_transactions.csv was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # Fraud statistics
    # --------------------------------------------------------

    total_transactions = len(fraud_data)

    fraud_count = int(
        fraud_data["is_fraud"].sum()
    )

    normal_count = (
        total_transactions -
        fraud_count
    )

    fraud_rate = (
        fraud_count /
        total_transactions *
        100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Normal Transactions",
        f"{normal_count:,}"
    )

    col3.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

    col4.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%"
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    st.subheader("Fraud Class Distribution")

    class_counts = (
        fraud_data["is_fraud"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "is_fraud",
        "count"
    ]

    class_counts["label"] = (
        class_counts["is_fraud"]
        .map({
            0: "Normal",
            1: "Fraud"
        })
    )

    fig = px.pie(
        class_counts,
        names="label",
        values="count",
        title="Normal vs Fraud Transactions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Fraud model results
    # --------------------------------------------------------

    if fraud_results is not None:

        st.subheader("ANN Model Performance")

        st.dataframe(
            fraud_results,
            use_container_width=True,
            hide_index=True
        )

        result = fraud_results.iloc[0]

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Accuracy",
            f"{result['Accuracy']:.3f}"
        )

        c2.metric(
            "Precision",
            f"{result['Precision']:.3f}"
        )

        c3.metric(
            "Recall",
            f"{result['Recall']:.3f}"
        )

        c4.metric(
            "F1 Score",
            f"{result['F1']:.3f}"
        )

        c5.metric(
            "ROC-AUC",
            f"{result['ROC_AUC']:.3f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # Fraud probability
    # --------------------------------------------------------

    if fraud_predictions is not None:

        st.subheader("Fraud Predictions")

        prediction_df = fraud_predictions.copy()

        if "fraud_probability" in prediction_df.columns:

            fig = px.histogram(
                prediction_df,
                x="fraud_probability",
                nbins=30,
                title="Fraud Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Fraud transactions only
        # ----------------------------------------------------

        if "fraud_prediction" in prediction_df.columns:

            fraud_only = prediction_df[
                prediction_df["fraud_prediction"] == 1
            ]

            st.subheader(
                f"🚨 Flagged Transactions ({len(fraud_only)})"
            )

            st.dataframe(
                fraud_only,
                use_container_width=True,
                hide_index=True
            )

        with st.expander(
            "View All Transaction Predictions"
        ):

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.warning(
            "Fraud prediction file was not found."
        )


# ============================================================
# DATA EXPLORATION
# ============================================================

elif page == "🔍 Data Exploration":

    st.header("🔍 Data Exploration")

    if forecast_data is None:

        st.error(
            "Forecasting dataset was not found."
        )

        st.stop()

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    st.subheader("Forecasting Dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{forecast_data.shape[0]:,}"
    )

    c2.metric(
        "Columns",
        f"{forecast_data.shape[1]:,}"
    )

    c3.metric(
        "Missing Values",
        f"{forecast_data.isna().sum().sum():,}"
    )

    # --------------------------------------------------------
    # Date filter
    # --------------------------------------------------------

    st.subheader("Date Filter")

    min_date = forecast_data["date"].min().date()
    max_date = forecast_data["date"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(selected_dates) == 2:

        start_date = pd.Timestamp(
            selected_dates[0]
        )

        end_date = pd.Timestamp(
            selected_dates[1]
        )

        filtered_data = forecast_data[
            (
                forecast_data["date"] >= start_date
            )
            &
            (
                forecast_data["date"] <= end_date
            )
        ]

    else:

        filtered_data = forecast_data

    # --------------------------------------------------------
    # Sales trend
    # --------------------------------------------------------

    st.subheader("Demand During Selected Period")

    daily = (
        filtered_data
        .groupby("date", as_index=False)["sales"]
        .sum()
    )

    fig = px.line(
        daily,
        x="date",
        y="sales",
        title="Demand Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Event analysis
    # --------------------------------------------------------

    if "is_event" in filtered_data.columns:

        st.subheader("Event vs Non-Event Demand")

        event_data = (
            filtered_data
            .groupby("is_event")["sales"]
            .mean()
            .reset_index()
        )

        event_data["type"] = (
            event_data["is_event"]
            .map({
                0: "Non-Event",
                1: "Event"
            })
        )

        fig = px.bar(
            event_data,
            x="type",
            y="sales",
            title="Average Demand: Event vs Non-Event"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    with st.expander("View Dataset"):

        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Retail Demand Forecasting + Fraud Detection System | "
    "Machine Learning Project"
)