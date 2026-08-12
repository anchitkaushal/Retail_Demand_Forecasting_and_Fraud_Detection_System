import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.forecasting.predict import forecast_future
from src.fraud.predict import predict_fraud


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Retail AI System",
    page_icon="📊",
    layout="wide"
)


# MODEL CACHING

@st.cache_resource
def load_models():
    """
    Cache model and feature paths.
    """
    forecasting_model_path = Path("models/ridge_forecasting_model.pkl")
    forecasting_features_path = Path("models/forecasting_features.pkl")
    fraud_model_path = Path("models/fraud_ann.pkl")
    fraud_features_path = Path("models/fraud_features.pkl")

    return (
        forecasting_model_path,
        forecasting_features_path,
        fraud_model_path,
        fraud_features_path
    )


# SIDEBAR & NAVIGATION

st.sidebar.title("🤖 Retail AI System")
page = st.sidebar.radio(
    "Navigation",
    ["Demand Forecasting", "Fraud Detection"]
)

st.title("Retail Demand Forecasting & Fraud Detection System")


# MODULE 1: DEMAND FORECASTING

if page == "Demand Forecasting":
    st.subheader("📈 Retail Demand Forecasting")
    st.markdown(
        "Upload historical daily sales data to generate future demand predictions using trained inference pipelines."
    )

    uploaded_file = st.file_uploader(
        "Upload Historical Sales CSV (Required columns: date, sales)",
        type=["csv"],
        key="forecast_csv_uploader"
    )

    if "use_sample_forecast" not in st.session_state:
        st.session_state["use_sample_forecast"] = False

    if uploaded_file is not None:
        st.session_state["use_sample_forecast"] = False

    if uploaded_file is None and not st.session_state["use_sample_forecast"]:
        sample_path = Path("data/interim/daily_total_sales.csv")
        if sample_path.exists():
            st.info("💡 No CSV uploaded. Click below to load sample daily sales dataset for testing.")
            if st.button("Load Sample Sales Data"):
                st.session_state["use_sample_forecast"] = True
                st.session_state["forecast_df"] = None
                st.rerun()

    use_sample = st.session_state["use_sample_forecast"] and (uploaded_file is None)

    if use_sample:
        col_s1, col_s2 = st.columns([4, 1])
        with col_s1:
            st.success("✅ Sample daily sales dataset loaded (`data/interim/daily_total_sales.csv`).")
        with col_s2:
            if st.button("Unload Sample Data", key="unload_forecast_sample"):
                st.session_state["use_sample_forecast"] = False
                st.session_state["forecast_df"] = None
                st.rerun()

    if uploaded_file is not None or use_sample:
        try:
            if uploaded_file is not None:
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_csv("data/interim/daily_total_sales.csv")

            if df_raw.empty:
                st.error("Uploaded CSV file is empty.")
                st.stop()

            # Column validation
            required_cols = ["date", "sales"]
            missing_cols = [c for c in required_cols if c not in df_raw.columns]

            if missing_cols:
                st.error(
                    f"Uploaded dataset is missing required column(s): {', '.join(missing_cols)}"
                )
                st.info("The forecasting dataset must contain at least `date` and `sales` columns.")
                st.stop()

            # Clean and prepare historical dataframe
            df_clean = df_raw.copy()
            df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
            df_clean["sales"] = pd.to_numeric(df_clean["sales"], errors="coerce")
            df_clean = df_clean.dropna(subset=["date", "sales"]).sort_values("date").reset_index(drop=True)

            if df_clean.empty:
                st.error("No valid historical data rows were found after parsing dates and sales.")
                st.stop()

            # Dataset Overview
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Historical Rows", f"{len(df_clean):,}")
                c2.metric("Total Columns", len(df_clean.columns))
                c3.metric("Latest Available Date", str(df_clean["date"].max().date()))

                with st.expander("Preview Historical Dataset", expanded=False):
                    st.dataframe(df_clean.head(10))

            # Horizon Selector
            st.subheader("Forecast Settings")
            horizon = st.slider(
                "Select Forecast Horizon (Days)",
                min_value=1,
                max_value=90,
                value=30,
                step=1
            )

            # Generate Forecast Button
            if st.button("Generate Forecast", type="primary"):
                f_model_path, f_feature_path, _, _ = load_models()

                if not f_model_path.exists() or not f_feature_path.exists():
                    st.error("Saved forecasting model/feature files not found in models/ directory. Please run 'python main.py' to generate model artifacts.")
                    st.stop()

                try:
                    with st.spinner(f"Generating recursive {horizon}-day demand forecast..."):
                        forecast_df = forecast_future(
                            df=df_clean,
                            model_path=f_model_path,
                            feature_path=f_feature_path,
                            horizon=horizon
                        )
                    st.session_state["forecast_df"] = forecast_df
                    st.session_state["forecast_horizon"] = horizon
                except Exception as e:
                    st.error(f"Failed to generate forecast: {str(e)}")

            if st.session_state.get("forecast_df") is not None:
                forecast_df = st.session_state["forecast_df"]
                f_horizon = st.session_state.get("forecast_horizon", horizon)

                st.success(f"Forecast generated successfully for next {len(forecast_df)} days!")

                # Summary Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Forecast Horizon", f"{len(forecast_df)} Days")
                m2.metric("Total Projected Sales", f"{forecast_df['forecast'].sum():,.0f}")
                m3.metric("Average Daily Forecast", f"{forecast_df['forecast'].mean():,.0f}")
                m4.metric("Peak Daily Forecast", f"{forecast_df['forecast'].max():,.0f}")

                # Chart: Historical + Future Forecast
                st.subheader("Demand Forecast Visualization")
                recent_hist = df_clean.tail(180)

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(
                    recent_hist["date"],
                    recent_hist["sales"],
                    label="Historical Sales",
                    color="#1f77b4",
                    linewidth=1.5
                )
                ax.plot(
                    forecast_df["date"],
                    forecast_df["forecast"],
                    label="Future Forecast",
                    color="#ff7f0e",
                    linestyle="--",
                    linewidth=2.0
                )
                ax.axvline(
                    x=recent_hist["date"].max(),
                    color="gray",
                    linestyle=":",
                    alpha=0.8,
                    label="Forecast Start"
                )
                ax.set_title("Retail Sales Demand - Historical vs Future Forecast", fontsize=12)
                ax.set_xlabel("Date")
                ax.set_ylabel("Sales")
                ax.legend(loc="upper left")
                ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

                # Forecast Results Table
                st.subheader("Forecast Data Table")
                st.dataframe(forecast_df)

                # CSV Download
                csv_data = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast CSV",
                    data=csv_data,
                    file_name=f"demand_forecast_{f_horizon}d.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading or parsing uploaded file: {str(e)}")



# MODULE 2: FRAUD DETECTION

elif page == "Fraud Detection":
    st.subheader("🚨 Transaction Fraud Detection")
    st.markdown(
        "Upload retail transaction records to evaluate fraud risk using the trained ANN classifier."
    )

    uploaded_file = st.file_uploader(
        "Upload Transaction CSV",
        type=["csv"],
        key="fraud_csv_uploader"
    )

    if "use_sample_fraud" not in st.session_state:
        st.session_state["use_sample_fraud"] = False

    if uploaded_file is not None:
        st.session_state["use_sample_fraud"] = False

    if uploaded_file is None and not st.session_state["use_sample_fraud"]:
        sample_path = Path("data/simulated/fraud_transactions.csv")
        if sample_path.exists():
            st.info("💡 No CSV uploaded. Click below to load simulated transaction dataset for testing.")
            if st.button("Load Sample Fraud Data"):
                st.session_state["use_sample_fraud"] = True
                st.session_state["fraud_results_df"] = None
                st.rerun()

    use_sample = st.session_state["use_sample_fraud"] and (uploaded_file is None)

    if use_sample:
        col_s1, col_s2 = st.columns([4, 1])
        with col_s1:
            st.success("✅ Sample transaction dataset loaded (`data/simulated/fraud_transactions.csv`).")
        with col_s2:
            if st.button("Unload Sample Data", key="unload_fraud_sample"):
                st.session_state["use_sample_fraud"] = False
                st.session_state["fraud_results_df"] = None
                st.rerun()

    if uploaded_file is not None or use_sample:
        try:
            if uploaded_file is not None:
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_csv("data/simulated/fraud_transactions.csv")

            if df_raw.empty:
                st.error("Uploaded transaction file is empty.")
                st.stop()

            _, _, fraud_model_path, fraud_feature_path = load_models()

            if not fraud_model_path.exists() or not fraud_feature_path.exists():
                st.error("Saved fraud ANN model/feature files not found in models/ directory. Please run 'python main.py' to train models first.")
                st.stop()

            feature_names = joblib.load(fraud_feature_path)

            # Feature validation
            missing_features = [f for f in feature_names if f not in df_raw.columns]
            if missing_features:
                st.error(
                    f"Uploaded dataset is missing required fraud feature(s): {', '.join(missing_features)}"
                )
                st.info(f"Required transaction features: {', '.join(feature_names)}")
                st.stop()

            with st.container(border=True):
                st.write(f"Dataset preview ({len(df_raw):,} transactions):")
                st.dataframe(df_raw.head(10))

            if st.button("Run Fraud Detection", type="primary"):
                try:
                    with st.spinner("Analyzing transaction risk with Neural Network..."):
                        results_df = predict_fraud(
                            df=df_raw,
                            model_path=fraud_model_path,
                            feature_path=fraud_feature_path
                        )
                    st.session_state["fraud_results_df"] = results_df
                except Exception as e:
                    st.error(f"Error executing fraud prediction: {str(e)}")

            if st.session_state.get("fraud_results_df") is not None:
                results_df = st.session_state["fraud_results_df"]
                st.success("Fraud evaluation complete!")

                # Summary Metrics
                total_tx = len(results_df)
                fraud_detected = int((results_df["fraud_prediction"] == 1).sum())
                fraud_pct = (fraud_detected / total_tx * 100) if total_tx > 0 else 0.0
                high_risk_cnt = int((results_df["risk_level"] == "HIGH").sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Transactions", f"{total_tx:,}")
                c2.metric("Fraud Detected", f"{fraud_detected:,}")
                c3.metric("Fraud Percentage", f"{fraud_pct:.2f}%")
                c4.metric("High-Risk Transactions", f"{high_risk_cnt:,}")

                # Tabular Output
                tab1, tab2 = st.tabs(["All Predictions", "High-Risk Transactions Only"])

                with tab1:
                    st.subheader("Complete Prediction Results")
                    st.dataframe(results_df)

                with tab2:
                    st.subheader("High-Risk Transactions (Probability ≥ 0.75)")
                    high_risk_df = results_df[results_df["risk_level"] == "HIGH"]
                    if high_risk_df.empty:
                        st.info("No high-risk transactions detected.")
                    else:
                        st.dataframe(high_risk_df)

                # Visualizations
                st.subheader("Fraud Risk Visualizations")
                col_vis1, col_vis2 = st.columns(2)

                with col_vis1:
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.hist(
                        results_df["fraud_probability"],
                        bins=20,
                        color="#d62728",
                        edgecolor="black",
                        alpha=0.7
                    )
                    ax.set_title("Fraud Probability Distribution", fontsize=10)
                    ax.set_xlabel("Probability")
                    ax.set_ylabel("Transaction Count")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    fig.tight_layout()
                    st.pyplot(fig)

                with col_vis2:
                    risk_counts = results_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
                    ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="black", alpha=0.8)
                    ax.set_title("Risk Level Classification", fontsize=10)
                    ax.set_xlabel("Risk Level")
                    ax.set_ylabel("Transaction Count")
                    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
                    fig.tight_layout()
                    st.pyplot(fig)

                # Download CSV
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Fraud Results CSV",
                    data=csv_data,
                    file_name="fraud_detection_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing uploaded transaction file: {str(e)}")