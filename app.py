import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from datetime import timedelta
from typing import Literal, Tuple, Optional


st.set_page_config(
    page_title="Demand Forecasting for Inventory Planning",
    layout="wide",
)


@st.cache_data
def generate_example_data(n_days: int = 365 * 2) -> pd.DataFrame:
    """Create a simple synthetic daily demand series with trend + seasonality."""
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    df = pd.DataFrame({"ds": dates})

    # Trend
    df["trend"] = 50 + 0.02 * (df.index.values)

    # Weekly seasonality (higher demand Mon-Fri, lower on weekends)
    df["dow"] = df["ds"].dt.dayofweek
    df["weekly"] = df["dow"].map(
        {0: 10, 1: 12, 2: 14, 3: 16, 4: 18, 5: 8, 6: 6}
    )

    # Yearly seasonality (simple sine wave)
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["yearly"] = 8 * np.sin(2 * np.pi * df["day_of_year"] / 365.25)

    # Random noise
    rng = np.random.default_rng(42)
    df["noise"] = rng.normal(0, 5, size=len(df))

    df["y"] = (df["trend"] + df["weekly"] + df["yearly"] + df["noise"]).clip(lower=0)

    return df[["ds", "y"]]


UploadedKind = Literal["history", "forecast"]


def load_uploaded_csv(uploaded_file) -> Tuple[UploadedKind, pd.DataFrame]:
    df = pd.read_csv(uploaded_file)
    if "ds" not in df.columns:
        raise ValueError("CSV must contain a 'ds' (date) column.")

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds")

    if "y" in df.columns:
        return "history", df[["ds", "y"]]

    forecast_cols = {"yhat", "yhat_lower", "yhat_upper"}
    if forecast_cols.issubset(set(df.columns)):
        return "forecast", df[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    raise ValueError(
        "CSV must be either:\n"
        "- history format: columns 'ds' and 'y'\n"
        "- forecast format: columns 'ds', 'yhat', 'yhat_lower', 'yhat_upper'\n"
        f"Detected columns: {list(df.columns)}"
    )


def fit_prophet_model(
    df: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
        stan_backend="CMDSTANPY",
    )
    m.fit(df)
    return m


def compute_inventory_metrics(
    forecast: pd.DataFrame,
    anchor_ds: pd.Timestamp,
    service_level: float,
    lead_time_days: int,
) -> pd.DataFrame:
    """
    Simple inventory metrics:
    - expected demand during lead time
    - standard deviation proxy from Prophet uncertainty (yhat_upper - yhat_lower)
    - safety stock & reorder point.
    """
    # Use the forecast horizon equal to lead_time_days for metrics
    lead_time_end = anchor_ds + timedelta(days=lead_time_days)
    horizon = forecast[(forecast["ds"] > anchor_ds) & (forecast["ds"] <= lead_time_end)]

    # Approximate mean demand during lead time
    expected_demand_lt = horizon["yhat"].sum()

    # Approximate demand uncertainty during lead time from Prophet intervals
    std_proxy = ((horizon["yhat_upper"] - horizon["yhat_lower"]) / 4).sum()

    # Z-values for common service levels (normal approx)
    z_table = {
        0.80: 0.84,
        0.85: 1.04,
        0.90: 1.28,
        0.95: 1.64,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }
    z = z_table.get(service_level, 1.64)

    safety_stock = z * std_proxy
    reorder_point = expected_demand_lt + safety_stock

    metrics = pd.DataFrame(
        {
            "metric": [
                "Expected demand during lead time",
                "Uncertainty proxy (sum of std)",
                "Safety stock",
                "Reorder point",
            ],
            "value": [
                expected_demand_lt,
                std_proxy,
                safety_stock,
                reorder_point,
            ],
        }
    )
    return metrics


def plot_forecast_only(forecast: pd.DataFrame) -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_upper"],
            line=dict(width=0),
            name="Upper",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat_lower"],
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.2)",
            line=dict(width=0),
            name="Uncertainty interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="Forecast (yhat)",
            line=dict(color="rgba(31, 119, 180, 1)"),
        )
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Demand",
    )
    return fig


def main():
    st.title("Demand Forecasting for Inventory Planning")
    st.markdown(
        """
This app demonstrates **time series forecasting for inventory planning** using the
**Prophet** library from Meta's Core Data Science team.

Prophet is an additive model that captures **trend**, **seasonality** and **holiday effects**
and is well-suited for business time series that are:

- **Daily or sub-daily**
- **Strongly seasonal**
- **Possibly messy** (missing values, outliers, level shifts)

Learn more in the [Prophet documentation](https://facebook.github.io/prophet/).
"""
    )

    st.sidebar.header("1. Data")
    use_example = st.sidebar.checkbox("Use example synthetic data", value=True)
    uploaded_file = st.sidebar.file_uploader("Or upload your own CSV", type=["csv"])

    if use_example and uploaded_file is not None:
        st.sidebar.warning("Both example data and upload selected. Using uploaded data.")
        use_example = False

    history_df: Optional[pd.DataFrame] = None
    uploaded_forecast_df: Optional[pd.DataFrame] = None

    if uploaded_file is not None and not use_example:
        try:
            kind, loaded = load_uploaded_csv(uploaded_file)
        except Exception as e:  # pylint: disable=broad-except
            st.error(f"Failed to load data: {e}")
            return

        if kind == "history":
            history_df = loaded
        else:
            uploaded_forecast_df = loaded
    else:
        history_df = generate_example_data()

    with st.expander("Show raw data"):
        if history_df is not None:
            st.dataframe(history_df, use_container_width=True, height=320)
            st.write(f"Number of observations: **{len(history_df)}**")
        else:
            st.dataframe(uploaded_forecast_df, use_container_width=True, height=320)
            st.write(f"Number of observations: **{len(uploaded_forecast_df)}**")

    st.sidebar.header("2. Prophet settings")
    if history_df is None:
        st.sidebar.info(
            "You uploaded a **forecast-format** CSV (ds, yhat, yhat_lower, yhat_upper). "
            "Prophet settings are disabled because no model fitting is needed."
        )
        horizon_days = 0
        yearly = weekly = daily = False
        cps = 0.05
    else:
        horizon_days = st.sidebar.slider("Forecast horizon (days)", 7, 365, 90, step=7)
        yearly = st.sidebar.checkbox("Yearly seasonality", value=True)
        weekly = st.sidebar.checkbox("Weekly seasonality", value=True)
        daily = st.sidebar.checkbox("Daily seasonality", value=False)
        cps = st.sidebar.slider(
            "Changepoint prior scale (trend flexibility)",
            0.01,
            0.5,
            0.05,
            step=0.01,
        )

    st.sidebar.header("3. Inventory settings")
    lead_time_days = st.sidebar.slider("Lead time (days)", 1, 90, 14)
    service_level = st.sidebar.select_slider(
        "Target cycle service level",
        options=[0.80, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99],
        value=0.95,
    )

    if st.button("Run forecast", type="primary"):
        if history_df is not None:
            with st.spinner("Fitting Prophet model..."):
                model = fit_prophet_model(
                    history_df,
                    yearly_seasonality=yearly,
                    weekly_seasonality=weekly,
                    daily_seasonality=daily,
                    changepoint_prior_scale=cps,
                )

                future = model.make_future_dataframe(periods=horizon_days, freq="D")
                forecast = model.predict(future)

            st.subheader("Forecast")
            fig = plot_plotly(model, forecast)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Inventory planning metrics")
            anchor_ds = history_df["ds"].max()
            metrics = compute_inventory_metrics(
                forecast=forecast,
                anchor_ds=anchor_ds,
                service_level=service_level,
                lead_time_days=lead_time_days,
            )
            st.table(metrics.style.format({"value": "{:,.1f}"}))

            st.subheader("Forecast components")
            components_fig = model.plot_components(forecast)
            st.pyplot(components_fig)
        else:
            forecast = uploaded_forecast_df
            st.subheader("Forecast (uploaded)")
            fig = plot_forecast_only(forecast)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Inventory planning metrics")
            anchor_date = st.date_input(
                "Anchor date (typically your last observed demand date)",
                value=forecast["ds"].min().date(),
                min_value=forecast["ds"].min().date(),
                max_value=forecast["ds"].max().date(),
            )
            anchor_ds = pd.Timestamp(anchor_date)
            metrics = compute_inventory_metrics(
                forecast=forecast,
                anchor_ds=anchor_ds,
                service_level=service_level,
                lead_time_days=lead_time_days,
            )
            st.table(metrics.style.format({"value": "{:,.1f}"}))

            st.info(
                "Component plots are only available when fitting a Prophet model "
                "(i.e., when you upload history-format data: ds,y)."
            )


if __name__ == "__main__":
    main()

