## Demand Forecasting for Inventory Planning (Streamlit + Prophet)

This project is a small **Streamlit** app that demonstrates **demand forecasting for inventory planning**
using **[Prophet](https://facebook.github.io/prophet/)** as the core forecasting library.

It supports two workflows:

- **Fit + forecast (Prophet-driven)**: upload historical demand (`ds,y`), tune Prophet settings, generate a future forecast.
- **Visualize an existing forecast**: upload a Prophet-like forecast file (`ds,yhat,yhat_lower,yhat_upper`) and use it for inventory planning metrics.

## What the app does

- **Forecasting**:
  - Fits a Prophet model on daily demand (trend + seasonality) when you provide `ds,y`.
  - Produces a forecast for a chosen horizon and plots:
    - forecast line (`yhat`)
    - uncertainty interval (`yhat_lower` / `yhat_upper`)
  - Shows Prophet **components** (trend + seasonalities) only in the “fit + forecast” workflow.

- **Inventory planning (simple demo metrics)**:
  - Uses the forecast to estimate demand over a **lead time window**.
  - Converts forecast uncertainty into an **uncertainty proxy**.
  - Calculates:
    - **Expected demand during lead time**
    - **Safety stock** (based on a target cycle service level)
    - **Reorder point** = expected lead-time demand + safety stock

This is a teaching/demo app: the inventory math is intentionally simple so you can see the end-to-end pipeline.

## Quick start (Docker Compose)

From the folder containing `docker-compose.yml`:

```bash
docker compose up --build
```

Open:

- `http://localhost:8501`

To rebuild from scratch (recommended when changing dependencies):

```bash
docker compose build --no-cache
docker compose up
```

## Run locally (without Docker)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

## Data formats (CSV upload)

In the sidebar, uncheck **“Use example synthetic data”** and upload a CSV.

### Option A: History format (recommended, Prophet-driven)

Use this when you want the app to **fit Prophet** and generate a forecast.

Required columns:

- `ds`: date/time column (parseable by pandas, e.g. `YYYY-MM-DD`)
- `y`: actual demand (numeric)

Notes:

- One row per day is the intended format (Prophet can handle missing days, but daily is best here).
- Sort order does not matter; the app sorts by `ds`.

### Option B: Forecast format (already predicted)

Use this when you already have predictions (e.g. exported from Prophet `predict()` or another system).

Required columns:

- `ds`: date
- `yhat`: forecast point estimate
- `yhat_lower`: lower prediction bound
- `yhat_upper`: upper prediction bound

Notes:

- In this mode the app **does not fit Prophet**, so **Prophet settings + component plots are disabled**.
- Inventory metrics still work because they only need `yhat` + interval columns.

## Settings explained (what to use and when)

All settings are in the Streamlit sidebar.

### 1) Data

- **Use example synthetic data**
  - **What it does**: generates a synthetic daily demand series with trend + weekly/yearly seasonality.
  - **When to use**: quick sanity check that the app runs end-to-end.

- **Upload CSV**
  - **What it does**: lets you bring your own `ds,y` history (recommended) or `ds,yhat,yhat_lower,yhat_upper` forecast.
  - **When to use**: validating the model on your own demand data / presenting results for a real SKU.

### 2) Prophet settings (only when you upload `ds,y`)

- **Forecast horizon (days)**
  - **What it does**: how far into the future Prophet will predict.
  - **Typical choices**:
    - Replenishment planning: 30–90 days
    - Longer strategic planning: 90–365 days
  - **Rule of thumb**: don’t forecast far beyond the time scale where you can act (or where your data is stable).

- **Yearly seasonality**
  - **What it does**: allows recurring annual patterns (holidays, summer/winter effects).
  - **Use it when**:
    - you have **many months to years** of data (ideally 1–2 years or more)
    - you expect annual cycles
  - **Turn it off when**:
    - you only have a few weeks/months of history (yearly seasonality can overfit)

- **Weekly seasonality**
  - **What it does**: models day-of-week effects (Mon/Tue/…/Sun patterns).
  - **Use it when**:
    - demand differs across weekdays vs weekends (very common in retail/e-comm)
  - **Turn it off when**:
    - your demand is not tied to weekdays (rare), or your data isn’t daily

- **Daily seasonality**
  - **What it does**: intra-day patterns (hourly cycles). This app is mainly designed for daily data.
  - **Use it when**:
    - you have sub-daily timestamps (e.g. hourly) and meaningful daily cycles
  - **Recommendation here**: keep it **off** for daily demand CSVs.

- **Changepoint prior scale (trend flexibility)**
  - **What it does**: how easily Prophet adapts the trend to sudden changes.
  - **Interpretation**:
    - lower values → smoother trend, fewer sharp changes
    - higher values → more flexible trend, can follow sudden level shifts
  - **Typical values**:
    - stable demand: 0.01–0.05
    - changing demand / growth / promotions: 0.05–0.2
    - very volatile series: 0.2–0.5 (risk of overfitting)
  - **How to choose**:
    - if forecasts lag behind a real shift → increase it
    - if the trend looks “wiggly” and implausible → decrease it

### 3) Inventory settings (works for both CSV types)

These settings drive the demo inventory calculations.

- **Lead time (days)**
  - **What it does**: the number of days between placing an order and receiving stock.
  - **Typical choices**:
    - domestic supplier: 3–14 days
    - overseas / complex supply chain: 21–90 days
  - **How it affects results**: longer lead time → higher expected lead-time demand and typically higher safety stock.

- **Target cycle service level**
  - **What it does**: desired probability of not stocking out during lead time (cycle service level).
  - **Typical choices**:
    - low-value / easy-to-replenish: 0.85–0.90
    - mainstream: 0.95
    - critical / high-penalty stockouts: 0.98–0.99
  - **How it affects results**: higher service level → higher safety stock → higher reorder point.

### Forecast-upload-only setting

When you upload `ds,yhat,yhat_lower,yhat_upper`, the app shows:

- **Anchor date**
  - **What it does**: tells the app where “today / last observed demand date” is, so it can compute metrics over the next `lead_time_days`.
  - **What to set**:
    - set it to your last real observed date (end of history) that the forecast is meant to extend from.

## How to interpret the outputs

- **Forecast plot**
  - `yhat` is the expected demand.
  - `yhat_lower` / `yhat_upper` represent uncertainty (wider band = more uncertainty).

- **Inventory planning metrics**
  - **Expected demand during lead time**: sum of predicted demand over the next lead-time window.
  - **Uncertainty proxy**: derived from the prediction interval width; used as a rough stand-in for variability.
  - **Safety stock**: increases with service level and uncertainty.
  - **Reorder point**: the demand threshold where you’d trigger replenishment in a simple \(s\) policy.

## Limitations (important)

- The inventory computation is a **simplified demonstration**:
  - It uses a proxy derived from Prophet intervals, not a full probabilistic lead-time demand model.
  - It assumes lead time is fixed (not variable).
  - It does not model on-hand stock, order quantity, backorders, MOQ, capacity constraints, etc.

If you want, we can extend it to a more standard policy (e.g. \(s, Q\) or \(R, S\)) and/or incorporate lead-time variability.

## Troubleshooting

- **Docker shows old behavior**
  - Rebuild without cache:

```bash
docker compose build --no-cache
docker compose up
```

- **CSV upload error**
  - Ensure the file has either:
    - `ds,y`, or
    - `ds,yhat,yhat_lower,yhat_upper`
  - Ensure dates are parseable (try `YYYY-MM-DD`).
