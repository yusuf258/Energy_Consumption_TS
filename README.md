# Household Energy Consumption Forecasting | LSTM Time Series

Deep learning LSTM model for forecasting household power consumption using high-frequency time series data.

## Problem Statement
Forecast household **global active power** consumption at hourly resolution using historical measurements from a smart meter dataset spanning ~4 years.

## Dataset
| Attribute | Detail |
|---|---|
| File | `household_power_consumption.txt` |
| Records | ~2,075,259 minute-level measurements |
| Resampled | Hourly aggregation for model training |
| Features | Global_active_power, Global_reactive_power, Voltage, etc. |
| Target | `Global_active_power` (kW) |
| Period | December 2006 – November 2010 |

## Methodology
1. **Data Loading** — Large dataset ingestion with dtype handling and missing value treatment
2. **EDA & Visualization** — Time series plots, seasonal patterns, rolling mean/std
3. **Resampling** — Minute-level → hourly aggregation
4. **Scaling** — `MinMaxScaler` normalization
5. **Sequence Construction** — Sliding window approach for LSTM input
6. **LSTM Model** — 2-layer LSTM architecture with Dropout regularization
7. **Evaluation** — MSE, RMSE on test set; training vs validation loss curves

## Results
| Model | Metric | Value |
|---|---|---|
| **LSTM (2-layer)** | val_loss (MSE, normalized) | ~0.006 |

## Technologies
`Python` · `TensorFlow/Keras` · `Pandas` · `NumPy` · `Matplotlib` · `scikit-learn` · `joblib`

## File Structure
```
07_Energy_Consumption_TS/
├── project_notebook.ipynb              # Main notebook
├── household_power_consumption.txt     # Raw dataset
└── models/                             # Saved LSTM model
```

## How to Run
```bash
cd 07_Energy_Consumption_TS
jupyter notebook project_notebook.ipynb
```
