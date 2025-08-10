"""
Use ARIMA model to forecast future electricity prices based on historical hourly market data.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from itertools import product

def aicc(res):
    k, n = res.params.size, res.nobs
    return res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Load data
df = pd.read_csv("data/power_data_clean.csv")

# Set and change index to datetime type
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df.set_index("utc_timestamp", inplace=True)

# Set hourly frequency
df = df.asfreq("h")

df.sort_index(inplace=True)

price = df["Price"]

# Fit basic ARIMA model
model = ARIMA(price, order=(1, 1, 1))
fitted_model = model.fit()

# Summary
logger.info(fitted_model.summary())

# Forecasting
forecast = fitted_model.forecast(steps=24)
logger.info(forecast)

# Values before forecast
history = price[-100:]

# Index starts where history ends
forecast_index = pd.date_range(start=history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(history.index, history, label="Historical Price")
plt.plot(forecast_index, forecast, label="Forecast (Next 24 Hours)", color="red")
plt.xlabel("Timestamp")
plt.ylabel("Electricity Price (€/MWh)")
plt.title("ARIMA Forecast of Electricity Prices")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Find better parameters for ARIMA
best = None
for d in (0, 1):
    for p, q in product(range(0, 4), range(0, 4)):
        try:
            res = ARIMA(price, order=(p, d, q)).fit()
            score = aicc(res)
            if (best is None) or (score < best[0]):
                best = (score, (p, d, q), res)
        except Exception:
            pass

best_aicc, best_order, fitted_model = best
logger.info(f"Selected ARIMA order={best_order} | AICc={best_aicc}")

forecast = fitted_model.get_forecast(steps=24)
mean = forecast.predicted_mean
ci = forecast.conf_int()
ci_low, ci_high = ci.iloc[:, 0], ci.iloc[:, 1]

history = price.iloc[-100:]

plt.figure(figsize=(12, 6))
plt.plot(history.index, history.values, label="Historical (last 100h)", linewidth=1.5)
plt.axvline(history.index[-1], linestyle="--", alpha=0.7, label="Forecast start")
plt.plot(mean.index, mean.values, label=f"ARIMA{best_order} forecast (24h)", linewidth=2)
plt.fill_between(mean.index, ci_low.values, ci_high.values, alpha=0.2, label="95% PI")
plt.xlabel("Timestamp")
plt.ylabel("Electricity Price (€/MWh)")
plt.title("ARIMA Forecast of Electricity Prices")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()