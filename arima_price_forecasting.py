"""
Use ARIMA model to forecast future electricity prices based on historical hourly market data.
"""

import pandas as pd
import logging

from statsmodels.tsa.arima.model import ARIMA

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