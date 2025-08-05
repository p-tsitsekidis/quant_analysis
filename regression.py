"""
Perform linear and multiple linear regression on cleaned power market data
to quantify the influence of Load, Wind, and Hour on electricity Price.
"""

import pandas as pd
import logging
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import seaborn as sns

from statsmodels.formula.api import ols

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# File paths
CLEANED_CSV = "data/power_data_clean.csv"
CLEANED_DB = "data/market_data.db"

# SQLite connection
conn = sqlite3.connect(CLEANED_DB)

logger.info("Loading CSV data...")
csv_file_df = pd.read_csv(CLEANED_CSV)

logger.info("Loading DB data...")
db_file_df = pd.read_sql("SELECT * FROM power WHERE Hour BETWEEN 8 AND 20", conn)
conn.close()

# Simple linear regression
linear_model = ols(formula="Price ~ Load", data=csv_file_df).fit()
logger.info(linear_model.summary())

# Multi-linear regression on a subset
multi_model = ols("Price ~ Load + Wind + Hour", data=db_file_df).fit()
logger.info(multi_model.summary())

# Price versus Load plot
plt.scatter(csv_file_df["Load"], csv_file_df["Price"], alpha=0.3, s=10)
plt.xlabel("Load (MW)")
plt.ylabel("Price (€/MWh)")
plt.title("Electricity Price vs Load")
plt.grid(True)
plt.show()

# Regression line plot (Price versus Load)
x = csv_file_df["Load"]
y = csv_file_df["Price"]
y_pred = linear_model.predict(csv_file_df)

plt.scatter(x, y, alpha=0.3, label="Observed", s=10)
plt.plot(x, y_pred, color="red", label="Fitted Line")
plt.xlabel("Load (MW)")
plt.ylabel("Price (€/MWh)")
plt.title("Linear Regression: Price ~ Load")
plt.legend()
plt.show()

# Residual Plot
residuals = linear_model.resid

plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residuals versus Fitted Values")
plt.grid(True)
plt.show()

# Histogram of Residuals
plt.hist(residuals, bins=50, edgecolor="black")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.show()

# Heatmap of Correlations
corr = db_file_df[["Price", "Load", "Wind", "Hour"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()