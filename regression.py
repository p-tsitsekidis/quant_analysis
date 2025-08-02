"""
Perform simple linear regression on power market data to analyze
the relationship between electricity Load and Price.
"""

import pandas as pd
import logging
import matplotlib
import sqlite3

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
db_file_df = pd.read_sql("SELECT * FROM power", conn)
conn.close()

# Simple linear regression
linear_model = ols(formula="Price ~ Load", data=csv_file_df).fit()

logger.info(linear_model.summary())