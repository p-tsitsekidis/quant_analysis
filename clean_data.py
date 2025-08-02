"""
Clean raw power market data and save it to CSV and SQLite.
"""

import pandas as pd
import sqlite3
import logging
import os

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# File paths
RAW_FILE = "data/power_data_raw.csv"
CLEAN_FILE = "data/power_data_clean.csv"
SQLITE_FILE = "data/market_data.db"
TABLE_NAME = "power"

def clean_data():
    logger.info(f"Loading raw data from {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, index_col=0, parse_dates=True)
    
    logger.info("Selecting relevant German market columns")
    df = df[[
        "DE_LU_price_day_ahead",
        "DE_load_actual_entsoe_transparency",
        "DE_wind_onshore_generation_actual"
    ]]
    
    logger.info("Renaming columns")
    df.rename(columns={
        "DE_LU_price_day_ahead": "Price",
        "DE_load_actual_entsoe_transparency": "Load",
        "DE_wind_onshore_generation_actual": "Wind"
    }, inplace=True)
    
    logger.info("Dropping rows with missing values")
    df.dropna(inplace=True)
    
    logger.info("Adding Hour column from timestamp index")
    df["Hour"] = df.index.hour
    
    logger.info(f"Saving cleaned data to CSV: {CLEAN_FILE}")
    df.to_csv(CLEAN_FILE)
    
    logger.info(f"Saving cleaned data to SQLite DB: {SQLITE_FILE}")
    with sqlite3.connect(SQLITE_FILE) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index_label="Timestamp")
    
    logger.info("Cleaned data successfully and stored the file in both CSV and SQLite")

if __name__ == "__main__":
    if os.path.exists(RAW_FILE):
        clean_data()
    else:
        logger.error(f"{RAW_FILE} not found. Please run the data download step first.")