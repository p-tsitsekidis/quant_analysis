"""
    Download and save raw market data CSV
"""

import pandas as pd
import logging

# Setup Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Download and save CSV. One time only
logger.info("Downloading power data...")
url = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

df = pd.read_csv(url)
logger.info("Saving to data/power_data_raw.csv")
df.to_csv("data/power_data_raw.csv", index=False)

logger.info("Process completed.")