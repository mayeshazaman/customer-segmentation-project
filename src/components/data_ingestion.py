import os
import sys
import pandas as pd
from dataclasses import dataclass
from ..pipeline.exception import CustomException
from ..pipeline.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    rfm_data_path: str = os.path.join("artifacts", "rfm.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            df = pd.read_excel(os.path.join(BASE_DIR, "dataset", "Customer Segmentation.xlsx"))
            logging.info(f"Dataset loaded: {df.shape}")

            # Drop nulls in CustomerID
            df.dropna(subset=["CustomerID"], inplace=True)

            # Remove cancelled invoices
            df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

            # Remove non-positive quantities and prices
            df = df[df["Quantity"] > 0]
            df = df[df["UnitPrice"] > 0]

            # Parse date
            df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

            # TotalPrice
            df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

            # Save raw cleaned data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # ---- RFM Feature Engineering ----
            snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

            rfm = df.groupby("CustomerID").agg(
                Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
                Frequency=("InvoiceNo", "nunique"),
                Monetary=("TotalPrice", "sum")
            ).reset_index()

            rfm.to_csv(self.ingestion_config.rfm_data_path, index=False)
            logging.info(f"RFM data saved: {rfm.shape}")

            return self.ingestion_config.rfm_data_path

        except Exception as e:
            raise CustomException(e, sys)