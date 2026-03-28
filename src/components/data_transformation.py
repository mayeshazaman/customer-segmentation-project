import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..pipeline.exception import CustomException
from ..pipeline.logger import logging
from ..pipeline.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, rfm_data_path):
        try:
            rfm = pd.read_csv(rfm_data_path)
            logging.info("RFM data loaded for transformation")

            features = ["Recency", "Frequency", "Monetary"]
            X = rfm[features]

            # Log transform Monetary and Frequency to reduce skew
            X = X.copy()
            X["Frequency"] = np.log1p(X["Frequency"])
            X["Monetary"] = np.log1p(X["Monetary"])

            preprocessor = self.get_data_transformer_object()
            X_scaled = preprocessor.fit_transform(X)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("Preprocessor saved")

            return X_scaled, rfm["CustomerID"].values

        except Exception as e:
            raise CustomException(e, sys)