import sys
import os
import numpy as np
import pandas as pd
from .exception import CustomException
from .utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, recency, frequency, monetary):
        try:
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model = load_object(os.path.join(BASE_DIR, "artifacts", "model.pkl"))
            preprocessor = load_object(os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl"))

            freq_log = np.log1p(frequency)
            monetary_log = np.log1p(monetary)
            X = np.array([[recency, freq_log, monetary_log]])
            X_scaled = preprocessor.transform(X)
            cluster = model.predict(X_scaled)[0]

            labels = {
                0: "Cluster 0 — Low Value",
                1: "Cluster 1 — At Risk",
                2: "Cluster 2 — Loyal",
                3: "Cluster 3 — Champions",
            }
            return cluster, labels.get(cluster, f"Cluster {cluster}")
        except Exception as e:
            raise CustomException(e, sys)