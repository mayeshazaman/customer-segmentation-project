import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..pipeline.exception import CustomException
from ..pipeline.logger import logging
from ..pipeline.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    cluster_result_path: str = os.path.join("artifacts", "clustered_customers.csv")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def find_optimal_k(self, X_scaled, k_range=range(2, 11)):
        """Use silhouette score to find best k"""
        best_k = 4
        best_score = -1
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            logging.info(f"k={k}, silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
        logging.info(f"Optimal k={best_k} with silhouette={best_score:.4f}")
        return best_k

    def initiate_model_training(self, X_scaled, customer_ids, rfm_data_path):
        try:
            optimal_k = self.find_optimal_k(X_scaled)

            model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)

            silhouette = silhouette_score(X_scaled, labels)
            logging.info(f"Final model silhouette score: {silhouette:.4f}")

            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            # Save cluster assignments back to customers
            rfm = pd.read_csv(rfm_data_path)
            rfm["Cluster"] = labels
            rfm.to_csv(self.model_trainer_config.cluster_result_path, index=False)
            logging.info("Clustered customer data saved")

            return silhouette

        except Exception as e:
            raise CustomException(e, sys)