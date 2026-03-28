import sys
from .exception import CustomException
from .logger import logging
from ..components.data_ingestion import DataIngestion
from ..components.data_transformation import DataTransformation
from ..components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("=== Training Pipeline Started ===")

            # Step 1: Ingestion + RFM
            data_ingestion = DataIngestion()
            rfm_path = data_ingestion.initiate_data_ingestion()

            # Step 2: Transform
            data_transformation = DataTransformation()
            X_scaled, customer_ids = data_transformation.initiate_data_transformation(rfm_path)

            # Step 3: Train KMeans
            model_trainer = ModelTrainer()
            silhouette = model_trainer.initiate_model_training(X_scaled, customer_ids, rfm_path)

            logging.info(f"=== Training Pipeline Completed | Silhouette Score: {silhouette:.4f} ===")
            return silhouette

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    score = pipeline.run_pipeline()
    print(f"Training complete. Silhouette Score: {score:.4f}")