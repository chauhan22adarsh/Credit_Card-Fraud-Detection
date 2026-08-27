import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the real Kaggle Credit Card Fraud dataset and performs a
        STRATIFIED train/test split on it, BEFORE any resampling happens
        downstream in data_transformation.py.

        This ordering is deliberate and important: fraud is only ~0.17%
        of transactions, so splitting first (and only rebalancing the
        training fold later) is what keeps the test set an honest,
        untouched sample of real-world conditions. Rebalancing before
        splitting is a common data-leakage mistake that inflates every
        metric reported later in the pipeline.
        """
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("notebook/data/creditcard.csv")
            logging.info(f"Read dataset as dataframe: shape={df.shape}")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Performing stratified train/test split on raw, imbalanced data")
            train_set, test_set = train_test_split(
                df, test_size=0.2, stratify=df["Class"], random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
