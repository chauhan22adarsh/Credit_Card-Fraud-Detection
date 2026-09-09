import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        V1-V28 are already PCA-scaled in the raw dataset. Amount and Time
        are on very different raw scales, so RobustScaler (uses
        median/IQR) is fit on them specifically — robust to the handful
        of very large transactions that would distort a standard scaler.
        """
        try:
            scaler = RobustScaler()
            return scaler
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        NOTE ON IMBALANCE HANDLING: this pipeline does NOT resample the
        training data here (no SMOTE, no undersampling). That's a
        deliberate choice, not an oversight — see
        ../experiments/compare_techniques.py and
        ../experiments/results_summary.txt for the full comparison, but
        in short: a 5-fold stratified CV comparison of 3 models x 3
        imbalance techniques (SMOTE, class_weight, random undersampling)
        found that `class_weight="balanced"` passed directly to the
        model (see model_trainer.py) outperformed both resampling
        techniques for the winning model (XGBoost), and does so without
        synthetically generating data or discarding real rows. The
        model receives the training data as-is; class weighting happens
        inside model_trainer.py at fit time instead.

        Same leakage-avoidance principle as before still applies to
        everything that DOES happen here: scaling is fit on the training
        fold only and applied (not re-fit) to the test fold, which stays
        untouched and imbalanced throughout.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data for transformation")

            target_column = "Class"
            scale_columns = ["Amount", "Time"]

            preprocessing_obj = self.get_data_transformer_object()

            train_df[["scaled_amount", "scaled_time"]] = preprocessing_obj.fit_transform(
                train_df[scale_columns]
            )
            test_df[["scaled_amount", "scaled_time"]] = preprocessing_obj.transform(
                test_df[scale_columns]
            )

            train_df = train_df.drop(columns=scale_columns)
            test_df = test_df.drop(columns=scale_columns)

            input_feature_train_df = train_df.drop(columns=[target_column])
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column])
            target_feature_test_df = test_df[target_column]

            logging.info(
                f"Training class distribution (no resampling applied): "
                f"{target_feature_train_df.value_counts().to_dict()}"
            )

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]
            # test set: stays untouched and imbalanced throughout
            test_arr = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
