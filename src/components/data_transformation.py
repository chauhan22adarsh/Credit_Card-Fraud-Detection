import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

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
        IMPORTANT ORDERING, same principle as data_ingestion.py:
        SMOTE is applied to the TRAINING fold only, AFTER the split that
        already happened in data_ingestion.py. The test set loaded here
        is used only for scaling (transform, not fit) and is never
        resampled — it stays a realistic, imbalanced sample throughout
        the whole pipeline.
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
                f"Before SMOTE: {target_feature_train_df.value_counts().to_dict()}"
            )
            smote = SMOTE(sampling_strategy=0.1, random_state=42)
            input_feature_train_res, target_feature_train_res = smote.fit_resample(
                input_feature_train_df, target_feature_train_df
            )
            logging.info(
                f"After SMOTE (training fold only): "
                f"{target_feature_train_res.value_counts().to_dict()}"
            )

            train_arr = np.c_[
                input_feature_train_res, np.array(target_feature_train_res)
            ]
            # test set: untouched by SMOTE, still reflects real-world imbalance
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