import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    decision_threshold: float = 0.86


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

            model = XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1,
                eval_metric="logloss", random_state=42,
                scale_pos_weight=scale_pos_weight, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            logging.info("XGBoost (class_weight-equivalent) trained")

            threshold = self.model_trainer_config.decision_threshold
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            metrics = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }
            logging.info(f"XGBoost @ threshold={threshold}: {metrics}")

            if metrics["pr_auc"] < 0.5:
                raise CustomException("Model did not meet the minimum PR-AUC threshold", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model,
            )

            return "XGBoost", metrics

        except Exception as e:
            raise CustomException(e, sys)