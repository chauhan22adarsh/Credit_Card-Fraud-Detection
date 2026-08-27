import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        One model from each major family — enough range for breadth,
        few enough to go deep on each: Logistic Regression (linear),
        Random Forest (bagging ensemble), XGBoost (boosting ensemble).

        The best model is picked by PR-AUC, not accuracy or even ROC-AUC —
        PR-AUC is the most informative single metric on data this
        imbalanced, since it's less forgiving of a high false-positive
        rate on the majority class than ROC-AUC can be.
        """
        try:
            logging.info("Splitting training and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "LogisticRegression": LogisticRegression(
                    C=0.1, max_iter=1000, random_state=42,
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, max_depth=14, min_samples_split=10,
                    random_state=42
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=150, max_depth=6, learning_rate=0.1,
                    eval_metric="logloss", random_state=42
                ),
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)
            for name, metrics in model_report.items():
                logging.info(f"{name}: {metrics}")

            best_model_name = max(model_report, key=lambda k: model_report[k]["pr_auc"])
            best_model_score = model_report[best_model_name]["pr_auc"]
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} (PR-AUC={best_model_score:.4f})")

            if best_model_score < 0.5:
                raise CustomException("No model met the minimum PR-AUC threshold", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return best_model_name, model_report

        except Exception as e:
            raise CustomException(e, sys)