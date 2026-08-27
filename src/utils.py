import os
import sys
import dill

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Trains each model on the (already SMOTE-balanced) training fold, then
    scores it on the untouched, still-imbalanced test fold using metrics
    that are actually meaningful for imbalanced classification —
    accuracy is deliberately NOT reported here, since with ~0.17% fraud
    a model predicting "normal" every time would score ~99.8% accuracy
    while catching zero fraud.
    """
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            report[name] = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "pr_auc": average_precision_score(y_test, y_proba),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)
