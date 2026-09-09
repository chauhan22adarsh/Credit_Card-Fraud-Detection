import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, precision_score, recall_score,
)

import shap

RANDOM_STATE = 42
N_SPLITS = 5
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("../notebook/data/creditcard.csv")
print(f"Loaded {df.shape[0]:,} transactions, {df['Class'].sum()} fraud "
      f"({df['Class'].mean()*100:.3f}%)")

scaler = RobustScaler()
df[["scaled_amount", "scaled_time"]] = scaler.fit_transform(df[["Amount", "Time"]])
df = df.drop(columns=["Amount", "Time"])

X = df.drop(columns=["Class"]).values
y = df["Class"].values
feature_names = df.drop(columns=["Class"]).columns.tolist()

# ---------------------------------------------------------------------------
# 2. STRATIFIED K-FOLD CROSS-VALIDATION SETUP
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n{N_SPLITS}-fold stratified split — fraud ratio per fold (sanity check):")
for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"  Fold {i}: train fraud ratio = {y[train_idx].mean():.5f}, "
          f"test fraud ratio = {y[test_idx].mean():.5f}")

# ---------------------------------------------------------------------------
# 3. THREE IMBALANCE-HANDLING TECHNIQUES, COMPARED SIDE BY SIDE
# ---------------------------------------------------------------------------

IMBALANCE_TECHNIQUES = ["smote", "class_weight", "undersample"]

MODEL_BUILDERS = {
    "LogisticRegression": lambda class_weight=None: LogisticRegression(
        C=0.1, max_iter=1000, random_state=RANDOM_STATE,
        class_weight=class_weight, n_jobs=-1
    ),
    "RandomForest": lambda class_weight=None: RandomForestClassifier(
        n_estimators=50, max_depth=14, min_samples_split=10,
        random_state=RANDOM_STATE, class_weight=class_weight, n_jobs=-1
    ),
    "XGBoost": lambda class_weight=None: XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE,
        # XGBoost doesn't take class_weight; scale_pos_weight is its
        # equivalent lever for the "class_weight" technique below.
        scale_pos_weight=(1 / y.mean() - 1) if class_weight == "balanced" else 1,
        n_jobs=-1,
    ),
}


def resample_training_fold(X_train, y_train, technique):
    if technique == "smote":
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res, None
    elif technique == "undersample":
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        return X_res, y_res, None
    elif technique == "class_weight":
        return X_train, y_train, "balanced"
    else:
        raise ValueError(f"Unknown technique: {technique}")


def evaluate_fold(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_one_combination(model_name, technique):
    build_model = MODEL_BUILDERS[model_name]
    fold_metrics = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_res, y_train_res, class_weight = resample_training_fold(
            X_train, y_train, technique
        )

        model = build_model(class_weight=class_weight)
        model.fit(X_train_res, y_train_res)

        fold_metrics.append(evaluate_fold(model, X_test, y_test))

    return fold_metrics


if __name__ == "__main__":
    import sys
    import json
    import os

    RESULTS_FILE = "cv_results.json"

    if len(sys.argv) == 4:
        model_name, technique, fold_idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
        build_model = MODEL_BUILDERS[model_name]

        splits = list(skf.split(X, y))
        train_idx, test_idx = splits[fold_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_res, y_train_res, class_weight = resample_training_fold(
            X_train, y_train, technique
        )
        model = build_model(class_weight=class_weight)
        model.fit(X_train_res, y_train_res)
        fold_result = evaluate_fold(model, X_test, y_test)

        partial_file = f"partial_{model_name}_{technique}.json"
        partial = {}
        if os.path.exists(partial_file):
            with open(partial_file) as f:
                partial = json.load(f)
        partial[str(fold_idx)] = fold_result
        with open(partial_file, "w") as f:
            json.dump(partial, f, indent=2)

        print(f"{model_name} x {technique} x fold {fold_idx}: "
              f"F1={fold_result['f1']:.4f}  ROC-AUC={fold_result['roc_auc']:.4f}")
        print(f"Saved to {partial_file} ({len(partial)}/{N_SPLITS} folds done)")

        if len(partial) == N_SPLITS:
            # All folds done — merge into the main results file
            fold_metrics = [partial[str(i)] for i in range(N_SPLITS)]
            results = {}
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE) as f:
                    results = json.load(f)
            results[f"{model_name}__{technique}"] = fold_metrics
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"All {N_SPLITS} folds complete — merged into {RESULTS_FILE}")

    elif len(sys.argv) == 3:
        # Single-combination mode: `python fraud_detection_upgraded.py RandomForest smote`
        model_name, technique = sys.argv[1], sys.argv[2]
        print(f"\nRunning {model_name} x {technique} across {N_SPLITS} folds...")
        fold_metrics = run_one_combination(model_name, technique)

        results = {}
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        results[f"{model_name}__{technique}"] = fold_metrics
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        mean_f1 = np.mean([m["f1"] for m in fold_metrics])
        mean_roc = np.mean([m["roc_auc"] for m in fold_metrics])
        mean_pr = np.mean([m["pr_auc"] for m in fold_metrics])
        print(f"{model_name:18s} | {technique:12s} -> "
              f"F1={mean_f1:.4f}  ROC-AUC={mean_roc:.4f}  PR-AUC={mean_pr:.4f}")
        print(f"Saved to {RESULTS_FILE}")

    elif len(sys.argv) == 2 and sys.argv[1] == "summarize":
        if not os.path.exists(RESULTS_FILE):
            raise FileNotFoundError(
                f"{RESULTS_FILE} not found. Run the 9 comparison "
                "combinations first (see README), then re-run this."
            )

        with open(RESULTS_FILE) as f:
            results = json.load(f)

        missing = [k for k in [f"{m}__{t}" for m in MODEL_BUILDERS for t in IMBALANCE_TECHNIQUES]
                   if k not in results]
        if missing:
            print(f"WARNING: {len(missing)}/9 combinations missing from {RESULTS_FILE}: {missing}")
            print("Summary below only covers what's actually present.\n")

        rows = []
        for key, folds in results.items():
            model_name, technique = key.split("__")
            rows.append({
                "model": model_name,
                "technique": technique,
                "f1": np.mean([f["f1"] for f in folds]),
                "f1_std": np.std([f["f1"] for f in folds]),
                "precision": np.mean([f["precision"] for f in folds]),
                "recall": np.mean([f["recall"] for f in folds]),
                "roc_auc": np.mean([f["roc_auc"] for f in folds]),
                "pr_auc": np.mean([f["pr_auc"] for f in folds]),
                "n_folds": len(folds),
            })
        rows.sort(key=lambda r: -r["f1"])

        best = rows[0] if rows else None

        lines = []
        lines.append("CREDIT CARD FRAUD DETECTION — MODEL COMPARISON SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Generated from: {RESULTS_FILE} ({len(rows)}/9 combinations present)")
        lines.append("")
        lines.append(f"{'Model':20s} {'Technique':14s} {'F1 (+/-std)':16s} "
                      f"{'Precision':10s} {'Recall':8s} {'ROC-AUC':8s} {'PR-AUC':8s} {'Folds':5s}")
        lines.append("-" * 100)
        for r in rows:
            f1_str = f"{r['f1']:.4f} (+/-{r['f1_std']:.4f})"
            lines.append(
                f"{r['model']:20s} {r['technique']:14s} {f1_str:16s} "
                f"{r['precision']:10.4f} {r['recall']:8.4f} {r['roc_auc']:8.4f} "
                f"{r['pr_auc']:8.4f} {r['n_folds']:5d}"
            )
        lines.append("")
        if best:
            lines.append(f"Best by F1: {best['model']} + {best['technique']} "
                          f"(F1={best['f1']:.4f}, PR-AUC={best['pr_auc']:.4f})")

        summary_text = "\n".join(lines)
        with open("results_summary.txt", "w") as f:
            f.write(summary_text + "\n")

        print(summary_text)
        print("\nSaved to results_summary.txt")

    elif len(sys.argv) == 2 and sys.argv[1] == "tune-threshold":
        # -------------------------------------------------------------------
        # 4. THRESHOLD TUNING (on the best combination found above)
        # -------------------------------------------------------------------
        with open(RESULTS_FILE) as f:
            results = json.load(f)

        # Pick the best (model, technique) combination by mean F1 across folds
        best_key, best_mean_f1 = None, -1
        for key, folds in results.items():
            mean_f1 = np.mean([f["f1"] for f in folds])
            if mean_f1 > best_mean_f1:
                best_key, best_mean_f1 = key, mean_f1
        best_model_name, best_technique = best_key.split("__")
        print(f"Best combination: {best_model_name} + {best_technique} "
              f"(mean F1={best_mean_f1:.4f}) — tuning its decision threshold.\n")

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        X_train_res, y_train_res, class_weight = resample_training_fold(
            X_train, y_train, best_technique
        )
        best_model = MODEL_BUILDERS[best_model_name](class_weight=class_weight)
        best_model.fit(X_train_res, y_train_res)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.05, 0.96, 0.01)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
            f1s.append(f1_score(y_test, y_pred_t, zero_division=0))

        best_idx = int(np.argmax(f1s))
        best_threshold = thresholds[best_idx]
        print(f"Optimal threshold (max F1): {best_threshold:.2f}")
        print(f"  At t=0.50 (default): precision={precisions[np.argmin(np.abs(thresholds-0.5))]:.4f}, "
              f"recall={recalls[np.argmin(np.abs(thresholds-0.5))]:.4f}, "
              f"F1={f1s[np.argmin(np.abs(thresholds-0.5))]:.4f}")
        print(f"  At t={best_threshold:.2f} (tuned): precision={precisions[best_idx]:.4f}, "
              f"recall={recalls[best_idx]:.4f}, F1={f1s[best_idx]:.4f}")

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1", linewidth=2.5)
        plt.axvline(best_threshold, color="red", linestyle="--",
                    label=f"Optimal threshold = {best_threshold:.2f}")
        plt.axvline(0.5, color="gray", linestyle=":", label="Default = 0.50")
        plt.xlabel("Decision threshold")
        plt.ylabel("Score")
        plt.title(f"Precision / Recall / F1 vs. Threshold\n{best_model_name} + {best_technique}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("threshold_tuning.png", dpi=120)
        print("Saved plot to threshold_tuning.png")

        # Save the tuned threshold + fitted model for the SHAP step
        import pickle
        with open("best_model.pkl", "wb") as f:
            pickle.dump({
                "model": best_model, "model_name": best_model_name,
                "technique": best_technique, "threshold": float(best_threshold),
                "X_test": X_test, "y_test": y_test, "feature_names": feature_names,
            }, f)
        print("Saved fitted model + test data to best_model.pkl")

    else:
        all_results = {}
        for model_name in MODEL_BUILDERS:
            for technique in IMBALANCE_TECHNIQUES:
                fold_metrics = run_one_combination(model_name, technique)
                all_results[f"{model_name}__{technique}"] = fold_metrics
                mean_f1 = np.mean([m["f1"] for m in fold_metrics])
                mean_roc = np.mean([m["roc_auc"] for m in fold_metrics])
                mean_pr = np.mean([m["pr_auc"] for m in fold_metrics])
                print(f"{model_name:18s} | {technique:12s} -> "
                      f"F1={mean_f1:.4f}  ROC-AUC={mean_roc:.4f}  PR-AUC={mean_pr:.4f}")
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)