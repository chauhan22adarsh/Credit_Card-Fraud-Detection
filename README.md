# Credit Card Fraud Detection

A fraud detection project split into three layers that each do a
different job and run in a specific order: `experiments/` first (finds
the best model/technique/threshold), then `notebook/` (documents that
process end-to-end, viewable on GitHub), then `src/` + `application.py`
(the fast, deployable pipeline that uses what the other two decided).

## The three layers, in the order you actually run them

| Layer | What it does |
|---|---|
| `experiments/` | Runs the 9 model×technique combinations under 5-fold CV, tunes the decision threshold, runs SHAP. This is where the winning configuration gets decided. |
| `notebook/` | Loads the results `experiments/` already produced and walks through EDA → the comparison → threshold tuning → SHAP, with charts. Doesn't redo the comparison — just reads and explains it. |
| `src/` + `application.py` | Trains one model (the winner from `experiments/`, hardcoded) and serves it via Flask. No comparison, no tuning — just executes the already-made decision, fast. |

**Why this order matters:** the notebook's Section 5 reads
`experiments/cv_results.json` directly — if you open the notebook before
running `experiments/`, that cell fails with a clear error telling you
to run `experiments/` first (see "How to run each layer" below).

The comparison work in `experiments/` decided a fixed configuration —
**XGBoost, `class_weight`-equivalent weighting, decision threshold
0.86** — and `src/model_trainer.py` trains directly with that
configuration instead of re-running the full comparison every time the
pipeline executes. This is deliberate: research/comparison code and
production code have different jobs, and conflating them either makes
the deployable app slow (if it re-compares every run) or makes the
comparison work hard to find/reproduce (if it's not kept anywhere).

## Project structure

```
fraud-detection-project/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── notebook/
│   ├── fraud_detection.ipynb       ← full story: EDA → CV comparison → threshold tuning → SHAP
│   └── data/
│       └── creditcard.csv          ← real Kaggle dataset (284,807 rows)
│
├── experiments/                    ← one-time comparison work, not on the fast pipeline's path
│   ├── compare_techniques.py       ← the 9-combination stratified CV grid + threshold tuning + summarize
│   ├── shap_analysis.py            ← SHAP analysis on the winning model
│   ├── cv_results.json             ← saved results (so the notebook loads instantly, no re-run needed)
│   ├── results_summary.txt         ← plain-text summary, regenerated from cv_results.json via `summarize`
│   ├── best_model.pkl              ← the fitted winning model + test data (for shap_analysis.py)
│   └── *.png                       ← threshold-tuning curve, SHAP plots
│
├── src/
│   ├── logger.py                   ← timestamped file logging, used everywhere
│   ├── exception.py                ← custom exception with file/line detail
│   ├── utils.py                    ← save_object / load_object
│   ├── components/
│   │   ├── data_ingestion.py       ← loads real data, stratified split (before anything else)
│   │   ├── data_transformation.py  ← RobustScaler on Amount/Time; NO resampling (see below)
│   │   └── model_trainer.py        ← trains XGBoost w/ class-weighting, tuned 0.86 threshold, fast
│   └── pipeline/
│       └── predict_pipeline.py     ← applies the SAME tuned threshold at prediction time
│
├── artifacts/                      ← generated: train.csv, test.csv, model.pkl, preprocessor.pkl
│
├── templates/
│   ├── index.html                  ← landing page
│   └── home.html                   ← pick-a-real-transaction demo (see below)
│
└── application.py                  ← Flask app: 2 routes, transaction picker → prediction
```

## Dataset

`creditcard.csv` :-
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## How to run each layer

Run these in order — the notebook depends on `experiments/` having
already produced `cv_results.json`.

**1. Experiments** :
```bash
cd experiments
python compare_techniques.py LogisticRegression smote
python compare_techniques.py LogisticRegression class_weight
python compare_techniques.py LogisticRegression undersample
python compare_techniques.py RandomForest smote
python compare_techniques.py RandomForest class_weight
python compare_techniques.py RandomForest undersample
python compare_techniques.py XGBoost smote
python compare_techniques.py XGBoost class_weight
python compare_techniques.py XGBoost undersample

python compare_techniques.py tune-threshold
python shap_analysis.py
python compare_techniques.py summarize
cd ..
```

**2. Notebook** (reads what step 1 produced, doesn't recompute it):
```bash
pip install -r requirements.txt
jupyter notebook notebook/fraud_detection.ipynb
```

**3. Production pipeline** (independent of steps 1-2, uses the hardcoded winning config):
```bash
python -m src.components.data_ingestion
python application.py                      # starts the Flask demo
```

## What the comparison in `experiments/` found

5-fold stratified cross-validation, 3 models × 3 imbalance-handling
techniques (SMOTE, `class_weight`, random undersampling), 9 combinations
— full table, generated by `compare_techniques.py summarize` directly
from `cv_results.json`:

| Model | Technique | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **XGBoost** | **class_weight** | **0.851** | 0.978 | **0.855** |
| Random Forest | class_weight | 0.826 | 0.968 | 0.841 |
| Random Forest | SMOTE | 0.730 | 0.977 | 0.837 |
| XGBoost | SMOTE | 0.679 | 0.977 | 0.852 |
| Logistic Regression | class_weight | 0.121 | 0.979 | 0.729 |
| Logistic Regression | undersample | 0.116 | 0.980 | 0.692 |
| Logistic Regression | SMOTE | 0.113 | 0.978 | 0.729 |
| Random Forest | undersample | 0.099 | 0.978 | 0.741 |
| XGBoost | undersample | 0.085 | 0.978 | 0.739 |

**Winner: XGBoost + `class_weight`** — best F1 and PR-AUC, and notably
without needing synthetic data or discarded training rows.

**Threshold tuning:** sweeping thresholds against the winning model
found F1 peaking near the default. Rather than lock in the single
sharpest peak from one sweep, the threshold was chosen by testing
candidate values against the *full* 5-fold comparison and picking the
one that held up most consistently across folds: **0.86**. This is what
`predict_pipeline.py` actually uses — not the default 0.5.

**Confirmed on a real production run** (`python -m src.components.data_ingestion`,
independent 80/20 split, threshold 0.86):

| Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|
| 0.899 | 0.816 | 0.856 | 0.965 | 0.874 | 0.856 |

This is a live, reproducible result — not a re-derived estimate — and it
lands consistent with the 5-fold CV table above (F1 ~0.85 either way),
which is itself a useful sanity check: the CV comparison and the
independently-trained production model agree.

**SHAP**: V14, V4, V12, V10, V11 dominate — consistent with Random
Forest's separate `.feature_importances_` ranking, which is a stronger
claim than either view alone (two different models, same signal).

Full details and plots are in `experiments/results_summary.txt`
(regenerate any time with `python compare_techniques.py summarize`) and
the notebook's Section 5.

## The Flask demo — an honest design note

The demo picks from **real transactions, unlabeled** — you choose by
amount/time only, the model predicts, then the real answer is revealed.
An earlier version let you type Amount/Time by hand for any transaction,
but the model's real signal comes almost entirely from 28 other
(anonymized) features no person can meaningfully type — that version was
removed because it couldn't actually demonstrate the model. See
`src/pipeline/predict_pipeline.py`'s `CustomData` docstring for the full
reasoning.
