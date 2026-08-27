# Credit Card Fraud Detection — Packaged ML Project

The same fraud detection project as the notebook version, restructured
as an installable Python package with a working Flask deployment —
modeled on a common "production-style ML project" template (data
ingestion → transformation → training as separate components, custom
logging, custom exceptions, a prediction pipeline, a web front end).

**This is a different deliverable, not a replacement.** The notebook is
easier to read top-to-bottom and better for explaining your reasoning
step by step. This structure demonstrates something the notebook can't:
that you can turn a model into something someone could actually deploy
and run.

## Project structure
```
fraud_mlproject/
├── app.py                          ← Flask app: 2 routes, form → prediction
├── setup.py                        ← makes this project pip-installable
├── requirements.txt
├── notebook/data/creditcard.csv    ← real Kaggle dataset (284,807 rows)
├── templates/
│   ├── index.html                  ← landing page
│   └── home.html                   ← prediction form + result
├── src/
│   ├── logger.py                   ← timestamped file logging, used everywhere
│   ├── exception.py                ← custom exception with file/line detail
│   ├── utils.py                    ← save/load model objects, evaluate_models()
│   ├── components/
│   │   ├── data_ingestion.py       ← loads real data, STRATIFIED split (before any resampling)
│   │   ├── data_transformation.py  ← RobustScaler + SMOTE (training fold ONLY)
│   │   └── model_trainer.py        ← trains 3 models, picks best by PR-AUC
│   └── pipeline/
│       └── predict_pipeline.py     ← loads saved model, serves predictions to app.py
└── artifacts/                      ← generated: train.csv, test.csv, model.pkl, preprocessor.pkl
```

## How to run it

**Train the pipeline** (regenerates everything in `artifacts/`):
```bash
pip install -r requirements.txt
python -m src.components.data_ingestion
```
This runs the whole chain: ingestion → transformation → training, and
logs every step to `logs/`.

**Run the web app:**
```bash
python app.py
```
Visit `http://localhost:5002`, click through to the prediction form,
enter a transaction amount and time, and get a live prediction back.

## The same methodology as the notebook version, just spread across files
- **`data_ingestion.py`**: stratified train/test split on the real,
  imbalanced data — done *before* any resampling, so the test set stays
  an honest, untouched sample.
- **`data_transformation.py`**: `RobustScaler` on Amount/Time, then
  **SMOTE applied to the training fold only** (`sampling_strategy=0.1`,
  not a full 50/50) — the test data loaded here is only ever
  `.transform()`-ed, never resampled.
- **`model_trainer.py`**: Logistic Regression, Random Forest, XGBoost —
  same three models as the notebook, picked by PR-AUC (not accuracy,
  not even plain ROC-AUC) since that's the most informative single
  metric on data this imbalanced.

## Actual results (this run, real data)
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.354 | 0.888 | 0.506 | 0.967 | 0.751 | 0.559 |
| **Random Forest** | **0.832** | 0.857 | **0.844** | **0.980** | **0.878** | **0.844** |
| XGBoost | 0.780 | 0.867 | 0.821 | 0.977 | 0.876 | 0.822 |

Matches the notebook version's numbers closely — same data, same
methodology, different packaging.

## An honest limitation of the web demo, worth stating upfront
The trained model's real predictive power comes almost entirely from the
28 anonymized PCA features (V1-V28) in the dataset — a bank's internal
fraud-detection pipeline gets these automatically from the transaction
itself, but there's no way for a person to meaningfully type them into a
web form by hand. The demo form only takes Amount and Time (the two
genuinely human-readable fields) and defaults V1-V28 to 0. This means the
**web demo shows the deployment mechanism working end-to-end**
(form → preprocessing → model → prediction), not the model's full
accuracy — that's demonstrated properly in the evaluation table above,
using the real, complete feature set. Say this proactively if asked "does
the demo actually catch fraud well" — it's honest, and it shows you
understand the difference between a demo interface and a production
integration (which would pull V1-V28 from the bank's own systems, not a
form).

## Why this structure, and when to use it over the notebook
This is worth showing when a role emphasizes deployment, MLOps, or
"can you ship this as software" — the custom logging and exception
handling in particular are meant to signal that. For a role focused on
analysis and modeling depth, the notebook version remains the stronger,
easier-to-defend choice; see `interview_prep.md` there for the full
methodology writeup (SMOTE-leakage, metric choices, etc.) — all of it
still applies here, just spread across files instead of cells.
