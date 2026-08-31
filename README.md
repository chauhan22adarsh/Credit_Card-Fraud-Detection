# Credit Card Fraud Detection — ML Project

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
