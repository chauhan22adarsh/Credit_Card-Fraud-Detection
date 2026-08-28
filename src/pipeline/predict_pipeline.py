import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Loads the saved preprocessor + trained model from artifacts/ and
        returns both the predicted class (0/1) and the fraud probability,
        so the web form can show a probability, not just a hard label.
        """
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data = features.copy()
            data[["scaled_amount", "scaled_time"]] = preprocessor.transform(
                data[["Amount", "Time"]]
            )
            data = data.drop(columns=["Amount", "Time"])
            data = data.to_numpy()  # avoid sklearn's "fitted without feature names" warning,
                                      # since the model was trained on a plain numpy array

            prediction = model.predict(data)
            probability = model.predict_proba(data)[:, 1]

            return prediction, probability

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Maps a dataset row (real Amount, Time, and 28 PCA features V1-V28)
    into the same 30-column layout the model was trained on.

    IMPORTANT: this is only ever called with real feature values, loaded
    from DEMO_EXAMPLES below — never with V1-V28 defaulted to 0. An
    earlier version of this project let the web form collect only
    Amount and Time, silently defaulting the other 28 features to 0.
    That produced a prediction, but not a meaningful one: Amount's
    feature importance in the trained model is ~0.004 (dead last of 30
    features; V13, V16, V9 dominate at 0.24, 0.15, 0.12), and the model
    saturates on Amount above ~5,000 — every value from 5,000 to
    1,000,000 produces the exact same output, because the trees rarely
    split on it more than once. Rather than keep a form that looks
    functional but can't actually demonstrate the model, that path was
    removed — see app.py and home.html.
    """

    def __init__(self, amount: float, time: float, v_features: dict):
        self.amount = amount
        self.time = time
        self.v_features = v_features

    def get_data_as_dataframe(self):
        try:
            data = {"Amount": [self.amount], "Time": [self.time]}
            for i in range(1, 29):
                col = f"V{i}"
                data[col] = [self.v_features.get(col, 0.0)]
            df = pd.DataFrame(data)
            # match training column order: Time, V1..V28, Amount
            df = df[["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]]
            return df
        except Exception as e:
            raise CustomException(e, sys)


# 8 REAL rows from the dataset, shuffled to a genuinely mixed order (4
# fraud, 4 normal — but NOT labeled as such anywhere in the UI or here
# by position). The user picks by transaction id/amount/time only, with
# no way to know the answer in advance — the model tells them, and they
# can check themselves against the source CSV if they want to verify.
# This is the actual fix for the "answer key" problem: the earlier
# version's buttons literally said "Real fraud example #1", which let
# the user see the ground truth before predicting anything.
DEMO_EXAMPLES = {
    "txn_1": {
        "time": 160364.0, "amount": 105.04,
        "v": {"V1": 1.8267, "V2": -0.3612, "V3": -1.7056, "V4": 0.5683, "V5": -0.2135,
              "V6": -1.2397, "V7": 0.1228, "V8": -0.1927, "V9": 1.2728, "V10": -0.8382,
              "V11": -0.7245, "V12": -0.5719, "V13": -1.8393, "V14": -1.0788, "V15": 0.1142,
              "V16": 0.0106, "V17": 1.2386, "V18": 0.1499, "V19": 0.0328, "V20": -0.1084,
              "V21": -0.2024, "V22": -0.6073, "V23": 0.1299, "V24": -0.1704, "V25": -0.1721,
              "V26": -0.09, "V27": -0.0326, "V28": -0.0088},
    },
    "txn_2": {
        "time": 95628.0, "amount": 1.63,
        "v": {"V1": -17.5189, "V2": 12.5721, "V3": -19.0385, "V4": 11.1909, "V5": -13.5547,
              "V6": -0.4119, "V7": -23.1894, "V8": -5.3014, "V9": -8.6304, "V10": -16.2556,
              "V11": 5.7021, "V12": -13.8124, "V13": -0.3068, "V14": -7.0136, "V15": -1.1823,
              "V16": -9.3602, "V17": -16.6684, "V18": -7.1475, "V19": 2.1621, "V20": 2.5285,
              "V21": -4.9695, "V22": 0.9761, "V23": 1.8412, "V24": 0.3344, "V25": -0.7201,
              "V26": -0.2326, "V27": -3.022, "V28": -0.4782},
    },
    "txn_3": {
        "time": 118231.0, "amount": 10.00,
        "v": {"V1": 2.2797, "V2": -0.5072, "V3": -2.5108, "V4": -1.0197, "V5": 0.2754,
              "V6": -1.4644, "V7": 0.3562, "V8": -0.5241, "V9": -1.2046, "V10": 1.1305,
              "V11": 0.6004, "V12": -0.4012, "V13": -0.7235, "V14": 0.7332, "V15": -1.1565,
              "V16": 0.3711, "V17": 0.2222, "V18": -1.0972, "V19": 1.1633, "V20": -0.0958,
              "V21": 0.5007, "V22": 1.4811, "V23": -0.2724, "V24": -0.2481, "V25": 0.7583,
              "V26": 0.3026, "V27": -0.103, "V28": -0.108},
    },
    "txn_4": {
        "time": 62348.0, "amount": 10.00,
        "v": {"V1": 0.2444, "V2": -0.4146, "V3": 0.0787, "V4": -2.6274, "V5": -0.6417,
              "V6": 0.7218, "V7": -1.8892, "V8": -2.5067, "V9": -2.3106, "V10": 0.6288,
              "V11": -0.7608, "V12": -1.2247, "V13": -0.817, "V14": 0.4234, "V15": -0.5221,
              "V16": 0.7978, "V17": -0.4805, "V18": 1.1406, "V19": -0.2807, "V20": 0.237,
              "V21": -1.5274, "V22": -0.1417, "V23": -0.1731, "V24": -1.4628, "V25": 0.7574,
              "V26": -0.2397, "V27": 0.0894, "V28": 0.2237},
    },
    "txn_5": {
        "time": 50808.0, "amount": 99.99,
        "v": {"V1": -9.1698, "V2": 7.0922, "V3": -12.354, "V4": 4.2431, "V5": -7.1764,
              "V6": -3.3866, "V7": -8.058, "V8": 6.4429, "V9": -2.413, "V10": -6.1349,
              "V11": 2.8267, "V12": -6.3098, "V13": -0.623, "V14": -7.2799, "V15": 0.9242,
              "V16": -4.2155, "V17": -7.1717, "V18": -2.5503, "V19": 0.5964, "V20": 0.8167,
              "V21": 0.9262, "V22": -0.8177, "V23": -0.1504, "V24": -0.0394, "V25": 0.4856,
              "V26": -0.2643, "V27": 1.1597, "V28": 0.2328},
    },
    "txn_6": {
        "time": 55180.0, "amount": 84.82,
        "v": {"V1": -0.7102, "V2": 0.4277, "V3": 0.7828, "V4": 0.8249, "V5": 0.8069,
              "V6": 0.7002, "V7": 0.6567, "V8": 0.0253, "V9": -0.4393, "V10": -0.1374,
              "V11": -1.495, "V12": -0.3851, "V13": 0.6845, "V14": 0.0678, "V15": 1.6587,
              "V16": -0.2141, "V17": -0.394, "V18": 0.3623, "V19": 0.8204, "V20": 0.0519,
              "V21": 0.2195, "V22": 0.8236, "V23": -0.0435, "V24": -1.2819, "V25": 0.0573,
              "V26": -0.0306, "V27": 0.1211, "V28": 0.1154},
    },
    "txn_7": {
        "time": 149640.0, "amount": 2.00,
        "v": {"V1": 0.7543, "V2": 2.3798, "V3": -5.1373, "V4": 3.8184, "V5": 0.0432,
              "V6": -1.2855, "V7": -1.7667, "V8": 0.7567, "V9": -1.7657, "V10": -3.263,
              "V11": 3.5928, "V12": -2.7723, "V13": -0.0745, "V14": -6.2811, "V15": 0.166,
              "V16": -2.6792, "V17": -1.3856, "V18": 0.2491, "V19": 2.3535, "V20": 0.3697,
              "V21": 0.3971, "V22": 0.1412, "V23": 0.172, "V24": 0.3943, "V25": -0.4446,
              "V26": -0.2632, "V27": 0.3047, "V28": -0.0444},
    },
    "txn_8": {
        "time": 71033.0, "amount": 426.40,
        "v": {"V1": -3.1708, "V2": 0.1857, "V3": -3.3999, "V4": 3.7612, "V5": -2.148,
              "V6": -1.5989, "V7": -2.5196, "V8": 1.3162, "V9": -2.4001, "V10": -4.9934,
              "V11": 4.4472, "V12": -5.2938, "V13": -1.4196, "V14": -6.4253, "V15": 0.9141,
              "V16": -3.5152, "V17": -6.3475, "V18": -0.9035, "V19": 1.1916, "V20": 1.9323,
              "V21": 1.0921, "V22": -0.0411, "V23": 0.9044, "V24": 0.18, "V25": 0.05,
              "V26": -0.2571, "V27": 0.8593, "V28": 0.2259},
    },
}

# Ground truth for each transaction above — kept in a SEPARATE dict on
# purpose, and never passed to the template before a prediction is made.
# This is what makes the demo honest: the user picks a transaction by
# its amount/time only, with no way to see the label in advance, and the
# reveal happens only after the model has made its prediction — so they
# can compare the model's call against the real answer, the same way a
# fraud analyst would review a flagged case after the fact.
GROUND_TRUTH = {
    "txn_1": 0, "txn_2": 1, "txn_3": 0, "txn_4": 0,
    "txn_5": 1, "txn_6": 0, "txn_7": 1, "txn_8": 1,
}
