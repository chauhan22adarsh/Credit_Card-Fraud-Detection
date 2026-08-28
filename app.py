from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline, DEMO_EXAMPLES, GROUND_TRUTH

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # Only amount/time/id are shown to pick from — no fraud/normal
        # labels anywhere, so the user genuinely doesn't know the answer.
        return render_template("home.html", demo_examples=DEMO_EXAMPLES)

    else:
        txn_key = request.form.get("txn")

        if txn_key not in DEMO_EXAMPLES:
            return render_template(
                "home.html", demo_examples=DEMO_EXAMPLES,
                error="Please choose one of the transactions above.",
            )

        example = DEMO_EXAMPLES[txn_key]
        data = CustomData(
            amount=example["amount"], time=example["time"], v_features=example["v"]
        )
        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()
        prediction, probability = predict_pipeline.predict(pred_df)

        model_said = "FRAUD" if prediction[0] == 1 else "Normal"
        prob_pct = round(probability[0] * 100, 2)

        actual = "FRAUD" if GROUND_TRUTH[txn_key] == 1 else "Normal"
        model_was_correct = model_said == actual

        return render_template(
            "home.html",
            demo_examples=DEMO_EXAMPLES,
            picked_txn=txn_key,
            results=model_said,
            probability=prob_pct,
            actual=actual,
            model_was_correct=model_was_correct,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
