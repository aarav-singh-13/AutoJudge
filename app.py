from flask import Flask, render_template, request
import joblib

from src.preprocess import clean_text
from src.features import build_features

app = Flask(__name__)

clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_class = None
    prediction_score = None

    if request.method == "POST":
        user_text = request.form["problem_text"]

        clean = clean_text(user_text)

        X, _, _ = build_features(
            [{"full_text": clean}],
            tfidf=tfidf,
            scaler=scaler,
            fit=False
        )

        prediction_class = clf.predict(X)[0]
        prediction_score = round(reg.predict(X)[0], 2)

    return render_template(
        "index.html",
        prediction_class=prediction_class,
        prediction_score=prediction_score
    )


if __name__ == "__main__":
    app.run(debug=True)
