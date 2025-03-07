from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("calories_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    calories_burned = None
    inputs = {}

    if request.method == "POST":
        # Get user inputs
        inputs["Duration"] = float(request.form["duration"])
        inputs["Heart Rate"] = float(request.form["heart_rate"])
        inputs["Body Temperature"] = float(request.form["body_temp"])
        inputs["Gender"] = 1 if request.form["gender"] == "male" else 0
        inputs["Age"] = int(request.form["age"])
        inputs["BMI"] = float(request.form["bmi"])

        # Prepare input data for the model
        features = np.array([
            inputs["Duration"],
            inputs["Heart Rate"],
            inputs["Body Temperature"],
            inputs["Gender"],
            inputs["Age"],
            inputs["BMI"]
        ]).reshape(1, -1)

        # Predict calories burned
        calories_burned = model.predict(features)[0]

    return render_template("index.html", result=calories_burned, inputs=inputs)

if __name__ == "__main__":
    app.run(debug=True)
