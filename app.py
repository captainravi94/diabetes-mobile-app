from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""

    if request.method == "POST":
        input_data = np.array([[
            float(request.form["age"]),
            float(request.form["hypertension"]),
            float(request.form["heart_disease"]),
            float(request.form["bmi"]),
            float(request.form["HbA1c_level"]),
            float(request.form["blood_glucose_level"])
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        prediction_text = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
