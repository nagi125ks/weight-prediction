import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained LinearRegression model
with open("weightpredict.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        feature_value = float(data.get("feature", 0))

        # Model expects a 2D array with 1 feature: [[x]]
        input_array = np.array([[feature_value]])
        prediction = model.predict(input_array)[0]

        return jsonify({
            "success": True,
            "input": feature_value,
            "prediction": round(float(prediction), 4)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)