from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

model_path = "model_rf.pkl"
scaler_path = "scaler_rf.pkl"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def validate_input(data):
    if "year" not in data or "month" not in data:
        return "Invalid input. 'year' and 'month' are required.", False
    try:
        year = int(data["year"])
        month = int(data["month"])
    except ValueError:
        return "Invalid input type. 'year' and 'month' must be integers.", False
    if not (1 <= month <= 12):
        return "Invalid month. Please provide a month in the range 1 to 12.", False
    return {"year": year, "month": month}, True

@app.route("/")
def welcome():
    return jsonify({
        "message": "Welcome to the server. To make predictions, go to the /predict route!",
        "example_request": {
            "year": 2021,
            "month": 1
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid input. Ensure the request contains valid JSON data."}), 400
        
        validated_data, valid = validate_input(data)
        if not valid:
            return jsonify({"error": validated_data}), 422

        custom_input = np.array([[validated_data["year"], validated_data["month"]]])
        custom_input_scaled = scaler.transform(custom_input)
        predicted_value = model.predict(custom_input_scaled)

        prediction_value = predicted_value[0]

        response_body = {"prediction": prediction_value}
        return jsonify(response_body)
    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
