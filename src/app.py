# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 17:58:58 2026

@author: shambhavic
"""

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

from preprocessing import preprocess

app = Flask(__name__)

# Load artifacts once
model = joblib.load("models/final_lasso.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/feature_columns.json") as f:
    feature_cols = json.load(f)


@app.route("/")
def home():
    return "Insurance Premium Prediction API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    df = pd.DataFrame([data])

    # Preprocess
    X, _, _ = preprocess(df, scaler=scaler, training=False)

    # Align columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # Predict
    pred = model.predict(X)[0]

    return jsonify({"predicted_premium": float(pred)})


if __name__ == "__main__":
    app.run(debug=True)
