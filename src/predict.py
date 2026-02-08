# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 12:17:18 2026

@author: shambhavic
"""

# -*- coding: utf-8 -*-
"""
Predict insurance premiums on production data
"""

import pandas as pd
import sqlite3
import joblib

from preprocessing import preprocess

# -----------------------
# 1. Load data
# -----------------------

conn = sqlite3.connect("data/regression.db")
df = pd.read_sql("SELECT * FROM Insurance_Prediction", conn)
conn.close()

print("Data loaded:", df.shape)

# -----------------------
# 2. Select production data
# -----------------------

prod_df = df.iloc[900000:].reset_index(drop=True)
print("Production data shape:", prod_df.shape)

# -----------------------
# 3. Load model and scaler
# -----------------------

model = joblib.load("models/final_lasso.pkl")
scaler = joblib.load("models/scaler.pkl")

# -----------------------
# 4. Preprocess production data
# -----------------------

X_prod, _, _ = preprocess(prod_df, scaler=scaler, training=False)

# -----------------------
# Align columns with training schema
# -----------------------
import json

with open("models/feature_columns.json") as f:
    feature_cols = json.load(f)

for col in feature_cols:
    if col not in X_prod.columns:
        X_prod[col] = 0

X_prod = X_prod[feature_cols]

# -----------------------
# 5. Generate predictions
# -----------------------

predictions = model.predict(X_prod)

# -----------------------
# 6. Attach predictions to data
# -----------------------

prod_df["predicted_charges"] = predictions

# -----------------------
# 7. Output results
# -----------------------

print(prod_df[["predicted_charges"]].head())

# Optional: save to CSV
prod_df.to_csv("predicted_premiums.csv", index=False)
print("Predictions saved to predicted_premiums.csv")
