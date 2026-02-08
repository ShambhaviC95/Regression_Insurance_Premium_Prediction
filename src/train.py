# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 00:34:11 2026

@author: shambhavic
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error

from preprocessing import preprocess

# -----------------------
# 1. Load data from DB
# -----------------------

conn = sqlite3.connect("data/regression.db")

df = pd.read_sql("SELECT * FROM Insurance_Prediction", conn)

conn.close()

print("Data loaded:", df.shape)

# -----------------------
# # 2. Business split
# -----------------------

train_df = df.iloc[:700000].reset_index(drop=True)
val_df   = df.iloc[700000:900000].reset_index(drop=True)
prod_df  = df.iloc[900000:].reset_index(drop=True)

print(train_df.shape, val_df.shape, prod_df.shape)


# -----------------------
# 3. Preprocess training data
# -----------------------

X_train, y_train, scaler = preprocess(train_df, training=True)

# -----------------------
# 4. Preprocess test data
# -----------------------

X_test, y_test, _ = preprocess(val_df, scaler=scaler, training=False)

# -----------------------
# 5. Train Random Forest
# -----------------------

model = Lasso(alpha=0.5)

model.fit(X_train, y_train)


# -----------------------
# 6. Evaluate model
# -----------------------

preds = model.predict(X_test)

r2 = r2_score(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse=np.sqrt(mse)

print(f"Validation R2: {r2:.4f}")
print(f"Validation RMSE: {rmse:.2f}")

# -----------------------
# 7. Save model + scaler
# ----------------------

import json
feature_columns = list(X_train.columns)

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)


import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/final_lasso.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved.")
