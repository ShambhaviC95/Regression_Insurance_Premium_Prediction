# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 01:57:58 2026

@author: shambhavic
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess(df, scaler=None, training=True):

    df = df.copy()

    # --------------------
    # Numerical missing
    # --------------------
    for col in ['age', 'children']:
        df[col] = df[col].fillna(df[col].median())

    df['children'] = df['children'].round().astype(int)

    # --------------------
    # Categorical missing
    # --------------------
    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

    for c in ['medical_history','family_medical_history']:
        df[c] = df[c].fillna('No_Record')

    for c in ['exercise_frequency','occupation']:
        df[c] = df[c].fillna('Unknown')

    # --------------------
    # Normalize text
    # --------------------
    cat_cols = [
        'gender','smoker','medical_history',
        'family_medical_history','exercise_frequency',
        'occupation','coverage_level'
    ]

    for c in cat_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # --------------------
    # Binary encoding
    # --------------------
    df['gender'] = df['gender'].map({'male':0,'female':1})
    df['smoker'] = df['smoker'].map({'yes':1,'no':0})

    # --------------------
    # Ordinal
    # --------------------
    df['exercise_frequency'] = df['exercise_frequency'].map({
        'never':0,'rarely':1,'occasionally':2,'frequently':3
    })

    df['coverage_level'] = df['coverage_level'].map({
        'basic':0,'standard':1,'premium':2
    })

    # --------------------
    # One hot
    # --------------------
    df = pd.get_dummies(
        df,
        columns=['medical_history','family_medical_history','occupation'],
        prefix=['med_h','fam_med_h','occ'],
        drop_first=True,
        dtype=int
    )

    # --------------------
    # Drop region
    # --------------------
    df = df.drop(columns=["region"], errors="ignore")

    # --------------------
    # Separate target if present
    # --------------------
    if 'charges' in df.columns:
        y = df['charges']
        X = df.drop('charges', axis=1)
    else:
        X = df
        y = None

    # --------------------
    # Scaling
    # --------------------
    num_cols = ['age','bmi','children']

    if training:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])

    return X, y, scaler
