# Insurance Premium Prediction (End-to-End ML Project)

## Problem Statement
Insurance companies need accurate premium estimates to reduce risk and improve customer satisfaction.  
Goal: Predict health insurance premiums using demographic, lifestyle, and medical history data.  
Demonstrates an **end-to-end ML workflow**: training, preprocessing, batch prediction, and real-time API.
---

## Dataset
- Source: `Regression.db` → `Insurance_Prediction` table  
- Size: 1,000,000 records (Train: 700k, Validation: 200k, Production: 100k)  
- Features: Age, BMI, Children, Gender, Smoker, Medical/Family History, Occupation, Exercise, Coverage

---

## Model & Results
**Chosen Model:** Lasso Regression  
- High accuracy (R² ~0.985)  
- Lightweight (~0.75 KB)  
- Fast inference (~0.007 sec)

Other models: Linear Regression, Random Forest (slower/larger)

---Structure---
insurance-premium-prediction/
│
├── data/ regression.db 
├── models/ # Saved ML models and artifacts
├── preprocessing.py
├── train.py
├── predict.py
├── app.py # Flask API
├── notebook/ # EDA & modeling
├── requirements.txt
└── README.md


---
### Run Train Model & Predict
```bash
python train.py
python predict.py       # Outputs: predicted_premiums.csv
python app.py           # Start Flask API

-----

### **Test API**
POST http://127.0.0.1:5000/predict

------
### **Tech Stack**

Python, Pandas, NumPy, Scikit-learn, SQLite, Flask, Joblib

------
### **Author**

Shambhavi Chaudhary
End-to-End ML Insurance Pricing Capstone Project


## Project Structure
