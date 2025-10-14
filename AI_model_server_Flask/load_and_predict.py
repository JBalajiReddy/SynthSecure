"""
Load the trained pipeline `best_xgboost_model.pkl` and run a single sample prediction from the CSV dataset.

Usage:
    python load_and_predict.py
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / 'best_xgboost_model.pkl'
CSV_PATH = HERE.parent / 'AI_model_Py_Scripts' / 'fraud_dataset_Generator_using_numpy.csv'


def main():
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}. Run train_xg.py first.")
        return
    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print('Model loaded. Pipeline steps:', model.named_steps.keys())

    if not CSV_PATH.exists():
        print(f"CSV not found at {CSV_PATH}. Can't run a sample prediction.")
        return
    df = pd.read_csv(CSV_PATH)
    # pick a sample row
    sample = df.drop(columns=[col for col in df.columns if col.lower() in ('isfraud','fraud','label','target','class')], errors='ignore').iloc[[0]]
    print('Sample input shape:', sample.shape)
    pred = model.predict(sample)
    print('Prediction for sample:', pred)

if __name__ == '__main__':
    main()
