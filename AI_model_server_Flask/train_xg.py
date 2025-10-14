# train_xg.py
# Train and evaluate an XGBoost model with feature engineering, encoding, scaling, and hyperparameter tuning.
# Saves a pipeline compatible with Flask API usage.

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib

try:
	from xgboost import XGBClassifier
except ImportError:
	raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

# --- Config ---
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / 'AI_model_Py_Scripts' / 'fraud_dataset_Generator_using_numpy.csv'
MODEL_PATH = HERE / 'best_xgboost_model.pkl'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Load Data ---
def load_data(csv_path=CSV_PATH):
	df = pd.read_csv(csv_path)
	# Try to find label column
	label_candidates = ['label', 'Label', 'is_fraud', 'Class', 'target', 'fraud']
	label_col = None
	for c in label_candidates:
		if c in df.columns:
			label_col = c
			break
	if label_col is None:
		# fallback: last column if binary
		last = df.columns[-1]
		if set(df[last].unique()).issubset({0,1}):
			label_col = last
		else:
			raise ValueError(f"No label column found in {csv_path}")
	X = df.drop(columns=[label_col])
	y = df[label_col]
	return X, y

# --- Preprocessing ---
def build_preprocessor(X):
	numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
	preprocessor = ColumnTransformer([
		('num', StandardScaler(), numeric_features),
		('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
	])
	return preprocessor, numeric_features, categorical_features

# --- Model & Hyperparameter Tuning ---
def build_model():
	clf = XGBClassifier(
		objective='binary:logistic',
		eval_metric='logloss',
		use_label_encoder=False,
		random_state=RANDOM_STATE,
		n_jobs=-1
	)
	return clf

def tune_hyperparameters(pipeline, X, y):
	param_grid = {
		'clf__n_estimators': [100, 200, 300],
		'clf__max_depth': [3, 5, 7, 10],
		'clf__learning_rate': [0.01, 0.05, 0.1, 0.2],
		'clf__subsample': [0.7, 0.8, 1.0],
		'clf__colsample_bytree': [0.7, 0.8, 1.0],
		'clf__scale_pos_weight': [1, 2, 5, 10],
	}
	cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
	search = RandomizedSearchCV(
		pipeline, param_grid, n_iter=10, scoring='roc_auc', cv=cv, verbose=2, n_jobs=-1, random_state=RANDOM_STATE
	)
	search.fit(X, y)
	print(f"Best params: {search.best_params_}")
	return search.best_estimator_

# --- Main Training Pipeline ---
def main():
	print(f"Loading data from {CSV_PATH}")
	X, y = load_data()
	preprocessor, num_feats, cat_feats = build_preprocessor(X)
	print(f"Numeric features: {num_feats}")
	print(f"Categorical features: {cat_feats}")

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

	pipeline = Pipeline([
		('pre', preprocessor),
		('clf', build_model())
	])

	print("Tuning hyperparameters (this may take a while)...")
	best_pipeline = tune_hyperparameters(pipeline, X_train, y_train)

	print("Evaluating on test set...")
	y_pred = best_pipeline.predict(X_test)
	y_proba = best_pipeline.predict_proba(X_test)[:,1]
	print(classification_report(y_test, y_pred, digits=4))
	print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
	print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

	print(f"Saving model to {MODEL_PATH}")
	joblib.dump(best_pipeline, MODEL_PATH)
	print("Done.")

if __name__ == '__main__':
	main()
