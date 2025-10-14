from flask import Flask, request, jsonify
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Try a few candidate model filenames (choose the first existing)
MODEL_CANDIDATES = [
    "best_xgboost_model.pkl",
    "best_xgboost_gan_model.pkl",
    "best_rf_model (1).pkl",
    "best_rf_model.pkl",
    "best_xgboost_model.joblib",
]

def find_model_path():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model file found. Checked: {MODEL_CANDIDATES}")

model_path = find_model_path()
print(f"Loading model from: {model_path}")

# Load with joblib first (handles joblib dumps), fallback to pickle
artifact = None
try:
    artifact = joblib.load(model_path)
except Exception:
    try:
        with open(model_path, 'rb') as f:
            artifact = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

# Normalize artifact to obtain pipeline and optional preprocessor
model_pipeline = None
preprocessor = None
features_meta = None
model_core = None  # classifier without preprocessor if pipeline detected
if isinstance(artifact, dict):
    # common keys used by our training scripts
    model_pipeline = artifact.get('pipeline') or artifact.get('model') or artifact.get('clf')
    preprocessor = artifact.get('preprocessor') or artifact.get('scaler')
    features_meta = artifact.get('features')
else:
    model_pipeline = artifact

if model_pipeline is None:
    raise RuntimeError(f"Loaded artifact did not contain a model pipeline. Keys: {list(artifact.keys()) if isinstance(artifact, dict) else 'N/A'}")

# If the artifact is a sklearn Pipeline with a preprocessor step, extract it and the classifier.
try:
    if hasattr(model_pipeline, 'named_steps') and isinstance(model_pipeline.named_steps, dict):
        preprocessor = model_pipeline.named_steps.get('pre') or model_pipeline.named_steps.get('preprocessor')
        model_core = model_pipeline.named_steps.get('clf') or model_pipeline.named_steps.get('model') or model_pipeline.named_steps.get('classifier')
        # Attempt to derive original input feature names (column order) used during fit
        if preprocessor is not None:
            try:
                # Available on fitted transformers when input was a DataFrame
                features_meta = list(preprocessor.feature_names_in_)
            except Exception:
                try:
                    # Fallback: combine declared numeric and categorical columns
                    transformers = getattr(preprocessor, 'transformers_', None)
                    if transformers:
                        cols = []
                        for _, _, sel in transformers:
                            if isinstance(sel, (list, tuple)):
                                cols.extend(list(sel))
                        if cols:
                            features_meta = list(cols)
                except Exception:
                    pass
        # If we didn't find a separate classifier, keep using the full pipeline
        if model_core is None:
            model_core = model_pipeline
except Exception:
    # If any introspection fails, proceed with the loaded model as-is
    model_core = model_pipeline

@app.route('/')
def home():
    return "Welcome to the XGBoost Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(silent=True, force=False)
    if not data or 'features' not in data:
        return jsonify({"error": "No input data provided or 'features' key missing"}), 400

    try:
        feats_in = data['features']

        X = None
        # Support both array-like and object-like inputs
        if isinstance(feats_in, dict):
            X = pd.DataFrame([feats_in])
            # Reorder/align to feature metadata if available
            if features_meta is not None:
                # Missing columns will be introduced with NaN (model may handle or raise)
                X = X.reindex(columns=features_meta)
        else:
            raw_feats = np.array(feats_in).reshape(1, -1)
            # Wrap into DataFrame when we know the expected column names
            if features_meta is not None:
                X = pd.DataFrame(raw_feats, columns=features_meta)
            else:
                X = raw_feats

        # Avoid double-preprocessing: if we extracted a preprocessor and a core model, use those.
        if preprocessor is not None and model_core is not None and model_core is not model_pipeline:
            feats = preprocessor.transform(X)
            prediction = model_core.predict(feats)
            try:
                proba = model_core.predict_proba(feats).tolist()
            except Exception:
                proba = None
        else:
            prediction = model_pipeline.predict(X)
            try:
                proba = model_pipeline.predict_proba(X).tolist()
            except Exception:
                proba = None

        return jsonify({"prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction, "probability": proba})
    except Exception as e:
        # Include error string in response to help debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
