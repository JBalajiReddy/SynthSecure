from flask import Flask, request, jsonify
import os
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
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

# ---- Baseline statistics (optional, used for simple explanations) ----
baseline_stats = {}
baseline_columns = []

def try_load_baseline():
    global baseline_stats, baseline_columns
    try:
        # Attempt to find the CSV relative to project structure
        here = Path(__file__).resolve().parent
        csv_path = here.parent / 'AI_model_Py_Scripts' / 'fraud_dataset_Generator_using_numpy.csv'
        if not csv_path.exists():
            return
        df = pd.read_csv(csv_path)
        # Prefer the features the model expects
        cols = list(features_meta) if features_meta else df.columns.tolist()
        # Keep only columns present in df
        cols = [c for c in cols if c in df.columns]
        num_df = df[cols].select_dtypes(include=['number']).copy()
        baseline_columns = list(num_df.columns)
        stats = {}
        for c in baseline_columns:
            s = num_df[c].dropna()
            if s.shape[0] == 0:
                continue
            stats[c] = {
                'mean': float(s.mean()),
                'std': float(max(s.std(ddof=0), 1e-8)),
            }
        baseline_stats = stats
    except Exception:
        # Non-fatal if baseline cannot be loaded
        baseline_stats = {}
        baseline_columns = []

try_load_baseline()

# ---- Feature importance (global) ----
def compute_feature_importances(top_n: int = 20):
    """Best-effort extraction of global feature importances.
    - If the core model exposes feature_importances_, use that.
    - If a preprocessor exists and exposes feature names, aggregate back to original columns by substring match.
    - Fallback to positional feature names when mapping is ambiguous.
    """
    try:
        core = model_core or model_pipeline
        if not hasattr(core, 'feature_importances_'):
            return []

        importances = getattr(core, 'feature_importances_', None)
        if importances is None:
            return []

        importances = np.array(importances).ravel()

        # Attempt to get output feature names from preprocessor
        out_feature_names = None
        if preprocessor is not None:
            try:
                out_feature_names = preprocessor.get_feature_names_out()
            except Exception:
                out_feature_names = None

        items = []
        if out_feature_names is not None and len(out_feature_names) == len(importances):
            # Try to aggregate derived features to original columns by substring match
            agg = {}
            base_cols = list(features_meta) if features_meta else []
            for name, imp in zip(out_feature_names, importances):
                name_str = str(name)
                base = None
                # Heuristic: pick the first base column name that appears in the transformed name
                for c in base_cols:
                    if c in name_str:
                        base = c
                        break
                if base is None:
                    # As fallback, try to trim common prefixes like 'num__', 'cat__'
                    parts = name_str.split('__')
                    if len(parts) > 1:
                        base = parts[1]
                    else:
                        base = name_str
                agg[base] = agg.get(base, 0.0) + float(imp)
            items = [{'feature': k, 'importance': float(v)} for k, v in agg.items()]
        else:
            # No mapping available; map positionally if possible
            if features_meta is not None and len(features_meta) == len(importances):
                items = [
                    {'feature': str(col), 'importance': float(imp)}
                    for col, imp in zip(features_meta, importances)
                ]
            else:
                items = [
                    {'feature': f'f{i}', 'importance': float(imp)}
                    for i, imp in enumerate(importances)
                ]

        # Normalize to sum to 1 for readability
        total = sum(abs(x['importance']) for x in items) or 1.0
        for x in items:
            x['importance'] = float(abs(x['importance']) / total)

        items.sort(key=lambda x: x['importance'], reverse=True)
        return items[:top_n]
    except Exception:
        return []

@app.route('/')
def home():
    return "Welcome to the XGBoost Prediction API!"

@app.route('/metrics/baseline', methods=['GET'])
def metrics_baseline():
    return jsonify({
        'columns': baseline_columns,
        'stats': baseline_stats,
    })

@app.route('/metrics/feature-importance', methods=['GET'])
def metrics_feature_importance():
    try:
        top_n = int(request.args.get('top', 10))
    except Exception:
        top_n = 10
    items = compute_feature_importances(top_n=top_n)
    return jsonify({ 'items': items })

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

        # Simple explanations: compute top absolute z-scores for numeric inputs versus baseline
        explanations = []
        try:
            if isinstance(X, pd.DataFrame) and baseline_stats:
                row = X.iloc[0]
                for col, v in row.items():
                    if col in baseline_stats and isinstance(v, (int, float)) and pd.notna(v):
                        m = baseline_stats[col]['mean']
                        s = baseline_stats[col]['std']
                        z = 0.0 if s == 0 else (float(v) - m) / s
                        explanations.append({
                            'feature': col,
                            'value': float(v),
                            'z': float(z)
                        })
                # Sort by absolute z and keep top 5
                explanations.sort(key=lambda x: abs(x['z']), reverse=True)
                explanations = explanations[:5]
        except Exception:
            explanations = []

        return jsonify({
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "probability": proba,
            "explanations": explanations
        })
    except Exception as e:
        # Include error string in response to help debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
