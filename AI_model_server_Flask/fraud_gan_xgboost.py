"""
fraud_gan_xgboost.py
End-to-end pipeline: GAN for fraud sample synthesis + XGBoost classifier.
Saves a model artifact (dict with pipeline, features, scaler) for Flask API compatibility.
"""
import os
import sys
import math
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, losses, optimizers, Model
except Exception as e:
    print("ERROR: TensorFlow/Keras is required to train the GAN. Install with: pip install tensorflow")
    raise

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import pickle

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "AI_model_Py_Scripts" / "fraud_dataset_Generator_using_numpy.csv"
DEFAULT_OUT = REPO_ROOT / "AI_model_server_Flask" / "best_xgboost_gan_model.pkl"
RNG_SEED = 42
np.random.seed(RNG_SEED)
tf.random.set_seed(RNG_SEED)

LABEL_CANDIDATES = ["label", "Label", "is_fraud", "Class", "target", "fraud"]

def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    for c in df.columns:
        unique_vals = set(df[c].dropna().unique().tolist())
        if unique_vals.issubset({0, 1}) and df[c].nunique() == 2:
            return c
    raise ValueError("Could not infer label column. Please rename your binary label to one of: " + ", ".join(LABEL_CANDIDATES))

def select_feature_columns(df: pd.DataFrame, label_col: str):
    # Use all columns except label
    return [c for c in df.columns if c != label_col]

# --- GAN classes (same as before) ---
class VectorGenerator(Model):
    def __init__(self, noise_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Input(shape=(noise_dim,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(out_dim, activation="tanh"),
        ])
    def call(self, z, training=False):
        return self.net(z, training=training)

class VectorDiscriminator(Model):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Input(shape=(in_dim,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
    def call(self, x, training=False):
        return self.net(x, training=training)

@tf.function
def train_gan_step(G, D, G_opt, D_opt, real_batch, noise_dim):
    batch_size = tf.shape(real_batch)[0]
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as d_tape:
        fake = G(noise, training=True)
        real_logits = D(real_batch, training=True)
        fake_logits = D(fake, training=True)
        d_loss = losses.binary_crossentropy(tf.ones_like(real_logits), real_logits) + \
                 losses.binary_crossentropy(tf.zeros_like(fake_logits), fake_logits)
        d_loss = tf.reduce_mean(d_loss)
    d_grads = d_tape.gradient(d_loss, D.trainable_variables)
    D_opt.apply_gradients(zip(d_grads, D.trainable_variables))
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as g_tape:
        fake = G(noise, training=True)
        fake_logits = D(fake, training=True)
        g_loss = losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits)
        g_loss = tf.reduce_mean(g_loss)
    g_grads = g_tape.gradient(g_loss, G.trainable_variables)
    G_opt.apply_gradients(zip(g_grads, G.trainable_variables))
    return d_loss, g_loss

def train_gan_on_minority(minority_data: np.ndarray,
                          epochs: int = 200,
                          batch_size: int = 256,
                          noise_dim: int = 64,
                          hidden: int = 128,
                          lr: float = 1e-4):
    dataset = tf.data.Dataset.from_tensor_slices(minority_data.astype("float32")).shuffle(4096, seed=RNG_SEED).batch(batch_size, drop_remainder=True)
    feature_dim = minority_data.shape[1]
    G = VectorGenerator(noise_dim, feature_dim, hidden)
    D = VectorDiscriminator(feature_dim, hidden)
    G_opt = optimizers.Adam(lr, beta_1=0.5)
    D_opt = optimizers.Adam(lr, beta_1=0.5)
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for real_batch in dataset:
            d_loss, g_loss = train_gan_step(G, D, G_opt, D_opt, real_batch, noise_dim)
            d_losses.append(float(d_loss.numpy()))
            g_losses.append(float(g_loss.numpy()))
        if epoch % max(1, epochs // 10) == 0:
            print(f"[GAN] Epoch {epoch:4d}/{epochs}  D_loss={np.mean(d_losses):.4f}  G_loss={np.mean(g_losses):.4f}")
    return G

def synthesize(G: VectorGenerator, n: int, noise_dim: int = 64) -> np.ndarray:
    batches = math.ceil(n / 512)
    outs = []
    for _ in range(batches):
        bs = min(512, n - len(outs) * 512)
        if bs <= 0:
            break
        z = tf.random.normal([bs, noise_dim])
        x = G(z, training=False).numpy()
        outs.append(x)
    return np.vstack(outs)

# --- Pipeline: load -> augment -> train XGBoost -> save ---
def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.shape[0] < 100 or df.shape[1] < 2:
        raise ValueError("Dataset appears too small or malformed.")
    return df

def prepare_data(df: pd.DataFrame):
    label_col = find_label_column(df)
    feat_cols = select_feature_columns(df, label_col)
    X = df[feat_cols].copy()
    y = df[label_col].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG_SEED)
    # Preprocessing: scale numeric, encode categoricals
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    return (X_train, y_train, X_test, y_test, X_train_proc, X_test_proc, preprocessor, feat_cols, label_col)

def augment_with_gan(X_train_proc: np.ndarray, y_train: np.ndarray,
                     epochs: int = 200, noise_dim: int = 64) -> np.ndarray:
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0:
        raise ValueError("No positive (fraud) samples found in training set.")
    if n_neg == 0:
        print("WARNING: No negative samples found; skipping augmentation.")
        return X_train_proc, y_train
    minority_data = X_train_proc[pos_idx]
    print(f"Minority samples: {n_pos}, Majority samples: {n_neg}")
    G = train_gan_on_minority(minority_data, epochs=epochs, noise_dim=noise_dim)
    n_to_gen = max(0, n_neg - n_pos)
    if n_to_gen == 0:
        print("Classes are balanced; no synthesis needed.")
        return X_train_proc, y_train
    print(f"Synthesizing {n_to_gen} fraud-like samples via GAN...")
    X_syn = synthesize(G, n_to_gen, noise_dim=noise_dim)
    X_aug = np.vstack([X_train_proc, X_syn])
    y_aug = np.concatenate([y_train, np.ones(len(X_syn), dtype=int)])
    return X_aug, y_aug

def train_xgboost(X_train_proc: np.ndarray, y_train: np.ndarray) -> Pipeline:
    clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=RNG_SEED,
        n_jobs=-1,
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2
    )
    # Wrap in pipeline for API compatibility (preprocessor is identity here)
    pipe = Pipeline([
        ('identity', 'passthrough'),
        ('clf', clf)
    ])
    pipe.fit(X_train_proc, y_train)
    return pipe

def evaluate_model(clf: Pipeline, X_test_proc: np.ndarray, y_test: np.ndarray):
    y_pred = clf.predict(X_test_proc)
    try:
        y_proba = clf.predict_proba(X_test_proc)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        y_proba = None
        auc = None
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")

def save_model(clf: Pipeline, preprocessor, feat_cols: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": clf,
        "preprocessor": preprocessor,
        "features": feat_cols
    }
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSaved model artifact to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Train GAN+XGBoost fraud detector and export a pickle model.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=200, help="GAN training epochs")
    parser.add_argument("--noise_dim", type=int, default=64, help="GAN noise dimension")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output pickle path")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    print(f"Loading dataset: {csv_path}")
    df = load_dataset(csv_path)
    X_train, y_train, X_test, y_test, X_train_proc, X_test_proc, preprocessor, feat_cols, label_col = prepare_data(df)
    print(f"Detected label column: {label_col}")
    print(f"Feature count: {len(feat_cols)}")
    X_aug, y_aug = augment_with_gan(X_train_proc, y_train.values, epochs=args.epochs, noise_dim=args.noise_dim)
    clf = train_xgboost(X_aug, y_aug)
    evaluate_model(clf, X_test_proc, y_test.values)
    save_model(clf, preprocessor, feat_cols, out_path)
    # Quick smoke prediction
    try:
        smoke_feats = X_test_proc[0].reshape(1, -1)
        result = clf.predict(smoke_feats)
        print(f"\nSmoke test prediction on one sample: {result}")
    except Exception as e:
        print(f"Smoke test skipped: {e}")

if __name__ == "__main__":
    main()
