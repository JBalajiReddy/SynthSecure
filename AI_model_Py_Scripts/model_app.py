# Imports and Environment Setup
import sys
import subprocess

# Try to ensure xgboost is available (optional)
try:
    import xgboost as xgb  # noqa: F401
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    except Exception as e:
        print("Warning: Failed to install xgboost, proceeding if already available:", e)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

print("Libraries imported.")

# Load dataset robustly
from pathlib import Path

candidate_paths = [
    Path('fraud_dataset_Generator_using_numpy.csv'),
    Path('../AI_model_Py_Scripts/fraud_dataset_Generator_using_numpy.csv'),
    Path('./fraud_dataset.csv'),
]

csv_path = None
for p in candidate_paths:
    if p.exists():
        csv_path = p
        break

if csv_path is None:
    raise FileNotFoundError("Could not find dataset in candidate paths: " + ", ".join(map(str, candidate_paths)))

data = pd.read_csv(csv_path)
print("Loaded:", csv_path)
print(data.head())
print(data.info())
print(data.describe())
if 'Label' in data.columns:
    print(data['Label'].value_counts())
else:
    raise KeyError("Expected 'Label' column in dataset.")

# Preprocess: scale numerical, one-hot encode categorical

numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns.drop('Label', errors='ignore')
categorical_cols = data.select_dtypes(include=['object', 'category']).columns

scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

print("Preprocessed head:\n", data.head())

# Split X/y
X = data.drop('Label', axis=1)
y = data['Label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print("Class distribution (train):\n", y_train.value_counts())
print("Class distribution (test):\n", y_test.value_counts())

# Build simple GAN (generator, discriminator, gan)

latent_dim = 100
input_dim = X_train.shape[1]

# Generator
def build_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(0.2))
    model.add(Dense(output_dim, activation='sigmoid'))
    return model

# Discriminator
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Compose GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.models.Model(gan_input, gan_output)
    return gan

generator = build_generator(latent_dim, input_dim)
discriminator = build_discriminator(input_dim)

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
    metrics=['accuracy']
)

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

print("GAN built. Input dim:", input_dim)

# Train GAN
import numpy as np

epochs = 200
batch_size = 64
half_batch = batch_size // 2


def train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim):
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        real_data = X_train.iloc[idx].values.astype(np.float32)

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        synthetic_data = generator.predict(noise, verbose=0)

        real_labels = np.random.uniform(0.9, 1.0, (half_batch, 1)).astype(np.float32)
        fake_labels = np.random.uniform(0.0, 0.1, (half_batch, 1)).astype(np.float32)

        real_data += np.random.normal(0, 0.01, real_data.shape)

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)

        # Train Generator (via GAN)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 50 == 0:
            sample_noise = np.random.normal(0, 1, (5, latent_dim))
            gen_samples = generator.predict(sample_noise, verbose=0)
            print(f"Epoch {epoch}: D_real={d_loss_real[0]:.4f}, D_fake={d_loss_fake[0]:.4f}, G={g_loss:.4f}, range=({gen_samples.min():.3f},{gen_samples.max():.3f})")

train_gan(generator, discriminator, gan, X_train, epochs, batch_size, latent_dim)
print("GAN training complete.")

# Generate synthetic samples equal to |X_train|
num_samples = X_train.shape[0]
noise = np.random.normal(0, 1, (num_samples, latent_dim))
synthetic_data = generator.predict(noise, verbose=0).astype(np.float32)
print("Synthetic generated:", synthetic_data.shape)

# Visual diagnostics: histograms and KDE for a few features
plt.figure(figsize=(12, 5))
num_features = min(2, X_train.shape[1])
real_subset = X_train.iloc[:, :num_features].values
synth_subset = synthetic_data[:, :num_features]

for i in range(num_features):
    plt.subplot(1, num_features, i + 1)
    plt.hist(real_subset[:, i], bins=40, alpha=0.5, label='Real', color='blue')
    plt.hist(synth_subset[:, i], bins=40, alpha=0.5, label='Synthetic', color='red')
    sns.kdeplot(real_subset[:, i], color='blue', label='Real KDE', fill=True, alpha=0.2)
    sns.kdeplot(synth_subset[:, i], color='red', label='Synthetic KDE', fill=True, alpha=0.2)
    plt.title(f"Feature {i+1} Distribution")
    plt.legend()
plt.tight_layout()
plt.show()


# Statistical comparison: means and variances
real_mean = np.mean(X_train, axis=0)
real_var = np.var(X_train, axis=0)
synth_mean = np.mean(synthetic_data, axis=0)
synth_var = np.var(synthetic_data, axis=0)
print("Mean diff (|real - synth|) sample:", np.abs(real_mean - synth_mean)[:5])
print("Var  diff (|real - synth|) sample:", np.abs(real_var - synth_var)[:5])

# Build augmented dataset and labels
num_synth = synthetic_data.shape[0]
fraud_ratio = float(np.mean(y_train))
num_fraud_samples = int(fraud_ratio * num_synth)

synthetic_labels = np.zeros((num_synth, 1), dtype=np.int32)
synthetic_labels[:num_fraud_samples] = 1

# y_train to 2D
y_train_2d = y_train.values.reshape(-1, 1)

X_train_augmented = np.concatenate((X_train.values, synthetic_data), axis=0)
y_train_augmented = np.concatenate((y_train_2d, synthetic_labels), axis=0)

print("Augmented shapes:", X_train_augmented.shape, y_train_augmented.shape)

# Compute scale_pos_weight = N_neg / N_pos
pos = int(y_train_augmented.sum())
neg = int(y_train_augmented.shape[0] - pos)
scale_pos_weight = (neg / max(1, pos))
print(f"scale_pos_weight: {scale_pos_weight:.3f}")

# Train/Validation split for augmented data
y_aug_1d = y_train_augmented.ravel()
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_augmented, y_aug_1d, test_size=0.2, random_state=42, stratify=y_aug_1d
)
print("Train/Val shapes:", X_train_split.shape, X_val_split.shape, y_train_split.shape, y_val_split.shape)


# Install/Import XGBoost (idempotent)
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except Exception as e:
    print("If xgboost import fails, please install it via pip.", e)
    raise


# Baseline XGBoost training
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
)

xgb_clf.fit(X_train_split, y_train_split)

y_pred_val = xgb_clf.predict(X_val_split)
acc = accuracy_score(y_val_split, y_pred_val)
print(f"Baseline XGBoost Accuracy: {acc * 100:.2f}%")


# Hyperparameter tuning with GridSearchCV (optional - can be time-consuming)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'reg_alpha': [0.0, 0.1],
    'reg_lambda': [1.0, 2.0],
}

base = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    scale_pos_weight=scale_pos_weight,
)

gs = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
)

gs.fit(X_train_split, y_train_split)
print("Best params:", gs.best_params_)
print("Best CV ROC-AUC:", gs.best_score_)

best_xgb = gs.best_estimator_
best_xgb.fit(X_train_split, y_train_split)
print("Refit best XGB on training split.")


# Evaluation metrics and plots (using best_xgb if available, else xgb_clf)
model = globals().get('best_xgb', None) or xgb_clf

# Predictions and probabilities
y_pred = model.predict(X_val_split)
y_proba = model.predict_proba(X_val_split)[:, 1]

acc = accuracy_score(y_val_split, y_pred)
prec = precision_score(y_val_split, y_pred)
rec = recall_score(y_val_split, y_pred)
f1 = f1_score(y_val_split, y_pred)
auc = roc_auc_score(y_val_split, y_proba)

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"AUC-ROC: {auc:.3f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val_split, y_proba)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.3f})', color='blue')
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val_split, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud','Fraud'], yticklabels=['Not Fraud','Fraud'])
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save XGBoost model as xg_model.pkl
import pickle

final_model = globals().get('best_xgb', None) or xgb_clf
with open('xg_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Model saved as 'xg_model.pkl'")

# Optional: auto-download in Colab
try:
    from google.colab import files
    files.download('xg_model.pkl')
except Exception:
    pass
