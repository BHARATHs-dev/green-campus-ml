import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "backend", "uploads", "csv", "field_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ai", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "catboost_agb_model.cbm")

PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "catboost_agb_preprocessor.joblib")

METADATA_PATH = os.path.join(MODEL_DIR, "catboost_agb_metadata.json")

FEATURES = ["diameter", "height", "year", "group", "site"]

TARGET = "AGB"

CATEGORICAL_FEATURES = ["group", "site"]

RANDOM_STATE = 42


# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(DATASET_PATH)

df = df[FEATURES + [TARGET]]

df = df.dropna()

df = df[
    (df["diameter"] > 0) &
    (df["height"] > 0) &
    (df["AGB"] >= 0)
]

print("AGB dataset after filtering:", df.shape)


# -------------------------
# DATA LEAKAGE DOCUMENTATION
# -------------------------
# CRITICAL FINDING:
# AGB in this dataset is mathematically derived from diameter using
# a power-law relationship approximately:
#
#     AGB = 0.1208 * diameter^1.98
#
# This means AGB is DIRECTLY CALCULATED from diameter, not independently
# measured. Training a model on this data will essentially learn this
# formula, producing artificially high R² values that do NOT represent
# real-world prediction accuracy on independently measured biomass.
#
# The model should therefore be understood as learning the existing
# allometric relationship embedded in the dataset, not as an independent
# biomass predictor.
# -------------------------


X = df[FEATURES]
y = df[TARGET]


# -------------------------
# Train/test split
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# -------------------------
# CatBoost Pools
# -------------------------

categorical_indices = [
    FEATURES.index(f) for f in CATEGORICAL_FEATURES
]

train_pool = Pool(
    X_train, y_train, cat_features=categorical_indices
)

test_pool = Pool(
    X_test, y_test, cat_features=categorical_indices
)


# -------------------------
# CatBoost Regressor
# -------------------------

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    loss_function="RMSE",
    eval_metric="RMSE",
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)


print("\nTraining CatBoost AGB model...")

model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=100,
)


# -------------------------
# Evaluation on test set
# -------------------------

y_pred = model.predict(test_pool)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== TEST SET METRICS ===")
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


# -------------------------
# Cross-validation
# -------------------------

print("\nRunning 5-fold cross-validation...")

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_cv_train = X.iloc[train_idx]
    y_cv_train = y.iloc[train_idx]
    X_cv_val = X.iloc[val_idx]
    y_cv_val = y.iloc[val_idx]

    cv_train_pool = Pool(
        X_cv_train, y_cv_train, cat_features=categorical_indices
    )
    cv_val_pool = Pool(
        X_cv_val, y_cv_val, cat_features=categorical_indices
    )

    cv_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    cv_model.fit(
        cv_train_pool,
        eval_set=cv_val_pool,
        verbose=0,
    )

    cv_pred = cv_model.predict(cv_val_pool)
    cv_r2 = r2_score(y_cv_val, cv_pred)
    cv_scores.append(cv_r2)
    print(f"  Fold {fold + 1}: R² = {cv_r2:.4f}")

cv_r2_mean = np.mean(cv_scores)
cv_r2_std = np.std(cv_scores)

print(f"\nCV R²: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")


# -------------------------
# Feature importance
# -------------------------

importance = model.get_feature_importance()
feature_importance = {
    f: float(importance[i]) for i, f in enumerate(FEATURES)
}

print("\n=== FEATURE IMPORTANCE ===")
for f, imp in sorted(
    feature_importance.items(), key=lambda x: x[1], reverse=True
):
    print(f"  {f}: {imp:.4f}")


# -------------------------
# Save model
# -------------------------

os.makedirs(MODEL_DIR, exist_ok=True)

model.save_model(MODEL_PATH)

print(f"\nModel saved: {MODEL_PATH}")


# -------------------------
# Save metadata
# -------------------------

metadata = {
    "model_type": "CatBoostRegressor",
    "model_version": "catboost-agb-v1",
    "target": TARGET,
    "features": FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "random_state": RANDOM_STATE,
    "training_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "total_samples": int(X.shape[0]),
    "metrics": {
        "test_r2": round(r2, 4),
        "test_mae": round(mae, 4),
        "test_rmse": round(rmse, 4),
        "cv_r2_mean": round(cv_r2_mean, 4),
        "cv_r2_std": round(cv_r2_std, 4),
        "cv_r2_folds": [round(s, 4) for s in cv_scores],
    },
    "feature_importance": feature_importance,
    "data_leakage_warning": {
        "detected": True,
        "description": (
            "AGB is mathematically derived from diameter using a "
            "power-law relationship (AGB = 0.1208 * diameter^1.98). "
            "The model learns this embedded formula rather than "
            "predicting independently measured biomass. High R² values "
            "reflect this derivation, not independent prediction accuracy."
        ),
    },
    "validation_method": "5-Fold Cross Validation + Hold-out Test Set",
    "parameters": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "RMSE",
        "early_stopping_rounds": 50,
    },
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {METADATA_PATH}")

print("\nCatBoost AGB training complete.")
