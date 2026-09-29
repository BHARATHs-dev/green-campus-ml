import os
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = os.path.join(BASE_DIR, "backend", "uploads", "csv", "vegetation_ecological_degradation_dataset_6000.csv")
MODEL_DIR = os.path.join(BASE_DIR, "ai", "models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "catboost_vegetation_model.cbm"
)

METADATA_PATH = os.path.join(
    MODEL_DIR, "catboost_vegetation_metadata.json"
)

FEATURES = [
    "NDVI",
    "SAVI",
    "RVI",
    "ARI",
    "MSI",
    "PRI",
    "Canopy_Cover_Percent",
    "Vegetation_Cover_Percent",
    "Species_Richness_Count",
    "Soil_Moisture_Percent",
    "Soil_Organic_Carbon_Percent",
    "Erosion_Risk_Index",
    "Human_Disturbance_Index",
    "Ecosystem_Stability_Score",
]

TARGET = "Degradation_Level"

RANDOM_STATE = 42


# -------------------------
# Load dataset
# -------------------------

df = pd.read_csv(DATASET_PATH)

df = df[FEATURES + [TARGET]]

df = df.dropna()

print("Vegetation dataset:", df.shape)
print("\nTarget distribution:")
print(df[TARGET].value_counts())


X = df[FEATURES]
y = df[TARGET]


# -------------------------
# Train/test split (stratified)
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# -------------------------
# CatBoost Pools
# -------------------------

train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)


# -------------------------
# CatBoost Classifier
# -------------------------

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
)


print("\nTraining CatBoost vegetation model...")

model.fit(
    train_pool,
    eval_set=test_pool,
    verbose=100,
)


# -------------------------
# Evaluation on test set
# -------------------------

y_pred = model.predict(test_pool).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

cm = confusion_matrix(y_test, y_pred)

print("\n=== TEST SET METRICS ===")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision (macro): {precision_macro:.4f}")
print(f"Recall (macro):    {recall_macro:.4f}")
print(f"F1 (macro):        {f1_macro:.4f}")
print(f"F1 (weighted):     {f1_weighted:.4f}")

print("\n=== CONFUSION MATRIX ===")
print(cm)

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))


# -------------------------
# Cross-validation
# -------------------------

print("\nRunning 5-fold stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=RANDOM_STATE
)

cv_accuracies = []
cv_f1_macros = []
cv_f1_weighteds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_cv_train = X.iloc[train_idx]
    y_cv_train = y.iloc[train_idx]
    X_cv_val = X.iloc[val_idx]
    y_cv_val = y.iloc[val_idx]

    cv_train_pool = Pool(X_cv_train, y_cv_train)
    cv_val_pool = Pool(X_cv_val, y_cv_val)

    cv_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        verbose=0,
        early_stopping_rounds=50,
        use_best_model=True,
    )

    cv_model.fit(
        cv_train_pool,
        eval_set=cv_val_pool,
        verbose=0,
    )

    cv_pred = cv_model.predict(cv_val_pool).flatten()
    cv_acc = accuracy_score(y_cv_val, cv_pred)
    cv_f1_mac = f1_score(y_cv_val, cv_pred, average="macro")
    cv_f1_w = f1_score(y_cv_val, cv_pred, average="weighted")

    cv_accuracies.append(cv_acc)
    cv_f1_macros.append(cv_f1_mac)
    cv_f1_weighteds.append(cv_f1_w)

    print(
        f"  Fold {fold + 1}: "
        f"Accuracy = {cv_acc:.4f}, "
        f"F1 (macro) = {cv_f1_mac:.4f}"
    )

cv_acc_mean = np.mean(cv_accuracies)
cv_acc_std = np.std(cv_accuracies)
cv_f1_macro_mean = np.mean(cv_f1_macros)
cv_f1_macro_std = np.std(cv_f1_macros)
cv_f1_weighted_mean = np.mean(cv_f1_weighteds)
cv_f1_weighted_std = np.std(cv_f1_weighteds)

print(f"\nCV Accuracy:        {cv_acc_mean:.4f} (+/- {cv_acc_std:.4f})")
print(f"CV F1 (macro):      {cv_f1_macro_mean:.4f} (+/- {cv_f1_macro_std:.4f})")
print(f"CV F1 (weighted):   {cv_f1_weighted_mean:.4f} (+/- {cv_f1_weighted_std:.4f})")


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
    "model_type": "CatBoostClassifier",
    "model_version": "catboost-vegetation-v1",
    "target": TARGET,
    "classes": model.classes_.tolist(),
    "features": FEATURES,
    "random_state": RANDOM_STATE,
    "training_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "total_samples": int(X.shape[0]),
    "metrics": {
        "test_accuracy": round(accuracy, 4),
        "test_precision_macro": round(precision_macro, 4),
        "test_recall_macro": round(recall_macro, 4),
        "test_f1_macro": round(f1_macro, 4),
        "test_f1_weighted": round(f1_weighted, 4),
        "cv_accuracy_mean": round(cv_acc_mean, 4),
        "cv_accuracy_std": round(cv_acc_std, 4),
        "cv_f1_macro_mean": round(cv_f1_macro_mean, 4),
        "cv_f1_macro_std": round(cv_f1_macro_std, 4),
        "cv_f1_weighted_mean": round(cv_f1_weighted_mean, 4),
        "cv_f1_weighted_std": round(cv_f1_weighted_std, 4),
        "cv_accuracy_folds": [round(s, 4) for s in cv_accuracies],
        "cv_f1_macro_folds": [round(s, 4) for s in cv_f1_macros],
        "cv_f1_weighted_folds": [round(s, 4) for s in cv_f1_weighteds],
    },
    "confusion_matrix": cm.tolist(),
    "feature_importance": feature_importance,
    "validation_method": "5-Fold Stratified Cross Validation + Hold-out Test Set",
    "parameters": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "loss_function": "MultiClass",
        "early_stopping_rounds": 50,
    },
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Metadata saved: {METADATA_PATH}")

print("\nCatBoost vegetation training complete.")
