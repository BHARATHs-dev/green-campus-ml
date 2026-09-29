import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from ai.preprocessing.vegetation_preprocessing import prepare_data


DATASET_PATH = (
    "backend/uploads/csv/"
    "vegetation_ecological_degradation_dataset_6000.csv"
)

MODEL_DIR = "ai/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "degradation_random_forest.joblib"
)


# Load data
X_train, X_test, y_train, y_test, scaler = prepare_data(
    DATASET_PATH
)

print("\nTraining Random Forest degradation model...")


# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluation
accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======= DEGRADATION RANDOM FOREST RESULTS =======")

print(f"Accuracy: {accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# Save
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "scaler": scaler
    },
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)
