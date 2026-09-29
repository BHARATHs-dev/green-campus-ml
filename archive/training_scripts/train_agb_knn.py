import os
import joblib

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ai.preprocessing.field_preprocessing import prepare_data


DATASET_PATH = "backend/uploads/csv/field_data.csv"

MODEL_DIR = "ai/models"
MODEL_PATH = os.path.join(MODEL_DIR, "agb_knn.joblib")


# Load and preprocess data
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    DATASET_PATH
)

print("\nTraining KNN AGB model...")

# Create model
model = KNeighborsRegressor(
    n_neighbors=5,
    weights="distance",
    metric="minkowski",
    p=2
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n========== AGB KNN RESULTS ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# Create model directory
os.makedirs(MODEL_DIR, exist_ok=True)


# Save model + preprocessing
joblib.dump(
    {
        "model": model,
        "preprocessor": preprocessor
    },
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)
