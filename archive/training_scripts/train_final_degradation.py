import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


DATASET_PATH = (
    "backend/uploads/csv/"
    "vegetation_ecological_degradation_dataset_6000.csv"
)

MODEL_DIR = "ai/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_degradation_model.joblib"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "final_degradation_preprocessor.joblib"
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
    "Ecosystem_Stability_Score"
]

TARGET = "Degradation_Level"


# -------------------------
# Load
# -------------------------

df = pd.read_csv(
    DATASET_PATH
)

df = df[
    FEATURES + [TARGET]
]

df = df.dropna()


X = df[FEATURES]
y = df[TARGET]


print(
    "Final degradation dataset:",
    df.shape
)


# -------------------------
# Scaling
# -------------------------

scaler = StandardScaler()

X_processed = scaler.fit_transform(X)


# -------------------------
# Final Random Forest
# -------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


print(
    "\nTraining final degradation model..."
)


model.fit(
    X_processed,
    y
)


# -------------------------
# Save
# -------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print(
    "\nFinal degradation model saved:"
)

print(MODEL_PATH)

print(
    "\nPreprocessor saved:"
)

print(SCALER_PATH)
