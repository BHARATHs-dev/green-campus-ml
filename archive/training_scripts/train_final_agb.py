import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = "backend/uploads/csv/field_data.csv"

MODEL_DIR = "ai/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_agb_model.joblib"
)

PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "final_agb_preprocessor.joblib"
)


FEATURES = [
    "diameter",
    "height",
    "year",
    "group",
    "site"
]

TARGET = "AGB"


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


X = df[FEATURES]
y = df[TARGET]


print("Final AGB dataset:", df.shape)


# -------------------------
# Preprocessing
# -------------------------

numeric_features = [
    "diameter",
    "height",
    "year"
]

categorical_features = [
    "group",
    "site"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


X_processed = preprocessor.fit_transform(X)


# -------------------------
# Final Random Forest
# -------------------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


print("\nTraining final AGB model...")

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
    preprocessor,
    PREPROCESSOR_PATH
)


print("\nFinal AGB model saved:")
print(MODEL_PATH)

print("\nPreprocessor saved:")
print(PREPROCESSOR_PATH)
