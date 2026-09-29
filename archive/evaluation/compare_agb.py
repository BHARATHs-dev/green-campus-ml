import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = "backend/uploads/csv/field_data.csv"

FEATURES = [
    "diameter",
    "height",
    "year",
    "group",
    "site"
]

TARGET = "AGB"


# Load dataset
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


# Preprocessor
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


# Models
models = {

    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="distance"
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Neural Network": MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )
}


# 5-fold CV
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("\n========== AGB 5-FOLD CROSS VALIDATION ==========")

results = []


for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        },
        n_jobs=-1
    )

    r2 = scores["test_r2"]

    mae = -scores["test_mae"]

    rmse = -scores["test_rmse"]

    results.append({
        "Model": name,
        "Mean R2": r2.mean(),
        "Std R2": r2.std(),
        "Mean MAE": mae.mean(),
        "Mean RMSE": rmse.mean()
    })


results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))
