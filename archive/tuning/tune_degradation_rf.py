import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


DATASET_PATH = (
    "backend/uploads/csv/"
    "vegetation_ecological_degradation_dataset_6000.csv"
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


df = pd.read_csv(DATASET_PATH)

df = df[FEATURES + [TARGET]]

df = df.dropna()


X = df[FEATURES]
y = df[TARGET]


# Split into 80/20 train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


param_grid = {

    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        10,
        20,
        30
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)


print(
    "\n======= TUNING DEGRADATION RANDOM FOREST ======="
)


grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Macro F1:")
print(grid_search.best_score_)


# Final evaluation on held-out 20% test set
best_model = grid_search.best_estimator_

y_pred = best_model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n========== FINAL TEST RESULTS ==========")

print(f"Accuracy   : {accuracy:.4f}")
print(f"Macro F1   : {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)
