import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


DATASET_PATH = "backend/uploads/csv/field_data.csv"

FEATURES = [
    "diameter",
    "height",
    "year",
    "group",
    "site"
]

TARGET = "AGB"


# Load
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


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Preprocessing
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


# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        )
    )
])


# Parameters to test
param_grid = {
    "model__n_estimators": [
        100,
        200,
        300
    ],

    "model__max_depth": [
        None,
        5,
        10,
        20
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


cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)


print("\n========== TUNING AGB RANDOM FOREST ==========")

grid_search.fit(
    X_train,
    y_train
)


print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV R²:")
print(grid_search.best_score_)


# Final test evaluation
best_model = grid_search.best_estimator_

y_pred = best_model.predict(
    X_test
)


r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5


print("\n========== FINAL TEST RESULTS ==========")

print(f"R²   : {r2:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
