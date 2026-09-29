import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


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


# Load
df = pd.read_csv(DATASET_PATH)

df = df[FEATURES + [TARGET]]

df = df.dropna()

X = df[FEATURES]

y = df[TARGET]


# Models
models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=500,
        random_state=42
    )
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print(
    "\n======= DEGRADATION 5-FOLD CROSS VALIDATION ======="
)


results = []


for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "f1": "f1_macro"
        },
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Mean Accuracy": scores["test_accuracy"].mean(),
        "Std Accuracy": scores["test_accuracy"].std(),
        "Mean Macro F1": scores["test_f1"].mean()
    })


results_df = pd.DataFrame(results)

print("\n")
print(results_df.to_string(index=False))
