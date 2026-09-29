import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def prepare_data(path):

    df = pd.read_csv(path)

    print("Original dataset:", df.shape)

    df = df[FEATURES + [TARGET]]

    df = df.dropna()

    print("Clean dataset:", df.shape)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(X_train)

    X_test_processed = scaler.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        scaler
    )
