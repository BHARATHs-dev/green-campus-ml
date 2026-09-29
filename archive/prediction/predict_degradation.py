import joblib
import pandas as pd


MODEL_PATH = (
    "ai/models/final_degradation_model.joblib"
)

SCALER_PATH = (
    "ai/models/final_degradation_preprocessor.joblib"
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


# Load model
model = joblib.load(
    MODEL_PATH
)

# Load scaler
scaler = joblib.load(
    SCALER_PATH
)


def predict_degradation(data):

    input_data = pd.DataFrame(
        [data]
    )

    input_data = input_data[
        FEATURES
    ]

    # Scale
    scaled_data = scaler.transform(
        input_data
    )

    # Prediction
    prediction = model.predict(
        scaled_data
    )[0]

    # Probability
    probabilities = model.predict_proba(
        scaled_data
    )[0]

    confidence = max(
        probabilities
    )

    return {
        "prediction": str(prediction),
        "confidence": float(confidence)
    }


if __name__ == "__main__":

    sample = {
        "NDVI": 0.72,
        "SAVI": 0.61,
        "RVI": 18.0,
        "ARI": 8.09,
        "MSI": 0.24,
        "PRI": 0.45,
        "Canopy_Cover_Percent": 84,
        "Vegetation_Cover_Percent": 80,
        "Species_Richness_Count": 12,
        "Soil_Moisture_Percent": 42,
        "Soil_Organic_Carbon_Percent": 2.5,
        "Erosion_Risk_Index": 20,
        "Human_Disturbance_Index": 15,
        "Ecosystem_Stability_Score": 80
    }

    result = predict_degradation(
        sample
    )

    print(
        "Prediction:",
        result["prediction"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )
