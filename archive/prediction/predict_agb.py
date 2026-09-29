import joblib
import pandas as pd


MODEL_PATH = "ai/models/final_agb_model.joblib"

PREPROCESSOR_PATH = (
    "ai/models/final_agb_preprocessor.joblib"
)


FEATURES = [
    "diameter",
    "height",
    "year",
    "group",
    "site"
]


# Load model
model = joblib.load(MODEL_PATH)

# Load preprocessor
preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


def predict_agb(
    diameter,
    height,
    year,
    group,
    site
):

    data = pd.DataFrame([
        {
            "diameter": diameter,
            "height": height,
            "year": year,
            "group": group,
            "site": site
        }
    ])

    # Preprocess
    processed_data = preprocessor.transform(
        data[FEATURES]
    )

    # Prediction
    prediction = model.predict(
        processed_data
    )

    return float(prediction[0])


if __name__ == "__main__":

    result = predict_agb(
        diameter=20,
        height=12,
        year=2024,
        group="fruit",
        site="site1"
    )

    print("Predicted AGB:", result)
