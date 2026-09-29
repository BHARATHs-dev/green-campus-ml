import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


FEATURES = [
    "diameter",
    "height",
    "year",
    "group",
    "site"
]

TARGET = "AGB"


def prepare_data(path):

    df = pd.read_csv(path)

    print("Original dataset:", df.shape)

    df = df[FEATURES + [TARGET]]

    df = df.dropna()

    df = df[
        (df["diameter"] > 0) &
        (df["height"] > 0) &
        (df["AGB"] >= 0)
    ]

    print("Clean dataset:", df.shape)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    numerical_features = [
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
                numerical_features
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

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )
