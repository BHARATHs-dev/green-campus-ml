import os
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ai.preprocessing.field_preprocessing import prepare_data


DATASET_PATH = "backend/uploads/csv/field_data.csv"

MODEL_DIR = "ai/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "agb_neural_network.keras"
)


# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Load data
X_train, X_test, y_train, y_test, preprocessor = prepare_data(
    DATASET_PATH
)

print("\nTraining AGB Neural Network...")
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Build model
model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),

    Dense(16, activation="relu"),
    Dropout(0.2),

    Dense(8, activation="relu"),

    Dense(1)
])


# Compile
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# Early stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)


# Train
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)


# Prediction
y_pred = model.predict(
    X_test,
    verbose=0
).flatten()


# Evaluation
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n========== AGB NEURAL NETWORK RESULTS ==========")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# Save model
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)

# Save preprocessing separately
joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "agb_nn_preprocessor.joblib"
    )
)

print("\nModel saved:")
print(MODEL_PATH)
