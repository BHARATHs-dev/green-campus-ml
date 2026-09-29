import os
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from ai.preprocessing.vegetation_preprocessing import prepare_data


DATASET_PATH = (
    "backend/uploads/csv/"
    "vegetation_ecological_degradation_dataset_6000.csv"
)

MODEL_DIR = "ai/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "degradation_neural_network.keras"
)


np.random.seed(42)
tf.random.set_seed(42)


# Load data
X_train, X_test, y_train, y_test, scaler = prepare_data(
    DATASET_PATH
)


print("\nTraining Degradation Neural Network...")

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


# Encode target classes
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(
    y_train
)

y_test_encoded = label_encoder.transform(
    y_test
)


num_classes = len(
    label_encoder.classes_
)


print("\nClasses:")
print(label_encoder.classes_)


# Build model
model = Sequential([
    Dense(
        64,
        activation="relu",
        input_shape=(X_train.shape[1],)
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        num_classes,
        activation="softmax"
    )
])


# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
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
    y_train_encoded,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# Prediction
probabilities = model.predict(
    X_test,
    verbose=0
)

y_pred_encoded = np.argmax(
    probabilities,
    axis=1
)


# Evaluation
accuracy = accuracy_score(
    y_test_encoded,
    y_pred_encoded
)


print("\n======= DEGRADATION NEURAL NETWORK RESULTS =======")

print(f"Accuracy: {accuracy:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test_encoded,
        y_pred_encoded,
        target_names=label_encoder.classes_
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test_encoded,
        y_pred_encoded
    )
)


# Save model
os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(MODEL_PATH)


# Save preprocessing
joblib.dump(
    {
        "scaler": scaler,
        "label_encoder": label_encoder
    },
    os.path.join(
        MODEL_DIR,
        "degradation_nn_preprocessor.joblib"
    )
)


print("\nModel saved:")
print(MODEL_PATH)
