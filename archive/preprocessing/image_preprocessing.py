import os
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


def build_datasets(data_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    data_dir = Path(data_dir)

    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not validation_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected train/validation/test folders under {data_dir}"
        )

    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(class_names)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_dataset = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        seed=SEED,
    )

    validation_dataset = image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        seed=SEED,
    )

    test_dataset = image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        seed=SEED,
    )

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names, num_classes


def get_class_weights(train_dir):
    class_names = sorted([d.name for d in Path(train_dir).iterdir() if d.is_dir()])
    total = sum(
        len(list((Path(train_dir) / c).glob("*.*"))) for c in class_names
    )
    class_weights = {}
    for idx, name in enumerate(class_names):
        count = len(list((Path(train_dir) / name).glob("*.*")))
        class_weights[idx] = total / (len(class_names) * count)
    return class_weights
