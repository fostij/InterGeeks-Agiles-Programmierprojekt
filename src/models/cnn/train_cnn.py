# ---------------------------------------------------------------------
# train_cnn.py — CNN-Training zur Klassifikation der Schadensschwere
# ---------------------------------------------------------------------

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------------------------------------------------------------
# Konfiguration - zentral an einer Stelle
# ---------------------------------------------------------------------
DATA_DIR     = Path("data/raw/car_damage")
TRAIN_DIR    = DATA_DIR / "training" 
VAL_DIR      = DATA_DIR / "validation" 
from src.models.cnn.cnn_config import MODEL_PATH, CLASSES_PATH, IMAGE_SIZE
BATCH_SIZE   = 32
EPOCHS       = 10
SEED         = 42


def load_datasets():
    """Lädt Trainings- und Validierungsbilder aus den jeweils vorgegebenen
    Ordnern (training/ und validation/). Der Datensatz ist bereits geteilt,
    daher wird KEIN validation_split verwendet.
    Gibt (train_ds, val_ds, class_names) zurück."""

    # Trainingsdaten aus dem Ordner training/ (Klassen = Unterordner)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    # Validierungsdaten aus dem Ordner validation/
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names

    # Performance: Daten im Speicher zwischenpuffern (schnelleres Training)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    """Baut das Transfer-Learning-Modell:
    Augmentation -> MobileNetV2 (eingefroren) -> Klassifikationskopf."""

    # --- Data Augmentation: kuenstliche Varianten gegen Overfitting ---
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    # --- Vortrainierte Basis laden, OHNE den originalen Klassenkopf ---
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), 
        include_top=False, 
        weights="imagenet",
    )
    base_model.trainable = False 

    # --- Gesamtes Modell zusammensetzen ---
    inputs = layers.Input(shape=IMAGE_SIZE + (3,))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)      
    x = layers.Dropout(0.2)(x)                 
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  
        metrics=["accuracy"],
    )
    return model


def train() -> None:
    """Kompletter Trainingslauf: Daten laden, Modell bauen, trainieren,
    speichern."""
    train_ds, val_ds, class_names = load_datasets()
    print(f"Classes: {class_names}")

    model = build_model(len(class_names))
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Modell und Klassennamen speichern (Klassen brauchen wir spaeter
    # in predict_image.py, um Index -> Name zuzuordnen)
    model.save(MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(class_names, ensure_ascii=False))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Classes saved: {CLASSES_PATH}")


if __name__ == "__main__":
    train()

