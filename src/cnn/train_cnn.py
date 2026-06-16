# =====================================================================
# train_cnn.py — CNN-Training zur Klassifikation der Schadensschwere
# ---------------------------------------------------------------------
# Framework: TensorFlow / Keras.
# Ansatz: TRANSFER LEARNING mit vortrainiertem MobileNetV2.
#   Statt ein CNN von Grund auf zu trainieren (braucht riesige
#   Datenmengen), nutzen wir ein auf ImageNet vortrainiertes Netz.
#   Die Basis bleibt eingefroren, nur ein kleiner Klassifikationskopf
#   wird neu trainiert. Vorteile: funktioniert mit wenigen Bildern,
#   schnell, auch ohne GPU lauffaehig.
#
# WICHTIG: TensorFlow benoetigt eine kompatible Python-Version
#   (z. B. 3.12) im virtuellen Umfeld (.venv).
#
# Erwartete Ordnerstruktur der Bilddaten (Klassen = Unterordner):
#   data/raw/car_damage/
#       minor/      *.jpg
#       moderate/   *.jpg
#       severe/     *.jpg
#
# Ausfuehren aus dem Projektordner:
#   python src/cnn/train_cnn.py
# =====================================================================

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# ---------------------------------------------------------------------
# Konfiguration - zentral an einer Stelle (leicht anzupassen)
# ---------------------------------------------------------------------
DATA_DIR     = Path("data/raw/car_damage")   # Bildordner (in .gitignore!)
MODELL_PFAD  = Path("src/cnn/model.keras")   # Speicherort des Modells
KLASSEN_PFAD = Path("src/cnn/klassen.json")  # Klassennamen fuer die Vorhersage
BILD_GROESSE = (224, 224)                     # MobileNetV2-Standardgroesse
BATCH_SIZE   = 32
EPOCHEN      = 10
SEED         = 42                             # fuer reproduzierbare Splits


def lade_datensaetze():
    """Laedt Bilder aus DATA_DIR und teilt sie in Trainings- und
    Validierungsdaten (80/20). Gibt (train, val, klassennamen) zurueck."""
    # 80 % Trainingsdaten
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=BILD_GROESSE,
        batch_size=BATCH_SIZE,
    )
    # 20 % Validierungsdaten
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=BILD_GROESSE,
        batch_size=BATCH_SIZE,
    )
    klassen = train_ds.class_names   # z. B. ['minor', 'moderate', 'severe']

    # Performance: Daten im Speicher zwischenpuffern (schnelleres Training)
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, klassen


def baue_modell(anzahl_klassen: int) -> tf.keras.Model:
    """Baut das Transfer-Learning-Modell:
    Augmentation -> MobileNetV2 (eingefroren) -> Klassifikationskopf."""

    # --- Data Augmentation: kuenstliche Varianten gegen Overfitting ---
    # (Spiegeln, Drehen, Zoomen - nur beim Training aktiv)
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    # --- Vortrainierte Basis laden, OHNE den originalen Klassenkopf ---
    basis = MobileNetV2(
        input_shape=BILD_GROESSE + (3,),   # (224, 224, 3) RGB
        include_top=False,                 # eigenen Kopf anbauen
        weights="imagenet",
    )
    basis.trainable = False                # Basis einfrieren (Transfer Learning)

    # --- Gesamtes Modell zusammensetzen ---
    inputs = layers.Input(shape=BILD_GROESSE + (3,))
    x = augmentation(inputs)
    # MobileNetV2 erwartet Pixelwerte im Bereich [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = basis(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)      # Feature-Vektor je Bild
    x = layers.Dropout(0.2)(x)                  # Regularisierung
    outputs = layers.Dense(anzahl_klassen, activation="softmax")(x)

    modell = models.Model(inputs, outputs)
    modell.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # ganzzahlige Labels
        metrics=["accuracy"],
    )
    return modell


def trainiere() -> None:
    """Kompletter Trainingslauf: Daten laden, Modell bauen, trainieren,
    speichern."""
    train_ds, val_ds, klassen = lade_datensaetze()
    print(f"Klassen: {klassen}")

    modell = baue_modell(len(klassen))
    modell.fit(train_ds, validation_data=val_ds, epochs=EPOCHEN)

    # Modell und Klassennamen speichern (Klassen brauchen wir spaeter
    # in predict_image.py, um Index -> Name zuzuordnen)
    modell.save(MODELL_PFAD)
    KLASSEN_PFAD.write_text(json.dumps(klassen, ensure_ascii=False))
    print(f"\nModell gespeichert: {MODELL_PFAD}")
    print(f"Klassen gespeichert: {KLASSEN_PFAD}")


if __name__ == "__main__":
    trainiere()

