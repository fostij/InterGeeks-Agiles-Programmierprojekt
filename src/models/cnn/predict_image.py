# ---------------------------------------------------------------------
# predict_image.py — Schadensschwere aus einem Foto vorhersagen
# ---------------------------------------------------------------------

import logging
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Konfiguration
from src.models.cnn.cnn_config import MODEL_PATH, CLASSES_PATH, SEVERITY_DE, IMAGE_SIZE


@lru_cache(maxsize=1)
def _load_model_and_classes() -> tuple[tf.keras.Model, list[str]]:
    """Lädt Modell und Klassennamen einmalig und hält sie im Cache.
    @lru_cache sorgt dafür, dass das Modell nur beim ersten Aufruf
    geladen wird (wichtig für das Dashboard: kein Neuladen pro Klick)."""
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASSES_PATH.read_text())
    return model, class_names


def predict_severity(image_path: str) -> tuple[str, float]:
    """Sagt die Schadensschwere für ein einzelnes Foto vorher.

    Args:
        image_path: Pfad zur Bilddatei.

    Returns:
        Tupel aus deutschem Schweregrad-Namen und Wahrscheinlichkeit
        (0.0 bis 1.0) der vorhergesagten Klasse.
    """
    model, class_names = _load_model_and_classes()

    # --- Bild laden und auf die Trainingsgroesse bringen ---
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image)
    # Modell erwartet einen Batch -> zusätzliche Dimension voranstellen
    batch = np.expand_dims(array, axis=0)

    # --- Vorhersage: Wahrscheinlichkeiten je Klasse ---
    # (preprocess_input ist bereits im Modell enthalten, siehe train_cnn.py)
    probabilities = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])

    # Internen Klassennamen auf deutschen Anzeigenamen abbilden
    raw_class = class_names[best_index]
    label_de = SEVERITY_DE.get(raw_class, raw_class)
    return label_de, confidence


# ---------------------------------------------------------------------
# Manueller Test über die Kommandozeile:
# python src/cnn/predict_image.py pfad/zum/foto.jpg
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python src/cnn/predict_image.py <image_path>")
        sys.exit(1)

    label, conf = predict_severity(sys.argv[1])
    logger.info(f"Predicted severity: {label} ({conf:.1%})")