"""
# Gemeinsame Konstanten des CNN-Moduls
"""

from pathlib import Path

# Pfade zum trainierten Modell und zu den Klassennamen
MODEL_PATH = Path("src/models/cnn/model.keras")
CLASSES_PATH = Path("src/models/cnn/classes.json")

# Bildgröße — muss in Training und Vorhersage identisch sein!
IMAGE_SIZE = (224, 224)

# Abbildung der CNN-Klassen auf deutsche Anzeigenamen
SEVERITY_DE = {
    "01-minor": "Leichter Schaden",
    "02-moderate": "Mittlerer Schaden",
    "03-severe": "Erheblicher Schaden",
}

# Abbildung der CNN-Klassen auf die Schwere-Kategorien des Datensatzes
CNN_TO_SEVERITY = {
    "01-minor": "Minor Damage",
    "02-moderate": "Major Damage",
    "03-severe": "Total Loss",
}