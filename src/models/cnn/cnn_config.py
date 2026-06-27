"""
# Gemeinsame Konstanten des CNN-Moduls
"""

# Trainingsparameter
BATCH_SIZE   = 32
EPOCHS       = 10
SEED         = 42

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