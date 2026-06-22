"""Zentrale Konfiguration für Datenpfade, Modellpfade und Feldlisten.
 
Dieses Modul bündelt alle projektweiten Konstanten (Dateipfade, gültige
Kategorien, Zielfelder), damit sie nicht in einzelnen Skripten verstreut
und dupliziert werden.
"""

from pathlib import Path

INSURANCE_DATASET_PATH = Path("data/raw/dataset.csv")
VEHICLE_DATASET_PATH = Path("data/raw/vehicle_dataset.csv")
CLEANED_DATA_OUTPUT_PATH = Path("data/output/dataset_prepared.csv")
CLEANUP_LOG_FILE_PATH = Path("data/output/dataset_cleanup_report.txt")
MULTI_HEAD_DATASET_PATH = Path("data/output/output.jsonl")
NETWORK_CONFIG_PATH = Path("network_config.json")
OUTPUT_CHECKPOINT_PATH = Path("data/output/last_checkpoint.pt")
BEST_CHECKPOINT_PATH = Path("data/output/best_checkpoint.pt")

MULTI_HEAD_MODEL_PATH = Path("checkpoints/multi_head_model.pt")
CNN_MODEL_PATH = Path("checkpoints/cnn_model.keras")
REGRESSION_MODEL_PATH = Path("checkpoints/regression_model.pkl")

VALID_CASE_TYPES = {
    "Front Collision",
    "Parked Car",
    "Rear Collision",
    "Side Collision",
    "Vehicle Theft",
}

TARGET_FIELDS = [
        "auto_year", 
        "auto_make", 
        "incident_type", 
        "incident_severity", 
        "collision_type", 
        "number_of_vehicles_involved",
        "witnesses",
        "police_report_available",
        "bodily_injuries",
        "property_damage",
    ]

FIELDS_TO_DELETE = [
    "policy_number",
    "incident_location",
    "injury_claim",
    "total_claim_amount",
    "property_claim",
    "_c39",
    "auto_model",
    "incident_city", 
    "incident_state",
    "authorities_contacted", 
]

PREDICTION_FIELD = "vehicle_claim"

INTEGER_FIELDS = ["auto_year", "witnesses", "number_of_vehicles_involved", "bodily_injuries"]
NUMERIC_FIELDS = []
CATEGORICAL_FIELDS = [f for f in TARGET_FIELDS if f not in NUMERIC_FIELDS]

# ---------------------------------------------------------------------
# - gemeinsame Konstanten des CNN-Moduls
# ---------------------------------------------------------------------

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

# Abbildung der Datensatz-Klassen auf deutsche Anzeigenamen.
# Die Schluessel entsprechen den Ordnernamen im Trainingsdatensatz.
CNN_TO_SEVERITY = {
    "01-minor": "Minor Damage",
    "02-moderate": "Major Damage",
    "03-severe": "Total Loss",
}

# Abbildung der CSV-Kategorien auf deutsche Anzeigenamen.
# Interne englische Werte bleiben erhalten (Datensatz/Modell-Kompatibilität).
SEVERITY__DE = {
    "Trivial Damage": "Bagatellschaden",
    "Minor Damage": "Leichter Schaden",
    "Major Damage": "Erheblicher Schaden",
    "Total Loss": "Totalschaden",
}

# Durchschnittliche Schadenhöhe je Schwere (aus dem Datensatz abgeleitet).
# Dient als Basis für die regelbasierte Platzhalter-Prognose.
BASE_AMOUNT = {
    "Trivial Damage": 5_000,
    "Minor Damage": 35_000,
    "Major Damage": 62_000,
    "Total Loss": 65_000,
}
