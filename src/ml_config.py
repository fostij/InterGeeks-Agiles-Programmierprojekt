"""Zentrale Konfiguration für Modellpfade und Feldlisten.
 
Dieses Modul bündelt alle projektweiten Konstanten (gültige
Kategorien, Zielfelder), damit sie nicht in einzelnen Skripten verstreut
und dupliziert werden.
"""

from pathlib import Path

def _find_project_root(start: Path) -> Path:
    current = start.resolve()

    while current != current.parent:
        if (current / "data").exists() and (current / "src").exists():
            return current
        current = current.parent

    raise RuntimeError("Project root not found")

_BASE_DIR = _find_project_root(Path(__file__).parent)

INSURANCE_DATASET_PATH = _BASE_DIR / "data" / "raw" / "dataset.csv"
VEHICLE_DATASET_PATH = _BASE_DIR / "data" / "raw" / "vehicle_dataset.csv"

MULTI_HEAD_MODEL_PATH = _BASE_DIR / "src" / "models" / "multi_head_insurance" / "multi_head_model.pt"
CNN_MODEL_PATH = _BASE_DIR / "src" / "models" / "cnn" / "cnn_model.keras"
CNN_CLASSES_PATH = _BASE_DIR / "src" / "models" / "cnn" / "classes.json"
REGRESSION_MODEL_PATH = _BASE_DIR / "src" / "models" / "regression" / "regression_model.pkl"

SCHEMA_PATH = _BASE_DIR / "sql" / "schema.sql"

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