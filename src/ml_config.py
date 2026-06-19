
from pathlib import Path


INSURANCE_DATASET_PATH = Path("data/output/dataset.csv")
BEST_CHECKPOINT_PATH = Path("data/output/best_checkpoint.pt")


VEHICLE_DATASET_PATH = Path("data/raw/vehicle_dataset.csv")
CLEANED_DATA_OUTPUT_PATH = Path("data/output/dataset_prepared.csv")
CLEANUP_LOG_FILE_PATH = Path("data/output/dataset_cleanup_report.txt")
MULTI_HEAD_DATASET_PATH = Path("data/output/output.jsonl")
NETWORK_CONFIG_PATH = Path("network_config.json")
OUTPUT_CHECKPOINT_PATH = Path("data/output/last_checkpoint.pt")

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

REGRESSION_TARGETS = NUMERIC_FIELDS + [PREDICTION_FIELD]