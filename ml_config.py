from pathlib import Path

INPUT_FILE = Path("data/raw/dataset.csv")
OUTPUT_FILE = Path("data/output/dataset_prepared.csv")
CLEANUP_LOG_FILE = Path("data/output/dataset_cleanup_report.txt")
DATASET_PATH = Path("data/output/output.jsonl")
NETWORK_CONFIG_PATH = Path("network_config.json")

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
        "auto_model",
        "incident_type", 
        "incident_severity", 
        "incident_city", 
        "incident_state",
        "collision_type", 
        "property_damage", 
        "witnesses",
        "authorities_contacted", 
        "police_report_available",
        "number_of_vehicles_involved", 
        "bodily_injuries"
    ]

TRASH_FIELDS = [
    "policy_number",
    "incident_location",
    "injury_claim",
    "total_claim_amount",
    "property_claim",
    "_c39"
]

PREDICTION_FIELD = "vehicle_claim"

NUMERIC_FIELDS = ["auto_year", "number_of_vehicles_involved", "witnesses"]
CATEGORICAL_FIELDS = [f for f in TARGET_FIELDS if f not in NUMERIC_FIELDS]