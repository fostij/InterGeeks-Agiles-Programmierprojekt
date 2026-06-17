from pathlib import Path

INPUT_FILE = Path("data/raw/dataset.csv")
OUTPUT_FILE = Path("data/output/dataset_prepared.csv")
CLEANUP_LOG_FILE = Path("data/output/dataset_cleanup_report.txt")
DATASET_PATH = Path("data/output/output.jsonl")
NETWORK_CONFIG_PATH = Path("network_config.json")
OUTPUT_CHECKPOINT = Path("data/output/last_checkpoint.pt")
BEST_CHECKPOINT = Path("data/output/best_checkpoint.pt")

VALID_CASE_TYPES = {
    "Front Collision",
    "Parked Car",
    "Rear Collision",
    "Side Collision",
    "Vehicle Theft",
}

# 1. Keep TARGET_FIELDS for your 14 multi-head attributes
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
    "bodily_injuries",
]

# 2. Keep these dropped fields, but ensure vehicle_claim is NOT here
TRASH_FIELDS = [
    "policy_number",
    "incident_location",
    "injury_claim",
    "total_claim_amount",
    "property_claim",
    "_c39",
]

# 3. Define your explicit final target
PREDICTION_FIELD = "vehicle_claim"

# 4. Define your field categories for preprocessing math
NUMERIC_FIELDS = ["auto_year", "number_of_vehicles_involved", "witnesses"]
CATEGORICAL_FIELDS = [f for f in TARGET_FIELDS if f not in NUMERIC_FIELDS]

# 5. Add this helper line at the bottom so your dataset cleaner/loader
# knows to normalize the financial Euro amounts during preprocessing!
REGRESSION_TARGETS = NUMERIC_FIELDS + [PREDICTION_FIELD]
