import pandas as pd
from src.utils.data_loader import get_insurence_dataset
from src.ml_config import (
    INSURANCE_DATASET_PATH,
    TARGET_FIELDS,
    PREDICTION_FIELD,
)

REGRESSION_NUMERIC_COLS = [
    "auto_year",
    "number_of_vehicles_involved",
    "witnesses",
    "bodily_injuries",
]
REGRESSION_BINARY_COLS = ["police_report_available", "property_damage"]
REGRESSION_CATEGORICAL_COLS = [
    f for f in TARGET_FIELDS
    if f not in REGRESSION_NUMERIC_COLS + REGRESSION_BINARY_COLS
]

def get_processed_data() -> pd.DataFrame:
    if not INSURANCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"❌ File not found at: {INSURANCE_DATASET_PATH}")

    df = get_insurence_dataset()

    target_fields = TARGET_FIELDS + [PREDICTION_FIELD]
    df = df[target_fields].copy()

    df["property_damage"] = df["property_damage"].map({"YES": 1, "NO": 0})
    df["police_report_available"] = df["police_report_available"].map({"YES": 1, "NO": 0})

    df = df.fillna(0)
    return df
