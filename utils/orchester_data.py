import pandas as pd
import json
from pathlib import Path

from utils.dataset_cleaner import INPUT_FILE, clean_dataset, fix_name_errors, build_prepared_dataset
from utils.fake_data_generators.accident_description_generator import generate_accident_description, add_generated_text_column

TAGS_KEYS = ["AUTO_MAKE",
        "AUTO_MODEL",
        "AUTO_YEAR",
        "INCIDENT_TYPE",
        "INCIDENT_SEVERITY",
        "COLLISION_TYPE",
        "AUTHORITIES_CONTACTED",
        "PROPERTY_DAMAGE",
        "POLICE_REPORT_AVAILABLE",
        "WITNESSES",
        "BODILY_INJURIES",
        "INCIDENT_STATE",
        "INCIDENT_CITY"]

def get_cleaned_dataset() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_FILE, na_values=["?", ""])
    cleaned = clean_dataset(dataframe)
    cleaned = fix_name_errors(cleaned)
    return cleaned

def get_dataset_with_descriptions() -> pd.DataFrame:
    df = get_cleaned_dataset()
    if "description" not in df.columns:
        df["description"] = df.apply(generate_accident_description, axis=1)
    return df

def get_dataset_with_descriptions_and_tags() -> pd.DataFrame:
    df = get_cleaned_dataset()
    tags = {}
    

    return df