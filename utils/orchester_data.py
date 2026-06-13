from typing import Dict, List, Any
import pandas as pd

from utils.dataset_cleaner import INPUT_FILE, clean_dataset, fix_name_errors
from utils.fake_data_generators.accident_description_generator import generate_accident_description, get_labels

TARGET_FIELDS = [
        "auto_year", "auto_make", "auto_model",
        "incident_type", "incident_severity", "incident_city", "incident_state",
        "collision_type", "property_damage", "witnesses",
        "authorities_contacted", "police_report_available",
        "number_of_vehicles_involved", "bodily_injuries"
    ]

def get_cleaned_dataset() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_FILE, na_values=["?", ""])
    cleaned = clean_dataset(dataframe)
    cleaned = fix_name_errors(cleaned)
    return cleaned

def get_dataset_with_descriptions() -> pd.DataFrame:
    df = get_cleaned_dataset()
    if "description" not in df.columns:
        df["description"] = df.apply(
            lambda row: generate_accident_description(dict(row))[0], 
            axis=1
        )
    return df

def get_dataset_with_descriptions_and_tags() -> tuple[pd.DataFrame, list[dict]]:
    df = get_cleaned_dataset()
    if "description" not in df.columns:
        df[["description", "temp_metadata"]] = df.apply(
            lambda row: generate_accident_description(dict(row)), 
            axis=1, 
            result_type="expand"
        )

    metadata_list = df["temp_metadata"].tolist()

    df = df.drop(columns=["temp_metadata"])
    return df, metadata_list

def get_description_and_labels(count: int) ->  List[Dict[str, Any]]:
    if count <= 0:
        return []
    
    df = get_cleaned_dataset()

    i = 0
    row_index = 0
    results = []

    while i < count:
        if row_index >= len(df):
            row_index = 0

        row = df.iloc[row_index].to_dict()

        description = generate_accident_description(row)
        labels = get_labels(TARGET_FIELDS, row)

        results.append({
            "text": description,
            "labels": labels,
        })

        i += 1
        row_index += 1

    return results