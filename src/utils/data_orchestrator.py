from typing import Dict, List, Any
import pandas as pd
from src.utils.dataset_cleaner import clean_insurence_dataset, fix_name_errors, shuffle_dataset
from src.utils.fake_data_generators.accident_description_generator import AccidentDataGenerator
from src.ml_config import TARGET_FIELDS
from src.utils.data_loader import get_insurence_dataset, get_vehicle_dataset

def get_cleaned_dataset() -> pd.DataFrame:
    dataframe = get_insurence_dataset()
    cleaned = clean_insurence_dataset(dataframe)
    fixed = fix_name_errors(cleaned)
    return fixed

def get_dataset_with_descriptions() -> pd.DataFrame:
    df = get_cleaned_dataset()
    accident_data_generator = AccidentDataGenerator(TARGET_FIELDS)

    if "description" not in df.columns:
        df["description"] = df.apply(
            lambda row: accident_data_generator.generate_text(dict(row))[0], 
            axis=1
        )
    return df

def get_dataset_with_descriptions_and_tags() -> tuple[pd.DataFrame, list[dict]]:
    df = get_cleaned_dataset()
    accident_data_generator = AccidentDataGenerator(TARGET_FIELDS)

    if "description" not in df.columns:
        df[["description", "temp_metadata"]] = df.apply(
            lambda row: accident_data_generator.generate_with_labels(dict(row)), 
            axis=1, 
            result_type="expand"
        )

    metadata_list = df["temp_metadata"].tolist()

    df = df.drop(columns=["temp_metadata"])
    return df, metadata_list

def get_descriptions_and_labels(count: int, df: pd.DataFrame) ->  List[Dict[str, Any]]:
    if count <= 0:
        return []
    
    accident_data_generator = AccidentDataGenerator(TARGET_FIELDS)
    i = 0
    row_index = 0
    results = []

    while i < count:
        if row_index >= len(df):
            row_index = 0

        row_df = df.iloc[row_index]
        row = row_df.to_dict()
    
        description, labels = accident_data_generator.generate_with_labels(row)
        target_price = float(row_df.get("vehicle_claim", 0.0))
        
        results.append({
            "text": description,
            "labels": labels,
            "target_price": target_price
        })

        i += 1
        row_index += 1

    return results

def get_descriptions_labels_with_new_vehicles(count: int, df: pd.DataFrame) -> List[Dict[str, Any]]:
    vehicle_df = get_vehicle_dataset()

    if count <= 0:
        return []

    accident_data_generator = AccidentDataGenerator(TARGET_FIELDS)
    results = []

    makes = vehicle_df["make"].unique()
    samples_per_make = max(1, count // len(makes)) + 1
    
    groups = []
    for make, group in vehicle_df.groupby("make"):
        sampled = group.sample(n=min(len(group), samples_per_make), replace=True)
        groups.append(sampled)

    balanced_vehicles = (
        pd.concat(groups, ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    insurance_rows = df.sample(n=count, replace=True).reset_index(drop=True)

    for i in range(count):
        row = insurance_rows.iloc[i].to_dict()

        vehicle_row = balanced_vehicles.iloc[i % len(balanced_vehicles)]
        vehicle = (vehicle_row["make"], vehicle_row["model"], int(vehicle_row["year"]))

        description, labels = accident_data_generator.generate_with_labels_and_vehicles(row, vehicle)
        target_price = float(row.get("vehicle_claim", 0.0))

        results.append({
            "text": description,
            "labels": labels,
            "target_price": target_price
        })

    return results