from typing import Dict, List, Any
import pandas as pd
from src.utils.dataset_cleaner import clean_dataset, fix_name_errors
from src.utils.fake_data_generators.accident_description_generator import AccidentDataGenerator
from src.ml_config import INPUT_FILE, TARGET_FIELDS

def get_cleaned_dataset() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_FILE, na_values=["?", ""])
    cleaned = clean_dataset(dataframe)
    cleaned = fix_name_errors(cleaned)
    return cleaned

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