import pandas as pd
import numpy as np
from pathlib import Path
from src.ml_config import CLEANUP_LOG_FILE, TRASH_FIELDS

def clean_dataset(dataframe: pd.DataFrame, log_path: Path = CLEANUP_LOG_FILE) -> pd.DataFrame:
    cleaned = dataframe.copy()
    missing_markers = ["?", "NA", "N/A", "null", "None", ""]
    cleaned = cleaned.replace(missing_markers, np.nan)


    for col in cleaned.columns:
        if cleaned[col].dtype == 'object':
            try:
                converted_date = pd.to_datetime(cleaned[col], errors='raise')
                cleaned[f'{col}_year'] = converted_date.dt.year
                cleaned[f'{col}_month'] = converted_date.dt.month
                cleaned[f'{col}_day'] = converted_date.dt.day
                cleaned = cleaned.drop(columns=[col])
            except (ValueError, TypeError):
                pass


    num_cols = cleaned.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = cleaned.select_dtypes(include=["object", "category", "string"]).columns.tolist()


    for col in num_cols:
        if cleaned[col].isna().sum() > 0:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)


    for col in cat_cols:
        if cleaned[col].isna().sum() > 0:
            mode = cleaned[col].mode(dropna=True)
            if len(mode) > 0:
                cleaned[col] = cleaned[col].fillna(mode.iloc[0])
            else:
                cleaned[col] = cleaned[col].fillna("Unknown")

    drop_cols = TRASH_FIELDS
    cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])
    return cleaned


def fix_name_errors(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["auto_make"] = dataframe["auto_make"].replace(
        {"Accura": "Acura", "Suburu": "Subaru"}
        )
    return dataframe
