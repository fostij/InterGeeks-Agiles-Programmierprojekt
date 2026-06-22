"""Bereinigung des Versicherungsdatensatzes vor dem Training.
 
Behandelt fehlende Werte, leitet Jahr/Monat/Tag aus Datumsspalten ab,
füllt numerische Lücken mit dem Median und kategoriale Lücken mit dem
Modus (bzw. "Unknown") und entfernt nicht benötigte Spalten.
"""
 
import logging
import pandas as pd
import numpy as np
from src.ml_config import FIELDS_TO_DELETE

logger = logging.getLogger(__name__)

def clean_insurence_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Bereinigt den Versicherungsdatensatz für das Training.
 
    Args:
        dataframe: Roher Versicherungsdatensatz.
 
    Returns:
        Bereinigter DataFrame ohne fehlende Werte und ohne die in
        FIELDS_TO_DELETE konfigurierten Spalten.
    """

    rows_before = len(dataframe)
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

    drop_cols = FIELDS_TO_DELETE
    cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns])

    logger.info(
        "Datensatz bereinigt: %d Zeilen, %d -> %d Spalten.",
        rows_before, len(dataframe.columns), len(cleaned.columns),
    )
    return cleaned


def fix_name_errors(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Korrigiert bekannte Schreibfehler in der Spalte auto_make.
 
    Args:
        dataframe: Datensatz mit der Spalte "auto_make".
 
    Returns:
        DataFrame mit korrigierten Markennamen.
    """

    dataframe["auto_make"] = dataframe["auto_make"].replace(
        {"Accura": "Acura", "Suburu": "Subaru"}
        )
    return dataframe

def clean_vehicle_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Platzhalter für die Bereinigung des Fahrzeugkatalogs (aktuell keine Änderung)."""
    
    return dataframe

def shuffle_dataset(dataframe: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """Mischt die Zeilen eines DataFrames zufällig durch.
 
    Args:
        dataframe: Zu mischender DataFrame.
        seed: Optionaler Seed für reproduzierbares Mischen.
 
    Returns:
        Neuer DataFrame mit zufällig vertauschten Zeilen und neu
        gesetztem Index.
    """

    return dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)