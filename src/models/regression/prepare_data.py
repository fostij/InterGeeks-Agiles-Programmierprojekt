"""Datenaufbereitung für das Regressionsmodell.
 
Wählt die relevanten Spalten aus dem Versicherungsdatensatz aus, kodiert
binäre Felder numerisch und behandelt den Sonderfall "?" bei
collision_type als eigene Kategorie.
"""

import logging
import pandas as pd
from src.exceptions import DataPreparationError
from src.utils.data_loader import get_insurance_dataset
from src.ml_config import (
    TARGET_FIELDS,
    PREDICTION_FIELD,
)

logger = logging.getLogger(__name__)

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

# "?" bei collision_type ist kein zufälliger fehlender Wert, sondern bedeutet
# "nicht zutreffend" (z. B. bei Vehicle Theft oder Parked Car gibt es keinen
# Kollisionstyp). Das ist ein aussagekräftiges Signal und wird daher als
# eigene Kategorie kodiert, statt mit 0/Modus aufgefüllt zu werden.
COLLISION_TYPE_NA_LABEL = "NotApplicable"

def get_processed_data() -> pd.DataFrame:
    """Lädt den Versicherungsdatensatz und bereitet ihn für die Regression auf.
 
    Returns:
        DataFrame mit den Zielfeldern (TARGET_FIELDS) und der Vorhersage-
        spalte (PREDICTION_FIELD), bereit für das Training.
 
    Raises:
        DataPreparationError: Wenn der Datensatz nicht geladen werden kann
            oder erwartete Spalten fehlen.
    """
    
    df = get_insurance_dataset()

    target_fields = TARGET_FIELDS + [PREDICTION_FIELD]
    missing_fields = [f for f in target_fields if f not in df.columns]
    if missing_fields:
        raise DataPreparationError(
            f"Im Datensatz fehlen erwartete Spalten: {missing_fields}"
        )

    df = df[target_fields].copy()

    df["property_damage"] = df["property_damage"].map({"YES": 1, "NO": 0})
    df["police_report_available"] = df["police_report_available"].map({"YES": 1, "NO": 0})

    if "collision_type" in df.columns:
        df["collision_type"] = df["collision_type"].fillna(COLLISION_TYPE_NA_LABEL)

    df = df.fillna(0)

    if df.empty:
        raise DataPreparationError("Nach der Aufbereitung sind keine Zeilen mehr vorhanden.")
 
    logger.info("Daten für Regression aufbereitet: %d Zeilen, %d Spalten.", len(df), len(df.columns))

    return df