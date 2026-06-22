"""Deskriptive Statistik für numerische Felder des Versicherungsdatensatzes.
 
Berechnet Mittelwert und Standardabweichung für die wichtigsten
numerischen Spalten, z. B. zur Normalisierung oder für die
explorative Datenanalyse.
"""

import logging
import numpy as np
import pandas as pd
from src.exceptions import DataPreparationError
 
logger = logging.getLogger(__name__)

NUMERIC_FIELDS = ["auto_year", "number_of_vehicles_involved", "witnesses"]

def get_descriptive_statistics_for_numeric(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Berechnet Mittelwert und Standardabweichung für ausgewählte numerische Felder.
 
    Args:
        df: Datensatz mit den Spalten auto_year, number_of_vehicles_involved
            und witnesses.
 
    Returns:
        Mapping {Feldname: {"mean": ..., "std": ...}}. Die Standard-
        abweichung enthält einen kleinen Offset (1e-8), um eine Division
        durch 0 bei nachgelagerter Normalisierung zu vermeiden.
 
    Raises:
        DataPreparationError: Wenn eine der erwarteten Spalten fehlt.
    """

    missing = [f for f in NUMERIC_FIELDS if f not in df.columns]
    if missing:
        raise DataPreparationError(f"Für die Statistik fehlen Spalten: {missing}")

    stats = {
        field: {
            "mean": float(np.mean(df[field].values)),
            "std": float(np.std(df[field].values)) + 1e-8,
        }
        for field in NUMERIC_FIELDS
    }

    logger.debug("Deskriptive Statistik berechnet für: %s", NUMERIC_FIELDS)
    return stats