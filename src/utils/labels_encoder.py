"""Label-Encoding und JSONL-Export für das Multi-Head-Training.
 
Erzeugt für jedes Zielfeld einen LabelEncoder anhand der bekannten Werte
(aus dem Versicherungsdatensatz bzw. dem Fahrzeugkatalog), kodiert die
synthetisch erzeugten Trainingsbeispiele und speichert sie als JSONL-Datei.
"""

from pathlib import Path
import logging
import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.exceptions import DataPreparationError
from src.utils.data_orchestrator import get_descriptions_labels_with_new_vehicles
from src.ml_config import TARGET_FIELDS
from src.utils.data_loader import get_vehicle_dataset

logger = logging.getLogger(__name__)

def prepare_pipeline_and_save_jsonl(
    count: int, output_file: str, dataset: pd.DataFrame
) -> tuple[dict[str, dict], dict[str, LabelEncoder]]:
    """Erzeugt Trainingsbeispiele, kodiert die Labels und speichert sie als JSONL.
 
    Args:
        count: Anzahl der zu erzeugenden synthetischen Trainingsbeispiele.
        output_file: Zielpfad der JSONL-Datei.
        dataset: Bereinigter Versicherungsdatensatz, aus dem die Werte
            für die nicht-fahrzeugbezogenen Felder stammen.
 
    Returns:
        Tupel (network_config, label_encoders): network_config enthält je
        Feld Typ und Klassenanzahl, label_encoders die zugehörigen
        gefitteten LabelEncoder-Instanzen.
 
    Raises:
        DataPreparationError: Wenn die JSONL-Datei nicht geschrieben
            werden kann.
    """

    raw_samples = get_descriptions_labels_with_new_vehicles(count, dataset)
    vehicle_df = get_vehicle_dataset()
    
    label_encoders = {}
    network_config = {}
    
    for field in TARGET_FIELDS:
        if field == "auto_year":
            from_vehicles = vehicle_df["year"].dropna().astype(str).unique().tolist()
            unique_values = list(set(from_vehicles))
        elif field == "auto_make":
            from_vehicles = vehicle_df["make"].dropna().astype(str).unique().tolist()
            unique_values = list(set(from_vehicles))
        else:
            unique_values = dataset[field].dropna().astype(str).unique().tolist()
            unique_values = list(set(unique_values))
        le = LabelEncoder()
        le.fit(unique_values)

        label_encoders[field] = le
        network_config[field] = {
            "type": "classification",
            "size": len(le.classes_)
        }


    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in raw_samples:
                text = sample["text"]
                raw_labels = sample["labels"]
                
                encoded_labels = {}
                for field, val in raw_labels.items():
                    if val is None:
                        encoded_labels[field] = -100
                    else:
                        encoded_labels[field] = int(label_encoders[field].transform([str(val)])[0])
                        
                json_line = {
                    "text": text,
                    "labels": encoded_labels,
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    except OSError as exc:
        raise DataPreparationError(f"JSONL-Datei konnte nicht geschrieben werden unter {output_file}: {exc}") from exc

    logger.info("Datensatz erfolgreich gespeichert unter: %s (%d Beispiele).", output_file, len(raw_samples))
    return network_config, label_encoders