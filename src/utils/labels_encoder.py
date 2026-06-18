import json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from src.utils.data_orchestrator import get_descriptions_labels_with_new_vehicles
from src.ml_config import TARGET_FIELDS
from src.utils.data_loader import get_vehicle_dataset

def prepare_pipeline_and_save_jsonl(count: int, output_file: str, dataset: pd.DataFrame) -> tuple[dict, dict]:
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
                "target_price": sample.get("target_price", 0.0)
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            
    print(f"✓ Dataset saved successfully to: {output_file}")
    print()
    return network_config, label_encoders