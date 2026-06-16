import json
from sklearn.preprocessing import LabelEncoder
from utils.orchester_data import get_descriptions_and_labels, get_cleaned_dataset
from ml_config import TARGET_FIELDS, NUMERIC_FIELDS

def prepare_pipeline_and_save_jsonl(count: int, output_file: str) -> dict:
    df_cleaned = get_cleaned_dataset()
    raw_samples = get_descriptions_and_labels(count, df_cleaned)
    
    label_encoders = {}
    network_config = {}
    
    for field in TARGET_FIELDS:
        if field in NUMERIC_FIELDS:
            network_config[field] = {
                "type": "regression"
            }
        else:
            unique_values = df_cleaned[field].fillna("None").astype(str).unique().tolist()
                
            le = LabelEncoder()
            le.fit(unique_values)
            
            label_encoders[field] = le
            network_config[field] = {
                "type": "classification",
                "num_classes": len(le.classes_)
            }


    with open(output_file, "w", encoding="utf-8") as f:
        for sample in raw_samples:
            text = sample["text"]
            raw_labels = sample["labels"]
            
            encoded_labels = {}
            for field, val in raw_labels.items():
                if val is None:
                    # Map the python None object to the integer ID of the "None" class
                    encoded_labels[field] = int(label_encoders[field].transform(["None"])[0])
                elif field in NUMERIC_FIELDS:
                    # Keep numbers as floats
                    encoded_labels[field] = float(val)
                else:
                    # Convert normal text to integer ID
                    encoded_labels[field] = int(label_encoders[field].transform([str(val)])[0])
                    
            json_line = {
                "text": text,
                "labels": encoded_labels,
                "target_price": sample.get("target_price", 0.0)
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            
    print(f"✓ Dataset saved successfully to: {output_file}")
    return network_config