import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from utils.orchester_data import TARGET_FIELDS, get_descriptions_and_labels, get_cleaned_dataset


NUMERIC_FIELDS = ["auto_year", "number_of_vehicles_involved", "witnesses"]
CATEGORICAL_FIELDS = [f for f in TARGET_FIELDS if f not in NUMERIC_FIELDS]

def prepare_pipeline_and_save_jsonl(count: int, output_file: str) -> dict:
    """
    1. Collects raw text and labels using your looping method.
    2. Builds LabelEncoder instances recognizing 'None' as a valid class.
    3. Transforms the list into encoded indices and exports to JSONL.
    """
    # Step 1: Use your list loop generator to fetch raw samples
    # (Temporarily returns string names and None values)
    raw_samples = get_descriptions_and_labels(count)
    
    # Extract unique values to train our LabelEncoders
    df_cleaned = get_cleaned_dataset()
    label_encoders = {}
    num_classes_dict = {}
    
    # Step 2: Initialize Encoders
    for field in TARGET_FIELDS:
        if field in NUMERIC_FIELDS:
            num_classes_dict[field] = 1  # 1 node for regression
        else:
            # Force the column to string and ensure 'None' text is a known category
            unique_values = df_cleaned[field].fillna("None").astype(str).unique().tolist()
            if "None" not in unique_values:
                unique_values.append("None")
                
            le = LabelEncoder()
            le.fit(unique_values)
            
            label_encoders[field] = le
            num_classes_dict[field] = len(le.classes_)

    # Step 3: Transform and Write to File
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
                    
            # Wrap data package
            json_line = {
                "text": text,
                "labels": encoded_labels,
                "target_price": sample.get("target_price", 0.0)
            }
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            
    print(f"✓ Dataset saved successfully to: {output_file}")
    return num_classes_dict