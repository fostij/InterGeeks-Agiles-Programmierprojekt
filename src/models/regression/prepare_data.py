import pandas as pd
<<<<<<< HEAD
from src.utils.data_loader import get_insurence_dataset
from src.ml_config import (
    INSURANCE_DATASET_PATH,
    TARGET_FIELDS,
    PREDICTION_FIELD,
)

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

def get_processed_data() -> pd.DataFrame:
    if not INSURANCE_DATASET_PATH.exists():
        raise FileNotFoundError(f"❌ File not found at: {INSURANCE_DATASET_PATH}")

    df = get_insurence_dataset()

    target_fields = TARGET_FIELDS + [PREDICTION_FIELD]
    df = df[target_fields].copy()

    df["property_damage"] = df["property_damage"].map({"YES": 1, "NO": 0})
    df["police_report_available"] = df["police_report_available"].map({"YES": 1, "NO": 0})

    df = df.fillna(0)
    return df
=======
import os

def get_processed_data():
    """
    Loads the dataset from a fixed path, selects relevant columns, 
    cleans binary/categorical data, and returns a processed DataFrame.
    """
    # Define the absolute path to the dataset
    file_path = r"E:\Final_Project\InterGeeks-Agiles-Programmierprojekt\data\output\dataset.csv"
    
    # Check if the file exists before attempting to load
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found at: {file_path}")
        
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Select only the features required for the project
    target_fields = [
        "auto_year", "auto_make", "incident_type", "incident_severity",
        "collision_type", "number_of_vehicles_involved", "witnesses",
        "police_report_available", "bodily_injuries", "property_damage", "vehicle_claim"
    ]
    
    # Work on a copy of the dataframe to avoid SettingWithCopy warnings
    df = df[target_fields].copy()
    
    # Convert binary 'YES'/'NO' columns into numerical 1/0
    if 'property_damage' in df.columns:
        df['property_damage'] = df['property_damage'].map({'YES': 1, 'NO': 0})
    
    if 'police_report_available' in df.columns:
        df['police_report_available'] = df['police_report_available'].map({'YES': 1, 'NO': 0})
    
    # Handle missing values by filling them with 0
    df = df.fillna(0)
    
    # Convert categorical text data into dummy (one-hot encoded) variables
    categorical_cols = ["auto_make", "incident_type", "incident_severity", "collision_type"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

# Optional: Test script to verify data loading
if __name__ == "__main__":
    try:
        data = get_processed_data()
        print("✅ Data loaded and processed successfully!")
        print(f"Data shape: {data.shape}")
    except Exception as e:
        print(f"❌ Error: {e}")
>>>>>>> 3021e37f5e8ba506c913b8dd6540222ba7b85146
