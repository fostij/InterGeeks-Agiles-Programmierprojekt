import pandas as pd
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