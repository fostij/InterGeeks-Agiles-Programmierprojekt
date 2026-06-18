import pandas as pd
from src.ml_config import INSURANCE_DATASET_PATH, VEHICLE_DATASET_PATH

def get_insurence_dataset() -> pd.DataFrame:
    df = pd.read_csv(INSURANCE_DATASET_PATH)
    return df

def get_vehicle_dataset() -> pd.DataFrame:
    df = pd.read_csv(VEHICLE_DATASET_PATH)
    return df