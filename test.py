import pandas as pd
from src.utils.data_loader import get_vehicle_dataset, get_insurence_dataset
from scipy.stats import spearmanr

df = get_insurence_dataset()
print(df.groupby("auto_model")["vehicle_claim"].mean().std())  # разброс средних выплат по моделям

# Корреляция severity с выплатой  
print(df.groupby("incident_severity")["vehicle_claim"].mean())