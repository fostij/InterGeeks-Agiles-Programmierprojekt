import numpy as np

def get_descriptive_statistics_for_numeric(df) -> dict:
    years = df["auto_year"].values
    vehicles = df["number_of_vehicles_involved"].values
    witnesses = df["witnesses"].values
    stats = {
        "auto_year": {
            "mean": np.mean(years),
            "std": np.std(years) + 1e-8
        },
        "number_of_vehicles_involved": {
            "mean": np.mean(vehicles),
            "std": np.std(vehicles) + 1e-8
        },
        "witnesses": {
            "mean": np.mean(witnesses),
            "std": np.std(witnesses) + 1e-8
        }
    }
    return stats