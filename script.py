from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/raw/dataset.csv")

print("start")

df = pd.read_csv(INPUT_FILE)

print("loaded")
print(df.head())