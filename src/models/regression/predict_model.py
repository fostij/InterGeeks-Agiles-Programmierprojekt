import joblib
import pandas as pd
from src.ml_config import REGRESSION_MODEL_PATH


def predict_new_claim(input_data: dict) -> float:
    pipeline = joblib.load(REGRESSION_MODEL_PATH)
    df_input = pd.DataFrame([input_data])
    prediction = pipeline.predict(df_input)
    return float(prediction[0])