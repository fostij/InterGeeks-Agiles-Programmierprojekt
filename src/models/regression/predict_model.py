import joblib
import pandas as pd
from src.ml_config import REGRESSION_MODEL_PATH


def predict_new_claim(input_data: dict) -> float:
    """
    Sagt vehicle_claim für einen einzelnen neuen Fall voraus.
    Wichtig: input_data muss die rohen, unkodierten Werte enthalten
    (z. B. auto_make='Audi' als Klartext-String), KEIN manuelles
    pd.get_dummies()/One-Hot-Encoding vorab anwenden. Das One-Hot-Encoding
    ist Teil der gespeicherten Pipeline (siehe build_pipeline() in
    train_model.py) und wird intern beim Aufruf von .predict() angewendet.
    Wird bereits kodierter Input übergeben, erkennt die Pipeline die
    Spalten nicht und liefert nahezu konstante, falsche Vorhersagen
    (siehe zusammenfassung.md, Abschnitt 2, für den ursprünglichen Bug
    in test_model.py, der genau dieses Problem verursachte).
    """


    pipeline = joblib.load(REGRESSION_MODEL_PATH)
    df_input = pd.DataFrame([input_data])
    prediction = pipeline.predict(df_input)
    return float(prediction[0])