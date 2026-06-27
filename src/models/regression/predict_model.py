import logging
import joblib
import pandas as pd
from src.exceptions import InferenceError, ModelLoadError
from src.ml_config import REGRESSION_MODEL_PATH

logger = logging.getLogger(__name__)

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
 
    Raises:
        ModelLoadError: Wenn die gespeicherte Modell-Pipeline nicht
            geladen werden kann (z. B. Datei fehlt oder ist beschädigt).
        InferenceError: Wenn die Vorhersage selbst fehlschlägt.
    """

    if not REGRESSION_MODEL_PATH.exists():
        raise ModelLoadError(
            f"Regressionsmodell wurde nicht gefunden unter: {REGRESSION_MODEL_PATH}. "
            "Bitte zuerst train_model.py ausführen."
        )
    
    try:
        pipeline = joblib.load(REGRESSION_MODEL_PATH)
    except Exception as exc:
        raise ModelLoadError(f"Regressionsmodell konnte nicht geladen werden: {exc}") from exc
 
    try:
        df_input = pd.DataFrame([input_data])
        prediction = pipeline.predict(df_input)
    except Exception as exc:
        raise InferenceError(f"Vorhersage für neuen Schadensfall fehlgeschlagen: {exc}") from exc

    result = float(prediction[0])
    logger.debug("Regressions-Vorhersage: %.2f", result)
    return result
