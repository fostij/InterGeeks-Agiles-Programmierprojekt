"""Manueller Smoke-Test für das gespeicherte Regressionsmodell.
 
Lädt das trainierte Modell und prüft anhand von zwei Beispielfällen
(Kollision vs. Diebstahl), ob die Vorhersagen plausibel auseinanderliegen.
Dient der schnellen manuellen Kontrolle, nicht als automatisierter Test.
"""

import logging
import joblib
import pandas as pd
from src.exceptions import ModelLoadError
from src.constants_paths import REGRESSION_MODEL_PATH

logger = logging.getLogger(__name__)

def test_model() -> None:
    """Lädt das Regressionsmodell und vergleicht zwei Beispielvorhersagen.
 
    Raises:
        ModelLoadError: Wenn die Modell-Datei nicht existiert oder nicht
            geladen werden kann.
    """

    model_path = REGRESSION_MODEL_PATH
    
    if not model_path.exists():
        raise ModelLoadError(f"Modell-Datei nicht gefunden unter {model_path}.")


    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Modell konnte nicht geladen werden: {exc}") from exc

    
    collision_case = pd.DataFrame([{
        'auto_year': 2020,
        'auto_make': 'Audi',
        'incident_type': 'Multi-vehicle collision',
        'incident_severity': 'Major Damage',
        'collision_type': 'Rear Collision',
        'number_of_vehicles_involved': 2,
        'witnesses': 1,
        'police_report_available': 1,
        'bodily_injuries': 0,
        'property_damage': 1
    }])

    theft_case = pd.DataFrame([{
        'auto_year': 2018,
        'auto_make': 'Honda',
        'incident_type': 'Vehicle Theft',
        'incident_severity': 'Trivial Damage',
        'collision_type': 'NotApplicable',
        'number_of_vehicles_involved': 1,
        'witnesses': 0,
        'police_report_available': 1,
        'bodily_injuries': 0,
        'property_damage': 0
    }])


    collision_pred = model.predict(collision_case)[0]
    theft_pred = model.predict(theft_case)[0]

    logger.info("Kollisionsfall — vorhergesagter Wert: %.2f", collision_pred)
    logger.info("Diebstahl/Parked-Car-Fall — vorhergesagter Wert: %.2f", theft_pred)


    if abs(collision_pred - theft_pred) < 1000:
        logger.warning("Die Vorhersagen für beide Fälle liegen sehr nah beieinander.")


if __name__ == "__main__":
    from src.logging_config import setup_logging
 
    setup_logging()
    test_model()