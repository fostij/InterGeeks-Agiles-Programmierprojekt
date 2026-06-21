"""Training und Speicherung des Regressionsmodells.
 
Baut eine sklearn-Pipeline (One-Hot-Encoding + RandomForestRegressor),
trainiert sie auf den aufbereiteten Versicherungsdaten und speichert das
Ergebnis als .pkl-Datei.
"""

import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exceptions import TrainingError
from src.ml_config import REGRESSION_MODEL_PATH, PREDICTION_FIELD
from src.models.regression.prepare_data import get_processed_data, REGRESSION_CATEGORICAL_COLS

logger = logging.getLogger(__name__)

def build_pipeline() -> Pipeline:
    """Erstellt die sklearn-Pipeline aus Vorverarbeitung und Regressor.
 
    Returns:
        Unfitted Pipeline: One-Hot-Encoding der kategorialen Spalten,
        gefolgt von einem RandomForestRegressor.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), REGRESSION_CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])

def train_and_save_model() -> None:
    """Lädt die Daten, trainiert die Regressions-Pipeline und speichert sie.
 
    Raises:
        TrainingError: Wenn das Training oder das Speichern des Modells
            fehlschlägt.
    """

    df = get_processed_data()

    X = df.drop(columns=[PREDICTION_FIELD])
    y = df[PREDICTION_FIELD]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    
    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        raise TrainingError(f"Training der Regressions-Pipeline fehlgeschlagen: {exc}") from exc


    test_r2 = r2_score(y_test, pipeline.predict(X_test))
    logger.info("Test R2: %.4f", test_r2)

    try:
        REGRESSION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, REGRESSION_MODEL_PATH)
    except OSError as exc:
        raise TrainingError(f"Modell konnte nicht gespeichert werden unter {REGRESSION_MODEL_PATH}: {exc}") from exc

    logger.info("Modell gespeichert unter: %s", REGRESSION_MODEL_PATH)

if __name__ == "__main__":
    from src.logging_config import setup_logging
 
    setup_logging()
    train_and_save_model()