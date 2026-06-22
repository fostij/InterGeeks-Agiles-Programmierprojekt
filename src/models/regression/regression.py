"""CLI-Einstiegspunkt für die vollständige Regressions-Pipeline.
 
Führt das Training des Regressionsmodells aus und beendet den Prozess
mit einem Exit-Code ungleich null, falls die Pipeline fehlschlägt.
"""

import logging
import sys
from src.exceptions import ProjectError
from src.logging_config import setup_logging
from src.models.regression.train_model import train_and_save_model

logger = logging.getLogger(__name__)

def run_full_pipeline() -> None:
    """Führt die komplette Regressions-Pipeline aus (Training + Speichern)."""
    logger.info("Starte Regressions-Pipeline.")
    train_and_save_model()
    logger.info("Regressions-Pipeline erfolgreich abgeschlossen.")


if __name__ == "__main__":
    setup_logging()
    try:
        run_full_pipeline()
    except ProjectError as exc:
        logger.error("Regressions-Pipeline fehlgeschlagen: %s", exc)
        sys.exit(1)