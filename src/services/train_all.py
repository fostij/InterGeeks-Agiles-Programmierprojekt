"""
train_all.py — Vollständige Trainings-Pipeline

Führt der Reihe nach aus:
  1. Versicherungsdaten laden (CSV -> 5 normalisierte Tabellen, legt Schema an)
  2. Fahrzeugkatalog laden (CSV -> fahrzeug_katalog)
  3. multi-head Modell trainieren (Text -> strukturierte Felder)
  4. CNN Modell trainieren (Foto -> Schweregrad)
  5. Regressionsmodell trainieren (Felder -> vehicle_claim)

Jeder Schritt kann einzeln übersprungen werden (z.B. wenn die DB
bereits befüllt ist oder ein Modell schon trainiert wurde) — einfach
das entsprechende RUN_*-Flag unten auf False setzen.

Jeder Schritt läuft als eigener Subprozess (siehe run_step()), damit ein
Fehler in einem Schritt (z. B. ein Absturz beim Modelltraining) den
Gesamtprozess nicht durch einen gemeinsamen, geteilten Zustand
verfälschen kann.

Ausführen aus dem PROJEKTORDNER:
    python scripts/train_all.py
"""

import logging
import subprocess
import sys
import os
from pathlib import Path

from src.logging_config import setup_logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(".") + os.pathsep + env.get("PYTHONPATH", "")

# Auf False setzen, um einen Schritt zu überspringen
RUN_LOAD_INSURANCE_DATA = True
RUN_LOAD_VEHICLE_CATALOG = True
RUN_TRAIN_MULTIHEAD = True
RUN_TRAIN_CNN = False
RUN_TRAIN_REGRESSION = False


def run_step(title: str, command: list[str]) -> None:
    """Führt einen einzelnen Pipeline-Schritt als Subprozess aus.

    Args:
        title: Anzeigename des Schritts (für die Konsolenausgabe).
        command: Auszuführender Befehl (z. B. [sys.executable, "-m", "..."]).

    Beendet den Gesamtprozess mit Exit-Code 1, falls der Subprozess
    fehlschlägt (returncode != 0).
    """
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        logger.error("Schritt fehlgeschlagen: %s", title)
        sys.exit(1)
    logger.info("%s abgeschlossen.", title)


def main() -> None:
    """Führt alle aktivierten Schritte der Trainings-Pipeline der Reihe nach aus."""
    if RUN_LOAD_INSURANCE_DATA:
        run_step(
            "1/5  Versicherungsdaten laden (CSV -> PostgreSQL)",
            [sys.executable, "-m", "src.db.load_insurance_data"],
        )

    if RUN_LOAD_VEHICLE_CATALOG:
        run_step(
            "2/5  Fahrzeugkatalog laden (CSV -> PostgreSQL)",
            [sys.executable, "-m", "src.db.load_vehicle_catalog"],
        )

    if RUN_TRAIN_MULTIHEAD:
        run_step(
            "3/5  multi-head Modell trainieren",
            [sys.executable, "-m", "src.models.multi_head_insurance.train"],
        )

    if RUN_TRAIN_CNN:
        run_step(
            "4/5  CNN Modell trainieren",
            [sys.executable, "-m", "src.models.cnn.train_cnn"],
        )

    if RUN_TRAIN_REGRESSION:
        run_step(
            "5/5  Regressionsmodell trainieren",
            [sys.executable, "-m", "src.models.regression.train_model"],
        )

    print("\n" + "=" * 70)
    print("  Vollständige Pipeline abgeschlossen.")
    print("    Modelle liegen in checkpoints/, bereit für dashboard.py / pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    setup_logging()
    main()