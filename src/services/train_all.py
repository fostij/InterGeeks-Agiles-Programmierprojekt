"""
train_all.py — Vollständige Trainings-Pipeline

Führt der Reihe nach aus:
  1. Datenbankschema anlegen (sql/schema.sql)
  2. Versicherungsdaten laden (CSV -> 5 normalisierte Tabellen)
  3. Fahrzeugkatalog laden (CSV -> fahrzeug_katalog)
  4. multi-head Modell trainieren (Text -> strukturierte Felder)
  5. CNN Modell trainieren (Foto -> Schweregrad)
  6. Regressionsmodell trainieren (Felder -> vehicle_claim)

Jeder Schritt kann einzeln übersprungen werden (z.B. wenn die DB
bereits befüllt ist), siehe SKIP_* Flags unten.

Ausführen aus dem PROJEKTORDNER:
    python scripts/train_all.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Auf False setzen, um einen Schritt zu überspringen
RUN_SCHEMA = True
RUN_LOAD_INSURANCE_DATA = True
RUN_LOAD_VEHICLE_CATALOG = True
RUN_TRAIN_MULTIHEAD = True
RUN_TRAIN_CNN = True
RUN_TRAIN_REGRESSION = True


def run_step(title: str, command: list[str], cwd: Path = PROJECT_ROOT):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        print(f"\n❌ Schritt fehlgeschlagen: {title}")
        sys.exit(1)
    print(f"✓ {title} abgeschlossen")


def main():
    if RUN_SCHEMA:
        run_step(
            "1/6  Datenbankschema anlegen",
            ["psql", "-U", "postgres", "-d", "kfz_schaden", "-f", "sql/schema.sql"],
        )

    if RUN_LOAD_INSURANCE_DATA:
        run_step(
            "2/6  Versicherungsdaten laden",
            [sys.executable, "src/db/load_data.py"],
        )

    if RUN_LOAD_VEHICLE_CATALOG:
        run_step(
            "3/6  Fahrzeugkatalog laden",
            [sys.executable, "src/db/load_vehicle_catalog.py"],
        )

    if RUN_TRAIN_MULTIHEAD:
        run_step(
            "4/6  multi-head Modell trainieren",
            [sys.executable, "-m", "src.models.multi_head_insurance.train"],
        )

    if RUN_TRAIN_CNN:
        run_step(
            "5/6  CNN Modell trainieren",
            [sys.executable, "src/models/cnn/train_cnn.py"],
        )

    if RUN_TRAIN_REGRESSION:
        run_step(
            "6/6  Regressionsmodell trainieren",
            [sys.executable, "regression/train_model.py"],
        )

    print("\n" + "=" * 70)
    print("  ✓ Vollständige Pipeline abgeschlossen.")
    print("    Modelle liegen in checkpoints/, bereit für dashboard.py / pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()