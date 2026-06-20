# =====================================================================
# load_vehicle_catalog.py — vehicle_dataset.csv in PostgreSQL laden
# ---------------------------------------------------------------------
# CSV-Format: year,model,make
#   1992,Integra,Acura
#
# Befüllt die Tabelle fahrzeug_katalog (siehe sql/schema.sql).
# Genutzt für:
#   1. Textgenerierung des multi-head Trainingsdatensatzes
#      (realistische Marke/Modell/Baujahr-Kombinationen)
#   2. Validierung der vom Modell vorhergesagten auto_make/auto_year
#      Kombination zur Inferenzzeit (validate_vehicle_with() in inference.py)
#
# Ausführen aus dem PROJEKTORDNER:
#   python src/db/load_vehicle_catalog.py
# =====================================================================

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.ml_config import VEHICLE_DATASET_PATH
from src.db.connection import get_engine

CSV_PATH = VEHICLE_DATASET_PATH

def extract() -> pd.DataFrame:
    """Schritt 1 (EXTRACT): Roh-CSV einlesen."""
    df = pd.read_csv(CSV_PATH)
    print(f"[EXTRACT]   {len(df)} Zeilen aus {CSV_PATH} eingelesen")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Schritt 2 (TRANSFORM): Bereinigen und auf DB-Spaltennamen abbilden."""
    df = df.dropna(subset=["year", "model", "make"]).copy()

    df["year"] = df["year"].astype(int)
    df["model"] = df["model"].astype(str).str.strip()
    df["make"] = df["make"].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates(subset=["year", "model", "make"])
    print(f"[TRANSFORM] {before - len(df)} Duplikate entfernt, {len(df)} verbleiben")

    df = df.rename(columns={"year": "baujahr", "model": "modell", "make": "marke"})
    return df[["baujahr", "modell", "marke"]]


def load(df: pd.DataFrame) -> None:
    """Schritt 3 (LOAD): Tabelle leeren und neu befüllen."""
    engine = get_engine()

    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fahrzeug_katalog RESTART IDENTITY"))
    print("[LOAD]      Tabelle 'fahrzeug_katalog' geleert")

    df.to_sql("fahrzeug_katalog", engine, if_exists="append", index=False)
    print(f"[LOAD]      Tabelle 'fahrzeug_katalog': {len(df)} Zeilen eingefügt")


def verify() -> None:
    """Abschließende Kontrolle."""
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM fahrzeug_katalog")).scalar()
        makes = conn.execute(text("SELECT COUNT(DISTINCT marke) FROM fahrzeug_katalog")).scalar()
        years = conn.execute(
            text("SELECT MIN(baujahr), MAX(baujahr) FROM fahrzeug_katalog")
        ).fetchone()

    print(f"[PRÜFUNG]   {total} Fahrzeuge, {makes} verschiedene Marken, "
          f"Baujahre {years[0]}–{years[1]}")


if __name__ == "__main__":
    df = extract()
    df = transform(df)
    load(df)
    verify()
    print("\nFertig! Fahrzeugkatalog liegt jetzt in PostgreSQL.")