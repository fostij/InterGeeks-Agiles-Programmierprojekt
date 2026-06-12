# =====================================================================
# load_data.py — ETL: CSV einlesen, bereinigen und in PostgreSQL laden
# ---------------------------------------------------------------------
# Ablauf (Extract -> Transform -> Load):
#   1. EXTRACT:   Roh-CSV aus data/raw/ einlesen
#   2. TRANSFORM: '?' -> NULL, Leerspalte entfernen, Typen umwandeln,
#                 flache Tabelle in 5 normalisierte Tabellen aufteilen
#   3. LOAD:      Schema anlegen (schema.sql) und Tabellen befüllen
#
# Ausführen aus dem PROJEKTORDNER (wichtig wegen relativer Pfade):
#   python src/db/load_data.py
# =====================================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

# Projektordner zum Python-Pfad hinzufügen, damit der Import
# "from src.db.connection import ..." auch beim direkten Start klappt.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.db.connection import get_engine

# Pfade zentral definieren
CSV_PFAD = Path("data/raw/dataset.csv")
SCHEMA_PFAD = Path("sql/schema.sql")


def extrahiere() -> pd.DataFrame:
    """Schritt 1 (EXTRACT): Roh-CSV einlesen."""
    df = pd.read_csv(CSV_PFAD)
    print(f"[EXTRACT]   {len(df)} Zeilen, {df.shape[1]} Spalten eingelesen")
    return df


def transformiere(df: pd.DataFrame) -> pd.DataFrame:
    """Schritt 2 (TRANSFORM): Datenbereinigung.
    Alle Entscheidungen hier sind Ergebnis der Datenqualitätsprüfung
    (siehe docs/datensatzbeschreibung.md)."""

    # --- 2.1 Komplett leere Spalte "_c39" entfernen (Artefakt der CSV) ---
    df = df.drop(columns=["_c39"], errors="ignore")

    # --- 2.2 Platzhalter '?' in echte fehlende Werte (NULL) umwandeln ---
    # Betroffen: collision_type, property_damage, police_report_available
    df = df.replace("?", np.nan)

    # --- 2.3 Datumsspalten von Text in echte Datumstypen umwandeln ---
    df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"]).dt.date
    df["incident_date"]   = pd.to_datetime(df["incident_date"]).dt.date

    # --- 2.4 Zielvariable Klassifikation: 'Y'/'N' -> Boolean ---
    df["fraud_reported"] = df["fraud_reported"].map({"Y": True, "N": False})

    # --- 2.5 Künstliche IDs erzeugen ---
    # Im Datensatz entspricht 1 Zeile = 1 Kunde + 1 Police + 1 Unfall.
    # Die fortlaufende Nummer dient daher als gemeinsamer Schlüssel
    # für alle fünf Tabellen (kunde_id = police_id = unfall_id = Zeile).
    df["lfd_id"] = range(1, len(df) + 1)

    print(f"[TRANSFORM] Bereinigt: '?'-Werte -> NULL, _c39 entfernt, Typen gesetzt")
    return df


def lade(df: pd.DataFrame) -> None:
    """Schritt 3 (LOAD): Schema anlegen und die 5 Tabellen befüllen."""
    engine = get_engine()

    # --- 3.1 Schema ausführen (legt alle Tabellen neu an) ---
    with engine.begin() as conn:                      # begin() = mit Transaktion
        conn.execute(text(SCHEMA_PFAD.read_text()))
    print("[LOAD]      Schema angelegt (5 Tabellen)")

    # --- 3.2 Flache Tabelle in die normalisierten Tabellen aufteilen ---
    # Jedes Dictionary: CSV-Spalte -> deutsche Spalte in der Datenbank.

    kunden = df[["lfd_id", "age", "insured_sex", "insured_education_level",
                 "insured_occupation", "insured_hobbies", "insured_relationship",
                 "insured_zip", "capital-gains", "capital-loss",
                 "months_as_customer"]].rename(columns={
        "lfd_id": "kunde_id", "age": "alter_jahre", "insured_sex": "geschlecht",
        "insured_education_level": "bildungsniveau", "insured_occupation": "beruf",
        "insured_hobbies": "hobbys", "insured_relationship": "familienverhaeltnis",
        "insured_zip": "plz", "capital-gains": "kapitalgewinne",
        "capital-loss": "kapitalverluste", "months_as_customer": "kunde_seit_monaten",
    })

    policen = df[["lfd_id", "policy_number", "policy_bind_date", "policy_state",
                  "policy_csl", "policy_deductable", "policy_annual_premium",
                  "umbrella_limit"]].rename(columns={
        "lfd_id": "police_id", "policy_number": "policen_nummer",
        "policy_bind_date": "vertragsbeginn", "policy_state": "bundesstaat",
        "policy_csl": "deckungsgrenze", "policy_deductable": "selbstbeteiligung",
        "policy_annual_premium": "jahrespraemie",
    })
    policen["kunde_id"] = df["lfd_id"] # Fremdschlüssel Kunde -> Police

    fahrzeuge = df[["lfd_id", "auto_make", "auto_model", "auto_year"]].rename(
        columns={"auto_make": "marke", "auto_model": "modell",
                 "auto_year": "baujahr"})
    fahrzeuge = fahrzeuge.rename(columns={"lfd_id": "police_id"})  # FK Police

    unfälle = df[["lfd_id", "incident_date", "incident_type", "collision_type",
                   "incident_severity", "authorities_contacted", "incident_state",
                   "incident_city", "incident_location", "incident_hour_of_the_day",
                   "number_of_vehicles_involved", "property_damage",
                   "bodily_injuries", "witnesses",
                   "police_report_available"]].rename(columns={
        "lfd_id": "unfall_id", "incident_date": "unfall_datum",
        "incident_type": "unfall_typ", "collision_type": "kollisions_typ",
        "incident_severity": "schadensschwere",
        "authorities_contacted": "behoerde", "incident_state": "bundesstaat",
        "incident_city": "stadt", "incident_location": "adresse",
        "incident_hour_of_the_day": "uhrzeit_stunde",
        "number_of_vehicles_involved": "anzahl_fahrzeuge",
        "property_damage": "sachschaden_dritter",
        "bodily_injuries": "anzahl_verletzte", "witnesses": "anzahl_zeugen",
        "police_report_available": "polizeibericht",
    })
    unfälle["police_id"] = df["lfd_id"]      # Fremdschlüssel Police -> Unfall

    schaeden = df[["lfd_id", "total_claim_amount", "injury_claim",
                   "property_claim", "vehicle_claim", "fraud_reported"]].rename(
        columns={"lfd_id": "unfall_id", "total_claim_amount": "gesamtschaden",
                 "injury_claim": "personenschaden",
                 "property_claim": "sachschaden",
                 "vehicle_claim": "fahrzeugschaden",
                 "fraud_reported": "betrug_gemeldet"})

    # --- 3.3 Befüllen — Reihenfolge wegen Fremdschlüsseln wichtig! ---
    # (Eltern-Tabellen zuerst, sonst verletzen die FKs die Integrität)
    for name, tabelle in [("kunden", kunden), ("policen", policen),
                          ("fahrzeuge", fahrzeuge), ("unfaelle", unfälle),
                          ("schaeden", schaeden)]:
        tabelle.to_sql(name, engine, if_exists="append", index=False)
        print(f"[LOAD]      Tabelle '{name}': {len(tabelle)} Zeilen eingefügt")


def prüfe() -> None:
    """Abschließende Kontrolle: Zeilenzahlen und eine Beispiel-JOIN-Abfrage."""
    engine = get_engine()
    with engine.connect() as conn:
        # JOIN über alle 5 Tabellen als Funktionsnachweis der Schlüssel
        ergebnis = conn.execute(text("""
            SELECT k.alter_jahre, f.marke, u.schadensschwere, s.gesamtschaden
            FROM schaeden s
            JOIN unfaelle  u ON u.unfall_id  = s.unfall_id
            JOIN policen   p ON p.police_id  = u.police_id
            JOIN fahrzeuge f ON f.police_id  = p.police_id
            JOIN kunden    k ON k.kunde_id   = p.kunde_id
            LIMIT 3
        """)).fetchall()
    print("[PRÜFUNG]   Beispiel-JOIN über alle Tabellen:")
    for zeile in ergebnis:
        print("           ", zeile)


# ---------------------------------------------------------------------
# Hauptprogramm: kompletter ETL-Lauf
# ---------------------------------------------------------------------
if __name__ == "__main__":
    df = extrahiere()
    df = transformiere(df)
    lade(df)
    prüfe()
    print("\nFertig! Alle Daten liegen jetzt in PostgreSQL.")