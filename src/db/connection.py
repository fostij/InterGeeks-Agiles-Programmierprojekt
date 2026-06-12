# =====================================================================
# connection.py — zentrale Datenbankverbindung (PostgreSQL)
# ---------------------------------------------------------------------
# Die Zugangsdaten liegen NICHT im Code, sondern in der Datei ".env"
# im Projektordner (steht in .gitignore -> landet NIE auf GitHub!).
#
# Beispiel-Inhalt der .env-Datei:
#   DB_USER=postgres
#   DB_PASSWORD=geheim
#   DB_HOST=localhost
#   DB_PORT=5432
#   DB_NAME=kfz_schaden
# =====================================================================

import os
from dotenv import load_dotenv          # liest die .env-Datei ein
from sqlalchemy import create_engine    # einheitliche DB-Schnittstelle

# .env-Variablen in die Umgebung laden
load_dotenv()


def get_engine():
    """Erstellt und liefert eine SQLAlchemy-Engine für PostgreSQL.
    Alle Module (Laden, Analyse, Dashboard) nutzen DIESE eine Funktion —
    so gibt es die Verbindungslogik nur an einer Stelle (DRY-Prinzip)."""
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    dbname   = os.getenv("DB_NAME", "kfz_schaden")

    # Verbindungs-URL: postgresql+psycopg2://user:passwort@host:port/datenbank
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url)