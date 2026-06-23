# KFZ-Schadensprognose-System

Ein KI-gestütztes System zur automatisierten Bewertung von Kfz-Schäden. Es kombiniert Textanalyse, Bildverarbeitung und Regressionsmodelle, um Schadenschweregrad und voraussichtliche Reparaturkosten vorherzusagen. Das System ist über drei Einstiegspunkte erreichbar: CLI, E-Mail-Worker und Streamlit-Dashboard.

---

## Inhaltsverzeichnis

- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Umgebungsvariablen](#umgebungsvariablen)
- [Modelle trainieren](#modelle-trainieren)
- [Starten der Anwendung](#starten-der-anwendung)
- [Projektstruktur](#projektstruktur)

---

## Voraussetzungen

- Python 3.10 oder höher
- PostgreSQL (Datenbankname: `kfz_schaden`)
- Internetzugang (für E-Mail-Funktionalität)

Abhängigkeiten installieren:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
git clone https://github.com/fostij/InterGeeks-Agiles-Programmierprojekt.git
cd InterGeeks-Agiles-Programmierprojekt
pip install -r requirements.txt
cp .env.example .env
# .env mit eigenen Werten befüllen (siehe unten)
```

---

## Umgebungsvariablen

Alle Konfigurationswerte werden über eine `.env`-Datei im Projektstammverzeichnis bereitgestellt. Eine Vorlage:

```env
# Datenbankverbindung (PostgreSQL)
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=kfz_schaden

# E-Mail-Konfiguration
EMAIL_IMAP_HOST=
EMAIL_SMTP_HOST=
EMAIL_SMTP_PORT=
EMAIL_USE_STARTTLS=
EMAIL_USERNAME=
EMAIL_PASSWORD=
EMAIL_POLL_INTERVAL=
EMAIL_IMAP_FOLDER=
```

### Beschreibung der Variablen

| Variable | Beschreibung | Beispielwert |
|---|---|---|
| `DB_USER` | PostgreSQL-Benutzername | `postgres` |
| `DB_PASSWORD` | Datenbankpasswort | `geheim123` |
| `DB_HOST` | Datenbankhost | `localhost` |
| `DB_PORT` | Datenbankport | `5432` |
| `DB_NAME` | Datenbankname | `kfz_schaden` |
| `EMAIL_IMAP_HOST` | IMAP-Server für den Empfang | `imap.gmail.com` |
| `EMAIL_SMTP_HOST` | SMTP-Server für den Versand | `smtp.gmail.com` |
| `EMAIL_SMTP_PORT` | SMTP-Port | `587` |
| `EMAIL_USE_STARTTLS` | STARTTLS aktivieren | `true` |
| `EMAIL_USERNAME` | E-Mail-Adresse | `robot@example.com` |
| `EMAIL_PASSWORD` | E-Mail-Passwort oder App-Passwort | `••••••••` |
| `EMAIL_POLL_INTERVAL` | Prüfintervall in Sekunden | `60` |
| `EMAIL_IMAP_FOLDER` | Zu überwachender Postfach-Ordner | `INBOX` |

---

## Datenbank vorbereiten

Vor dem ersten Start muss die Datenbank erstellt und mit den Basisdaten befüllt werden:

```bash
# Datenbank anlegen
psql -U postgres -c "CREATE DATABASE kfz_schaden;"

# Versicherungs- und Fahrzeugdaten laden
python src/db/load_insurance_data.py
python src/db/load_vehicle_catalog.py
```

---

## Modelle trainieren

### Vorausgesetzte Dateistruktur

Bevor das Training gestartet wird, müssen folgende Dateien und Verzeichnisse vorhanden sein:

```
data/
└── raw/
    ├── dataset.csv                  # Textdaten (Schadensbeschreibungen + Metadaten)
    ├── vehicle_dataset.csv          # Fahrzeugdaten für das Regressionsmodell
    └── car_damage/
        ├── training/
        │   ├── 01-minor/            # Bilder mit leichten Schäden
        │   ├── 02-moderate/         # Bilder mit mittleren Schäden
        │   └── 03-severe/           # Bilder mit schweren Schäden
        └── validation/
            ├── 01-minor/
            ├── 02-moderate/
            └── 03-severe/
```

### Training starten

```bash
python src/services/train_all.py
```

Das Skript trainiert alle drei Modelle nacheinander und speichert die Gewichte automatisch in die richtigen Pfade.

---

## Starten der Anwendung

### Voraussetzung: Modellgewichte

Für jeden Einstiegspunkt müssen die trainierten Modellgewichte vorhanden sein:

```
src/models/multi_head_insurance/multi_head_model.pt   # Textklassifikation (PyTorch)
src/models/cnn/cnn_model.keras                        # Bildklassifikation (TensorFlow/Keras)
src/models/regression/regression_model.pkl            # Kostenprognose (scikit-learn)
```

Ohne diese Dateien kann die Pipeline nicht ausgeführt werden.

---

### 1. CLI (Kommandozeile)

Direkter Aufruf der Vorhersage-Pipeline aus dem Terminal.

```bash
# Nur Text
python main.py --text "Frontschaden nach Auffahrunfall, Airbag ausgelöst"

# Text und Foto
python main.py --text "Seitenschaden am vorderen Kotflügel" --photo foto.jpg

# Nur Foto
python main.py --photo foto.jpg
```

**Parameter:**

| Parameter | Beschreibung |
|---|---|
| `--text` | Freitextbeschreibung des Schadens |
| `--photo` | Pfad zu einem Schadensfoto (JPG, PNG) |

Mindestens ein Parameter muss angegeben werden. Die Ergebnisse (Schweregrad, Konfidenz, Kostenschätzung) werden direkt in der Konsole ausgegeben.

---

### 2. E-Mail-Worker

Überwacht automatisch ein E-Mail-Postfach und verarbeitet eingehende Schadensanfragen.

```bash
python -m src.automation.email_worker
```

Der Worker liest die E-Mail-Konfiguration aus der `.env`-Datei. Eingehende E-Mails mit Schadensbeschreibungen und/oder angehängten Fotos werden automatisch durch die Pipeline verarbeitet und das Ergebnis per E-Mail zurückgesendet.

---

### 3. Streamlit-Dashboard

Grafische Weboberfläche zur Eingabe und Auswertung von Schadensanfragen.

```bash
streamlit run app/dashboard.py
```

Das Dashboard ist anschließend im Browser unter `http://localhost:8501` erreichbar.

---

## Projektstruktur

```
InterGeeks-Agiles-Programmierprojekt/
├── app/
│   └── dashboard.py              # Streamlit-Dashboard
├── data/
│   └── raw/                      # Rohdaten (nicht eingecheckt)
├── docs/                         # Dokumentation
├── sql/                          # Datenbankschema und Migrationen
├── src/
│   ├── automation/
│   │   └── email_worker.py       # E-Mail-Worker
│   ├── db/
│   │   └── pipeline_repository.py
│   ├── models/
│   │   ├── cnn/                  # CNN-Bildklassifikator
│   │   ├── multi_head_insurance/ # Multihead-Textmodell
│   │   └── regression/           # Kostenregressionsmodell
│   ├── services/
│   │   ├── pipeline.py           # Haupt-Pipeline
│   │   └── train_all.py          # Training aller Modelle
│   ├── exceptions.py
│   └── logging_config.py
├── utils/
│   └── fake_data_generators/     # Hilfsskripte für Testdaten
├── main.py                       # CLI-Einstiegspunkt
├── requirements.txt
└── .env                          # Lokale Konfiguration (nicht eingecheckt)
```
