<div align="center">

# 🚗 KFZ-Schadenprognose

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=4CAF50&center=true&vCenter=true&width=680&lines=Data+Science+%26+KI-Workflow;Von+der+Datenbank+bis+zur+Web-Oberfläche;Schadenprognose+aus+Text+und+Bild;Eine+Pipeline+%E2%80%93+drei+Zugänge" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.12-4CAF50?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Datenbank-4CAF50?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-4CAF50?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MultiHead-4CAF50?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-4CAF50?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-4CAF50?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-4CAF50?style=for-the-badge&logo=streamlit&logoColor=white)

**Gruppe InterGeeks · Hochschule Hannover · Fakultät III – Medien, Information und Design**

</div>

---

## 📋 Projektübersicht

Dieses Projekt setzt einen vollständigen **Data-Science- und KI-Workflow** rund um
KFZ-Versicherungsschäden um – von der Datenhaltung über die Analyse bis zur
Anwendung. Auf Basis eines offenen Datensatzes zu Versicherungsfällen werden der
**Schadenschweregrad** und die **voraussichtlichen Reparaturkosten** prognostiziert.

Das System kombiniert drei KI-Komponenten in einer **zentralen Pipeline**:

- **Multi-Head-Textmodell** (PyTorch) – rekonstruiert strukturierte Felder aus der Freitext-Schadensmeldung,
- **CNN-Bildklassifikator** (TensorFlow/Keras) – bestimmt die Schadensschwere aus einem Foto,
- **Regressionsmodell** (scikit-learn) – schätzt die erwarteten Reparaturkosten.

Die Pipeline ist über **drei Einstiegspunkte** erreichbar: **CLI**, **E-Mail-Worker**
und **Streamlit-Dashboard**.

> **Fragestellung:** Welche Faktoren beeinflussen die Schadenhöhe, und lässt sie
> sich aus Schadensmeldung und Foto zuverlässig prognostizieren?
>
> **Zielvariable:** `vehicle_claim` (Regression).

---

## ✨ Features

| Bereich | Umsetzung |
| --- | --- |
| 🗄️ **Datenhaltung** | Normalisiertes PostgreSQL-Schema (mehrere Tabellen, Fremdschlüssel, Indizes) |
| 🔄 **ETL** | CSV einlesen, bereinigen (`?`→NaN), Typen umwandeln und in die DB laden |
| 📊 **Deskriptive Statistik** | Lageparameter, Streuungsmaße, Verteilungen, Ausreißer, Datenqualität |
| 🔗 **Einfluss- & Zusammenhangsanalyse** | Korrelationen und Einfluss der Variablen auf die Zielvariable |
| 🧪 **Hypothesentests** | Formulierung, Voraussetzungsprüfung und Interpretation |
| 🤖 **Regression** (Pflicht) | Vorhersage der Schadenhöhe, Vergleich mehrerer Modelle |
| 🧠 **CNN-Bilderkennung** (Bonus) | Transfer Learning (MobileNetV2): Foto → Schadensschwere |
| 🔎 **Automatisierte Textanalyse** (Bonus) | Multi-Head-Modell: Freitext → strukturierte Felder |
| 🧩 **Zentrale Pipeline** | Ein Backend orchestriert alle Stufen für jeden Client |
| ✉️ **E-Mail-Worker** (Bonus) | Automatische Verarbeitung eingehender Schadens-E-Mails |
| 🖥️ **Web-Oberfläche** (Bonus) | Streamlit-Dashboard: Text- und Foto-Eingabe, Speicherung, Prognose, Diagramm |

---

## 🧩 Architektur

Die **Pipeline** (`src/services/pipeline.py`) ist der einzige Einstiegspunkt für jeden
Client. Sie orchestriert die fachliche Logik, während die Persistenz an ein
Repository delegiert wird (Single-Responsibility- und Dependency-Inversion-Prinzip).

```
        ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐
Clients │     CLI     │   │ E-Mail-Worker│   │    Dashboard    │
        └──────┬──────┘   └──────┬───────┘   └────────┬────────┘
               └─────────────────┼────────────────────┘
                                 ▼
                        ┌──────────────────┐
                        │     Pipeline     │   (src/services/pipeline.py)
                        └──────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
┌───────────────┐       ┌────────────────┐        ┌────────────────┐
│  Multi-Head   │       │      CNN       │        │   Regression   │
│  Text → Felder│       │  Foto → Schwere│        │ Felder → Kosten│
│   (PyTorch)   │       │  (TF/Keras)    │        │ (scikit-learn) │
└───────────────┘       └────────────────┘        └────────────────┘
        └────────────────────────┼────────────────────────┘
                                 ▼
                        ┌──────────────────┐
                        │   PostgreSQL     │   (Anfrage + Ergebnis je Stufe)
                        └──────────────────┘
```

**Ablauf eines Durchlaufs:** Anfrage speichern → Multi-Head-Textanalyse (falls Text)
→ CNN-Bildanalyse (falls Foto) → finale Schadensschwere bestimmen (höhere Konfidenz
gewinnt) → Regressions-Kostenschätzung. Jede Stufe persistiert ihr Ergebnis.

---

## 🛠️ Tech-Stack

- **Sprache:** Python 3.12
- **Datenbank:** PostgreSQL · SQLAlchemy · psycopg2
- **Datenverarbeitung:** pandas · NumPy
- **Machine Learning:** scikit-learn
- **Deep Learning:** PyTorch (Multi-Head) · TensorFlow / Keras – MobileNetV2 (CNN) · Transformers
- **Visualisierung:** Plotly · Matplotlib
- **Web-Oberfläche:** Streamlit
- **Versionsverwaltung & Agile:** Git · GitHub · Jira (Scrum)

---

## ⚙️ Installation

```bash
# Repository klonen
git clone https://github.com/fostij/InterGeeks-Agiles-Programmierprojekt.git
cd InterGeeks-Agiles-Programmierprojekt

# Virtuelle Umgebung (Python 3.12) anlegen und aktivieren
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

#Dieser Schritt registriert src/ und app/ als Python-Pakete im virtuellen Environment.
#Danach funktionieren alle Importe (from src.services.pipeline import ...) korrekt — unabhängig davon,
#aus welchem Verzeichnis ein Skript gestartet wird. Ohne diesen Schritt können ModuleNotFoundError-Fehler auftreten.
pip install -e .

# Zugangsdaten konfigurieren (.env aus Vorlage erstellen und ausfüllen)
cp .env.example .env
```

---

## 🔐 Umgebungsvariablen

Alle Konfigurationswerte werden über eine `.env`-Datei im Projektstammverzeichnis
bereitgestellt (nicht im Code, nicht im Repository). Vorlage:

```env
# Datenbankverbindung (PostgreSQL)
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_NAME=kfz_schaden

# E-Mail-Konfiguration (für den E-Mail-Worker)
EMAIL_IMAP_HOST=
EMAIL_SMTP_HOST=
EMAIL_SMTP_PORT=
EMAIL_USE_STARTTLS=
EMAIL_USERNAME=
EMAIL_PASSWORD=
EMAIL_POLL_INTERVAL=
EMAIL_IMAP_FOLDER=
```

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

## 🗄️ Datenbank vorbereiten

Vor dem ersten Start muss die Datenbank `kfz_schaden` angelegt werden:

```bash
# Datenbank anlegen
psql -U postgres -c "CREATE DATABASE kfz_schaden;"

```

Für die Befüllung der Datenbank gibt es zwei alternative Wege:

### Option A: Befüllung aus CSV-Rohdaten (Skripte)

Voraussetzung: Die Dateien `data/raw/dataset.csv` und `data/raw/vehicle_dataset.csv` müssen vorhanden sein. Die Skripte erstellen das Schema automatisch und laden die Daten.

```bash
# Versicherungs- und Fahrzeugdaten laden
python src/db/load_insurance_data.py
python src/db/load_vehicle_catalog.py

```

### Option B: Wiederherstellung aus einem SQL-Dump

Falls ein fertiger Datenbank-Dump (`backup.sql`) vorliegt, kann das Schema inklusive aller Daten direkt importiert werden:

```bash
# Daten aus SQL-Dump einspielen
psql -U postgres -d kfz_schaden -f backup.sql

```

### Schema

Der Original-Datensatz (eine flache CSV, 40 Spalten) wird in ein **normalisiertes
Schema** (3. Normalform) überführt; zusätzlich nehmen vier Tabellen die Ergebnisse
der Pipeline auf, eine weitere dient als Fahrzeug-Referenzkatalog.

```
kunden ──< policen ──< unfaelle ──< schaeden
               │
               └──< fahrzeuge

anfragen ──< multihead_ergebnisse
         ├──< cnn_ergebnisse
         └──< regression_ergebnisse

fahrzeug_katalog        (Referenzdaten: Marke / Modell / Baujahr)
```

| Tabelle | Inhalt |
|---|---|
| `kunden` | Stammdaten der Versicherungsnehmer |
| `policen` | Versicherungsverträge (1 Kunde → n Policen) |
| `fahrzeuge` | versicherte Fahrzeuge |
| `unfaelle` | gemeldete Vorfälle |
| `schaeden` | Schadenszahlungen (Zielvariable des Projekts) |
| `anfragen` | je eine Zeile pro eingehender Anfrage (Dashboard, E-Mail, CLI) |
| `multihead_ergebnisse` | aus dem Text rekonstruierte Felder inkl. Konfidenzwerten |
| `cnn_ergebnisse` | Foto-basierte Schweregrad-Erkennung |
| `regression_ergebnisse` | finale Kostenprognose (`vehicle_claim`) |
| `fahrzeug_katalog` | Marke/Modell/Baujahr-Referenzdaten |

Geldbeträge werden als `NUMERIC` gespeichert, fehlende Werte als `NULL`, Fremdschlüssel
sind indiziert. Die Verbindung erfolgt zentral über eine SQLAlchemy-Engine; die
Zugangsdaten stammen aus der `.env`-Datei.

---

## 🧠 Modelle trainieren

### Vorausgesetzte Dateistruktur

```
data/
└── raw/
    ├── dataset.csv                  # Schadensbeschreibungen + Metadaten
    ├── vehicle_dataset.csv          # Fahrzeugdaten (Referenzkatalog)
    └── car_damage/
        ├── training/
        │   ├── 01-minor/            # Bilder: leichte Schäden
        │   ├── 02-moderate/         # Bilder: mittlere Schäden
        │   └── 03-severe/           # Bilder: schwere Schäden
        └── validation/
            ├── 01-minor/
            ├── 02-moderate/
            └── 03-severe/
```

### Training starten

```bash
# Alle drei Modelle nacheinander trainieren
python -m src.services.train_all

# Alternativ einzeln, z. B. nur das CNN:
python -m src.models.cnn.train_cnn
```

Vor dem Start in `src/services/train_all.py` festlegen, was ausgeführt werden soll:

```python
# Datenbankbefüllung (nur nötig, wenn noch nicht geschehen)
RUN_LOAD_INSURANCE_DATA = False
RUN_LOAD_VEHICLE_CATALOG = False

# Modelle trainieren
RUN_TRAIN_MULTIHEAD = False
RUN_TRAIN_CNN = True
RUN_TRAIN_REGRESSION = True
```

Das Skript `train_all.py` trainiert alle Modelle und speichert die Gewichte
automatisch in die erwarteten Pfade.

Trainings-Hyperparameter und CNN-Konstanten (Bildgröße, Batch-Size, Epochen usw.) befinden sich in den jeweiligen Config-Dateien:

```
src/models/cnn/cnn_config.py                    # CNN-Trainingskonstanten
src/models/multi_head_insurance/multi_head_config.py        # Multi-Head-Hyperparameter
src/models/regression/regression_config.py                  # Regressions-Hyperparameter
```

---

## ▶️ Starten der Anwendung

### Voraussetzung: Modellgewichte

Für jeden Einstiegspunkt müssen die trainierten Modellgewichte vorhanden sein:

```
checkpoints/multi_head_model.pt   # Textklassifikation (PyTorch)
checkpoints/cnn_model.keras                        # Bildklassifikation (TensorFlow/Keras)
checkpoints/classes.json                           # Klassenreihenfolge ["01-minor","02-moderate","03-severe"]
checkpoints/regression_model.pkl            # Kostenprognose (scikit-learn)
```

> Die Gewichtsdateien werden **nicht** im Repository eingecheckt und müssen lokal
> vorliegen (durch Training oder Bereitstellung im Team).

### 1. CLI (Kommandozeile)

```bash
# Nur Text
python main.py --text "Frontschaden nach Auffahrunfall, Airbag ausgelöst"

# Text und Foto
python main.py --text "Seitenschaden am vorderen Kotflügel" --photo foto.jpg

# Nur Foto
python main.py --photo foto.jpg
```

| Parameter | Beschreibung |
|---|---|
| `--text` | Freitextbeschreibung des Schadens |
| `--photo` | Pfad zu einem Schadensfoto (JPG, PNG) |

Mindestens ein Parameter muss angegeben werden. Die Ergebnisse (Schweregrad,
Konfidenz, Kostenschätzung) werden direkt in der Konsole ausgegeben.

### 2. E-Mail-Worker

```bash
python -m src.automation.email_worker
```

Der Worker liest die E-Mail-Konfiguration aus der `.env`-Datei, verarbeitet
eingehende E-Mails mit Schadensbeschreibungen und/oder Fotos automatisch durch die
Pipeline und sendet das Ergebnis per E-Mail zurück.

### 3. Streamlit-Dashboard

```bash
streamlit run app/dashboard.py
```

Anschließend im Browser unter `http://localhost:8501` erreichbar.

Im Dashboard wird links eine **Schadensmeldung** als Freitext eingegeben und rechts
optional ein **Foto** hochgeladen. Nach dem Klick auf *Prognose erstellen* ruft das
Dashboard die zentrale Pipeline auf und zeigt die extrahierten Felder, die
Schadensschwere (aus Text und/oder Foto), die prognostizierten Kosten sowie ein
Vergleichsdiagramm an. Anfrage und Ergebnisse werden in der Datenbank gespeichert.

---

## 🔧 Funktionsweise (Text & Foto)

Text und Foto liefern unterschiedliche, sich ergänzende Informationen:

```
Foto  -> CNN (visuell)             -> Schwere (Konfidenz) |
                                                          +--> finale Schwere --> Regression --> Kosten
Text  -> Multi-Head (strukturiert) -> Schwere (Konfidenz) |
```

- **Nur Text:** Das Multi-Head-Modell rekonstruiert die Felder inkl. Schadensschwere.
- **Nur Foto:** Das CNN bestimmt die Schadensschwere.
- **Text + Foto:** Beide Quellen schätzen die Schwere unabhängig; die Schätzung mit
  der **höheren Konfidenz** gewinnt. Auf abweichende Einschätzungen weist das System hin.

---

## 📁 Projektstruktur

```
InterGeeks-Agiles-Programmierprojekt/
├── app/
│   ├── dashboard.py               # Streamlit-Dashboard (Frontend der Pipeline)
│   └── app_config.py              # Konstanten der Anwendungsebene
├── checkpoints/                   # Ordner für Modellgewichte (nicht eingecheckt)
├── data/
│   └── raw/                       # Rohdaten (nicht eingecheckt)
├── docs/                          # Dokumentation & Sprint-Protokolle
├── prassentation/                 # Präsentationsfolien
├── sql/
│   └── schema.sql                 # PostgreSQL-Schema
├── src/
│   ├── analysis/                  # Statistik-Notebooks
│   ├── automation/
│   │   └── email_worker.py        # E-Mail-Worker
│   ├── db/
│   │   ├── connection.py          # zentrale DB-Verbindung
│   │   ├── load_insurance_data.py # ETL: Versicherungsdaten
│   │   ├── load_vehicle_catalog.py# ETL: Fahrzeugkatalog
│   │   └── pipeline_repository.py # Persistenz der Pipeline-Ergebnisse
│   ├── models/
│   │   ├── cnn/                   # CNN: Training & Vorhersage
│   │   │   ├── cnn_config.py
│   │   │   └── ...
│   │   ├── multi_head_insurance/  # Multi-Head-Textmodell
│   │   │   ├── multi_head_config.py
│   │   │   └── ...
│   │   └── regression/            # Kostenregressionsmodell
│   │   │   ├── regression_config.py
│   │   │   └── ...
│   ├── services/
│   │   ├── pipeline.py            # zentrale Pipeline (Backend)
│   │   └── train_all.py           # Training aller Modelle
│   ├── exceptions.py
│   ├── logging_config.py
│   └── ml_config.py               # zentrale Pfad-/Modellkonfiguration
├── utils/
│   └── fake_data_generators/      # Hilfsskripte für Testdaten
├── .streamlit/
│   └── config.toml                # Theme im Farbschema der Hochschule
├── main.py                        # CLI-Einstiegspunkt
├── requirements.txt
├── pyproject.toml
└── .env                           # Lokale Konfiguration (nicht eingecheckt)
```

---

## 🔭 Ausblick

- Diskussion der **Modellgrenzen** (begrenzter Bilddatensatz, Genauigkeit der
  einzelnen Stufen).
- Verbesserung der CNN-Generalisierung durch einen größeren, ausgewogeneren
  Bilddatensatz.
- Erweiterungen außerhalb des Projektumfangs: Datenschutz, OCR für Bild-PDFs.

---

## ⚠️ Hinweis

Es werden ausschließlich **Testdaten** verwendet – keine realen Kundendaten. Die
Beträge des Originaldatensatzes (USD) werden im Projekt als EUR-Testwerte interpretiert.

---

## 👥 Team

* Oleg Fostii
* Anton Duschak
* Bernard Turikumana

Projektarbeit im Rahmen des Agilen
Programmierprojekts an der Hochschule Hannover – Fakultät III, Medien, Information
und Design.
