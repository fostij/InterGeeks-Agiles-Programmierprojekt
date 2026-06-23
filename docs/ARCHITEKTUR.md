# Systemarchitektur — KFZ-Schadensprognose

## Überblick

Das System besteht aus drei unabhängigen Einstiegspunkten, die alle auf dieselbe zentrale Pipeline zugreifen. Die Pipeline kombiniert drei spezialisierte KI-Modelle zu einer einzigen Schadensbewertung und persistiert die Ergebnisse in einer PostgreSQL-Datenbank.

```
         ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
Eingabe  │     CLI      │   │   E-Mail-    │   │   Streamlit-     │
         │  (main.py)   │   │   Worker     │   │   Dashboard      │
         └──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
                │                  │                     │
                └──────────────────┼─────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │    Pipeline     │
                          │ (services/      │
                          │  pipeline.py)   │
                          └────────┬────────┘
               ┌───────────────────┘
               │
      ┌────────▼──────┐  ┌────────────────┐
      │  Multihead-   │  │  CNN-Modell    │
      │  Textmodell   │  │  (Bild)        │
      │  (.pt)        │  │  (.keras)      │
      └───────┬───────┘  └───────┬────────┘
   Text-      │                  │  Bild-
   Features   └────────┬─────────┘  Features
                       │
              ┌────────▼──────┐
              │ Regressions-  │
              │ modell (.pkl) │
              └───────────────┘
                                   │
                          ┌────────▼────────┐
                          │   PostgreSQL    │
                          │  (kfz_schaden)  │
                          └─────────────────┘
```

---

## Einstiegspunkte

### CLI (`main.py`)

Der einfachste Einstiegspunkt. Wird direkt aus dem Terminal aufgerufen und gibt das Ergebnis auf der Konsole aus. Geeignet für manuelle Tests und Entwicklung.

- Liest Argumente mit `argparse` (`--text`, `--photo`)
- Lädt die `.env`-Datei mit `python-dotenv`
- Instanziiert `Pipeline` und ruft `pipeline.run()` auf
- Quelle im Ergebnis: `"cli"`

### E-Mail-Worker (`src/automation/email_worker.py`)

Ein dauerhaft laufender Prozess, der ein IMAP-Postfach in konfigurierbaren Intervallen abfragt. Eingehende E-Mails werden als Schadensanfragen interpretiert. Das Ergebnis wird per SMTP zurückgesendet.

- Verbindung über IMAP (Empfang) und SMTP mit STARTTLS (Versand)
- Anhänge (Fotos) werden automatisch erkannt und der Pipeline übergeben
- Konfiguration vollständig über `.env`
- Quelle im Ergebnis: `"email"`

### Streamlit-Dashboard (`app/dashboard.py`)

Eine interaktive Weboberfläche für manuelle Eingaben und zur Anzeige historischer Ergebnisse aus der Datenbank.

- Eingabe: Freitext und/oder Foto-Upload
- Zeigt Schweregrad, Konfidenz und Kostenschätzung
- Visualisiert historische Anfragen aus der Datenbank
- Quelle im Ergebnis: `"dashboard"`

---

## Kern-Pipeline (`src/services/pipeline.py`)

Die Pipeline ist der zentrale Verarbeitungsknoten. Sie nimmt Text und/oder Foto entgegen, koordiniert die Modelle und gibt ein strukturiertes Ergebnis zurück.

```
pipeline.run(text, photo_path, source)
        │
        ├─► Multihead-Textmodell ──► Merkmalsvektor (Text-Features)
        │                                       │
        ├─► CNN-Modell ──────────► Merkmalsvektor (Bild-Features)
        │                                       │
        │              ┌─────────────────────────┘
        │              │
        ├─► Vergleich & Zusammenführung der Feature-Vektoren
        │   (kombinierter Merkmalsvektor aus Text + Bild)
        │                                       │
        ├─► Regressionsmodell ──────────────────┘
        │       └──► Schweregrad + Kostenschätzung (EUR)
        │
        └─► PipelineResultRepository.save() ──► PostgreSQL
```

**Ausgabefelder des `PipelineResult`:**

| Feld | Typ | Beschreibung |
|---|---|---|
| `request_id` | UUID | Eindeutige Anfragen-ID |
| `final_severity` | str | `minor` / `moderate` / `severe` |
| `severity_source` | str | `text` / `photo` / `fusion` |
| `text_confidence` | float | Konfidenz des Textmodells (0–1) |
| `photo_confidence` | float | Konfidenz des CNN (0–1) |
| `predicted_amount` | float | Geschätzte Reparaturkosten in EUR |
| `missing_fields` | list | Fehlende Eingaben |

---

## KI-Modelle

### Multihead-Textmodell (`src/models/multi_head_insurance/`)

- **Typ:** PyTorch-Modell mit mehreren Ausgabe-Köpfen
- **Rolle:** Feature-Extraktor — extrahiert einen numerischen Merkmalsvektor aus dem Freitext
- **Eingabe:** Schadensbeschreibung als Text
- **Ausgabe:** Merkmalsvektor (Text-Features), der an das Regressionsmodell weitergegeben wird
- **Gewichte:** `multi_head_model.pt`
- **Basis:** Transformer-Architektur (HuggingFace `transformers`)

### CNN-Bildklassifikator (`src/models/cnn/`)

- **Typ:** Convolutional Neural Network (TensorFlow/Keras)
- **Rolle:** Feature-Extraktor — extrahiert einen numerischen Merkmalsvektor aus dem Schadensfoto
- **Eingabe:** Schadensfoto (JPG/PNG)
- **Ausgabe:** Merkmalsvektor (Bild-Features), der zusammen mit den Text-Features verglichen und kombiniert wird
- **Gewichte:** `cnn_model.keras`
- **Training:** Separate Trainings- und Validierungsordner je Klasse (`01-minor`, `02-moderate`, `03-severe`)

### Regressionsmodell (`src/models/regression/`)

- **Typ:** scikit-learn Regressor (serialisiert als Pickle)
- **Rolle:** Finale Vorhersage — nimmt den kombinierten Feature-Vektor aus Textmodell und CNN entgegen
- **Eingabe:** Zusammengeführte Merkmale aus Text + Bild (sowie Fahrzeugdaten)
- **Ausgabe:** Schweregrad der Beschädigung + geschätzter Reparaturbetrag in EUR
- **Gewichte:** `regression_model.pkl`

---

## Datenfluss beim Training

```
data/raw/dataset.csv              ──► Textmodell-Training
data/raw/vehicle_dataset.csv      ──► Regressionsmodell-Training
data/raw/car_damage/training/     ──► CNN-Training
data/raw/car_damage/validation/   ──► CNN-Validierung

python src/services/train_all.py
        │
        ├─► Trainiert Textmodell  ──► speichert multi_head_model.pt
        ├─► Trainiert CNN         ──► speichert cnn_model.keras
        └─► Trainiert Regression  ──► speichert regression_model.pkl
```

---

## Datenbankschicht (`src/db/`)

### `PipelineResultRepository`

Kapselt alle Datenbankoperationen. Verwendet SQLAlchemy als ORM.

- `save(result)` — persistiert ein `PipelineResult`-Objekt
- Verbindungsparameter kommen aus der `.env`-Datei
- Datenbankname: `kfz_schaden` (PostgreSQL)

Die SQL-Schemata und Migrationen befinden sich im Verzeichnis `sql/`.

---

## Konfiguration und Abhängigkeiten

Alle Laufzeitkonfiguration erfolgt über `.env`. Es gibt keine fest codierten Zugangsdaten.

| Bereich | Technologie |
|---|---|
| Datenspeicherung | PostgreSQL + SQLAlchemy |
| Textmodell | PyTorch + HuggingFace Transformers |
| Bildmodell | TensorFlow / Keras |
| Regressionsmodell | scikit-learn |
| Dashboard | Streamlit + Plotly |
| Bildverarbeitung | Pillow |
| E-Mail | IMAP / SMTP (STARTTLS) |
| Konfiguration | python-dotenv |

---

## Verzeichnisstruktur (Kurzübersicht)

```
src/
├── automation/
│   └── email_worker.py       # E-Mail-Einstiegspunkt
├── db/
│   └── pipeline_repository.py
├── models/
│   ├── cnn/
│   │   └── cnn_model.keras
│   ├── multi_head_insurance/
│   │   └── multi_head_model.pt
│   └── regression/
│       └── regression_model.pkl
├── services/
│   ├── pipeline.py           # Zentrale Pipeline
│   └── train_all.py          # Trainings-Orchestrator
├── exceptions.py
└── logging_config.py
```
