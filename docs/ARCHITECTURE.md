# Architektur

Dieses Dokument beschreibt die Code-Struktur des Projekts: welches Modul
wofür zuständig ist, welche Abhängigkeiten zwischen den Modulen bestehen
und welche Entwurfsentscheidungen dahinterstehen. Es ergänzt die
Projektübersicht in der README.md, die den fachlichen Workflow
beschreibt, um die Sicht auf den Code selbst.

## Überblick

```
app/                        Streamlit-Dashboard (eigenständiger Prototyp)
main.py                     Bisher ungenutzter Einstiegspunkt
sql/schema.sql               Datenbankschema (PostgreSQL, 3. Normalform)
src/
  exceptions.py              Projektweite Exception-Hierarchie
  logging_config.py          Zentrale Logging-Konfiguration
  ml_config.py                Zentrale Konfiguration (Pfade, Feldlisten)
  automation/                 Geplante, noch nicht implementierte Automatisierung
  db/                          Datenbankzugriff (Verbindung, ETL, Repository)
  models/                       Die drei ML-Modelle (Regression, Multi-Head, CNN)
  services/                     Orchestrierung (Pipeline, Trainings-Runner)
  utils/                        Datenaufbereitung, Statistik, synthetische Daten
```

Drei unabhängige Modelle sagen jeweils einen Teil des Schadensfalls voraus:

- **Regression** (`src/models/regression/`): strukturierte Felder → Schadenhöhe in EUR.
- **Multi-Head-Klassifikation** (`src/models/multi_head_insurance/`): Freitext → mehrere strukturierte Felder gleichzeitig (Marke, Baujahr, Schwere, ...).
- **CNN** (`src/models/cnn/`): Foto → Schadensschwere (3 Klassen).

`src/services/pipeline.py` verbindet alle drei zu einem Gesamtablauf für
die produktive Nutzung. `app/dashboard.py` ist davon unabhängig: ein
früher Streamlit-Prototyp mit eigener, regelbasierter Platzhalterlogik
(siehe Abschnitt dazu unten).

---

## Wurzelverzeichnis

### `main.py`
Vorgesehener Haupteinstiegspunkt des Projekts. Importiert aktuell
`Pipeline`, `PredictionResult` und `TrainConfig`, enthält aber noch keine
Logik (`main()` tut nichts). Dient als Platzhalter für eine zukünftige
CLI- oder API-Anbindung der Pipeline.

### `sql/schema.sql`
Definiert das PostgreSQL-Schema in zwei Gruppen von Tabellen:

1. **Stammdaten** (3. Normalform, aus dem Versicherungsdatensatz
   abgeleitet): `kunden`, `policen`, `fahrzeuge`, `unfaelle`, `schaeden`
   sowie `fahrzeug_katalog` (Marke/Modell/Baujahr-Referenz).
2. **Pipeline-Ergebnisse** (befüllt zur Laufzeit durch
   `PipelineResultRepository`): `anfragen` (Rohanfragen),
   `multihead_ergebnisse`, `cnn_ergebnisse`, `regression_ergebnisse`
   (je ein Ergebnis pro Pipeline-Stufe, referenziert über `anfrage_id`).

---

## `app/` — Dashboard (eigenständiger Prototyp)

### `app/dashboard.py`
Streamlit-Weboberfläche zur manuellen Eingabe einer Schadensmeldung
(Text und/oder Foto) und Anzeige einer geschätzten Schadenhöhe.

**Wichtig:** Dieses Modul verwendet **nicht** die echten ML-Modelle aus
`src/models/` bzw. `src/services/pipeline.py`. Stattdessen enthält es
eine eigene, regelbasierte Platzhalterlogik:

- `extract_features()` erkennt Schadensschwere und Zahlenwerte per
  Schlüsselwortsuche und regulären Ausdrücken im deutschen Freitext.
- `predict_amount()` berechnet die Schadenhöhe über feste Basiswerte je
  Schwere zuzüglich Zuschlägen (kein trainiertes Regressionsmodell).
- `predict_severity_from_photo()` ruft zwar das echte CNN auf
  (`src.models.cnn.predict_image.predict_severity`), bettet das Ergebnis
  aber in dieselbe eigenständige Logik ein.
- `save_prediction()` schreibt in eine eigene Tabelle `vorhersagen`,
  getrennt von den Pipeline-Tabellen aus `schema.sql`.

Diese Trennung ist im Code als TODO vermerkt ("VORLAEUFIGER PLATZHALTER
... durch das trainierte Regressionsmodell ersetzen"). Eine spätere
Anbindung an `Pipeline` würde Dashboard und Pipeline-Backend
vereinheitlichen.

---

## `src/exceptions.py` — Fehlerhierarchie

Definiert projektweite Exception-Klassen, alle abgeleitet von der
gemeinsamen Basisklasse `ProjectError`:

| Klasse | Bedeutung |
|---|---|
| `DataPreparationError` | Rohdaten können nicht geladen/bereinigt werden |
| `ModelLoadError` | Ein trainiertes Modell kann nicht geladen werden |
| `InferenceError` | Eine Modell-Vorhersage schlägt fehl |
| `TrainingError` | Das Training eines Modells schlägt fehl |
| `PersistenceError` | Ein Datenbankzugriff schlägt fehl |

Module fangen an Bibliotheksgrenzen typischerweise die eingebauten
Exceptions (z. B. `OSError`, `SQLAlchemyError`) ab und werfen stattdessen
die passende, aussagekräftige Exception aus dieser Hierarchie weiter.

## `src/logging_config.py` — Logging

Stellt `setup_logging()` bereit, die den Root-Logger einmalig pro
Prozess konfiguriert (Konsole, optional zusätzlich eine Log-Datei).
Jedes andere Modul erstellt nur seinen eigenen Logger über
`logging.getLogger(__name__)`; die Konfiguration übernimmt ausschließlich
der jeweilige Einstiegspunkt (z. B. `train_model.py`, `train_all.py`).

## `src/ml_config.py` — Zentrale Konfiguration

Bündelt projektweite Konstanten an einer Stelle: Datenpfade
(`INSURANCE_DATASET_PATH`, `VEHICLE_DATASET_PATH`, ...), Modellpfade
(`MULTI_HEAD_MODEL_PATH`, `CNN_MODEL_PATH`, `REGRESSION_MODEL_PATH`),
die Zielfelder des Versicherungsdatensatzes (`TARGET_FIELDS`) sowie
Hilfslisten wie `FIELDS_TO_DELETE` und `INTEGER_FIELDS`.

---

## `src/automation/` — Geplante Automatisierung

### `src/automation/email_parser.py`
Aktuell leere Datei. Vorgesehen für eine zukünftige Anbindung, bei der
Schadensmeldungen automatisiert aus eingehenden E-Mails extrahiert und
an die Pipeline übergeben werden, statt sie nur über das Dashboard
einzugeben.

---

## `src/db/` — Datenbankzugriff

### `src/db/connection.py`
Stellt `get_engine()` bereit: erzeugt eine SQLAlchemy-Engine für
PostgreSQL anhand von Umgebungsvariablen (`.env`-Datei). Einzige Stelle
im Projekt, die die Verbindungs-URL zusammensetzt (DRY-Prinzip) — alle
anderen Module, die eine Datenbankverbindung benötigen, rufen diese
Funktion auf.

### `src/db/load_insurance_data.py`
ETL-Skript (Extract – Transform – Load), das den rohen
Versicherungsdatensatz (CSV) einliest, bereinigt und auf die fünf
normalisierten Tabellen (`kunden`, `policen`, `fahrzeuge`, `unfaelle`,
`schaeden`) aufteilt. Legt dabei auch das Datenbankschema an
(`sql/schema.sql`). Wird einmalig vor dem Training ausgeführt (Schritt 1
in `train_all.py`).

### `src/db/load_vehicle_catalog.py`
ETL-Skript für den Fahrzeugkatalog (Marke/Modell/Baujahr-Kombinationen).
Befüllt die Tabelle `fahrzeug_katalog`, die sowohl für die
Trainingsdaten-Generierung (`AccidentDataGenerator`) als auch für die
Validierung vorhergesagter Fahrzeuge zur Inferenzzeit
(`validate_vehicle_with()`) verwendet wird. Wird einmalig vor dem
Training ausgeführt (Schritt 2 in `train_all.py`).

### `src/db/pipeline_repository.py`
`PipelineResultRepository` kapselt sämtliche SQL-Zugriffe, die beim
Durchlauf von `Pipeline` (siehe `src/services/pipeline.py`) anfallen:
das Speichern der Rohanfrage sowie der Ergebnisse jeder Pipeline-Stufe
(Multi-Head, CNN, Regression). Diese Trennung folgt dem
Single-Responsibility-Prinzip: `Pipeline` orchestriert die fachliche
Logik, das Repository ist ausschließlich für die Persistenz zuständig.
`Pipeline` hängt dabei nur von dieser Abstraktion ab, nicht von
SQLAlchemy direkt (Dependency-Inversion-Prinzip) — ein Austausch der
Speicherung erfordert keine Änderung an `Pipeline`.

---

## `src/models/regression/` — Schadenhöhen-Schätzung

Sagt `vehicle_claim` (Schadenhöhe in EUR) anhand strukturierter Felder
voraus (RandomForestRegressor mit One-Hot-Encoding der kategorialen
Spalten).

- **`prepare_data.py`** — Lädt den Versicherungsdatensatz und bereitet
  ihn auf: Auswahl der relevanten Spalten, Kodierung der Ja/Nein-Felder,
  Behandlung von `collision_type = "?"` als eigene Kategorie
  ("NotApplicable").
- **`train_model.py`** — Baut die sklearn-Pipeline
  (`ColumnTransformer` + `RandomForestRegressor`), trainiert sie und
  speichert sie als `.pkl`-Datei.
- **`predict_model.py`** — Lädt die gespeicherte Pipeline und sagt
  `vehicle_claim` für einen einzelnen neuen Schadensfall voraus. Erwartet
  rohe, unkodierte Eingabewerte (das One-Hot-Encoding ist Teil der
  gespeicherten Pipeline).
- **`test_model.py`** — Manueller Smoke-Test: lädt das Modell und
  vergleicht die Vorhersagen für einen Kollisions- und einen
  Diebstahlfall auf Plausibilität.
- **`regression.py`** — CLI-Einstiegspunkt, der `train_and_save_model()`
  ausführt und bei Fehlschlag mit Exit-Code 1 beendet.

## `src/models/multi_head_insurance/` — Textanalyse

Sagt aus einer deutschsprachigen Unfallbeschreibung mehrere strukturierte
Felder gleichzeitig voraus (z. B. `auto_make`, `auto_year`,
`incident_severity`), basierend auf einem GottBERT-Encoder mit einem
eigenen Klassifikationskopf pro Zielfeld.

- **`models.py`** — `GermanInsuranceClassifier`: Transformer-Encoder mit
  Mean-Pooling und mehreren parallelen MLP-Köpfen (einer pro Zielfeld).
- **`losses.py`** — `DynamicMultiHeadLoss`: gewichtete Summe der
  Cross-Entropy-Verluste aller Köpfe; ignoriert Padding-Labels (`-100`)
  und fängt NaN-Verluste ab (z. B. wenn ein Batch keine gültigen Labels
  für ein Feld enthält).
- **`dataset.py`** — `GermanInsuranceDataset`: liest die von
  `labels_encoder.py` erzeugte JSONL-Trainingsdatei ein und tokenisiert
  den Text pro Beispiel.
- **`engine.py`** — Metrik-Berechnung: `FieldEvaluator` (Accuracy je
  Feld), `MultiHeadEvaluator` (Aggregation über alle Felder, inkl.
  Exact-Match- und Partial-Score-Metrik), `ErrorAnalyzer` (sammelt
  Fehlklassifikationen zur Analyse) sowie die Funktion `evaluate()` für
  eine vollständige Validierungs-/Test-Schleife.
- **`train.py`** — Orchestriert den kompletten Trainingsablauf:
  Datenaufbereitung, DataLoader-Aufbau, Trainings-/Evaluierungsschleife
  über mehrere Epochen, Tracking des besten Checkpoints
  (`BestModelTracker`, anhand der Partial-Score-Metrik) sowie
  abschließende Evaluierung auf dem Testdatensatz.
- **`inference.py`** — `load_model()` lädt einen trainierten Checkpoint;
  `predict()` sagt für eine Liste von Texten alle Zielfelder samt
  Konfidenz voraus; `validate_vehicle_with()` gleicht die vorhergesagte
  Marke/Baujahr-Kombination gegen den Fahrzeugkatalog ab und korrigiert
  das Baujahr ggf. auf das nächstgelegene bekannte Baujahr derselben
  Marke.

## `src/models/cnn/` — Bildanalyse

Sagt die Schadensschwere (3 Klassen: leicht/mittel/erheblich) aus einem
Foto des beschädigten Fahrzeugs voraus (Transfer-Learning auf Basis von
MobileNetV2).

- **`train_cnn.py`** — Lädt Trainings-/Validierungsbilder aus
  vorgegebenen Ordnern, baut das Modell (eingefrorenes MobileNetV2 +
  Augmentation + Klassifikationskopf), trainiert es und speichert Modell
  sowie Klassennamen.
- **`predict_image.py`** — `predict_severity()` lädt das gespeicherte
  Modell (gecacht über `@lru_cache`, damit es nicht bei jedem Aufruf neu
  geladen wird) und sagt für ein einzelnes Foto die Schadensschwere samt
  Konfidenz voraus. Bildet die internen Klassennamen auf deutsche
  Anzeigenamen ab (`SEVERITY_DE`).

CNN ist in `Pipeline` als **optionale** Stufe eingebunden: Sind die
CNN-Abhängigkeiten (TensorFlow) nicht installiert, wird die Stufe
übersprungen, ohne die übrige Pipeline zu beeinträchtigen.

---

## `src/services/` — Orchestrierung

### `src/services/pipeline.py`
`Pipeline` ist der zentrale Backend-Einstiegspunkt für jeden Client
(Dashboard, E-Mail-Worker, CLI). Führt für eine eingehende
Anfrage (Text und/oder Foto) nacheinander aus:

1. Rohanfrage speichern (`PipelineResultRepository.save_request`)
2. Multi-Head-Textanalyse (falls Text vorhanden)
3. CNN-Bildanalyse (falls Foto vorhanden, optionale Stufe)
4. Bestimmung der finalen Schadensschwere — die Schätzung mit der
   höheren Konfidenz (Text vs. Foto) gewinnt
5. Regressions-Kostenschätzung (falls Regressionsmodell verfügbar und
   eine Schwere bestimmt wurde)

`Pipeline` enthält keinen SQL-Code; jede Stufe delegiert das Speichern
ihres Ergebnisses an ein über den Konstruktor übergebenes
`PipelineResultRepository`. Das Ergebnis eines Durchlaufs wird als
`PredictionResult`-Dataclass zurückgegeben.

### `src/services/train_all.py`
CLI-Runner, der die komplette Trainings-Pipeline der Reihe nach ausführt:
Versicherungsdaten laden → Fahrzeugkatalog laden → Multi-Head-Training →
CNN-Training → Regressions-Training. Jeder Schritt läuft als eigener
Subprozess (`subprocess.run`), damit ein Fehler in einem Schritt den
Gesamtprozess nicht durch geteilten Zustand verfälschen kann. Einzelne
Schritte lassen sich über `RUN_*`-Flags überspringen.

---

## `src/utils/` — Datenaufbereitung und Hilfsfunktionen

### `src/utils/data_loader.py`
Lädt die Datensätze aus PostgreSQL: `get_insurance_dataset()` (JOIN über
die fünf normalisierten Tabellen, zurückgemappt auf die ursprünglichen
englischen Spaltennamen) und `get_vehicle_dataset()` (Fahrzeugkatalog).

### `src/utils/data_orchestrator.py`
Verbindet Laden und Bereinigen (`get_cleaned_dataset()`) sowie die
Erzeugung synthetischer Trainingsbeispiele
(`get_descriptions_labels_with_new_vehicles()`): kombiniert zufällige
Zeilen aus dem Versicherungsdatensatz mit Fahrzeugen aus dem Katalog
(gleichmäßig über die Marken verteilt) und erzeugt daraus
Unfallbeschreibungen samt Labels über `AccidentDataGenerator`.

### `src/utils/dataset_cleaner.py`
Bereinigt den rohen Versicherungsdatensatz: ersetzt Platzhalter für
fehlende Werte (`"?"`, `"NA"`, ...) durch `NaN`, zerlegt Datumsspalten in
Jahr/Monat/Tag, füllt numerische Lücken mit dem Median und kategoriale
Lücken mit dem Modus, entfernt nicht benötigte Spalten
(`FIELDS_TO_DELETE`) und korrigiert bekannte Schreibfehler bei
Fahrzeugmarken (`fix_name_errors()`).

### `src/utils/labels_encoder.py`
`prepare_pipeline_and_save_jsonl()` erzeugt für jedes Zielfeld einen
`LabelEncoder` anhand der bekannten Werte, kodiert die synthetisch
erzeugten Trainingsbeispiele und speichert sie als JSONL-Datei — die
Eingabedatei für `GermanInsuranceDataset` beim Multi-Head-Training.

### `src/utils/fake_data_generators/`
- **`accident_description_generator.py`** — `AccidentDataGenerator`
  erzeugt synthetische, deutschsprachige Unfallbeschreibungen samt
  Zielfeld-Labels. Variiert dabei Detailgrad, Satzreihenfolge und
  Formulierung (über mehrere private Hilfsmethoden, je eine pro
  Teilaufgabe: Fallart bestimmen, Strategie wählen, Sätze aufbauen,
  Kontext einfügen, ...), um eine vielfältige Trainingsbasis für das
  Multi-Head-Modell zu erzeugen.
- **`accident_text_constants.py`** — Reine Datensammlung (Wortlisten,
  Satzschablonen, Konfiguration je Fallart) ohne eigene Logik, verwendet
  vom Generator.

---

## Bewusst nicht weiter aufgeteilte Stellen

- **`Pipeline`** orchestriert weiterhin alle fünf Stufen in einer Klasse,
  statt z. B. in einzelne Stufen-Objekte aufgeteilt zu werden. Die
  Persistenz wurde bewusst ausgelagert (siehe `pipeline_repository.py`),
  die fachliche Orchestrierung bleibt aber zentral, da die Stufen eng
  voneinander abhängen (z. B. bestimmt Stufe 4 die Eingabe für Stufe 5).
- **`app/dashboard.py`** verwendet weiterhin seine eigene, von der echten
  Pipeline unabhängige Platzhalterlogik (siehe oben). Eine Anbindung an
  `Pipeline` ist als zukünftige Erweiterung vorgesehen.
