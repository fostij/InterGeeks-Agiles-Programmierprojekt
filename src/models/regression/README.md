# Regressionsmodul — Vorhersage der Schadenshöhe (`vehicle_claim`)

Dieses Modul stellt eine automatisierte Machine-Learning-Pipeline zur Vorhersage
des Fahrzeugschadens in EUR (`vehicle_claim`) bereit. Es folgt einer modularen
Architektur, die Datenvorbereitung, Training und Inferenz klar voneinander trennt.

---

## Pipeline-Architektur

### 1. Datenvorbereitung (`prepare_data.py`)

- Lädt den Datensatz aus PostgreSQL über `get_insurance_dataset()`
  (`src/utils/data_loader.py`), das die 5 normalisierten Tabellen
  (`kunden`, `policen`, `fahrzeuge`, `unfaelle`, `schaeden`) zusammenführt
  und die Spaltennamen auf das englische Originalschema zurückbildet.
- Filtert auf die relevanten Felder (`TARGET_FIELDS` in `src/ml_config.py`).
- Bildet `YES`/`NO`-Felder (`property_damage`, `police_report_available`)
  auf `1`/`0` ab.
- Kodiert fehlende `collision_type`-Werte explizit als eigene Kategorie
  `"NotApplicable"`. Der ursprüngliche `"?"`-Platzhalter aus der CSV-Datei
  wird bereits beim ETL-Schritt (`src/db/load_insurance_data.py`) in `NaN`
  umgewandelt — `get_insurance_dataset()` liefert also `NaN`, nicht den
  String `"?"`. Dieser fehlende Wert ist kein zufälliger Datenfehler,
  sondern tritt systematisch bei Schadenstypen ohne Kollision auf
  (z. B. `Vehicle Theft`, `Parked Car`) und ist ein aussagekräftiges Signal,
  das erhalten bleiben soll.
- Füllt verbleibende fehlende numerische/binäre Werte mit `0`.
- **Hinweis:** Die kategoriale Kodierung (One-Hot-Encoding) wird hier
  **nicht** durchgeführt. Sie erfolgt innerhalb des `ColumnTransformer` in
  `train_model.py` als Teil der gespeicherten `sklearn`-Pipeline. Dieses
  Modul bereinigt und selektiert ausschließlich Spalten.

### 2. Modelltraining (`train_model.py`)

- Baut eine `sklearn`-`Pipeline` aus `ColumnTransformer` mit
  `OneHotEncoder` und `RandomForestRegressor`, sodass Kodierung und Modell
  gemeinsam als ein Artefakt serialisiert werden.
- Das am Ende von `train_and_save_model()` ausgegebene R² basiert auf einem
  einzelnen Train/Test-Split und dient nur als schnelle Plausibilitätsprüfung
  beim Training — den validierten Wert enthält der Abschnitt
  **Modellleistung** unten.
- Serialisiert die trainierte Pipeline via `joblib` nach
  `REGRESSION_MODEL_PATH` (`src/ml_config.py`, aktuell
  `checkpoints/regression_model.pkl`).

### 3. Inferenz (`predict_model.py`)

- `predict_new_claim(input_data: dict)` lädt die gespeicherte Pipeline
  und sagt einen einzelnen Schadenbetrag vorher.
- **Die Eingabe muss rohe, unkodierte Werte enthalten**
  (z. B. `auto_make="Audi"` als Klartext-String). Kein manuelles
  `pd.get_dummies()` oder sonstiges Vorcodieren der Eingabe — der
  `OneHotEncoder` der Pipeline erledigt das intern. Vorcodierte Eingaben
  führen zu Spalten-Mismatches und liefern nahezu konstante, fehlerhafte
  Vorhersagen.

### 4. Orchestrierung (`regression.py`)

- Führt die vollständige Trainings-Pipeline von Anfang bis Ende über
  `train_and_save_model()` aus.
- CLI-Einstiegspunkt: bricht bei Fehlern mit Exit-Code 1 ab.

### 5. Smoke-Test (`test_model.py`)

- Testet das gespeicherte Modell anhand zweier Beispielfälle:
  einem Kollisionsfall und einem Diebstahl-/Parked-Car-Fall.
- `vehicle_claim` ist bimodal — Kollisionsfälle liegen im Durchschnitt
  bei ~44.000–46.000 EUR, Diebstahl-/Parked-Car-Fälle bei ~3.800–4.000 EUR.
  Der Test warnt, wenn beide Vorhersagen verdächtig nah beieinanderliegen
  (das wäre ein Hinweis auf eine defekte Eingabepipeline).

---

## Modellleistung

Ein einzelner Train/Test-Split reagiert empfindlich auf die kleine
Datensatzgröße und wird **nicht** als gemeldete Metrik verwendet. Stattdessen
wurden RandomForest, XGBoost und CatBoost mit 5-facher Kreuzvalidierung
auf dem originalen Datensatz (~1.000 Zeilen) bewertet:

| Modell | R² (Mittelwert) | R² (Std) | MAE (Mittelwert) |
|---|:---:|:---:|:---:|
| RandomForest (eingesetzt) | 0,697 | 0,023 | 7.656 EUR |
| XGBoost (Standardparameter) | 0,642 | 0,026 | 8.483 EUR |
| CatBoost | 0,698 | 0,022 | 7.683 EUR |
| XGBoost (getunt) | 0,697 | 0,021 | 7.831 EUR |

Alle drei Algorithmen konvergieren nach korrekter Konfiguration auf
ungefähr dasselbe R² (~0,70). Das ist ein starkes Indiz dafür, dass
R² ≈ 0,70 die praktische Obergrenze für den aktuellen Merkmalssatz
darstellt — keine Einschränkung des gewählten Algorithmus.
RandomForest wird im Produktionsbetrieb verwendet, da es dieses Niveau
ohne Hyperparameter-Tuning erreicht.

**Bekannte Einschränkungen:**

- `injury_claim`, `property_claim` und `total_claim_amount` werden
  absichtlich aus dem Merkmalssatz ausgeschlossen.
  `total_claim_amount` korreliert stark mit `vehicle_claim` (r ≈ 0,98)
  und würde Datenleckage darstellen. `injury_claim`/`property_claim`
  werden ausgeschlossen, weil sie laut Geschäftslogik zum
  Vorhersagezeitpunkt nicht bekannt sind.
- Ein MAE von ~7.700 EUR bei einem Median-`vehicle_claim` von ~42.000 EUR
  entspricht einem relativen Fehler von ca. 18 % — ausreichend als
  unterstützendes Signal, sollte jedoch vor dem Einsatz als alleinige
  Entscheidungsgrundlage gegen die Geschäftsanforderungen geprüft werden.

---

## Dateneinrichtung

Trainingsdaten werden aus **PostgreSQL** gelesen, nicht direkt aus einer
CSV-Datei. `get_insurance_dataset()` (`src/utils/data_loader.py`) verbindet
5 normalisierte Tabellen (`kunden`, `policen`, `fahrzeuge`, `unfaelle`,
`schaeden`) und bildet die deutschen Spaltennamen auf das englische
Originalschema zurück.

Um die Datenbank aus der rohen CSV zu befüllen (einmalig / Neuimport):

```bash
python src/db/load_insurance_data.py
```

Dieser Schritt liest die CSV-Datei aus `INSURANCE_DATASET_PATH`
(`src/ml_config.py`, aktuell `data/raw/dataset.csv`), bereinigt sie
(`"?"` → `NULL`, Typkonvertierungen) und lädt sie in die 5 normalisierten
Tabellen aus `sql/schema.sql`. `INSURANCE_DATASET_PATH` wird
**ausschließlich** von diesem einmaligen ETL-Schritt verwendet — die
Trainings-Pipeline selbst (`prepare_data.py` → `train_model.py`) liest
immer aus der Datenbank über `get_insurance_dataset()`, unabhängig davon,
ob die Original-CSV noch auf der Festplatte vorhanden ist.

Datenbankverbindungseinstellungen werden in `src/db/connection.py`
(`get_engine()`) konfiguriert.

---

## Projektstruktur

```
src/
├── ml_config.py                    Pfade und Feldkonfiguration
├── db/
│   ├── connection.py               get_engine() — Datenbankverbindung
│   └── load_insurance_data.py      Einmaliger ETL: CSV → PostgreSQL
├── utils/
│   └── data_loader.py              get_insurance_dataset() — DB → DataFrame
└── models/
    └── regression/
        ├── prepare_data.py         Datenbereinigung und -selektion
        ├── train_model.py          Pipeline-Aufbau, Training und Serialisierung
        ├── predict_model.py        Einzelfall-Inferenz
        ├── test_model.py           Smoke-Test (Kollision + Diebstahl)
        └── regression.py           CLI-Einstiegspunkt

checkpoints/
└── regression_model.pkl            Gespeicherte sklearn-Pipeline (nach Training)
```

---

## Verwendung

Abhängigkeiten installieren:

```bash
pip install pandas scikit-learn joblib sqlalchemy psycopg2-binary
```

**Voraussetzung:** eine laufende, befüllte PostgreSQL-Datenbank
(siehe **Dateneinrichtung** oben) mit Verbindungseinstellungen in
`src/db/connection.py`. Das Training schlägt fehl, wenn die Datenbank
nicht erreichbar oder leer ist.

Modell trainieren und speichern:

```bash
python -m src.models.regression.regression
```

Smoke-Test auf dem gespeicherten Modell durchführen:

```bash
python -m src.models.regression.test_model
```

Einzelfall aus Python vorhersagen:

```python
from src.models.regression.predict_model import predict_new_claim

betrag = predict_new_claim({
    "auto_year": 2020,
    "auto_make": "Audi",
    "incident_type": "Multi-vehicle Collision",
    "incident_severity": "Major Damage",
    "collision_type": "Rear Collision",
    "number_of_vehicles_involved": 2,
    "witnesses": 1,
    "police_report_available": 1,
    "bodily_injuries": 0,
    "property_damage": 1,
})
print(f"Geschätzte Schadenshöhe: {betrag:,.2f} EUR")
```
