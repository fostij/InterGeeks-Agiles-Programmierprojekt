# Analyse des Regressionsmodells `vehicle_claim` — Zusammenfassung

Dieses Dokument fasst die gemeinsame Analyse des Modells zur Vorhersage von
`vehicle_claim` zusammen: gefundene Fehler, Erkenntnisse aus der explorativen
Datenanalyse (EDA), den Modellvergleich und offene Punkte.

---

## 1. Ursprüngliches Problem

Symptome:
- R² des Modells lag bei ca. **0,6**.
- Vorhersagen für unterschiedliche Eingaben lagen immer im engen Bereich
  **44.000–46.000**, unabhängig vom konkreten Input.

---

## 2. Gefundener Bug: falsche Eingabe in `test_model.py`

Das trainierte Modell ist eine `sklearn`-`Pipeline`, die einen
`OneHotEncoder` **intern** enthält (siehe `build_pipeline()` in
`train_model.py`). Die Pipeline erwartet daher **rohe** Eingabespalten
(z. B. `auto_make` als Klartext-String), nicht bereits one-hot-kodierte
Daten.

In `test_model.py` wurde stattdessen:

```python
df_input = pd.get_dummies(sample_data)
df_input = df_input.reindex(columns=X_train.columns, fill_value=0)
```

verwendet. Das ist falsch, weil:

1. `pd.get_dummies` auf einer einzelnen Beispielzeile eigene Spaltennamen
   erzeugt (z. B. `auto_make_Audi`).
2. `X_train.columns` enthält die **unverarbeiteten** Rohspalten (vor der
   Pipeline-Transformation).
3. Beim `reindex` matchen die Spaltennamen kaum, wodurch die meisten Merkmale
   auf `0` gesetzt werden.
4. Das Modell erhält dadurch einen fast leeren/ungültigen Input und sagt
   näherungsweise den Mittelwert der Trainingsdaten voraus — das erklärt den
   engen Vorhersagebereich 44k–46k.

**Lösung:** Den rohen `sample_data`-DataFrame direkt an `model.predict()`
übergeben, ohne manuelles `get_dummies`/`reindex` — genau wie es in
`predict_model.py` bereits korrekt gemacht wird.

---

## 3. Mögliche Ursachen für R² ≈ 0,6 (unabhängig vom Bug)

- `FIELDS_TO_DELETE` entfernt bewusst `injury_claim`, `total_claim_amount`
  und `property_claim` — vermutlich wegen Leakage-Bedenken.
- Die verbleibenden Merkmale in `TARGET_FIELDS` (`auto_year`, `auto_make`,
  `incident_type`, `incident_severity`, `collision_type`,
  `number_of_vehicles_involved`, `witnesses`, `police_report_available`,
  `bodily_injuries`, `property_damage`) sind größtenteils **schwache**
  indirekte Prädiktoren für die konkrete Schadenshöhe am Fahrzeug.
- `REGRESSION_CATEGORICAL_COLS` wird über Listendifferenz berechnet
  (`TARGET_FIELDS` minus numerische/binäre Spalten) — funktional korrekt,
  aber fragil bei künftigen Änderungen an `TARGET_FIELDS`.
- `df.fillna(0)` in `prepare_data.py` füllt fehlende Werte pauschal mit `0`,
  was nicht für jedes Feld semantisch neutral ist.

---

## 4. Frage: Ausreißer entfernen?

### Empfehlung der ersten EDA-Runde (auf fehlerhaftem Datensatz, siehe Abschnitt 6)

- **`vehicle_claim` ist bimodal**, nicht durch klassische Ausreißer geprägt:
  - "Leichte" Fälle (`Parked Car`, `Vehicle Theft`, `Trivial Damage`):
    Werte um 3.800–4.000, enger Streubereich.
  - "Schwere" Fälle (Kollisionen, `Minor/Major Damage`, `Total Loss`):
    Werte um 44.000–46.000, großer Streubereich.
- Die IQR-Methode fand **0 Ausreißer** — was zur Bimodalität passt: Es gibt
  keine klassischen Ausreißer, sondern zwei unterschiedliche Populationen.
- **Schlussfolgerung:** Ausreißer entfernen ist hier nicht der richtige
  Hebel. Stattdessen sollte das Modell die Bimodalität über vorhandene
  Merkmale (`incident_type`, `incident_severity`) lernen.

### Erkannte Datenqualitätsprobleme

- `collision_type` enthält `"?"` für Fälle ohne Kollision (Diebstahl,
  geparktes Auto) — das ist **kein zufälliger fehlender Wert**, sondern ein
  aussagekräftiges Signal. Es sollte als **eigene Kategorie** kodiert werden
  (z. B. `"NotApplicable"`), nicht mit dem Modus imputiert werden — sonst
  geht ein nützliches Trennsignal verloren.
- `total_claim_amount` korreliert mit `vehicle_claim` mit **r ≈ 0,98** —
  starkes Indiz für direktes Leakage (vermutlich
  `total = injury + property + vehicle`). Der Ausschluss aus den Merkmalen
  ist korrekt.
- `injury_claim` (r ≈ 0,72) und `property_claim` (r ≈ 0,73) haben **keine**
  derart direkte arithmetische Beziehung zu `vehicle_claim`. Ob sie als
  Merkmale verwendet werden dürfen, hängt von der Geschäftslogik ab.

  **Klargestellt:** Laut Geschäftslogik sind andere Schadensarten
  (`injury_claim`, `property_claim`) zum Vorhersagezeitpunkt **nicht im
  Voraus bekannt** — sie dürfen daher **nicht** als Merkmale verwendet
  werden.

---

## 5. Klassifikator für `incident_type`? — Nicht nötig

Es wurde kurzzeitig vorgeschlagen, einen separaten Klassifikator zu bauen,
der zwischen "leichten" und "schweren" Fällen unterscheidet. Das ist
**unnötig**, da `incident_type` bereits als fertiges Merkmal in den
Eingabedaten vorliegt. Ein Klassifikator würde nur etwas vorhersagen, das
bereits bekannt ist.

Stattdessen kann (und sollte) `incident_type` direkt als Signal in der
bestehenden Regressionspipeline genutzt werden — Tree-basierte Modelle
(RandomForest, XGBoost, CatBoost) können diesen Modus-Unterschied
grundsätzlich selbst über Splits lernen.

---

## 6. Wichtige Korrektur: falscher Datensatz in der EDA verwendet

**Die erste EDA-Runde (Abschnitt 4) wurde auf einem fehlerhaften Datensatz
durchgeführt.** Es stellte sich heraus:

- Der **originale** Datensatz hat **~1.000 Zeilen**.
- Der in der ersten EDA verwendete Datensatz hatte **~9.000–10.000 Zeilen**
  und war **synthetisch nachgeneriert, von schlechter Qualität**.

**Konsequenz:** Alle quantitativen Aussagen aus der ersten EDA-Runde
(Verteilungsform, exakte Korrelationswerte, Gruppenstatistiken nach
`incident_severity`/`collision_type`/`incident_type`) müssen auf dem
**originalen 1.000-Zeilen-Datensatz neu überprüft** werden. Die
grundsätzlichen *Mechanismen* (z. B. dass `collision_type = "?"`
aussagekräftig ist, dass `total_claim_amount` Leakage darstellt) sind
plausibel und vermutlich weiterhin gültig, aber die genauen Zahlen
(Korrelationen, Gruppenmittelwerte, Bimodalitätsstärke) nicht ungeprüft
übernehmen.

---

## 7. Modellvergleich: RandomForest vs. XGBoost vs. CatBoost

### Erster (nicht ganz fairer) Test

Erste Ergebnisse auf dem one-hot-kodierten Datensatz (RF/XGBoost) im
Vergleich zu nativ-kategorialem CatBoost zeigten XGBoost klar vorn. Dieser
Vergleich war jedoch nicht ganz fair, weil:

- `collision_type = "?"` mit dem Modus imputiert wurde, statt als eigene
  Kategorie erhalten zu bleiben (Signalverlust).
- RF/XGBoost und CatBoost technisch unterschiedliche Kodierungen erhielten,
  was zwei Variablen (Algorithmus *und* Encoding) gleichzeitig veränderte.

### Korrigierter, fairer Vergleich

Nach Korrektur (explizite `"NotApplicable"`-Kategorie, identische Bereinigung
für alle Modelle, Encoding-Unterschied bleibt algorithmusbedingt bestehen):
Alle drei Modelle wurden auf dem **falschen, synthetischen** 9.000-Zeilen-
Datensatz verglichen — die Ergebnisse aus dieser Runde sind aus demselben
Grund wie in Abschnitt 6 **nicht verlässlich** und müssen auf dem originalen
Datensatz wiederholt werden.

### Test auf dem originalen ~1.000-Zeilen-Datensatz

Beim Test auf dem echten Datensatz (Train: 800, Test: 200) zeigte sich ein
neues Problem: Bei der Aufteilung der Testmetriken nach Gruppen
(`incident_type`) blieben in der "leichten" Gruppe nur **37 Zeilen** übrig.
Bei so kleinen Stichproben kann eine einzelne ungewöhnliche Zeile das R²
stark verfälschen (bis hin zu stark negativen Werten), unabhängig von der
tatsächlichen Modellqualität. Diese gruppenweisen Ergebnisse sind bei
diesem Stichprobenumfang **nicht aussagekräftig**.

---

## 8. Wechsel zu Kreuzvalidierung

Wegen der geringen Datenmenge (~1.000 Zeilen) wurde von einem einzelnen
Train/Test-Split auf **5-fache Kreuzvalidierung (K-Fold)** umgestellt:

- Die Daten werden in 5 Teile aufgeteilt; das Modell wird 5-mal trainiert
  (jeweils auf 4 Teilen) und auf dem verbleibenden Teil getestet.
- Die Ergebnisse der 5 Durchläufe werden gemittelt, zusätzlich wird die
  Standardabweichung zwischen den Folds betrachtet.
- Dadurch landet jede Zeile irgendwann in der Testmenge, und die Bewertung
  hängt nicht von einem einzigen, womöglich unglücklichen Split ab.
- Eine Aufteilung der Metriken nach Untergruppen (leicht/Kollision) wird in
  diesem Setup **nicht mehr durchgeführt**, da der Datensatz dafür zu klein
  ist.
- Für CatBoost wurde die Kreuzvalidierung manuell mit derselben
  `KFold`-Aufteilung wie für RF/XGBoost implementiert (da `cat_features`
  nicht direkt mit `sklearn.cross_validate` kompatibel ist), um die
  Vergleichbarkeit zu wahren.
- `RandomizedSearchCV` für die XGBoost-Hyperparametersuche nutzt ebenfalls
  dieselbe `KFold`-Aufteilung.

**Wichtiger Hinweis für die Interpretation:** Bei ~1.000 Zeilen sind
Unterschiede von 0,02–0,05 im R² zwischen Modellen oft nicht bedeutsam.
Die Standardabweichung zwischen den Folds sollte immer mitbetrachtet werden
— eine hohe Std bedeutet, dass die Schätzung bei diesem Datensatzumfang
grundsätzlich unsicher ist, unabhängig vom verwendeten Modell.

---

## 9. Finale Ergebnisse: Kreuzvalidierung auf dem originalen Datensatz

Mit dem korrigierten Notebook (5-fache `KFold`-Kreuzvalidierung, `shuffle=True`,
`random_state=42`) wurde auf dem **originalen ~1.000-Zeilen-Datensatz**
folgendes Ergebnis erzielt (One-Hot-Form: 1000 × 28, kategoriale Form für
CatBoost: 1000 × 10):

| Modell                  | R² Mittelwert | R² Std | MAE Mittelwert | MAE Std |
|--------------------------|:---:|:---:|:---:|:---:|
| Random Forest             | 0,6968 | 0,0230 | 7.656,47 | 240,14 |
| XGBoost (Standardparameter) | 0,6415 | 0,0261 | 8.482,92 | 240,24 |
| CatBoost (nativ kategorial) | 0,6980 | 0,0220 | 7.683,34 | 256,55 |
| XGBoost (getunt)          | 0,6967 | 0,0205 | 7.830,87 | 182,25 |

Beste Parameter aus `RandomizedSearchCV` für XGBoost:
`subsample=0.6`, `reg_lambda=2.0`, `reg_alpha=0`, `n_estimators=300`,
`min_child_weight=1`, `max_depth=3`, `learning_rate=0.01`,
`colsample_bytree=0.8`.

### Interpretation

- Die Standardabweichung zwischen den Folds ist bei allen Modellen klein
  (R²-Std ≈ 0,02–0,03; MAE-Std ≈ 180–260), die Bewertung ist also stabil
  und nicht durch zufällige Splits verzerrt.
- **Random Forest, CatBoost und das getunte XGBoost landen praktisch auf
  demselben Niveau** (R² ≈ 0,696–0,698) — die Unterschiede liegen innerhalb
  der jeweiligen Standardabweichung und sind damit nicht bedeutsam.
- **XGBoost mit Standardparametern** schnitt deutlich schlechter ab
  (R² = 0,642). Die vom Tuning gefundenen besten Parameter
  (`max_depth=3`, `learning_rate=0.01`, also ein deutlich konservativeres,
  stärker regularisiertes Modell als die Standardwerte `max_depth=6`,
  `learning_rate=0.1`) deuten darauf hin, dass das ungetunte XGBoost auf
  diesem kleinen Datensatz **überangepasst** war. Nach dem Tuning erreicht
  es dasselbe Niveau wie die anderen Modelle.

### Schlussfolgerung

Wenn drei unterschiedliche Algorithmen (Bagging: Random Forest; zwei
Boosting-Varianten: XGBoost, CatBoost) nach korrekter Konfiguration
**unabhängig voneinander auf demselben Niveau** (R² ≈ 0,70) konvergieren,
ist das ein starkes Indiz dafür, dass **R² ≈ 0,70 die aktuelle Obergrenze
für den vorhandenen Merkmalssatz** ist — und kein Problem der Modellwahl
oder des Hyperparameter-Tunings.

Eine weitere Verbesserung ist mit dem aktuellen Merkmalssatz über
Modellwechsel oder weiteres Tuning **nicht mehr realistisch zu erwarten**.
Mögliche nächste Hebel wären ausschließlich:

1. **Neue Merkmale**, die zum Vorhersagezeitpunkt tatsächlich bekannt sind
   (`injury_claim`/`property_claim` scheiden laut Geschäftslogik aus,
   `total_claim_amount` stellt Leakage dar — beide bleiben also
   ausgeschlossen).
2. **Feature Engineering** auf Basis der bestehenden Felder (z. B.
   Interaktionsterme wie `auto_year` × `incident_severity`) — realistisch
   nur ein kleiner zusätzlicher Gewinn, keine grundlegende Verbesserung.
3. **Mehr Daten** — bei nur ~1.000 Zeilen ist die Stichprobe ohnehin klein;
   ein größerer (echter, nicht synthetisch nachgenerierter) Datensatz
   könnte sowohl die Modellqualität als auch die Zuverlässigkeit der
   Bewertung verbessern.

Da CatBoost im Projekt nicht eingesetzt werden soll, ist **Random Forest**
die empfohlene Wahl: gleichwertige Qualität wie das getunte XGBoost, jedoch
ohne den Aufwand einer Hyperparameter-Suche.

---

## 10. Offene Punkte / nächste Schritte

1. **EDA auf dem originalen ~1.000-Zeilen-Datensatz wiederholen**
   (Verteilung von `vehicle_claim`, Korrelationen, Gruppenstatistiken) —
   die bisherigen Zahlen stammen vom falschen, synthetischen Datensatz.
2. Prüfen, ob die Bimodalität (leicht vs. Kollision) im originalen
   1.000-Zeilen-Datensatz in ähnlicher Form vorhanden ist wie im
   synthetischen Datensatz vermutet.
3. `collision_type = "?"` weiterhin explizit als eigene Kategorie behandeln,
   nicht imputieren.
4. `injury_claim` und `property_claim` **nicht** als Merkmale verwenden
   (laut Geschäftslogik zum Vorhersagezeitpunkt nicht bekannt).
5. Bug in `test_model.py` beheben (rohen DataFrame statt
   `get_dummies`/`reindex` an die Pipeline übergeben).
6. Produktionsmodell auf **Random Forest** festlegen (CatBoost ist laut
   Vorgabe nicht erwünscht; XGBoost würde ein zusätzliches Tuning
   erfordern, um dasselbe Niveau zu erreichen, ohne einen messbaren Vorteil
   zu bieten).
7. Akzeptanzkriterium für R² ≈ 0,70 / MAE ≈ 7.700 mit dem Fachbereich
   klären — prüfen, ob diese Genauigkeit für den Anwendungsfall ausreicht.