# CNN-Modul — Dokumentation

## Übersicht

Das CNN-Modul klassifiziert die Schadensschwere eines Fahrzeugs anhand
eines Fotos. Es verwendet Transfer Learning auf Basis von **MobileNetV2**
(vortrainiert auf ImageNet) und unterscheidet drei Schadensklassen:

| Interne Klasse | Anzeigename | Pipeline-Schwere |
|---|---|---|
| `01-minor` | Leichter Schaden | Minor Damage |
| `02-moderate` | Mittlerer Schaden | Major Damage |
| `03-severe` | Erheblicher Schaden | Total Loss |

---

## Dateistruktur

```
src/models/cnn/
    cnn_config.py       Gemeinsame Konstanten (Pfade, Bildgröße, Mappings)
    train_cnn.py        Training des Modells
    predict_image.py    Vorhersage für ein einzelnes Foto
    model.keras         Gespeichertes Modell (nach dem Training)
    classes.json        Klassennamen in Modell-Reihenfolge (nach dem Training)
```

---

## Voraussetzungen

### Bilddatensatz

Der Datensatz muss vor dem Training in folgender Struktur vorliegen:

```
data/raw/car_damage/
    training/
        01-minor/       ← Fotos mit leichtem Schaden
        02-moderate/    ← Fotos mit mittlerem Schaden
        03-severe/      ← Fotos mit erheblichem Schaden
    validation/
        01-minor/
        02-moderate/
        03-severe/
```

Der Datensatz ist bereits in Training und Validierung aufgeteilt.
Es wird kein automatischer Split durchgeführt.

### Abhängigkeiten

```
tensorflow
pillow      ← wird von tf.keras.utils.load_img() benötigt
numpy
```

---

## Verwendung

### Training

```bash
python -m src.models.cnn.train_cnn
```

Nach dem Training werden zwei Dateien gespeichert:
- `src/models/cnn/model.keras` — das trainierte Modell
- `src/models/cnn/classes.json` — die Klassennamen in Modell-Reihenfolge

### Vorhersage (Kommandozeile)

```bash
python -m src.models.cnn.predict_image pfad/zum/foto.jpg
```

Gibt Schweregrad und Konfidenz aus, z. B.:
```
Schadensschwere: Mittlerer Schaden (Konfidenz: 78.3%)
```

### Vorhersage (im Code)

```python
from src.models.cnn.predict_image import predict_severity

label_de, confidence = predict_severity("foto.jpg")
# label_de   → "Leichter Schaden" | "Mittlerer Schaden" | "Erheblicher Schaden"
# confidence → float zwischen 0.0 und 1.0
```

---

## Architektur

```
Eingabebild (224 × 224 × 3)
    ↓
Augmentation (RandomFlip, RandomRotation, RandomZoom)
    ↓
MobileNetV2 (eingefroren, ImageNet-Gewichte)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.2)
    ↓
Dense (3 Klassen, Softmax)
```

**Transfer Learning:** Die MobileNetV2-Gewichte werden eingefroren
(`trainable = False`). Nur der Klassifikationskopf (Dense-Schicht) wird
auf dem Schadensdatensatz trainiert. Das reduziert den Datenbedarf
erheblich und beschleunigt das Training.

**Augmentation:** Horizontales Spiegeln, leichte Rotation (±10°) und
Zoom (±10%) erzeugen künstliche Trainingsvarianten und reduzieren
Overfitting.

**Preprocessing:** `mobilenet_v2.preprocess_input` normalisiert die
Pixelwerte auf den von MobileNetV2 erwarteten Bereich (−1 bis +1).
Diese Schicht ist direkt im Modell enthalten und wird daher sowohl beim
Training als auch bei der Vorhersage automatisch angewendet.

---

## Einbindung in die Pipeline

Das CNN ist eine **optionale Stufe** in `src/services/pipeline.py`.
Es wird nur ausgeführt, wenn ein Foto übergeben wird
(`photo_path` oder `photo_bytes`). Fehlen die CNN-Abhängigkeiten
(TensorFlow nicht installiert), wird die Stufe übersprungen ohne
die übrige Pipeline zu beeinträchtigen.

```
Pipeline.run(text=..., photo_bytes=...)
    ↓
_run_cnn()
    ↓
predict_severity(tmp_pfad)      ← gibt (label_de, confidence) zurück
    ↓
CNN_TO_SEVERITY[label_de]       ← Mapping auf Pipeline-Schwere
    ↓
result.severity_from_photo      ← gespeichert im PredictionResult
```

Das Ergebnis konkurriert mit der Textschwere aus dem Multi-Head-Modell.
Die Quelle mit der höheren Konfidenz gewinnt (`_resolve_severity()`).

---

## Konfiguration

Alle gemeinsamen Konstanten sind in `cnn_config.py` zentralisiert:

| Konstante | Wert | Bedeutung |
|---|---|---|
| `MODEL_PATH` | `src/models/cnn/model.keras` | Pfad zum Modell |
| `CLASSES_PATH` | `src/models/cnn/classes.json` | Pfad zu den Klassennamen |
| `IMAGE_SIZE` | `(224, 224)` | Bildgröße in Pixeln |
| `SEVERITY_DE` | dict | Mapping → deutsche Anzeigenamen |
| `CNN_TO_SEVERITY` | dict | Mapping → Pipeline-Schwere-Kategorien |

**Wichtig:** `IMAGE_SIZE` muss in Training und Vorhersage identisch sein.
Eine Änderung erfordert ein vollständiges Neutraining des Modells.

---

## Bekannte Grenzen

- Das Modell wurde auf einem begrenzten Bilddatensatz trainiert und kann
  bei unbekannten Schadensmuster niedrige Konfidenz zeigen.
- Bei Konfidenzwerten unter ca. 50 % sollte das Ergebnis als unsicher
  behandelt werden.
- Das Modell erkennt ausschließlich Fahrzeugschäden — andere Bildtypen
  (z. B. Innenraumfotos, Dokumente) führen zu bedeutungslosen Vorhersagen.
- TensorFlow ist eine schwere Abhängigkeit (~500 MB). In Umgebungen ohne
  TensorFlow wird die CNN-Stufe automatisch übersprungen.
