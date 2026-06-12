# Refactoring: Accident Description Generator

## Ziel

Der Generator wurde für NER/IE-Training umgebaut.  
Vorher produzierte er lesbare Berichte. Das ist für **Informationsextraktion** das falsche Ziel – ein Modell lernt dann Schablonen auswendig, nicht Entitäten.

---

## Was geändert wurde

### 1. Komponenten-basierte Generierung statt fixer Templates

**Vorher:**
```python
templates = config["templates"][detail_level]
text = random.choice(templates).format(make=..., model=..., year=..., ...)
```

**Nachher:**  
Jeder Satz wird aus unabhängigen Bausteinen zusammengesetzt:

| Komponente | Beispiele |
|---|---|
| `VEHICLE_INTRO_TEMPLATES` | `"Betroffen: {vehicle_ref}."`, `"Es handelt sich um {vehicle_ref}."` |
| `EVENT_STANDALONE_TEMPLATES` | `"Hergang: {incident_phrase}."`, `"Vorfall: {incident_phrase}."` |
| `DAMAGE_INTRO_TEMPLATES` | `"Schaden: {damage}."`, `"Schadensbild: {damage}."` |
| `YEAR_STANDALONE_TEMPLATES` | `"Baujahr: {year}."`, `"Das Fahrzeug ist Baujahr {year}."` |

**Warum:** Das Modell soll lernen, Entitäten im Text zu erkennen – unabhängig vom umgebenden Satz.

---

### 2. Zufällige Anordnung der Satzblöcke (Assembly-Strategien)

Statt immer `[VEHICLE] → [EVENT] → [DAMAGE]` gibt es 5 Strategien:

| Strategie | Beschreibung |
|---|---|
| `vehicle_in_event` | Fahrzeugreferenz eingebettet im Ereignissatz |
| `vehicle_then_event` | Fahrzeug → Ereignis → Schaden |
| `event_then_vehicle` | Ereignis → Fahrzeug → Schaden (oder Schaden vor Fahrzeug) |
| `distant` | Make/Model und Baujahr in **getrennten Sätzen**, Reihenfolge zufällig gemischt |
| `no_vehicle` | Kein Fahrzeugbezug (realistisch für kurze Meldungen) |

**Warum:** Das Modell darf nicht lernen „MAKE steht immer an Position 1". Distante Abhängigkeiten (strategy `distant`) sind besonders wichtig: Baujahr und Marke erscheinen in verschiedenen Sätzen.

---

### 3. Alternative Fahrzeugreferenz-Formen

**Vorher:**
```
Mein Audi A4 (2023) ...
```

**Nachher** (zufällig aus Pool):
```
Audi A4 (2023)
Audi A4, Baujahr 2023
2023er Audi A4
Audi A4 aus 2023
A4 von Audi, Baujahr 2023
Fahrzeug A4 (Audi)
Audi Modell A4
```

**Warum:** Das Modell muss MAKE/MODEL/YEAR in verschiedenen syntaktischen Formen erkennen, nicht nur in `Make Model (Year)`.

---

### 4. Kontexteinschub an zufälliger Position

Wetter und Tageszeit werden nicht mehr immer **vor** dem Ereignis eingefügt, sondern an einem zufälligen Index in der Satzliste.

**Warum:** Verhindert, dass das Modell lernt „Wetter kommt immer vor dem Unfall".

---

### 5. Rückgabetyp: `tuple[str, dict]`

**Vorher:**
```python
def generate_accident_description(row: dict) -> str:
    ...
    return text
```

**Nachher:**
```python
def generate_accident_description(row: dict) -> tuple[str, dict]:
    ...
    return text, {"entities": [(start, end, "MAKE"), ...]}
```

Die Entitäten werden **post-hoc** per Zeichenoffset-Suche im fertigen Text gefunden:

- Gesucht wird nach `YEAR` (längster Wert zuerst), dann `MAKE`, dann `MODEL`
- Längste Treffer zuerst, damit Teiltreffer nicht längere Matches überschatten
- `occupied`-Set verhindert überlappende Spans
- Format kompatibel mit spaCy-Trainingsformat: `(start, end, label)`

---

## Was unverändert blieb

- `resolve_case_type()` – Logik zur Falltyp-Erkennung
- `CASE_CONFIG` – Konfiguration der Schadensbilder und Ereignissätze pro Falltyp
- `DETAIL_LEVELS` / `DETAIL_LEVEL_WEIGHTS` – Verteilung der Detailstufen (minimal / normal / detailed / noisy)
- Kontextfelder (Wetter, Tageszeit, Zeug:innen, Polizei, Abschlepp) – Inhalte gleich, Position nun variabel

---

## Ergebnis

| Eigenschaft | Vorher | Nachher |
|---|---|---|
| Satzstruktur | fest (`VEHICLE → EVENT → DAMAGE`) | zufällig permutiert |
| Fahrzeugreferenz | immer `Make Model (Year)` | 10+ syntaktische Varianten |
| Make/Model/Year | immer im selben Satz | ggf. in getrennten Sätzen |
| Rückgabe | `str` | `tuple[str, dict]` mit Entitätsspans |
| Stilmix | vorhanden | vorhanden (unverändert) |
