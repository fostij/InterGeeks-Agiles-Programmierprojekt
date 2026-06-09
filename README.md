<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:2EA66E,100:14784F&height=200&section=header&text=Auto%20Insurance%20Claims&fontSize=44&fontColor=ffffff&fontAlignY=34&desc=Agiles%20Programmierprojekt%20%E2%80%93%20Hochschule%20Hannover&descSize=18&descAlignY=56" alt="Banner" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3500&pause=800&color=2EA66E&center=true&vCenter=true&width=640&lines=Data+Science+%26+KI+Workflow;Betrugserkennung+in+der+Kfz-Versicherung;PostgreSQL+%7C+Python+%7C+Machine+Learning" alt="Typing SVG"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-DB-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

![Status](https://img.shields.io/badge/Status-In%20Entwicklung-2EA66E?style=flat-square)
![Sprint](https://img.shields.io/badge/Sprint-1%20von%203-2EA66E?style=flat-square)
![Lizenz](https://img.shields.io/badge/Verwendung-Nur%20Testdaten-lightgrey?style=flat-square)
![Hochschule](https://img.shields.io/badge/Fakult%C3%A4t%20III-Medien%2C%20Information%20%26%20Design-444?style=flat-square)

</div>

---

## 🎯 Über das Projekt

Data-Science- und KI-Projekt im Rahmen der Weiterbildung an der **Hochschule Hannover**
(Fakultät III – Medien, Information und Design). Umgesetzt wird ein vollständiger Workflow:
von der Speicherung in **PostgreSQL** über deskriptive Statistik, Einfluss-, Zusammenhangs-
und Hypothesenanalysen bis hin zu **Machine-Learning-Modellen** und einem automatisierten
Verarbeitungssystem.

> **❓ Fragestellung:** Welche Faktoren beeinflussen die Höhe der Schadenssumme sowie die
> Wahrscheinlichkeit eines Versicherungsbetrugs – und lassen sich diese vorhersagen?

<div align="center">

| 🧩 Aufgabe | 🎯 Zielvariable | 📌 Typ |
|---|---|---|
| **Regression** _(Pflicht)_ | `total_claim_amount` | Vorhersage der Gesamtschadenssumme |
| **Klassifikation** _(Bonus)_ | `fraud_reported` | Erkennung von Betrugsfällen |

</div>

---

## 👥 Team

| 👤 Mitglied | 📧 E-Mail |
|---|---|
| Oleg Fostii | olegfostij@gmail.com |
| Anton Duschak | anton.duschak@gmail.com |
| Bernard Turikumana | mutabazi105@gmail.com |

- 🏷️ **Gruppenname:** _<ABO_Team>_
- 📋 **Projektmanagement-Tool:** _<Jira-/https://github.com/fostij/InterGeeks-Agiles-Programmierprojekt.git>_

---

## 📊 Datensatz

**Auto Insurance Claims Data** (Kaggle, *buntyshah*) · **1000 Zeilen** · **40 Spalten**

<div align="center">

![Zeilen](https://img.shields.io/badge/Zeilen-1000-2EA66E?style=flat-square)
![Spalten](https://img.shields.io/badge/Spalten-40-2EA66E?style=flat-square)
![Betrugsquote](https://img.shields.io/badge/Betrugsquote-~25%25-orange?style=flat-square)

</div>

**🔍 Besonderheiten der Datenqualität:**
- ⚠️ Fehlende Werte als `?` codiert (`collision_type`, `property_damage`, `police_report_available`)
- 🗑️ Spalte `_c39` ist leer → wird verworfen
- ⚖️ Betrugsfälle unausgewogen (ca. 25 % `Y`) → relevant für die Klassifikation

> 📁 Die Datei `insurance_claims.csv` muss manuell unter `data/raw/` abgelegt werden (nicht im Repository enthalten).

---

## 🛠️ Tech-Stack

<div align="center">

| Bereich | Technologie |
|---|---|
| 🐍 Datenverarbeitung & Analyse | Python (pandas, NumPy, SciPy, statsmodels) |
| 🗄️ Datenspeicherung | PostgreSQL |
| 🤖 Machine Learning | scikit-learn |
| 📈 Visualisierung | matplotlib, seaborn |
| 🔄 Versionsverwaltung | Git & GitHub |
| 🌀 Vorgehen | Agile Entwicklung mit 3 Sprints |

</div>

---

## 📂 Projektstruktur

```
auto-insurance-claims/
├── 📁 config/         Konfiguration (DB-Zugangsdaten, nicht eingecheckt)
├── 📁 data/
│   ├── raw/          Rohdaten (insurance_claims.csv hier ablegen)
│   └── processed/    Aufbereitete Daten
├── 📁 sql/            SQL-Skripte (Schema, Laden, Transformation)
├── 📁 src/
│   ├── db/           Datenbankanbindung und Laden
│   ├── descriptive/  Deskriptive Statistik
│   ├── analysis/     Einfluss-, Zusammenhangsanalyse, Hypothesentests
│   ├── ml/           Machine-Learning-Modelle
│   └── automation/   Automatisiertes Verarbeitungssystem
├── 📁 notebooks/      Jupyter-Notebooks für Exploration
├── 📁 reports/        Ergebnisse und Grafiken
└── 📁 docs/           Dokumentation und Sprint-Planung
```

---

## 🚀 Einrichtung

<details open>
<summary><b>Schritt-für-Schritt-Anleitung</b></summary>

**1️⃣ Abhängigkeiten installieren**
```bash
pip install -r requirements.txt
```

**2️⃣ Datenbank-Zugangsdaten konfigurieren**
```bash
cp config/db_config.example.ini config/db_config.ini
# anschließend config/db_config.ini mit den eigenen Daten ausfüllen
```

**3️⃣ Datensatz ablegen**
> `insurance_claims.csv` nach `data/raw/` kopieren.

**4️⃣ Schema anlegen und Daten laden** (aus dem Projekt-Hauptverzeichnis)
```bash
psql -d insurance -f sql/01_schema.sql
psql -d insurance -f sql/02_load_raw.sql
psql -d insurance -f sql/03_transform_core.sql
```

Alternativ kann das Laden in das raw-Schema mit Python erfolgen:
```bash
python -m src.db.load_data
```

</details>

---

## 🌀 Sprint-Planung

```mermaid
flowchart LR
    A([🟢 Sprint 1<br/>Setup &amp; Statistik]) --> B([🟡 Sprint 2<br/>Analyse &amp; ML])
    B --> C([🔵 Sprint 3<br/>Automatisierung &amp; Präsentation])
```

| Sprint | Inhalt |
|:--:|---|
| **1️⃣** | Setup, Datensatzauswahl, Datenbankaufbau, deskriptive Statistik |
| **2️⃣** | Einfluss- und Zusammenhangsanalyse, Hypothesentests, Machine Learning |
| **3️⃣** | Automatisierung, optionale Weboberfläche, Präsentation |

> 📄 Details siehe [`docs/sprint1_backlog.md`](docs/sprint1_backlog.md)

---

## 📅 Wichtige Termine

| 📌 Abgabe | ⏰ Frist |
|---|:--:|
| Gruppenname, Mitglieder, PM-Tool-Link | **09.06.2026, 10:00** |
| Präsentation (erste Version) | **24.06.2026, 12:00** |
| Finale Abgabe (Code, DB, Doku, GitHub) | **03.07.2026, 11:59** |

📩 Alle Abgaben an: `mohammad.al-nasouh@hs-hannover.de`

---

## 🔒 Datenschutzhinweis

> Es werden **ausschließlich Testdaten** verwendet. Es findet **keine Verarbeitung realer
> Kundendaten** statt. Datenschutzrechtliche Anforderungen werden bei der Entwicklung des
> automatisierten Systems grundsätzlich berücksichtigt.

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14784F,100:2EA66E&height=120&section=footer" alt="Footer" width="100%"/>

<sub>⭐ Hochschule Hannover · Fakultät III – Medien, Information und Design · 2026</sub>

</div>
