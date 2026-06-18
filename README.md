<div align="center">

# 🚗 KFZ-Schadenprognose

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3000&pause=800&color=4CAF50&center=true&vCenter=true&width=640&lines=Data+Science+%26+KI-Workflow;Von+der+Datenbank+bis+zur+Web-Oberfläche;Schadenprognose+aus+Text+und+Bild" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.12-4CAF50?style=for-the-badge&logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Datenbank-4CAF50?style=for-the-badge&logo=postgresql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-ORM-4CAF50?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-4CAF50?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-4CAF50?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualisierung-4CAF50?style=for-the-badge&logo=plotly&logoColor=white)

**Gruppe InterGeeks · Hochschule Hannover · Fakultät III – Medien, Information und Design**

</div>

---

## 📋 Projektübersicht

Dieses Projekt setzt einen vollständigen **Data-Science- und KI-Workflow** rund um
KFZ-Versicherungsschäden um – von der Datenhaltung über die Analyse bis zur
Anwendung. Auf Basis eines offenen Datensatzes zu Versicherungsfällen wird die
**erwartete Schadenhöhe** prognostiziert. Ergänzend bestimmt ein **CNN** die
Schadensschwere aus einem Foto des beschädigten Fahrzeugs. Über ein
**Streamlit-Dashboard** lassen sich Schadensmeldungen als Freitext eingeben und
Fotos hochladen; das System extrahiert die relevanten Merkmale, speichert sie in
der Datenbank und visualisiert das Ergebnis.

> **Fragestellung:** Welche Faktoren beeinflussen die Schadenhöhe, und lässt sie
> sich aus Schadensmeldung und Foto zuverlässig prognostizieren?
> **Zielvariablen:** `total_claim_amount` (Regression), `fraud_reported` (Klassifikation).

---

## ✨ Features

| Bereich | Umsetzung |
| --- | --- |
| 🗄️ **Datenhaltung** | Normalisiertes PostgreSQL-Schema (mehrere Tabellen, Fremdschlüssel, Indizes) |
| 🔄 **ETL** | CSV einlesen, bereinigen (`?`→NULL), Typen umwandeln und in die DB laden |
| 📊 **Deskriptive Statistik** | Lageparameter, Streuungsmaße, Verteilungen, Ausreißer, Datenqualität |
| 🔗 **Einfluss- & Zusammenhangsanalyse** | Korrelationen und Einfluss der Variablen auf die Zielvariable |
| 🧪 **Hypothesentests** | Formulierung, Voraussetzungsprüfung und Interpretation |
| 🤖 **Regression** (Pflicht) | Vorhersage der Schadenhöhe, Vergleich mehrerer Modelle |
| 🏷️ **Klassifikation** (Bonus) | Betrugserkennung (`fraud_reported`) mit mehreren Verfahren |
| 🧠 **CNN-Bilderkennung** (Bonus) | Transfer Learning (MobileNetV2): Foto → Schadensschwere |
| 🔎 **Automatisierte Textanalyse** | Extraktion relevanter Merkmale aus der Freitext-Schadensmeldung |
| 🖥️ **Web-Oberfläche** (Bonus) | Streamlit-Dashboard: Text- und Foto-Eingabe, Speicherung, Prognose, Diagramm |

---

## 🛠️ Tech-Stack

- **Sprache:** Python 3.12
- **Datenbank:** PostgreSQL · SQLAlchemy · psycopg2
- **Datenverarbeitung:** pandas · NumPy
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow / Keras (MobileNetV2)
- **Visualisierung:** Plotly · Matplotlib
- **Web-Oberfläche:** Streamlit
- **Versionsverwaltung & Agile:** Git · GitHub · Jira (Scrum)

---

## 📁 Projektstruktur

```
InterGeeks-Agiles-Programmierprojekt/
├── app/
│   └── dashboard.py          # Streamlit-Weboberfläche
├── sql/
│   ├── schema.sql            # PostgreSQL-Schema
│   └── export/               # SQL-Export
├── src/
│   ├── db/                   # Datenbankanbindung & ETL
│   │   ├── connection.py
│   │   └── load_data.py
│   ├── analysis/             # Statistik-Notebooks
│   ├── models/               # Regression & Klassifikation
│   ├── automation/           # Automatisierte Textverarbeitung
│   └── cnn/                  # CNN: Training & Vorhersage
│       ├── train_cnn.py
│       └── predict_image.py
├── utils/                    # Hilfsskripte
├── docs/                     # Dokumentation & Sprint-Protokolle
├── .streamlit/
│   └── config.toml           # Theme im Farbschema der Hochschule
├── main.py
└── requirements.txt
```

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

# Zugangsdaten konfigurieren (.env aus Vorlage erstellen und ausfüllen)
cp .env.example .env

# Datenbank anlegen und befüllen
psql -U postgres -c "CREATE DATABASE kfz_schaden;"
python src/db/load_data.py
```

---

## ▶️ Nutzung

```bash
# Dashboard starten
streamlit run app/dashboard.py

# CNN-Modell trainieren (Bilddaten unter data/raw/car_damage/ erforderlich)
python src/cnn/train_cnn.py

# Einzelnes Foto klassifizieren
python src/cnn/predict_image.py pfad/zum/foto.jpg
```

Im Dashboard wird links eine **Schadensmeldung** als Freitext eingegeben und rechts
optional ein **Foto** hochgeladen. Nach dem Klick auf *Prognose erstellen* zeigt die
Anwendung die extrahierten Merkmale, die Schadensschwere (aus Text und/oder Foto),
die prognostizierte Schadenhöhe sowie ein Vergleichsdiagramm an. Die Eingaben
werden in der Datenbank gespeichert.

---

## 🗄️ Datenbank

Der Datensatz wird in einem **normalisierten Schema** gespeichert:

```
kunden ──< policen ──< unfaelle ──< schaeden
               │
               └──< fahrzeuge
```

- **Schema:** Tabellen mit Primär-/Fremdschlüsseln und Indizes; Geldbeträge als
  `NUMERIC`, fehlende Werte als `NULL`.
- **Verbindung:** zentrale SQLAlchemy-Engine; Zugangsdaten liegen in einer
  `.env`-Datei (nicht im Code, nicht im Repository).
- **ETL:** liest die CSV ein, bereinigt sie (`?` → NULL, leere Spalte entfernt,
  Typen gesetzt) und verteilt die Daten auf die Tabellen.
- **Persistenz:** über das Dashboard erfasste Meldungen und Prognosen werden in
  der Tabelle `vorhersagen` gespeichert.

---

## 🔧 Funktionsweise (Text & Foto)

Text und Foto liefern unterschiedliche, sich ergänzende Informationen:

```
Foto  -> CNN -> Schadensschwere (visuell)            |
                                                      +--> Prognose -> Diagramm
Text  -> Extraktion -> Fahrzeuge, Verletzte, Polizei |
```

- **Nur Text:** Die Schwere wird aus Schlüsselwörtern abgeleitet.
- **Text + Foto:** Die Schwere stammt aus dem Foto (CNN, zuverlässiger), der
  Kontext aus dem Text. Weichen beide Einschätzungen ab, weist das System darauf hin.

---

## 📸 Screenshots

<div align="center">
  <em>Demo folgt – Screenshot bzw. GIF des Dashboards hier einfügen.</em>
  <!-- ![Dashboard](docs/screenshots/dashboard.gif) -->
</div>

---

## 🔭 Ausblick

- Anbindung des trainierten **Regressionsmodells** an das Dashboard anstelle der
  aktuellen regelbasierten Platzhalter-Schätzung.
- Diskussion der **Modellgrenzen** (kleiner Datensatz, begrenzte Genauigkeit).
- Erweiterungen außerhalb des Projektumfangs: Datenschutz, OCR für Bild-PDFs.

---

## ⚠️ Hinweis

Es werden ausschließlich **Testdaten** verwendet – keine realen Kundendaten. Die
Beträge des Originaldatensatzes (USD) werden im Projekt als EUR-Testwerte interpretiert.

---

## 👥 Team

Projektarbeit der Gruppe **InterGeeks** (3 Personen) im Rahmen des Agilen
Programmierprojekts an der Hochschule Hannover – Fakultät III, Medien, Information
und Design.