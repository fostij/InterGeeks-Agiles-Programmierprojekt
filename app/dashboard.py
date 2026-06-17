# ---------------------------------------------------------------------
# dashboard.py — Web interface for damage prediction (Streamlit)
# ---------------------------------------------------------------------

import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Projektordner zum Pfad hinzufügen, damit das CNN-Modul importierbar ist
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------
# 1) Seitenkonfiguration (muss der erste Streamlit-Befehl sein)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="KFZ-Schadenprognose",
    page_icon="🚗",
    layout="wide", 
)

# Abbildung der CSV-Kategorien auf deutsche Anzeigenamen.
# Interne englische Werte bleiben erhalten (Datensatz/Modell-Kompatibilitaet).
SEVERITY_DE = {
    "Trivial Damage": "Bagatellschaden",
    "Minor Damage": "Leichter Schaden",
    "Major Damage": "Erheblicher Schaden",
    "Total Loss": "Totalschaden",
}

# Durchschnittliche Schadenhoehe je Schwere (aus dem Datensatz abgeleitet).
# Dient als Basis für die regelbasierte Platzhalter-Prognose.
BASE_AMOUNT = {
    "Trivial Damage": 5_000,
    "Minor Damage": 35_000,
    "Major Damage": 62_000,
    "Total Loss": 65_000,
}

# Abbildung der CNN-Klassen auf die Schwere-Kategorien des Datensatzes.
# So spricht das Foto-Ergebnis dieselbe "Sprache" wie die Textanalyse.
CNN_TO_SEVERITY = {
    "01-minor": "Minor Damage",
    "02-moderate": "Major Damage",
    "03-severe": "Total Loss",
}


# ---------------------------------------------------------------------
# 2) Datensatz laden (für die Vergleichswerte im Diagramm)
# ---------------------------------------------------------------------
@st.cache_data
def load_dataset():
    """Lädt den Datensatz für den Vergleich der Prognose mit den
    historischen Durchschnittswerten. Gibt None zurück, falls die
    Datei fehlt (die App bleibt dann ohne Vergleichsdiagramm nutzbar)."""
    try:
        return pd.read_csv("data/raw/dataset.csv")
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------
# 3) Merkmalsextraktion aus dem Freitext
# ---------------------------------------------------------------------
def extract_features(text: str) -> dict:
    """Extrahiert strukturierte Merkmale aus einer deutschsprachigen
    Schadensmeldung. Liefert auch eine textbasierte Schwere-Einschaetzung,
    die spaeter ggf. durch das Foto-Ergebnis ueberschrieben wird."""

    text_lower = text.lower()

    # --- Schadensschwere aus Schluesselwoertern (Fallback ohne Foto) ---
    if any(w in text_lower for w in ["totalschaden", "total zerstört", "abgeschleppt"]):
        severity = "Total Loss"
    elif any(w in text_lower for w in ["erheblich", "schwer beschädigt", "großer schaden"]):
        severity = "Major Damage"
    elif any(w in text_lower for w in ["kratzer", "bagatelle", "geringfügig"]):
        severity = "Trivial Damage"
    else:
        severity = "Minor Damage"

    # --- Zahlenbasierte Merkmale per regulaerem Ausdruck ---
    m = re.search(r"(\d+)\s*fahrzeug", text_lower)
    vehicles = int(m.group(1)) if m else 1

    m = re.search(r"(\d+)\s*(personen?\s*)?verletzt", text_lower)
    injuries = int(m.group(1)) if m else 0

    m = re.search(r"(\d+)\s*zeug", text_lower)
    witnesses = int(m.group(1)) if m else 0

    police = "polizei" in text_lower

    return {
        "severity": severity, 
        "vehicles": vehicles,
        "injuries": injuries,
        "witnesses": witnesses,
        "police": police,
    }


# ---------------------------------------------------------------------
# 4) Schadensschwere aus dem Foto (CNN) bestimmen
# ---------------------------------------------------------------------
def predict_severity_from_photo(uploaded_file) -> tuple[str, float] | None:
    """Speichert das hochgeladene Foto temporaer und ruft das CNN auf.
    Gibt (interne Schwere-Kategorie, Wahrscheinlichkeit) zurueck oder None,
    falls kein Modell verfuegbar ist."""
    # Lazy-Import: TensorFlow nur laden, wenn wirklich ein Foto kommt
    try:
        from src.cnn.predict_image import predict_severity, SEVERITY_DE as CNN_LABELS
    except Exception:
        return None

    # Streamlit liefert die Datei im Speicher -> temporaer auf Platte schreiben,
    # da predict_severity einen Dateipfad erwartet.
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    label_de, confidence = predict_severity(tmp_path)

    # predict_severity liefert bereits einen deutschen Namen; wir brauchen
    # zusätzlich die interne Kategorie -> ueber die Umkehrung der CNN-Map.
    cnn_class = next((k for k, v in CNN_LABELS.items() if v == label_de), None)
    severity = CNN_TO_SEVERITY.get(cnn_class, "Minor Damage")
    return severity, confidence


# ---------------------------------------------------------------------
# 5) Prognose der Schadenhoehe — VORLAEUFIGER PLATZHALTER
#    TODO: Durch das trainierte Regressionsmodell ersetzen.
# ---------------------------------------------------------------------
def predict_amount(features: dict) -> float:
    """Schätzt die Schadenhoehe (EUR) regelbasiert anhand der Schwere
    sowie Zuschlaegen für beteiligte Fahrzeuge und Verletzte."""
    amount = BASE_AMOUNT[features["severity"]]
    amount += (features["vehicles"] - 1) * 4_000
    amount += features["injuries"] * 3_000
    return float(amount)


def format_eur(value: float) -> str:
    """Formatiert einen Betrag im deutschen Format (Punkt als Tausender)."""
    return f"{value:,.0f} EUR".replace(",", ".")


# =====================================================================
# 6) Benutzeroberflaeche
# =====================================================================
st.title("🚗 KFZ-Schadenprognose")
st.caption(
    "Schadensmeldung beschreiben und optional ein Foto des Schadens hochladen. "
    "Das System schätzt die Schadensschwere und die erwartete Schadenhoehe."
)

# --- Zweispaltige Eingabe: links Text, rechts Foto ---
col_text, col_photo = st.columns(2)

with col_text:
    st.subheader("1. Schadensmeldung")
    message = st.text_area(
        label="Beschreibung des Schadens",
        height=220,
        placeholder=(
            "Beispiel: Am 10.06.2026 kam es zu einem Unfall mit 2 Fahrzeugen. "
            "Mein Auto wurde erheblich beschädigt, 1 Person wurde verletzt. "
            "Die Polizei wurde informiert, es gibt 2 Zeugen."
        ),
    )

with col_photo:
    st.subheader("2. Schadenfoto (optional)")
    photo = st.file_uploader(
        label="Foto des beschädigten Fahrzeugs",
        type=["jpg", "jpeg", "png"],
    )
    if photo is not None:
        st.image(photo, caption="Hochgeladenes Foto", use_container_width=True)

st.divider()

# --- Aktion: Prognose nur nach Klick ---
if st.button("Prognose erstellen", type="primary"):

    if not message.strip() and photo is None:
        st.warning("Bitte eine Schadensmeldung eingeben oder ein Foto hochladen.")
        st.stop()

    # --- Merkmale aus dem Text (Kontext + Text-Schwere als Fallback) ---
    features = extract_features(message) if message.strip() else {
        "severity": "Minor Damage", "vehicles": 1,
        "injuries": 0, "witnesses": 0, "police": False,
    }
    severity_text = features["severity"]

    # --- Schwere aus dem Foto (CNN), falls vorhanden ---
    severity_photo, photo_confidence = None, None
    if photo is not None:
        result = predict_severity_from_photo(photo)
        if result is not None:
            severity_photo, photo_confidence = result
            # Foto hat Vorrang: es überschreibt die finale Schwere
            features["severity"] = severity_photo
        else:
            st.info("CNN-Modell nicht gefunden — es wird die Schwere aus dem Text verwendet.")

    # --- Prognose berechnen ---
    amount = predict_amount(features)

    # --- Ergebnisanzeige ---
    st.subheader("Ergebnis")
    res_left, res_right = st.columns([1, 1])

    with res_left:
        st.markdown("**Extrahierte Informationen**")
        info = {
            "Schadensschwere": SEVERITY_DE[features["severity"]],
            "Beteiligte Fahrzeuge": features["vehicles"],
            "Verletzte Personen": features["injuries"],
            "Zeugen": features["witnesses"],
            "Polizei informiert": "Ja" if features["police"] else "Nein",
        }
        st.table(pd.DataFrame(info.items(), columns=["Merkmal", "Wert"]))

    with res_right:
        st.markdown("**Schadensschwere**")
        # Quelle der Schwere transparent anzeigen
        if severity_photo is not None:
            st.write(f"Aus Foto (CNN): **{SEVERITY_DE[severity_photo]}** "
                     f"({photo_confidence:.1%})")
        if message.strip():
            st.write(f"Aus Text: {SEVERITY_DE[severity_text]}")
        # Hinweis, falls Text und Foto deutlich abweichen
        if severity_photo is not None and message.strip() \
                and severity_photo != severity_text:
            st.warning("Beschreibung und Foto weichen voneinander ab "
                       "— ggf. genauere Prüfung erforderlich.")

        st.metric("Erwartete Schadenhöhe", format_eur(amount))

    # --- Visualisierung: Prognose vs. historische Durchschnitte ---
    df = load_dataset()
    if df is not None:
        means = df.groupby("incident_severity")["total_claim_amount"].mean()
        labels_de = [SEVERITY_DE.get(s, s) for s in means.index]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels_de, means.values, color="#9aa7b1",
               label="Durchschnitt (Datensatz)")
        ax.axhline(amount, color="#c0392b", linewidth=2,
                   label=f"Prognose: {format_eur(amount)}")
        ax.set_ylabel("Schadenhöhe (EUR)")
        ax.set_title("Prognose im Vergleich zu historischen Schadensfällen")
        ax.legend()
        plt.xticks(rotation=15)
        st.pyplot(fig)
    else:
        st.info("Datensatz nicht gefunden — Vergleichsdiagramm erscheint, "
                "sobald data/raw/dataset.csv vorhanden ist.")

    st.caption(
        "Hinweis: Die Schadenhöhe wird aktuell regelbasiert geschätzt "
        "(Platzhalter). Nach Abschluss des Trainings wird hier das "
        "ML-Regressionsmodell eingesetzt."
    )