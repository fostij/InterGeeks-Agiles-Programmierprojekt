# ---------------------------------------------------------------------
# dashboard.py — Web interface for damage prediction (Streamlit)
# ---------------------------------------------------------------------

import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

# Projektordner zum Pfad hinzufügen, damit das CNN-Modul importierbar ist
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------
# 1) Seitenkonfiguration (muss der erste Streamlit-Befehl sein)
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="KFZ-Schadenprognose",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------
# Eigenes Styling: grüner Farbverlauf-Hintergrund + sanfte Einblend-
# Animation. Wird einmalig per CSS injiziert.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Sanfter grüner Verlauf als Hintergrund */
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 55%);
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    /* Abgerundete, weiche Karten-Optik fuer Tabellen und Bilder */
    .stDataFrame, .stImage img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Abbildung der CSV-Kategorien auf deutsche Anzeigenamen.
# Interne englische Werte bleiben erhalten (Datensatz/Modell-Kompatibilität).
SEVERITY_DE = {
    "Trivial Damage": "Bagatellschaden",
    "Minor Damage": "Leichter Schaden",
    "Major Damage": "Erheblicher Schaden",
    "Total Loss": "Totalschaden",
}

# Durchschnittliche Schadenhöhe je Schwere (aus dem Datensatz abgeleitet).
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
    die später ggf. durch das Foto-Ergebnis überschrieben wird."""

    text_lower = text.lower()

    # --- Schadensschwere aus Schlüsselwörtern (Fallback ohne Foto) ---
    if any(w in text_lower for w in ["totalschaden", "total zerstört", "abgeschleppt"]):
        severity = "Total Loss"
    elif any(w in text_lower for w in ["erheblich", "schwer beschädigt", "großer schaden"]):
        severity = "Major Damage"
    elif any(w in text_lower for w in ["kratzer", "bagatelle", "geringfügig"]):
        severity = "Trivial Damage"
    else:
        severity = "Minor Damage"

    # --- Zahlenbasierte Merkmale per regulärem Ausdruck ---
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
    Gibt (interne Schwere-Kategorie, Wahrscheinlichkeit) zurück oder None,
    falls kein Modell verfügbar ist."""
    # Lazy-Import: TensorFlow nur laden, wenn wirklich ein Foto kommt
    try:
        from src.models.cnn.predict_image import predict_severity, SEVERITY_DE as CNN_LABELS
    except Exception:
        return None

    # Streamlit liefert die Datei im Speicher -> temporär auf Platte schreiben,
    # da predict_severity einen Dateipfad erwartet.
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    label_de, confidence = predict_severity(tmp_path)

    # predict_severity liefert bereits einen deutschen Namen; wir brauchen
    # zusätzlich die interne Kategorie -> über die Umkehrung der CNN-Map.
    cnn_class = next((k for k, v in CNN_LABELS.items() if v == label_de), None)
    severity = CNN_TO_SEVERITY.get(cnn_class, "Minor Damage")
    return severity, confidence


# ---------------------------------------------------------------------
# 5) Prognose der Schadenhöhe — VORLAEUFIGER PLATZHALTER
#    TODO: Durch das trainierte Regressionsmodell ersetzen.
# ---------------------------------------------------------------------
def predict_amount(features: dict) -> float:
    """Schätzt die Schadenhöhe (EUR) regelbasiert anhand der Schwere
    sowie Zuschlägen für beteiligte Fahrzeuge und Verletzte."""
    amount = BASE_AMOUNT[features["severity"]]
    amount += (features["vehicles"] - 1) * 4_000
    amount += features["injuries"] * 3_000
    return float(amount)


def format_eur(value: float) -> str:
    """Formatiert einen Betrag im deutschen Format (Punkt als Tausender)."""
    return f"{value:,.0f} EUR".replace(",", ".")


# ---------------------------------------------------------------------
# 5b) Eingaben und Prognose in der PostgreSQL-Datenbank speichern
# ---------------------------------------------------------------------
def save_prediction(message: str, features: dict, amount: float,
                    severity_source: str) -> bool:
    """Speichert die eingegebene Schadensmeldung samt extrahierten Merkmalen
    und prognostizierter Schadenhöhe in der Tabelle `vorhersagen`.
    Gibt True bei Erfolg zurück, False falls keine DB-Verbindung möglich ist."""
    try:
        from src.db.connection import get_engine
        engine = get_engine()
        with engine.begin() as conn:
            # Tabelle bei Bedarf anlegen (macht das Dashboard unabhängig)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS vorhersagen (
                    vorhersage_id    SERIAL PRIMARY KEY,
                    erstellt_am      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    meldung          TEXT,
                    schadensschwere  VARCHAR(20),
                    quelle_schwere   VARCHAR(10),
                    anzahl_fahrzeuge INTEGER,
                    anzahl_verletzte INTEGER,
                    anzahl_zeugen    INTEGER,
                    polizei          BOOLEAN,
                    prognose_eur     NUMERIC(12,2)
                )
            """))
            # Eingaben + Prognose als neue Zeile einfügen (parametrisiert -> kein SQL-Injection)
            conn.execute(text("""
                INSERT INTO vorhersagen
                    (meldung, schadensschwere, quelle_schwere, anzahl_fahrzeuge,
                     anzahl_verletzte, anzahl_zeugen, polizei, prognose_eur)
                VALUES
                    (:meldung, :schwere, :quelle, :fahrzeuge,
                     :verletzte, :zeugen, :polizei, :prognose)
            """), {
                "meldung": message,
                "schwere": features["severity"],
                "quelle": severity_source,
                "fahrzeuge": features["vehicles"],
                "verletzte": features["injuries"],
                "zeugen": features["witnesses"],
                "polizei": features["police"],
                "prognose": amount,
            })
        return True
    except Exception:
        # DB nicht erreichbar -> App bleibt nutzbar, nur ohne Speicherung
        return False


# =====================================================================
# 6) Benutzeroberfläche
# =====================================================================
# Kopfzeile mit grünem Auto-Icon (Inline-SVG) statt rotem Emoji
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
      <svg width="42" height="42" viewBox="0 0 24 24" fill="#4CAF50"
           xmlns="http://www.w3.org/2000/svg">
        <path d="M5 11l1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11h1a1
                 1 0 0 1 1 1v4a1 1 0 0 1-1 1h-1v1a1 1 0 0 1-2 0v-1H7v1a1 1 0
                 0 1-2 0v-1H4a1 1 0 0 1-1-1v-4a1 1 0 0 1 1-1h1zm2.2 0h9.6l-1-3H8.2l-1
                 3zM6.5 15a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm11 0a1 1 0 1 0 0-2 1 1 0 0 0 0 2z"/>
      </svg>
      <h1 style="margin:0; color:#1B1B1B;">KFZ-Schadenprognose</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Schadensmeldung beschreiben und optional ein Foto des Schadens hochladen. "
    "Das System schätzt die Schadensschwere und die erwartete Schadenhöhe."
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

# --- Aktion: Vorhersage nur nach Klick ---
if st.button("Vorhersage erstellen", type="primary"):

    if not message.strip() and photo is None:
        st.warning("Bitte eine Schadensmeldung eingeben oder ein Foto hochladen.")
        st.stop()

    # --- Merkmale aus dem Text (Kontext + Text-Schwere als Fallback) ---
    features = extract_features(message) if message.strip() else {
        "severity": "Minor Damage", "vehicles": 1,
        "injuries": 0, "witnesses": 0, "police": False,
    }
    severity_text = features["severity"]    # Schwere laut Text

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

    # --- Vorhersage berechnen ---
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

    # --- Eingaben und Vorhersage in der Datenbank speichern ---
    severity_source = "Foto" if severity_photo is not None else "Text"
    if save_prediction(message, features, amount, severity_source):
        st.success("Eingaben und Vorhersage wurden in der Datenbank gespeichert.")
    else:
        st.info("Hinweis: keine Datenbankverbindung — Ergebnis wurde nicht gespeichert.")

    # --- Visualisierung: Vorhersage vs. historische Durchschnitte ---
    df = load_dataset()
    if df is not None:
        means = df.groupby("incident_severity")["total_claim_amount"].mean()
        labels_de = [SEVERITY_DE.get(s, s) for s in means.index]

        # Interaktives Balkendiagramm (Plotly) im grünen Farbschema.
        # Balken: historische Durchschnittswerte; Linie: aktuelle Vorhersage.
        fig = go.Figure()
        fig.add_bar(
            x=labels_de, y=means.values,
            marker=dict(color=means.values, colorscale="Greens",
                        line=dict(color="#2E7D32", width=1)),
            name="Durchschnitt (Datensatz)",
            hovertemplate="%{x}: %{y:,.0f} EUR<extra></extra>",
        )
        # Vorhersage als horizontale Referenzlinie
        fig.add_hline(
            y=amount, line_color="#1B5E20", line_width=3, line_dash="dash",
            annotation_text=f"Vorhersage: {format_eur(amount)}",
            annotation_position="top left",
        )
        fig.update_layout(
            title="Vorhersage im Vergleich zu historischen Schadensfällen",
            yaxis_title="Schadenhöhe (EUR)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=14),
            showlegend=False,
            margin=dict(t=60, b=40, l=60, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Datensatz nicht gefunden — Vergleichsdiagramm erscheint, "
                "sobald data/raw/dataset.csv vorhanden ist.")

    st.caption(
        "Hinweis: Die Schadenhöhe wird aktuell regelbasiert geschätzt "
        "(Platzhalter). Nach Abschluss des Trainings wird hier das "
        "ML-Regressionsmodell eingesetzt."
    )