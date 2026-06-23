# ---------------------------------------------------------------------
# dashboard.py — Web-Oberfläche (Streamlit) als Frontend für der Pipeline
# ---------------------------------------------------------------------

import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

# Projektordner zum Pfad hinzufügen, damit src-Module importierbar sind
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.pipeline import Pipeline, PredictionResult
from src.db.pipeline_repository import PipelineResultRepository
from src.logging_config import setup_logging
from app.app_config import SEVERITY_DE

# ---------------------------------------------------------------------
# 1) Seitenkonfiguration (muss der erste Streamlit-Befehl sein)
# ---------------------------------------------------------------------
st.set_page_config(page_title="KFZ-Schadenprognose", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 55%);
        animation: fadeIn 0.8s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .stDataFrame, .stImage img { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# 2) Pipeline einmalig laden
# ---------------------------------------------------------------------
@st.cache_resource
def get_pipeline() -> Pipeline:
    """Initialisiert die Pipeline einmalig pro Session. 
    st.cashe_resource sorgt dafür, dass die Modelle nicht bei jeder
    Interaktion neu geladen werden."""
    setup_logging()
    return Pipeline(PipelineResultRepository())

@st.cache_data
def load_dataset():
    """Lädt den Datensatz für den Diagramm-Vergleich. None, falls Datei fehlt."""
    try:
        return pd.read_csv("data/raw/dataset.csv")
    except FileNotFoundError:
        return None
    
def format_eur(value: float) -> str:
    """Formatiert einen Betrag im deutschen Format (Punkt als Tausender)."""
    return f"{value:,.0f} EUR".replace(",", ".")


# ---------------------------------------------------------------------
# 3) Darstellung des Pipeline-Ergebnisses in der Weboberfläche
# ---------------------------------------------------------------------
def render_result(result: PredictionResult) -> None:
    """Stellt das PredictionResult der Pipeline dar: extrahierte Felder,
    Schadensschwere (Quelle + beide Schätzungen) und Kostenschätzung."""
    st.subheader("Ergebnis")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Extrahierte Informationen**")
        if result.fields:
            rows = [(k, "—" if v is None else str(v)) for k, v in result.fields.items()]
            st.table(pd.DataFrame(rows, columns=["Feld", "Wert"]).astype(str))
        else:
            st.write("Keine Felder aus dem Text extrahiert.")

    with right:
        st.markdown("**Schadensschwere**")
        if result.severity_from_text:
            st.write(f"Aus Text: {SEVERITY_DE.get(result.severity_from_text, result.severity_from_text)}")
        if result.severity_from_photo:
            conf = f" ({result.photo_confidence:.1%})" if result.photo_confidence else ""
            st.write(f"Aus Foto (CNN): {SEVERITY_DE.get(result.severity_from_photo, result.severity_from_photo)}{conf}")

        if result.final_severity:
            quelle = "Foto" if result.severity_source == "photo" else "Text"
            st.write(f"**Finale Schwere:** {SEVERITY_DE.get(result.final_severity, result.final_severity)} "
                     f"(Quelle: {quelle})")

        if result.predicted_amount is not None:
            st.metric("Geschätzte Schadenhöhe (vehicle_claim)", format_eur(result.predicted_amount))
        else:
            st.info("Keine Kostenschätzung möglich (keine Schwere bestimmt).")


def render_comparison_chart(amount: float) -> None:
    """Zeigt das interaktive Vergleichsdiagramm (Prognose vs. Historie)."""
    df = load_dataset()
    if df is None:
        return
    means = df.groupby("incident_severity")["total_claim_amount"].mean()
    labels_de = [SEVERITY_DE.get(s, s) for s in means.index]

    fig = go.Figure()
    fig.add_bar(x=labels_de, y=means.values,
                marker=dict(color=means.values, colorscale="Greens",
                            line=dict(color="#2E7D32", width=1)),
                hovertemplate="%{x}: %{y:,.0f} EUR<extra></extra>")
    fig.add_hline(y=amount, line_color="#1B5E20", line_width=3, line_dash="dash",
                  annotation_text=f"Prognose: {format_eur(amount)}",
                  annotation_position="top left")
    fig.update_layout(title="Prognose im Vergleich zu historischen Schadensfällen",
                      yaxis_title="Schadenhöhe (EUR)",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=14), showlegend=False,
                      margin=dict(t=60, b=40, l=60, r=20))
    st.plotly_chart(fig, width="stretch")


# =====================================================================
# 4) Benutzeroberfläche
# =====================================================================
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
st.caption("Schadensmeldung beschreiben und optional ein Foto hochladen. "
           "Die zentrale Pipeline analysiert Text und Bild und schätzt die Schadenhöhe.")


col_text, col_photo = st.columns(2)

with col_text:
    st.subheader("1. Schadensmeldung")
    message = st.text_area("Beschreibung des Schadens", height=220,
                           placeholder="Beispiel: Unfall mit dem Subaru bei klarem Wetter, "
                                       "Kratzer an der Beifahrertür, Stoßfänger gebrochen ...")
with col_photo:
    st.subheader("2. Schadenfoto (optional)")
    photo = st.file_uploader("Foto des beschädigten Fahrzeugs", type=["jpg", "jpeg", "png"])
    if photo is not None:
        st.image(photo, caption="Hochgeladenes Foto", width="stretch")

st.divider()

if st.button("Prognose erstellen", type="primary"):
    if not message.strip() and photo is None:
        st.warning("Bitte eine Schadensmeldung eingeben oder ein Foto hochladen.")
        st.stop()

    # Foto (aus dem Speicher) temporär auf Platte schreiben, da die
    # Pipeline einen Dateipfad erwartet.
    photo_path = None
    if photo is not None:
        suffix = Path(photo.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(photo.getbuffer())
            photo_path = tmp.name

    with st.spinner("Analyse läuft ..."):
        pipeline = get_pipeline()
        result = pipeline.run(text=message, photo_path=photo_path, source="dashboard")

    render_result(result)
    if result.predicted_amount is not None:
        render_comparison_chart(result.predicted_amount)