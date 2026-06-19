import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from src.db.connection import get_engine
from src.models.regression.predict_model import predict_damage

# 1) Page Config
st.set_page_config(page_title="KFZ-Schadenprognose Pro", layout="wide")

# Styling
st.markdown("""<style>.stApp {background: #f8f9fa;}</style>""", unsafe_allow_html=True)

# 2) UI Components
st.title("🛡️ KFZ-Schadenprognose Professional")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Unfall-Parameter")
    message = st.text_area("Beschreibung des Schadens", height=150)
    vehicles = st.number_input("Anzahl Fahrzeuge", min_value=1, value=1)
    injuries = st.number_input("Anzahl Verletzte", min_value=0, value=0)
    witnesses = st.number_input("Anzahl Zeugen", min_value=0, value=0)
    
    if st.button("Vorhersage berechnen", type="primary"):
        # Gukoresha ML model
        amount = predict_damage(vehicles, injuries, witnesses)
        
        # Kwerekana ibisubizo
        st.metric("Erwartete Schadenhöhe", f"{amount:,.2f} EUR")
        
        # Visualization
        chart_data = pd.DataFrame({
            'Kategorie': ['Fahrzeuge', 'Verletzte', 'Zeugen'],
            'Wert': [vehicles, injuries, witnesses]
        })
        fig = px.bar(chart_data, x='Kategorie', y='Wert', title="Unfall-Analyse")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Historische Daten")
    # Gusoma amakuru muri Database
    try:
        engine = get_engine()
        query = "SELECT meldung, prognose_eur FROM vorhersagen ORDER BY erstellt_am DESC LIMIT 5"
        df_history = pd.read_sql(query, engine)
        st.dataframe(df_history, use_container_width=True)
    except Exception as e:
        st.info("Datenbank ist noch leer oder nicht verbunden.")

# Image of project architecture