# =====================================================================
# dashboard.py — Weboberfläche zur Schadenprognose (Streamlit)
# ---------------------------------------------------------------------
# Zweck (laut Aufgabenstellung):
#   - Eingabe von Kundeninformationen / Textdaten (Schadensmeldung)
#   - Automatisierte Extraktion relevanter Merkmale aus dem Text
#   - Anwendung eines Modells zur Vorhersage der Schadenhöhe
#   - Visualisierung der Prognose (Diagramm + Text)
#
# Start der App im Terminal (aus dem Projektordner):
#   streamlit run app/dashboard.py
# =====================================================================

import re                        # Reguläre Ausdrücke für die Merkmalextraktion
import pandas as pd              # Datenverarbeitung
import matplotlib.pyplot as plt  # Visualisierung
import streamlit as st           # Web-Framework für das Dashboard

# ---------------------------------------------------------------------
# Übersetzung der Datensatz-Kategorien für die Anzeige (UI)
# WICHTIG: Intern bleiben die englischen Originalwerte erhalten,
# da sie exakt den Kategorien im Datensatz/Modell entsprechen müssen.
# ---------------------------------------------------------------------
SCHWERE_DEUTSCH = {
    "Trivial Damage": "Bagatellschaden",
    "Minor Damage": "Leichter Schaden",
    "Major Damage": "Erheblicher Schaden",
    "Total Loss": "Totalschaden",
}

# ---------------------------------------------------------------------
# 1) Grundkonfiguration der Seite
#    Muss der ERSTE Streamlit-Befehl im Skript sein.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="KFZ-Schadenprognose",  # Titel der Seite im Browser-Tab
    page_icon="🚗",
    layout="centered",                 # Zentrierte Anordnung der Inhalte
)


# ---------------------------------------------------------------------
# 2) Daten laden (für Vergleichswerte im Diagramm)
#    @st.cache_data sorgt dafür, dass die CSV nur EINMAL gelesen wird
#    und nicht bei jedem Klick erneut (Performance-Leistung).
# ---------------------------------------------------------------------

@st.cache_data
def load_data():
    """Lädt den Datensatz, um die Prognose mit dem Durchschnitt
    aller bisherigen Schadensfälle vergleichen zu können.
    Gibt None zurück, falls die Datei (noch) nicht vorhanden ist —
    die App funktioniert dann trotzdem, nur ohne Vergleichswerte."""
    try:
        return pd.read_csv("data/raw/dataset.csv")
    except FileNotFoundError:
        return None
    
# ---------------------------------------------------------------------
# 3) Merkmals-Extraktion aus dem Freitext
#    Hier suchen wir mit einfachen Regeln (Schlüsselwörter + Regex)
#    die Informationen, die unser Modell als Eingabe braucht.
#    Später kann dieser Teil durch ein NLP-Verfahren ersetzt werden.
# ---------------------------------------------------------------------

def extrahiere_merkmale(text: str) -> dict:
    """Extrahiert strukturierte Merkmale aus einer deutschsprachigen
    Schadensmeldung (Freitext) und gibt sie als Dictionary zurück."""

    text_klein = text.lower() # Für die Suche ignorieren wir Groß-/Kleinschreibung

    # --- Schadensschwere: Schlüsselwörter -> Kategorien des Datensatzes ---
    # Die Kategorien entsprechen der Spalte `incident_severity`:
    # Trivial Damage < Minor Damage < Major Damage < Total Loss

    if any(w in text_klein for w in ["totalschaden", "total zerstört", "abgeschleppt"]):
        schwere = "Total Loss"
    elif any(w in text_klein for w in ["erheblich", "schwer beschädigt", "großer Schaden"]):
        schwere = "Major Damage"
    elif any(w in text_klein for w in ["kratzer", "bagatelle", "geringfügig"]):
        schwere = "Trivial Damage"
    else:
        schwere = "Minor Damage"  # Standardannahme, falls nichts erkannt wird

    # --- Anzahl beteiligter Fahrzeuge: Zahl vor dem Wort "Fahrzeug..." ---       
    m = re.search(r"(\d+)\s*fahrzeug", text_klein)
    fahrzeuge = int(m.group(1)) if m else 1 # Standard: 1 Fahrzeug, falls nichts gefunden

    # --- Verletzte Personen: Zahl vor "verletzt" ---
    m = re.search(r"(\d+)\s*(personen?\s*)?verletzt", text_klein)
    verletzte = int(m.group(1)) if m else 0 # Standard: 0 verletzte, falls nichts gefunden

    # --- Zeugen: Zahl vor "zeuge/zeugen" ---
    m = re.search(r"(\d+)\s*zeug", text_klein)
    zeugen = int(m.group(1)) if m else 0 # Standard: 0 Zeugen, falls nichts gefunden

    # --- Polizei informiert? (ja, falls das Wort vorkommt) ---
    polizei = "polizei" in text_klein

    return {
        "Schadensschwere": schwere,
        "Beteiligte Fahrzeuge": fahrzeuge,
        "Verletzte Personen": verletzte,
        "Zeugen": zeugen,
        "Polizei informiert": "Ja" if polizei else "Nein",
    }

# ---------------------------------------------------------------------
# 4) Prognosefunktion — VORLÄUFIGER PLATZHALTER!
#    TODO: Sobald das Regressionsmodell (src/models/regression.py)
#    trainiert ist, wird NUR diese Funktion ersetzt:
#    Modell laden (z. B. mit joblib) und model.predict(...) aufrufen.
#    Die Oberfläche bleibt unverändert — das ist der Vorteil der Trennung
#    von Frontend und Modell.
# ---------------------------------------------------------------------

def vorhersage_schadenhöhe(merkmale: dict) -> float:
    """Schätzt die Schadenhöhe (USD) anhand einfacher Regeln.
    Die Basiswerte orientieren sich an den Mittelwerten je
    Schadensschwere im Datensatz."""

    # Durchschnittliche die Schadenhöhe je Schwere (aus dem Datensatz abgeleitet)
    basis = {
        "Trivial Damage": 5_000,
        "Minor Damage": 35_000,
        "Major Damage": 62_000,
        "Total Loss": 65_000,
    }[merkmale["Schadensschwere"]]

    # Zuschläge: mehr Fahrzeuge / Verletzte erhöhen die erwartete Summe
    basis += (merkmale["Beteiligte Fahrzeuge"] - 1) * 4_000
    basis += merkmale["Verletzte Personen"] * 3_000

    return float(basis)

# ---------------------------------------------------------------------
# 5) Aufbau der Benutzeroberfläche
# ---------------------------------------------------------------------
st.title("🚗 KFZ-Schadenvorhersage")
st.caption(
    "Schadensmeldung eingeben — das System extrahiert die relevanten "
    "Informationen und prognostiziert die erwartete Schadenhöhe."
)

# Mehrzeiliges Textfeld für die Schadensmeldung mit Beispieltext
meldung = st.text_area(
    label="Schadensmeldung (Freitext)",
    height=180,
    placeholder=(
        "Beispiel: Am 10.06.2026 kam es zu einem Unfall mit 2 Fahrzeugen. "
        "Mein Auto wurde erheblich beschädigt, 1 Person wurde verletzt. "
        "Die Polizei wurde informiert, es gibt 2 Zeugen."
    ),
)

# Button: Die Vorhersage wird NUR nach dem Klick berechnet
if st.button("Vorhersage erstellen", type="primary"):
    # --- Eingabevalidierung: leere Eingabe abfangen ---
    if not meldung.strip():
        st.warning("Bitte zuerst eine Schadensmeldung eingeben.")
        st.stop()  # bricht die weitere Ausführung sauer ab

    # --- Schritt 1: Merkmale aus dem Text extrahieren ---
    merkmale = extrahiere_merkmale(meldung)

    # --- Schritt 2: Vorhersage berechnen ---
    prognose = vorhersage_schadenhöhe(merkmale)

    # --- Schritt 3: Ergebnisse anzeigen ---
    st.subheader("Extrahierte Merkmale")
    # Kopie nur für die Anzeige: Schadensschwere ins Deutsche übersetzen,
    # das Original-Dictionary bleibt unverändert (für die Prognose!)
    anzeige = merkmale.copy()
    anzeige["Schadensschwere"] = SCHWERE_DEUTSCH[merkmale["Schadensschwere"]]
    st.table(pd.DataFrame(anzeige.items(), columns=["Merkmal", "Wert"]))

    st.subheader("Vorhersage")
    st.metric(
        label="Erwartete Schadenhöhe",
        value=f"{prognose:,.0f} EUR".replace(",", "."),  # deutsches Zahlenformat
    )

    # --- Schritt 4: Visualisierung ---
    df = load_data()  # Datensatz laden (für Vergleichswerte)
    if df is not None:
        # Vergleich: Vorhersage vs. Durchschnitt je Schadensschwere
        mittelwerte = df.groupby("incident_severity")["total_claim_amount"].mean()

        fig, ax = plt.subplots(figsize=(7, 4))
        # X-Achse: Kategorien ins Deutsche übersetzen (nur Anzeige)
        labels_deutsch = [SCHWERE_DEUTSCH.get(s, s) for s in mittelwerte.index]
        # Balken: historische Durchschnittswerte aus dem Datensatz
        ax.bar(labels_deutsch, mittelwerte.values,
               color="#9aa7b1", label="Durchschnitt (Datensatz)")
        # Rote Linie: unsere aktuelle Prognose zum Vergleich
        ax.axhline(prognose, color="#c0392b", linewidth=2,
                   label=f"Vorhersage: {prognose:,.0f} EUR".replace(",", "."))
        ax.set_ylabel("Schadenhöhe (EUR)")
        ax.set_title("Vorhersage im Vergleich zu historischen Schadensfällen")
        ax.legend()
        plt.xticks(rotation=15)
        st.pyplot(fig)   # Diagramm im Dashboard anzeigen
    else:
        # Fallback, falls die CSV lokal nicht gefunden wurde
        st.info("Datensatz nicht gefunden — Vergleichsdiagramm wird "
                "angezeigt, sobald data/raw/dataset.csv vorhanden ist.")
        
    # Hinweis zur Transparenz (wichtig für die Bewertung!)
    st.caption("Hinweis: Aktuell wird ein regelbasierter Platzhalter verwendet. "
        "Nach Abschluss des Modelltrainings wird hier das ML-Regressionsmodell "
        "eingesetzt."
    )