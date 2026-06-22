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