import csv
from pathlib import Path

import random


VALID_CASE_TYPES = {
    "Front Collision",
    "Parked Car",
    "Rear Collision",
    "Side Collision",
    "Vehicle Theft",
}


def resolve_case_type(row: dict) -> str:
    incident_type = str(row.get("incident_type", "")).strip()
    collision_type = str(row.get("collision_type", "")).strip()

    for value in (collision_type, incident_type):
        if value in VALID_CASE_TYPES:
            return value

    return "Side Collision"

def generate_accident_description(row):

    vehicle = str(row.get('auto_make_model', '')).strip()
    if vehicle:
        make, _, model = vehicle.partition(' ')
        model = model.strip()
    else:
        make = str(row.get('auto_make', '')).strip()
        model = str(row.get('auto_model', '')).strip()
    year = row['auto_year']
    severity = row['incident_severity']
    case_type = resolve_case_type(row)

    # Schadensdetails

    details_minor = [
        "vorderer Stoßfänger gebrochen",
        "hinterer Stoßfänger beschädigt",
        "linker Kotflügel eingedellt",
        "rechter Kotflügel eingedellt",
        "Kratzer an der Fahrertür",
        "Kratzer an der Beifahrertür",
        "Außenspiegel beschädigt",
        "linker Scheinwerfer zerbrochen",
        "rechter Scheinwerfer zerbrochen",
        "Riss im Stoßfänger",
        "Motorhaube verformt",
        "Kofferraum beschädigt",
        "Nebelscheinwerfer gebrochen",
        "kleine Delle an der Karosserie",
        "Kühlergrill beschädigt"
    ]

    details_major = [
        "die gesamte Frontpartie ist stark deformiert",
        "der Motorraum ist schwer beschädigt",
        "die Windschutzscheibe ist zerborsten",
        "die Karosseriegeometrie ist verzogen",
        "das Fahrwerk ist beschädigt",
        "Türen sind blockiert",
        "der Kühler ist beschädigt",
        "Teile der A-Säule sind zerstört",
        "schwere Schäden am Unterbau",
        "das Fahrzeugdach ist beschädigt",
        "das Fahrzeug ist nicht mehr fahrbereit",
        "der tragende Karosserierahmen ist deformiert"
    ]

    # Unfallart

    front_collision_templates = [
        "es kam zu einem Frontalzusammenstoß",
        "das Fahrzeug prallte frontal gegen ein Hindernis",
        "die Fahrerin bzw. der Fahrer verlor die Kontrolle und verursachte eine Frontalkollision",
        "das Fahrzeug fuhr in eine Straßenbegrenzung",
        "die Front des Fahrzeugs wurde schwer getroffen"
    ]

    rear_collision_templates = [
        "das Fahrzeug wurde von hinten angefahren",
        "es kam zu einem Auffahrunfall im Stau",
        "der Aufprall traf das Fahrzeugheck",
        "das Auto wurde durch einen Heckaufprall beschädigt",
        "die Kollision ereignete sich während des Haltens an einer Ampel"
    ]

    side_collision_templates = [
        "an einer Kreuzung kam es zu einer Seitenkollision",
        "ein anderes Fahrzeug traf die Seitenpartie",
        "die Kollision passierte beim Spurwechsel",
        "das Fahrzeug erlitt einen starken Seitenaufprall",
        "beim Einfahren aus einer Nebenstraße kam es zum Zusammenstoß"
    ]

    parked_car_templates = [
        "das Fahrzeug war ordnungsgemaess geparkt und wurde im Stand beschaedigt",
        "waehrend der Parkzeit wurde das Fahrzeug durch ein anderes Auto touchiert",
        "am abgestellten Fahrzeug wurden nach der Rueckkehr Beschaedigungen festgestellt",
        "das geparkte Fahrzeug wurde bei einem Rangiermanoever getroffen"
    ]

    vehicle_theft_templates = [
        "das Fahrzeug wurde als entwendet gemeldet",
        "es wurde ein Diebstahlfall ohne direkte Kollision registriert",
        "das Fahrzeug war am Abstellort nicht mehr auffindbar",
        "im Rahmen der Erstmeldung wurde ein Fahrzeugdiebstahl angezeigt"
    ]

    # Zusaetzliche Umstaende

    weather_conditions = [
        "bei klarem Wetter",
        "waehrend Regens",
        "bei Schneefall",
        "bei Nebel",
        "auf nasser Fahrbahn",
        "bei starkem Wind",
        "bei eingeschraenkter Sicht"
    ]

    time_conditions = [
        "am Morgen",
        "am Tag",
        "am Abend",
        "in der Nacht",
        "zur Hauptverkehrszeit"
    ]

    speed_descriptions = [
        "mit etwa 30 km/h",
        "mit etwa 50 km/h",
        "mit etwa 70 km/h",
        "bei langsamer Fahrt",
        "bei dichtem Verkehr"
    ]

    witness_info = [
        "",
        "Am Unfallort waren Zeug:innen anwesend.",
        "Der Unfallhergang wurde durch Augenzeug:innen bestaetigt.",
        "Zeug:innen des Vorfalls wurden befragt."
    ]

    police_info = [
        "",
        "Die Polizei wurde zum Unfallort gerufen.",
        "Der Unfall wurde durch die Polizei aufgenommen.",
        "Der Vorfall ist bei der Verkehrspolizei registriert."
    ]

    tow_info = [
        "",
        "Das Fahrzeug wurde abgeschleppt.",
        "Ein Abschleppdienst war erforderlich.",
        "Das Fahrzeug wurde per Abschleppwagen auf einen Stellplatz gebracht."
    ]

    # Schadenauswahl nach Schweregrad

    if case_type == "Vehicle Theft":

        theft_details = [
            "Fahrzeugschluessel und Dokumente werden geprueft",
            "der Abstellort wurde fuer die Ermittlungen dokumentiert",
            "eine Fahndungsmeldung wurde eingeleitet",
            "es liegen derzeit keine kollisionsbedingten Schaeden vor"
        ]

        damage_count = random.randint(2, 3)
        damage = ", ".join(
            random.sample(
                theft_details,
                min(damage_count, len(theft_details))
            )
        )

        airbags = random.choice([
            "Eine Airbag-Ausloesung ist nicht relevant.",
            "Es wurde keine Aktivierung passiver Sicherheitssysteme festgestellt.",
            "Kollisionsbezogene Airbag-Daten liegen nicht vor."
        ])

    elif case_type == "Parked Car":

        damage_count = random.randint(2, 4)
        damage = ", ".join(
            random.sample(
                details_minor,
                min(damage_count, len(details_minor))
            )
        )

        airbags = random.choice([
            "Die Airbags wurden nicht aktiviert.",
            "Eine Airbag-Ausloesung wurde nicht festgestellt.",
            "Bei dem Parkschaden kam es zu keiner Aktivierung der Sicherheitssysteme."
        ])

    elif severity in ['Major Damage', 'Total Loss']:

        damage_count = random.randint(3, 5)

        damage = ", ".join(
            random.sample(
                details_major + details_minor,
                min(damage_count, len(details_major + details_minor))
            )
        )

        airbags = random.choice([
            "Die Airbags haben ausgeloest.",
            "Die Frontairbags wurden aktiviert.",
            "Die passiven Sicherheitssysteme wurden aktiviert."
        ])

    else:

        damage_count = random.randint(2, 4)

        damage = ", ".join(
            random.sample(
                details_minor,
                min(damage_count, len(details_minor))
            )
        )

        airbags = random.choice([
            "Die Airbags haben nicht ausgeloest.",
            "Die Sicherheitssysteme wurden nicht aktiviert.",
            "Es wurde keine Airbag-Ausloesung festgestellt."
        ])

    # Auswahl der Unfallart

    if case_type == "Vehicle Theft":
        accident_type = random.choice(vehicle_theft_templates)

    elif case_type == "Parked Car":
        accident_type = random.choice(parked_car_templates)

    elif case_type == "Front Collision":
        accident_type = random.choice(front_collision_templates)

    elif case_type == "Rear Collision":
        accident_type = random.choice(rear_collision_templates)

    else:
        accident_type = random.choice(side_collision_templates)

    # Zufallsparameter

    weather = random.choice(weather_conditions)
    time_of_day = random.choice(time_conditions)
    time_of_day_cap = time_of_day.capitalize()
    speed = random.choice(speed_descriptions)
    speed_part = speed if case_type not in ["Vehicle Theft", "Parked Car"] else ""

    witness = random.choice(witness_info)
    police = random.choice(police_info)
    tow = random.choice(tow_info)

    # Stilrichtung und Vorlagen
    style = random.choice(["umgangssprachlich", "laendlich", "hochdeutsch"])

    templates_by_style_collision = {
        "umgangssprachlich": [
            """
            Der {make} {model} ({year}) hatte einen Crash. {time_of_day_cap} {weather}
            kam es dazu: {accident_type} {speed}. Ergebnis: {damage}.
            {airbags} {witness} {police} {tow}
            """,
            """
            Laut Fahrerin bzw. Fahrer vom {make} {model} ist der Unfall so passiert:
            {time_of_day} {weather}, dann {accident_type}. Am Auto sieht man:
            {damage}. {airbags} {police}
            """,
            """
            Mit dem {make} {model} ({year}) lief es heute leider schief.
            {time_of_day_cap} {weather} kam es zum Knall: {accident_type}.
            Danach wurden diese Schaeden festgestellt: {damage}. {airbags} {witness} {tow}
            """,
            """
            Kurze Zusammenfassung zum {make} {model}: {time_of_day} {weather},
            dann {accident_type} {speed}. Sichtbar beschaedigt sind: {damage}.
            {airbags} {police} {tow}
            """
        ],
        "laendlich": [
            """
            Der {make} {model} mit Baujahr {year} geriet auf der Strecke in einen Unfall.
            {time_of_day_cap}, {weather}, und dann {accident_type} {speed}.
            Festgestellt wurden: {damage}. {airbags} {witness} {tow}
            """,
            """
            Beim Fahren ueber Land mit dem {make} {model} passierte der Schadenfall.
            {time_of_day} {weather} kam es zum Vorfall: {accident_type}.
            Nach der Besichtigung wurden folgende Schaeden notiert: {damage}.
            {airbags} {witness} {police} {tow}
            """,
            """
            Auf der Landstrasse war der {make} {model} ({year}) unterwegs.
            {time_of_day_cap} {weather} geschah dann Folgendes: {accident_type}.
            Dabei entstanden diese Schaeden: {damage}. {airbags} {witness} {police}
            """,
            """
            Im laendlichen Bereich kam es mit dem {make} {model} zu einem Unfallereignis.
            {time_of_day} {weather} trat ein {accident_type} auf, {speed}.
            Bei der Nachschau wurden vermerkt: {damage}. {airbags} {tow}
            """
        ],
        "hochdeutsch": [
            """
            Das Fahrzeug {make} {model}, Baujahr {year}, war in einen Verkehrsunfall verwickelt.
            {time_of_day_cap} {weather} ereignete sich folgender Hergang: {accident_type} {speed}.
            Im Rahmen der Begutachtung wurden folgende Schaeden festgestellt: {damage}.
            {airbags} {witness} {police} {tow}
            """,
            """
            Fuer das Fahrzeug {make} {model} wurde ein Versicherungsfall erfasst.
            Der Unfall ereignete sich {time_of_day} {weather}. Dabei gilt: {accident_type}.
            Dokumentierte Beschaedigungen: {damage}. {airbags} {police} {tow}
            """,
            """
            Im vorliegenden Fall betrifft der Schaden das Fahrzeug {make} {model} ({year}).
            Das Ereignis trat {time_of_day} {weather} ein; hierbei kam es dazu, dass {accident_type}.
            Die technische Erfassung dokumentiert folgende Schaeden: {damage}.
            {airbags} {witness} {police}
            """,
            """
            Die Erstaufnahme zum Fahrzeug {make} {model} beschreibt einen Unfallhergang,
            bei dem {time_of_day} {weather} ein Szenario mit {accident_type} vorlag.
            In der Schadensbewertung wurden folgende Positionen festgehalten: {damage}.
            {airbags} {witness} {tow}
            """
        ]
    }

    templates_by_style_parked = {
        "umgangssprachlich": [
            """
            Der {make} {model} ({year}) stand geparkt. {time_of_day_cap} {weather}
            passierte Folgendes: {accident_type}. Festgestellt wurde: {damage}.
            {airbags} {witness} {police} {tow}
            """,
            """
            Beim geparkten {make} {model} fiel nach der Rueckkehr ein Schaden auf.
            {time_of_day} {weather} zeigte sich: {accident_type}. Sichtbar sind {damage}.
            {airbags} {police}
            """
        ],
        "laendlich": [
            """
            Der geparkte {make} {model} ({year}) wurde im Stand beschaedigt.
            {time_of_day_cap} {weather} ergab sich folgender Sachverhalt: {accident_type}.
            Vermerkt wurden: {damage}. {airbags} {witness} {tow}
            """,
            """
            Beim abgestellten {make} {model} trat ein Parkschadenfall ein.
            {time_of_day} {weather} wurde gemeldet: {accident_type}. Schaeden: {damage}.
            {airbags} {police} {tow}
            """
        ],
        "hochdeutsch": [
            """
            Das Fahrzeug {make} {model}, Baujahr {year}, war zum Ereigniszeitpunkt geparkt.
            {time_of_day_cap} {weather} wurde folgender Hergang dokumentiert: {accident_type}.
            Bei der Aufnahme wurden folgende Beschaedigungen festgestellt: {damage}.
            {airbags} {witness} {police} {tow}
            """,
            """
            Im vorliegenden Parkschadenfall betrifft die Meldung den {make} {model}.
            Das Ereignis trat {time_of_day} {weather} ein; dabei gilt: {accident_type}.
            Dokumentierte Positionen: {damage}. {airbags} {police}
            """
        ]
    }

    templates_by_style_theft = {
        "umgangssprachlich": [
            """
            Beim {make} {model} ({year}) gibt es einen Diebstahlfall.
            {time_of_day_cap} {weather} wurde gemeldet: {accident_type}.
            Aktueller Stand: {damage}. {airbags} {witness} {police}
            """,
            """
            Der {make} {model} war ploetzlich weg. {time_of_day} {weather}
            wurde der Fall gemeldet: {accident_type}. Notiert ist: {damage}.
            {airbags} {police}
            """
        ],
        "laendlich": [
            """
            Fuer den {make} {model} ({year}) wurde ein Diebstahlvorfall aufgenommen.
            {time_of_day_cap} {weather} ergab die Meldung: {accident_type}.
            Vermerkt wurde folgender Sachstand: {damage}. {airbags} {witness} {police}
            """,
            """
            Beim abgestellten {make} {model} trat ein Entwendungsfall auf.
            {time_of_day} {weather} wurde festgestellt: {accident_type}.
            Die Akte fuehrt derzeit: {damage}. {airbags} {tow}
            """
        ],
        "hochdeutsch": [
            """
            Das Fahrzeug {make} {model}, Baujahr {year}, ist Gegenstand eines Diebstahlfalls.
            {time_of_day_cap} {weather} wurde folgender Sachverhalt dokumentiert: {accident_type}.
            Der aktuelle Bearbeitungsstand lautet: {damage}. {airbags} {police}
            """,
            """
            Fuer das Fahrzeug {make} {model} wurde ein Entwendungsereignis erfasst.
            Das Ereignis trat {time_of_day} {weather} ein; hierzu wurde festgehalten: {accident_type}.
            Dokumentierte Angaben: {damage}. {airbags} {witness} {police}
            """
        ]
    }

    if case_type == "Vehicle Theft":
        templates = templates_by_style_theft[style]
    elif case_type == "Parked Car":
        templates = templates_by_style_parked[style]
    else:
        templates = templates_by_style_collision[style]

    text = random.choice(templates).format(
        make=make,
        model=model,
        year=year,
        accident_type=accident_type,
        damage=damage,
        airbags=airbags,
        weather=weather,
        time_of_day=time_of_day,
        time_of_day_cap=time_of_day_cap,
        speed=speed,
        speed_part=speed_part,
        witness=witness,
        police=police,
        tow=tow
    )

    return " ".join(text.split())


def add_generated_text_column(input_path: Path, output_path: Path, column_name: str) -> int:
    with input_path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if column_name not in fieldnames:
        fieldnames.append(column_name)

    for row in rows:
        row[column_name] = generate_accident_description(row)

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)

