DETAILS_MINOR = [
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
    "Kühlergrill beschädigt",
]

DETAILS_MAJOR = [
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
    "der tragende Karosserierahmen ist deformiert",
]

FRONT_COLLISION_TEMPLATES = [
    "es kam zu einem Frontalzusammenstoß",
    "das Fahrzeug prallte frontal gegen ein Hindernis",
    "die Fahrerin bzw. der Fahrer verlor die Kontrolle und verursachte eine Frontalkollision",
    "das Fahrzeug fuhr in eine Straßenbegrenzung",
    "die Front des Fahrzeugs wurde schwer getroffen",
]

REAR_COLLISION_TEMPLATES = [
    "das Fahrzeug wurde von hinten angefahren",
    "es kam zu einem Auffahrunfall im Stau",
    "der Aufprall traf das Fahrzeugheck",
    "das Auto wurde durch einen Heckaufprall beschädigt",
    "die Kollision ereignete sich während des Haltens an einer Ampel",
]

SIDE_COLLISION_TEMPLATES = [
    "an einer Kreuzung kam es zu einer Seitenkollision",
    "ein anderes Fahrzeug traf die Seitenpartie",
    "die Kollision passierte beim Spurwechsel",
    "das Fahrzeug erlitt einen starken Seitenaufprall",
    "beim Einfahren aus einer Nebenstraße kam es zum Zusammenstoß",
]

PARKED_CAR_TEMPLATES = [
    "das Fahrzeug war ordnungsgemaess geparkt und wurde im Stand beschaedigt",
    "waehrend der Parkzeit wurde das Fahrzeug durch ein anderes Auto touchiert",
    "am abgestellten Fahrzeug wurden nach der Rueckkehr Beschaedigungen festgestellt",
    "das geparkte Fahrzeug wurde bei einem Rangiermanoever getroffen",
]

VEHICLE_THEFT_TEMPLATES = [
    "das Fahrzeug wurde als entwendet gemeldet",
    "es wurde ein Diebstahlfall ohne direkte Kollision registriert",
    "das Fahrzeug war am Abstellort nicht mehr auffindbar",
    "im Rahmen der Erstmeldung wurde ein Fahrzeugdiebstahl angezeigt",
]

WEATHER_CONDITIONS = [
    "bei klarem Wetter",
    "waehrend Regens",
    "bei Schneefall",
    "bei Nebel",
    "auf nasser Fahrbahn",
    "bei starkem Wind",
    "bei eingeschraenkter Sicht",
]

TIME_CONDITIONS = [
    "am Morgen",
    "am Tag",
    "am Abend",
    "in der Nacht",
    "zur Hauptverkehrszeit",
]

SPEED_DESCRIPTIONS = [
    "mit etwa 30 km/h",
    "mit etwa 50 km/h",
    "mit etwa 70 km/h",
    "bei langsamer Fahrt",
    "bei dichtem Verkehr",
]

WITNESS_INFO = [
    "",
    "Am Unfallort waren Zeug:innen anwesend.",
    "Der Unfallhergang wurde durch Augenzeug:innen bestaetigt.",
    "Zeug:innen des Vorfalls wurden befragt.",
]

POLICE_INFO = [
    "",
    "Die Polizei wurde zum Unfallort gerufen.",
    "Der Unfall wurde durch die Polizei aufgenommen.",
    "Der Vorfall ist bei der Verkehrspolizei registriert.",
]

TOW_INFO = [
    "",
    "Das Fahrzeug wurde abgeschleppt.",
    "Ein Abschleppdienst war erforderlich.",
    "Das Fahrzeug wurde per Abschleppwagen auf einen Stellplatz gebracht.",
]

THEFT_DETAILS = [
    "Fahrzeugschluessel und Dokumente werden geprueft",
    "der Abstellort wurde fuer die Ermittlungen dokumentiert",
    "eine Fahndungsmeldung wurde eingeleitet",
    "es liegen derzeit keine kollisionsbedingten Schaeden vor",
]

THEFT_AIRBAGS = [
    "Eine Airbag-Ausloesung ist nicht relevant.",
    "Es wurde keine Aktivierung passiver Sicherheitssysteme festgestellt.",
    "Kollisionsbezogene Airbag-Daten liegen nicht vor.",
]

PARKED_AIRBAGS = [
    "Die Airbags wurden nicht aktiviert.",
    "Eine Airbag-Ausloesung wurde nicht festgestellt.",
    "Bei dem Parkschaden kam es zu keiner Aktivierung der Sicherheitssysteme.",
]

MAJOR_AIRBAGS = [
    "Die Airbags haben ausgeloest.",
    "Die Frontairbags wurden aktiviert.",
    "Die passiven Sicherheitssysteme wurden aktiviert.",
]

MINOR_AIRBAGS = [
    "Die Airbags haben nicht ausgeloest.",
    "Die Sicherheitssysteme wurden nicht aktiviert.",
    "Es wurde keine Airbag-Ausloesung festgestellt.",
]

DETAIL_LEVELS = ["minimal", "normal", "detailed", "noisy"]
DETAIL_LEVEL_WEIGHTS = [0.15, 0.35, 0.35, 0.15]

# ── Collision templates ───────────────────────────────────────────────────────

TEMPLATES_COLLISION_MINIMAL = [
    "{incident_phrase}. Schaden: {damage}.",
    "Unfall gemeldet. {incident_phrase}. Beschaedigungen: {damage}.",
    "Schadensmeldung: {incident_phrase}. Betroffene Teile: {damage}.",
    "Es wurde ein Unfall registriert. Schadensbild: {damage}.",
    "Folgende Schaeden wurden nach dem Unfall festgestellt: {damage}.",
]

TEMPLATES_COLLISION_NORMAL = [
    "Mein {make} {model} ({year}) war in einen Unfall verwickelt. {incident_phrase}. Schaeden: {damage}. {airbags}",
    "Mit dem {make} {model} ({year}) gab es einen Unfall: {incident_phrase}. Beschaedigungen: {damage}.",
    "Der {make} {model} ({year}) wurde beschaedigt. {incident_phrase}. Schadensbild: {damage}. {airbags}",
    "Fuer den {make} {model} ({year}) liegt eine Unfallanzeige vor. {incident_phrase}. Festgestellt: {damage}. {airbags}",
    "Beim {make} {model} ({year}) ereignete sich ein Unfall: {incident_phrase}. Sichtbare Schaeden: {damage}.",
]

TEMPLATES_COLLISION_DETAILED = [
    (
        "{time_of_day_cap} {weather} war mein {make} {model} ({year}) in einen Unfall verwickelt. "
        "{incident_phrase} {speed}. Schaeden: {damage}. {airbags} {witness} {police} {tow}"
    ),
    (
        "Das Fahrzeug {make} {model}, Baujahr {year}, war in einen Unfall verwickelt. "
        "{time_of_day} {weather} ereignete sich: {incident_phrase} {speed}. "
        "Dokumentierte Schaeden: {damage}. {airbags} {police} {tow}"
    ),
    (
        "{time_of_day_cap} {weather} wurde das Fahrzeug {make} {model} ({year}) in einen Unfall "
        "verwickelt: {incident_phrase}. Schaeden: {damage}. {airbags} {witness} {tow}"
    ),
    (
        "Fuer das Fahrzeug {make} {model} ({year}) wurde ein Unfallhergang dokumentiert. "
        "{time_of_day_cap} {weather} kam es zu: {incident_phrase} {speed}. "
        "Schaeden: {damage}. {airbags} {police} {tow}"
    ),
    (
        "Im vorliegenden Fall betrifft der Schaden den {make} {model} ({year}). "
        "Das Ereignis trat {time_of_day} {weather} ein, dabei kam es zu: {incident_phrase}. "
        "Schaeden: {damage}. {airbags} {witness} {police} {tow}"
    ),
]

TEMPLATES_COLLISION_NOISY = [
    # without vehicle info
    "Ich war {time_of_day} {weather} unterwegs, als {incident_phrase}. {damage}. {airbags} {tow}",
    "War gerade {weather} unterwegs, dann {incident_phrase}. {damage} kaputt. {airbags}",
    "{time_of_day_cap} ist was passiert: {incident_phrase}. {damage} betroffen. {airbags} {police}",
    "Ich hab keinen genauen Ablauf, aber {incident_phrase}. Sieht nicht gut aus: {damage}. {airbags}",
    "Alles ging so schnell – {incident_phrase} {weather}. {damage} beschaedigt. {airbags} {tow}",
    "Ich glaube {incident_phrase}. {damage} auf jeden Fall kaputt. {airbags}",
    # with vehicle info
    "Mein {make} {model} hatte heute einen Crash: {incident_phrase}. {damage}. {airbags}",
    "War mit dem {make} {model} unterwegs als {incident_phrase}. {damage} kaputt. {airbags}",
    "{time_of_day_cap} ist mir mit dem {make} {model} was passiert {weather}. {damage} beschaedigt. {airbags}",
]

# ── Parked car templates ──────────────────────────────────────────────────────

TEMPLATES_PARKED_MINIMAL = [
    "Das geparkte Fahrzeug wurde beschaedigt. Schaden: {damage}.",
    "Am geparkten Auto wurden Schaeden festgestellt: {damage}.",
    "{incident_phrase}. Schaden: {damage}.",
    "Parkschaden gemeldet. Betroffene Teile: {damage}.",
    "Schaeden am stehenden Fahrzeug: {damage}.",
]

TEMPLATES_PARKED_NORMAL = [
    "Mein {make} {model} ({year}) stand geparkt und wurde beschaedigt. {incident_phrase}. Schaden: {damage}. {airbags}",
    "Am geparkten {make} {model} ({year}) wurden Schaeden gemeldet. {incident_phrase}. Sichtbare Schaeden: {damage}.",
    "Der {make} {model} ({year}) war abgestellt und wurde getroffen. {incident_phrase}. Schadensbild: {damage}. {airbags}",
    "Fuer den {make} {model} ({year}) wurde ein Parkschadenfall erfasst. {incident_phrase}. Schaeden: {damage}. {airbags}",
]

TEMPLATES_PARKED_DETAILED = [
    (
        "{time_of_day_cap} {weather} stand mein {make} {model} ({year}) geparkt und wurde beschaedigt. "
        "{incident_phrase}. Schaden: {damage}. {airbags} {witness} {police} {tow}"
    ),
    (
        "Das Fahrzeug {make} {model}, Baujahr {year}, war zum Ereigniszeitpunkt geparkt. "
        "{time_of_day} {weather} wurde folgender Hergang dokumentiert: {incident_phrase}. "
        "Schaeden: {damage}. {airbags} {witness} {police} {tow}"
    ),
    (
        "{time_of_day_cap} {weather} wurde der abgestellte {make} {model} ({year}) beschaedigt. "
        "{incident_phrase}. Festgestellt: {damage}. {airbags} {police} {tow}"
    ),
    (
        "Im vorliegenden Parkschadenfall betrifft die Meldung den {make} {model} ({year}). "
        "Das Ereignis trat {time_of_day} {weather} ein; dabei gilt: {incident_phrase}. "
        "Dokumentierte Schaeden: {damage}. {airbags} {police}"
    ),
]

TEMPLATES_PARKED_NOISY = [
    # without vehicle info
    "Ich bin zurueckgekommen und das Auto hatte einen Schaden. {incident_phrase}. {damage}. {airbags}",
    "Das Auto stand einfach da und jetzt ist {damage} beschaedigt. {airbags} {police}",
    "{time_of_day_cap} war das Auto noch okay, jetzt ist {damage} kaputt. {incident_phrase}.",
    "Jemand hat mein Auto gerammt. {damage} auf jeden Fall betroffen. {airbags}",
    "War weg und beim Zurueckkommen: {incident_phrase}. {damage}. {airbags} {police}",
    # with vehicle info
    "Parkschaden an meinem {make} {model}. {incident_phrase}. Schaden: {damage}. {airbags} {tow}",
    "Mein {make} {model} ({year}) hat jetzt nen Parkschaden. {damage}. {airbags}",
]

# ── Vehicle theft templates ───────────────────────────────────────────────────

TEMPLATES_THEFT_MINIMAL = [
    "Das Fahrzeug wurde entwendet. {damage}.",
    "Es liegt ein Diebstahlfall vor. Angaben: {damage}.",
    "{incident_phrase}. Stand: {damage}.",
    "Fahrzeugdiebstahl gemeldet. Aktuell: {damage}.",
    "Das Auto ist verschwunden. {damage}.",
]

TEMPLATES_THEFT_NORMAL = [
    "Mein {make} {model} ({year}) wurde gestohlen. {incident_phrase}. Stand: {damage}. {airbags}",
    "Der {make} {model} ({year}) ist entwendet worden. {incident_phrase}. Angaben: {damage}.",
    "Fuer den {make} {model} ({year}) wurde ein Diebstahlfall erfasst. {incident_phrase}. Aktuell: {damage}. {airbags}",
    "Diebstahl {make} {model} ({year}). {incident_phrase}. Dokumentierter Stand: {damage}.",
]

TEMPLATES_THEFT_DETAILED = [
    (
        "{time_of_day_cap} {weather} wurde mein {make} {model} ({year}) entwendet. "
        "{incident_phrase}. Aktueller Stand: {damage}. {airbags} {police}"
    ),
    (
        "Das Fahrzeug {make} {model}, Baujahr {year}, ist Gegenstand eines Diebstahlfalls. "
        "{time_of_day} {weather} wurde folgender Sachverhalt dokumentiert: {incident_phrase}. "
        "Stand: {damage}. {airbags} {police}"
    ),
    (
        "Fuer das Fahrzeug {make} {model} ({year}) wurde ein Entwendungsereignis erfasst. "
        "{time_of_day_cap} {weather} gilt: {incident_phrase}. "
        "Dokumentierte Angaben: {damage}. {airbags} {witness} {police}"
    ),
    (
        "Im vorliegenden Diebstahlfall ist das Fahrzeug {make} {model} ({year}) betroffen. "
        "Das Ereignis trat {time_of_day} {weather} ein; dabei wurde festgehalten: {incident_phrase}. "
        "Dokumentierte Angaben: {damage}. {airbags} {police}"
    ),
]

TEMPLATES_THEFT_NOISY = [
    # without vehicle info
    "Das Auto ist weg. {incident_phrase}. {damage}. {airbags} {police}",
    "Jemand hat mein Auto geklaut. {incident_phrase}. {damage}. {police}",
    "Diebstahl gemeldet. {incident_phrase}. {damage}. {airbags} {police}",
    # with vehicle info
    "Mein {make} {model} ist gestohlen worden. {incident_phrase}. {damage}. {police}",
    "{time_of_day_cap} war {make} {model} noch da, jetzt nicht mehr. {incident_phrase}. {damage}. {police}",
    "Ich hab meinen {make} {model} ({year}) nicht mehr gefunden. {incident_phrase}. {damage}.",
]

# ── Legacy placeholder (kept for backward compat) ─────────────────────────────
TEMPLATES_BY_STYLE_COLLISION = {
    "umgangssprachlich": [
        """
        Der {make} {model} ({year}) hatte einen Crash. {time_of_day_cap} {weather}
        kam es dazu: {incident_phrase} {speed}. Ergebnis: {damage}.
        {airbags} {witness} {police} {tow}
        """,
        """
        Laut Fahrerin bzw. Fahrer vom {make} {model} ist der Unfall so passiert:
        {time_of_day} {weather}, dann {incident_phrase}. Am Auto sieht man:
        {damage}. {airbags} {police}
        """,
        """
        Mit dem {make} {model} ({year}) lief es heute leider schief.
        {time_of_day_cap} {weather} kam es zum Knall: {incident_phrase}.
        Danach wurden diese Schaeden festgestellt: {damage}. {airbags} {witness} {tow}
        """,
        """
        Kurze Zusammenfassung zum {make} {model}: {time_of_day} {weather},
        dann {incident_phrase} {speed}. Sichtbar beschaedigt sind: {damage}.
        {airbags} {police} {tow}
        """,
    ],
    "laendlich": [
        """
        Der {make} {model} mit Baujahr {year} geriet auf der Strecke in einen Unfall.
        {time_of_day_cap}, {weather}, und dann {incident_phrase} {speed}.
        Festgestellt wurden: {damage}. {airbags} {witness} {tow}
        """,
        """
        Beim Fahren ueber Land mit dem {make} {model} passierte der Schadenfall.
        {time_of_day} {weather} kam es zum Vorfall: {incident_phrase}.
        Nach der Besichtigung wurden folgende Schaeden notiert: {damage}.
        {airbags} {witness} {police} {tow}
        """,
        """
        Auf der Landstrasse war der {make} {model} ({year}) unterwegs.
        {time_of_day_cap} {weather} geschah dann Folgendes: {incident_phrase}.
        Dabei entstanden diese Schaeden: {damage}. {airbags} {witness} {police}
        """,
        """
        Im laendlichen Bereich kam es mit dem {make} {model} zu einem Unfallereignis.
        {time_of_day} {weather} trat ein {incident_phrase} auf, {speed}.
        Bei der Nachschau wurden vermerkt: {damage}. {airbags} {tow}
        """,
    ],
    "hochdeutsch": [
        """
        Das Fahrzeug {make} {model}, Baujahr {year}, war in einen Verkehrsunfall verwickelt.
        {time_of_day_cap} {weather} ereignete sich folgender Hergang: {incident_phrase} {speed}.
        Im Rahmen der Begutachtung wurden folgende Schaeden festgestellt: {damage}.
        {airbags} {witness} {police} {tow}
        """,
        """
        Fuer das Fahrzeug {make} {model} wurde ein Versicherungsfall erfasst.
        Der Unfall ereignete sich {time_of_day} {weather}. Dabei gilt: {incident_phrase}.
        Dokumentierte Beschaedigungen: {damage}. {airbags} {police} {tow}
        """,
        """
        Im vorliegenden Fall betrifft der Schaden das Fahrzeug {make} {model} ({year}).
        Das Ereignis trat {time_of_day} {weather} ein; hierbei kam es dazu, dass {incident_phrase}.
        Die technische Erfassung dokumentiert folgende Schaeden: {damage}.
        {airbags} {witness} {police}
        """,
        """
        Die Erstaufnahme zum Fahrzeug {make} {model} beschreibt einen Unfallhergang,
        bei dem {time_of_day} {weather} ein Szenario mit {incident_phrase} vorlag.
        In der Schadensbewertung wurden folgende Positionen festgehalten: {damage}.
        {airbags} {witness} {tow}
        """,
    ],
}
TEMPLATES_BY_STYLE_PARKED = {
    "umgangssprachlich": [
        """
        Der {make} {model} ({year}) stand geparkt. {time_of_day_cap} {weather}
        passierte Folgendes: {incident_phrase}. Festgestellt wurde: {damage}.
        {airbags} {witness} {police} {tow}
        """,
        """
        Beim geparkten {make} {model} fiel nach der Rueckkehr ein Schaden auf.
        {time_of_day} {weather} zeigte sich: {incident_phrase}. Sichtbar sind {damage}.
        {airbags} {police}
        """,
    ],
    "laendlich": [
        """
        Der geparkte {make} {model} ({year}) wurde im Stand beschaedigt.
        {time_of_day_cap} {weather} ergab sich folgender Sachverhalt: {incident_phrase}.
        Vermerkt wurden: {damage}. {airbags} {witness} {tow}
        """,
        """
        Beim abgestellten {make} {model} trat ein Parkschadenfall ein.
        {time_of_day} {weather} wurde gemeldet: {incident_phrase}. Schaeden: {damage}.
        {airbags} {police} {tow}
        """,
    ],
    "hochdeutsch": [
        """
        Das Fahrzeug {make} {model}, Baujahr {year}, war zum Ereigniszeitpunkt geparkt.
        {time_of_day_cap} {weather} wurde folgender Hergang dokumentiert: {incident_phrase}.
        Bei der Aufnahme wurden folgende Beschaedigungen festgestellt: {damage}.
        {airbags} {witness} {police} {tow}
        """,
        """
        Im vorliegenden Parkschadenfall betrifft die Meldung den {make} {model}.
        Das Ereignis trat {time_of_day} {weather} ein; dabei gilt: {incident_phrase}.
        Dokumentierte Positionen: {damage}. {airbags} {police}
        """,
    ],
}

TEMPLATES_BY_STYLE_THEFT = {
    "umgangssprachlich": [
        """
        Beim {make} {model} ({year}) gibt es einen Diebstahlfall.
        {time_of_day_cap} {weather} wurde gemeldet: {incident_phrase}.
        Aktueller Stand: {damage}. {airbags} {witness} {police}
        """,
        """
        Der {make} {model} war ploetzlich weg. {time_of_day} {weather}
        wurde der Fall gemeldet: {incident_phrase}. Notiert ist: {damage}.
        {airbags} {police}
        """,
    ],
    "laendlich": [
        """
        Fuer den {make} {model} ({year}) wurde ein Diebstahlvorfall aufgenommen.
        {time_of_day_cap} {weather} ergab die Meldung: {incident_phrase}.
        Vermerkt wurde folgender Sachstand: {damage}. {airbags} {witness} {police}
        """,
        """
        Beim abgestellten {make} {model} trat ein Entwendungsfall auf.
        {time_of_day} {weather} wurde festgestellt: {incident_phrase}.
        Die Akte fuehrt derzeit: {damage}. {airbags} {tow}
        """,
    ],
    "hochdeutsch": [
        """
        Das Fahrzeug {make} {model}, Baujahr {year}, ist Gegenstand eines Diebstahlfalls.
        {time_of_day_cap} {weather} wurde folgender Sachverhalt dokumentiert: {incident_phrase}.
        Der aktuelle Bearbeitungsstand lautet: {damage}. {airbags} {police}
        """,
        """
        Fuer das Fahrzeug {make} {model} wurde ein Entwendungsereignis erfasst.
        Das Ereignis trat {time_of_day} {weather} ein; hierzu wurde festgehalten: {incident_phrase}.
        Dokumentierte Angaben: {damage}. {airbags} {witness} {police}
        """,
    ],
}


# ── Component-based NER training generation ───────────────────────────────────

VEHICLE_REF_WITH_YEAR = [
    "{make} {model} ({year})",
    "{make} {model}, Baujahr {year}",
    "{year}er {make} {model}",
    "{make} {model} aus {year}",
    "{model} von {make}, Baujahr {year}",
    "{make} {model} mit Baujahr {year}",
]

VEHICLE_REF_WITHOUT_YEAR = [
    "{make} {model}",
    "{make} Modell {model}",
    "{model} von {make}",
    "Fahrzeug {model} ({make})",
]

VEHICLE_INTRO_TEMPLATES = [
    "Betroffen: {vehicle_ref}.",
    "Fahrzeug: {vehicle_ref}.",
    "Das betroffene Fahrzeug ist {vehicle_ref}.",
    "Es handelt sich um {vehicle_ref}.",
    "Das Fahrzeug: {vehicle_ref}.",
    "Gemeldet wurde {vehicle_ref}.",
    "Das Fahrzeug wurde als {vehicle_ref} identifiziert.",
    "Meldung fuer {vehicle_ref}.",
]

VEHICLE_IN_EVENT_TEMPLATES = [
    "Es kam zu einem Unfall mit {vehicle_ref}: {incident_phrase}.",
    "{vehicle_ref} war in folgenden Unfall verwickelt: {incident_phrase}.",
    "Unfall mit {vehicle_ref} – {incident_phrase}.",
    "{incident_phrase} – Fahrzeug: {vehicle_ref}.",
    "Schadensmeldung fuer {vehicle_ref}: {incident_phrase}.",
    "Beim Fahrzeug {vehicle_ref}: {incident_phrase}.",
]

EVENT_STANDALONE_TEMPLATES = [
    "{incident_phrase}.",
    "Zum Vorfall: {incident_phrase}.",
    "Gemeldet wurde: {incident_phrase}.",
    "Vorfall: {incident_phrase}.",
    "Das Ereignis: {incident_phrase}.",
    "Hergang: {incident_phrase}.",
    "Schadensereignis: {incident_phrase}.",
    "Unfallhergang: {incident_phrase}.",
]

DAMAGE_INTRO_TEMPLATES = [
    "Schaden: {damage}.",
    "Festgestellte Schaeden: {damage}.",
    "Beschaedigungen: {damage}.",
    "Schadensbild: {damage}.",
    "Karosserieschaden: {damage}.",
    "Am Fahrzeug wurden folgende Schaeden festgestellt: {damage}.",
    "Es wurden folgende Schaeden notiert: {damage}.",
    "Betroffen sind: {damage}.",
    "Die Schaeden umfassen: {damage}.",
    "Sichtbare Schaeden: {damage}.",
    "Schadensbewertung: {damage}.",
]

YEAR_STANDALONE_TEMPLATES = [
    "Das Fahrzeug ist Baujahr {year}.",
    "Baujahr: {year}.",
    "Es handelt sich um ein Fahrzeug aus dem Jahr {year}.",
    "Das Modell stammt aus {year}.",
    "Erstzulassung: {year}.",
]

# ─────────────────────────────────────────────────────────────────────────────

MAJOR_SEVERITIES = {"Major Damage", "Total Loss"}

CASE_CONFIG = {
    "Front Collision": {
        "incident_phrases": FRONT_COLLISION_TEMPLATES,
        "damage": {
            "major": {"pool": DETAILS_MAJOR + DETAILS_MINOR, "count": (3, 5)},
            "minor": {"pool": DETAILS_MINOR, "count": (2, 4)},
        },
        "airbags": {
            "major": MAJOR_AIRBAGS,
            "minor": MINOR_AIRBAGS,
        },
        "templates": {
            "minimal": TEMPLATES_COLLISION_MINIMAL,
            "normal": TEMPLATES_COLLISION_NORMAL,
            "detailed": TEMPLATES_COLLISION_DETAILED,
            "noisy": TEMPLATES_COLLISION_NOISY,
        },
    },
    "Rear Collision": {
        "incident_phrases": REAR_COLLISION_TEMPLATES,
        "damage": {
            "major": {"pool": DETAILS_MAJOR + DETAILS_MINOR, "count": (3, 5)},
            "minor": {"pool": DETAILS_MINOR, "count": (2, 4)},
        },
        "airbags": {
            "major": MAJOR_AIRBAGS,
            "minor": MINOR_AIRBAGS,
        },
        "templates": {
            "minimal": TEMPLATES_COLLISION_MINIMAL,
            "normal": TEMPLATES_COLLISION_NORMAL,
            "detailed": TEMPLATES_COLLISION_DETAILED,
            "noisy": TEMPLATES_COLLISION_NOISY,
        },
    },
    "Side Collision": {
        "incident_phrases": SIDE_COLLISION_TEMPLATES,
        "damage": {
            "major": {"pool": DETAILS_MAJOR + DETAILS_MINOR, "count": (3, 5)},
            "minor": {"pool": DETAILS_MINOR, "count": (2, 4)},
        },
        "airbags": {
            "major": MAJOR_AIRBAGS,
            "minor": MINOR_AIRBAGS,
        },
        "templates": {
            "minimal": TEMPLATES_COLLISION_MINIMAL,
            "normal": TEMPLATES_COLLISION_NORMAL,
            "detailed": TEMPLATES_COLLISION_DETAILED,
            "noisy": TEMPLATES_COLLISION_NOISY,
        },
    },
    "Parked Car": {
        "incident_phrases": PARKED_CAR_TEMPLATES,
        "damage": {
            "major": {"pool": DETAILS_MINOR, "count": (2, 4)},
            "minor": {"pool": DETAILS_MINOR, "count": (2, 4)},
        },
        "airbags": {
            "major": PARKED_AIRBAGS,
            "minor": PARKED_AIRBAGS,
        },
        "templates": {
            "minimal": TEMPLATES_PARKED_MINIMAL,
            "normal": TEMPLATES_PARKED_NORMAL,
            "detailed": TEMPLATES_PARKED_DETAILED,
            "noisy": TEMPLATES_PARKED_NOISY,
        },
    },
    "Vehicle Theft": {
        "incident_phrases": VEHICLE_THEFT_TEMPLATES,
        "damage": {
            "major": {"pool": THEFT_DETAILS, "count": (2, 3)},
            "minor": {"pool": THEFT_DETAILS, "count": (2, 3)},
        },
        "airbags": {
            "major": THEFT_AIRBAGS,
            "minor": THEFT_AIRBAGS,
        },
        "templates": {
            "minimal": TEMPLATES_THEFT_MINIMAL,
            "normal": TEMPLATES_THEFT_NORMAL,
            "detailed": TEMPLATES_THEFT_DETAILED,
            "noisy": TEMPLATES_THEFT_NOISY,
        },
    },
}