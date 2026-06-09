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

STYLE_OPTIONS = ["umgangssprachlich", "laendlich", "hochdeutsch"]

TEMPLATES_BY_STYLE_COLLISION = {
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
        """,
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
        """,
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
        """,
    ],
}

TEMPLATES_BY_STYLE_PARKED = {
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
        """,
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
        """,
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
        """,
    ],
}

TEMPLATES_BY_STYLE_THEFT = {
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
        """,
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
        """,
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
        """,
    ],
}
