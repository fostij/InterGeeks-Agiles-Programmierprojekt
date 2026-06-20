-- ---------------------------------------------------------------------
-- schema.sql — Datenbankschema für das Projekt "KFZ-Schadenprognose"
-- ---------------------------------------------------------------------
-- Der Original-Datensatz (1 flache CSV-Tabelle, 40 Spalten) wird in
-- 5 normalisierte Tabellen aufgeteilt (3. Normalform):
--
--   kunden ──< policen ──< unfaelle ──< schaeden
--                  │
--                  └──< fahrzeuge
-- ----------------------------------------------------------------------



-- Bestehende Tabellen löschen (Reihenfolge wegen Fremdschlüsseln wichtig!)
-- CASCADE entfernt automatisch abhängige Objekte.
DROP TABLE IF EXISTS schaeden   CASCADE;
DROP TABLE IF EXISTS unfaelle   CASCADE;
DROP TABLE IF EXISTS fahrzeuge  CASCADE;
DROP TABLE IF EXISTS policen    CASCADE;
DROP TABLE IF EXISTS kunden     CASCADE;

-- ---------------------------------------------------------------------
-- Tabelle 1: kunden — Stammdaten der Versicherungsnehmer
-- ---------------------------------------------------------------------
CREATE TABLE kunden (
    kunde_id            SERIAL PRIMARY KEY,        
    alter_jahre         INTEGER NOT NULL,          
    geschlecht          VARCHAR(10),               
    bildungsniveau      VARCHAR(20),               
    beruf               VARCHAR(50),               
    hobbys              VARCHAR(50),              
    familienverhaeltnis VARCHAR(30),              
    plz                 VARCHAR(10),               
    kapitalgewinne      INTEGER,                  
    kapitalverluste     INTEGER,                   
    kunde_seit_monaten  INTEGER                    
);

-- ---------------------------------------------------------------------
-- Tabelle 2: policen — Versicherungsverträge (1 Kunde -> n Policen)
-- ---------------------------------------------------------------------
CREATE TABLE policen (
    police_id           SERIAL PRIMARY KEY,
    kunde_id            INTEGER NOT NULL REFERENCES kunden(kunde_id),  
    policen_nummer      INTEGER UNIQUE NOT NULL,   
    vertragsbeginn      DATE,                      
    bundesstaat         VARCHAR(5),                
    deckungsgrenze      VARCHAR(20),               
    selbstbeteiligung   INTEGER,                   
    jahrespraemie       NUMERIC(10,2),             
    umbrella_limit      BIGINT                     
);

-- ---------------------------------------------------------------------
-- Tabelle 3: fahrzeuge — versicherte Fahrzeuge (1 Police -> n Fahrzeuge)
-- ---------------------------------------------------------------------
CREATE TABLE fahrzeuge (
    fahrzeug_id         SERIAL PRIMARY KEY,
    police_id           INTEGER NOT NULL REFERENCES policen(police_id),
    marke               VARCHAR(30),               
    modell              VARCHAR(30),               
    baujahr             INTEGER                   
);

-- ---------------------------------------------------------------------
-- Tabelle 4: unfaelle — gemeldete Vorfälle (1 Police -> n Unfälle)
-- ---------------------------------------------------------------------
CREATE TABLE unfaelle (
    unfall_id           SERIAL PRIMARY KEY,
    police_id           INTEGER NOT NULL REFERENCES policen(police_id),
    unfall_datum        DATE,                      
    unfall_typ          VARCHAR(40),               
    kollisions_typ      VARCHAR(30),               
    schadensschwere     VARCHAR(20),              
    behoerde            VARCHAR(20),               
    bundesstaat         VARCHAR(5),                
    stadt               VARCHAR(40),              
    adresse             VARCHAR(80),               
    uhrzeit_stunde      INTEGER CHECK (uhrzeit_stunde BETWEEN 0 AND 23),
    anzahl_fahrzeuge    INTEGER,                   
    sachschaden_dritter VARCHAR(5),                
    anzahl_verletzte    INTEGER,                   
    anzahl_zeugen       INTEGER,                   
    polizeibericht      VARCHAR(5)                 
);

-- ---------------------------------------------------------------------
-- Tabelle 5: schaeden — Schadenszahlungen (Zielvariable des Projekts!)
-- ---------------------------------------------------------------------
CREATE TABLE schaeden (
    schaden_id          SERIAL PRIMARY KEY,
    unfall_id           INTEGER NOT NULL REFERENCES unfaelle(unfall_id),
    gesamtschaden       NUMERIC(12,2),             
    personenschaden     NUMERIC(12,2),             
    sachschaden         NUMERIC(12,2),             
    fahrzeugschaden     NUMERIC(12,2),             
    betrug_gemeldet     BOOLEAN                    
);

-- ---------------------------------------------------------------------
-- Tabelle 6: anfragen — eine Zeile pro eingehender Anfrage (jede Quelle:
-- Dashboard, E-Mail, Chatbot, ...)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anfragen (
    anfrage_id       SERIAL PRIMARY KEY,
    erstellt_am      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quelle           VARCHAR(20),       -- 'dashboard' | 'email' | 'chat'
    rohtext          TEXT,
    foto_pfad        VARCHAR(255)       -- NULL falls kein Foto
);

-- ---------------------------------------------------------------------
-- Tabelle 7: multihead_ergebnisse — aus dem Text rekonstruierte Felder
-- mit Konfidenzwert für JEDES einzelne Feld
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS multihead_ergebnisse (
    ergebnis_id              SERIAL PRIMARY KEY,
    anfrage_id                INTEGER NOT NULL REFERENCES anfragen(anfrage_id),
    erstellt_am                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    incident_severity            VARCHAR(20),
    incident_severity_conf       NUMERIC(5,4),

    incident_type                  VARCHAR(40),
    incident_type_conf             NUMERIC(5,4),

    collision_type                   VARCHAR(30),
    collision_type_conf              NUMERIC(5,4),

    number_of_vehicles                 INTEGER,
    number_of_vehicles_conf            NUMERIC(5,4),

    bodily_injuries                      INTEGER,
    bodily_injuries_conf                 NUMERIC(5,4),

    witnesses                              INTEGER,
    witnesses_conf                         NUMERIC(5,4),

    police_report                            VARCHAR(5),
    police_report_conf                       NUMERIC(5,4),

    property_damage                            VARCHAR(5),
    property_damage_conf                       NUMERIC(5,4),

    auto_make                                    VARCHAR(30),
    auto_make_conf                               NUMERIC(5,4),

    auto_year                                      INTEGER,
    auto_year_conf                                 NUMERIC(5,4),

    fehlende_felder                                  TEXT     -- kommagetrennte Liste niedrig-konfidenter Felder
);

-- ---------------------------------------------------------------------
-- Tabelle 8: cnn_ergebnisse — Foto-basierte Schweregrad-Erkennung
-- (unabhängig vom Text; konkurriert per Konfidenz mit dem NLP-Ergebnis)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cnn_ergebnisse (
    ergebnis_id      SERIAL PRIMARY KEY,
    anfrage_id       INTEGER NOT NULL REFERENCES anfragen(anfrage_id),
    erstellt_am      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    severity         VARCHAR(20),       -- englische Klasse (für Regression: Minor/Major/Total Loss)
    severity_de      VARCHAR(30),       -- deutsches CNN-Label (für UI-Anzeige)
    confidence       NUMERIC(5,4)
);

-- ---------------------------------------------------------------------
-- Tabelle 9: regression_ergebnisse — finale Schadenhöhe-Prognose (vehicle_claim)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS regression_ergebnisse (
    ergebnis_id      SERIAL PRIMARY KEY,
    anfrage_id       INTEGER NOT NULL REFERENCES anfragen(anfrage_id),
    erstellt_am      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    vehicle_claim    NUMERIC(12,2),     -- Zielvariable der Regression
    severity_quelle  VARCHAR(10)        -- 'text' | 'photo' — welche Quelle gewann
);

-- ---------------------------------------------------------------------
-- Tabelle 10: fahrzeug_katalog — Marke/Modell/Baujahr-Referenzdaten
-- (vehicle_dataset.csv: year,model,make)
-- Wird zum Trainieren des multi-head Modells (Textgenerierung mit
-- realen Fahrzeugkombinationen) sowie zur Validierung der vom Modell
-- vorhergesagten auto_make/auto_year-Kombination genutzt
-- (siehe validate_vehicle_with() in inference.py).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fahrzeug_katalog (
    katalog_id    SERIAL PRIMARY KEY,
    baujahr       INTEGER NOT NULL,
    modell        VARCHAR(50) NOT NULL,
    marke         VARCHAR(30) NOT NULL
);

-- ---------------------------------------------------------------------
-- Indizes auf Fremdschlüsseln: beschleunigen JOIN-Abfragen
-- ---------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_policen_kunde    ON policen(kunde_id);
CREATE INDEX IF NOT EXISTS idx_fahrzeuge_police ON fahrzeuge(police_id);
CREATE INDEX IF NOT EXISTS idx_unfaelle_police  ON unfaelle(police_id);
CREATE INDEX IF NOT EXISTS idx_schaeden_unfall  ON schaeden(unfall_id);
CREATE INDEX IF NOT EXISTS idx_mh_anfrage       ON multihead_ergebnisse(anfrage_id);
CREATE INDEX IF NOT EXISTS idx_cnn_anfrage      ON cnn_ergebnisse(anfrage_id);
CREATE INDEX IF NOT EXISTS idx_reg_anfrage      ON regression_ergebnisse(anfrage_id);
CREATE INDEX IF NOT EXISTS idx_katalog_marke    ON fahrzeug_katalog(marke);
CREATE INDEX IF NOT EXISTS idx_katalog_marke_jahr ON fahrzeug_katalog(marke, baujahr);