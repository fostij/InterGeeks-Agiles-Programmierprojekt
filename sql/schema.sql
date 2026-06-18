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
-- Tabelle 6: vorhersagen — über das Dashboard erfasste Eingaben + Prognosen
-- ---------------------------------------------------------------------
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
);

-- ---------------------------------------------------------------------
-- Indizes auf Fremdschlüsseln: beschleunigen JOIN-Abfragen
-- ---------------------------------------------------------------------
CREATE INDEX idx_policen_kunde    ON policen(kunde_id);
CREATE INDEX idx_fahrzeuge_police ON fahrzeuge(police_id);
CREATE INDEX idx_unfaelle_police  ON unfaelle(police_id);
CREATE INDEX idx_schaeden_unfall  ON schaeden(unfall_id);