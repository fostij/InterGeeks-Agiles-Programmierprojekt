-- =====================================================================
-- schema.sql — Datenbankschema für das Projekt "KFZ-Schadenprognose"
-- ---------------------------------------------------------------------
-- Der Original-Datensatz (1 flache CSV-Tabelle, 40 Spalten) wird in
-- 5 normalisierte Tabellen aufgeteilt (3. Normalform):
--
--   kunden ──< policen ──< unfaelle ──< schaeden
--                  │
--                  └──< fahrzeuge
--
-- Vorteile: keine Redundanz, klare Verantwortlichkeiten je Tabelle,
-- referenzielle Integrität über Fremdschlüssel (FOREIGN KEY).
--
-- Ausführen mit:  psql -U postgres -d kfz_schaden -f sql/schema.sql
-- =====================================================================

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
    kunde_id            SERIAL PRIMARY KEY,        -- künstlicher Schlüssel (auto-increment)
    alter_jahre         INTEGER NOT NULL,          -- Spalte "age" ("alter" ist in SQL reserviert!)
    geschlecht          VARCHAR(10),               -- insured_sex (MALE/FEMALE)
    bildungsniveau      VARCHAR(20),               -- insured_education_level (MD, PhD, ...)
    beruf               VARCHAR(50),               -- insured_occupation
    hobbys              VARCHAR(50),               -- insured_hobbies
    familienverhaeltnis VARCHAR(30),               -- insured_relationship
    plz                 VARCHAR(10),               -- insured_zip (als Text: führende Nullen!)
    kapitalgewinne      INTEGER,                   -- capital-gains
    kapitalverluste     INTEGER,                   -- capital-loss
    kunde_seit_monaten  INTEGER                    -- months_as_customer
);

-- ---------------------------------------------------------------------
-- Tabelle 2: policen — Versicherungsverträge (1 Kunde -> n Policen)
-- ---------------------------------------------------------------------
CREATE TABLE policen (
    police_id           SERIAL PRIMARY KEY,
    kunde_id            INTEGER NOT NULL REFERENCES kunden(kunde_id),  -- Fremdschlüssel
    policen_nummer      INTEGER UNIQUE NOT NULL,   -- policy_number (natürlicher Schlüssel)
    vertragsbeginn      DATE,                      -- policy_bind_date
    bundesstaat         VARCHAR(5),                -- policy_state
    deckungsgrenze      VARCHAR(20),               -- policy_csl (z. B. "250/500")
    selbstbeteiligung   INTEGER,                   -- policy_deductable
    jahrespraemie       NUMERIC(10,2),             -- policy_annual_premium (Geld -> NUMERIC!)
    umbrella_limit      BIGINT                     -- umbrella_limit (bis 10 Mio. -> BIGINT)
);

-- ---------------------------------------------------------------------
-- Tabelle 3: fahrzeuge — versicherte Fahrzeuge (1 Police -> n Fahrzeuge)
-- ---------------------------------------------------------------------
CREATE TABLE fahrzeuge (
    fahrzeug_id         SERIAL PRIMARY KEY,
    police_id           INTEGER NOT NULL REFERENCES policen(police_id),
    marke               VARCHAR(30),               -- auto_make
    modell              VARCHAR(30),               -- auto_model
    baujahr             INTEGER                    -- auto_year
);

-- ---------------------------------------------------------------------
-- Tabelle 4: unfaelle — gemeldete Vorfälle (1 Police -> n Unfälle)
-- ---------------------------------------------------------------------
CREATE TABLE unfaelle (
    unfall_id           SERIAL PRIMARY KEY,
    police_id           INTEGER NOT NULL REFERENCES policen(police_id),
    unfall_datum        DATE,                      -- incident_date
    unfall_typ          VARCHAR(40),               -- incident_type
    kollisions_typ      VARCHAR(30),               -- collision_type ('?' -> NULL!)
    schadensschwere     VARCHAR(20),               -- incident_severity
    behoerde            VARCHAR(20),               -- authorities_contacted
    bundesstaat         VARCHAR(5),                -- incident_state
    stadt               VARCHAR(40),               -- incident_city
    adresse             VARCHAR(80),               -- incident_location
    uhrzeit_stunde      INTEGER CHECK (uhrzeit_stunde BETWEEN 0 AND 23),
    anzahl_fahrzeuge    INTEGER,                   -- number_of_vehicles_involved
    sachschaden_dritter VARCHAR(5),                -- property_damage ('?' -> NULL)
    anzahl_verletzte    INTEGER,                   -- bodily_injuries
    anzahl_zeugen       INTEGER,                   -- witnesses
    polizeibericht      VARCHAR(5)                 -- police_report_available ('?' -> NULL)
);

-- ---------------------------------------------------------------------
-- Tabelle 5: schaeden — Schadenszahlungen (Zielvariable des Projekts!)
-- ---------------------------------------------------------------------
CREATE TABLE schaeden (
    schaden_id          SERIAL PRIMARY KEY,
    unfall_id           INTEGER NOT NULL REFERENCES unfaelle(unfall_id),
    gesamtschaden       NUMERIC(12,2),             -- total_claim_amount (ZIELVARIABLE Regression)
    personenschaden     NUMERIC(12,2),             -- injury_claim
    sachschaden         NUMERIC(12,2),             -- property_claim
    fahrzeugschaden     NUMERIC(12,2),             -- vehicle_claim
    betrug_gemeldet     BOOLEAN                    -- fraud_reported Y/N (ZIELVARIABLE Klassifikation)
);

-- ---------------------------------------------------------------------
-- Indizes auf Fremdschlüsseln: beschleunigen JOIN-Abfragen
-- ---------------------------------------------------------------------
CREATE INDEX idx_policen_kunde    ON policen(kunde_id);
CREATE INDEX idx_fahrzeuge_police ON fahrzeuge(police_id);
CREATE INDEX idx_unfaelle_police  ON unfaelle(police_id);
CREATE INDEX idx_schaeden_unfall  ON schaeden(unfall_id);