-- 1. Initialize the official Schema first
\i sql/schema.sql

-- 2. Create the staging table with exactly 40 columns (39 real + 1 extra trailing column)
CREATE TEMP TABLE staging_raw (
    months_as_customer INT, age INT, policy_number INT, policy_bind_date DATE, policy_state TEXT,
    policy_csl TEXT, policy_deductable INT, policy_annual_premium NUMERIC, umbrella_limit INT,
    insured_zip TEXT, insured_sex TEXT, insured_education_level TEXT, insured_occupation TEXT,
    insured_hobbies TEXT, insured_relationship TEXT, "capital-gains" INT, "capital-loss" INT,
    incident_date DATE, incident_type TEXT, collision_type TEXT, incident_severity TEXT,
    authorities_contacted TEXT, incident_state TEXT, incident_city TEXT, incident_location TEXT,
    incident_hour_of_the_day INT, number_of_vehicles_involved INT, property_damage TEXT,
    bodily_injuries INT, witnesses INT, police_report_available TEXT, total_claim_amount INT,
    injury_claim INT, property_claim INT, vehicle_claim INT, auto_make TEXT, auto_model TEXT,
    auto_year INT, fraud_reported TEXT,
    _c39 TEXT -- The hidden 40th column that was breaking things!
);

-- 3. EXTRACT: Directly load the CSV into our 40-column table
\copy staging_raw FROM 'data/raw/dataset.csv' WITH CSV HEADER NULL AS '?';

-- Add sequential ID to mimic Python's "lfd_id"
ALTER TABLE staging_raw ADD COLUMN lfd_id SERIAL;

-- 4. LOAD & TRANSFORM (Inserting into the 5 target tables)
INSERT INTO kunden (kunde_id, alter_jahre, geschlecht, bildungsniveau, beruf, hobbys, familienverhaeltnis, plz, kapitalgewinne, kapitalverluste, kunde_seit_monaten)
SELECT lfd_id, age, insured_sex, insured_education_level, insured_occupation, insured_hobbies, insured_relationship, insured_zip, "capital-gains", "capital-loss", months_as_customer
FROM staging_raw;

INSERT INTO policen (police_id, policen_nummer, vertragsbeginn, bundesstaat, deckungsgrenze, selbstbeteiligung, jahrespraemie, kunde_id)
SELECT lfd_id, policy_number, policy_bind_date, policy_state, policy_csl, policy_deductable, policy_annual_premium, lfd_id
FROM staging_raw;

INSERT INTO fahrzeuge (police_id, marke, modell, baujahr)
SELECT lfd_id, auto_make, auto_model, auto_year
FROM staging_raw;

INSERT INTO unfaelle (unfall_id, unfall_datum, unfall_typ, kollisions_typ, schadensschwere, behoerde, bundesstaat, stadt, adresse, uhrzeit_stunde, anzahl_fahrzeuge, sachschaden_dritter, anzahl_verletzte, anzahl_zeugen, polizeibericht, police_id)
SELECT 
    lfd_id, incident_date, incident_type, 
    NULLIF(collision_type, '?'), incident_severity, authorities_contacted, incident_state, incident_city, incident_location, incident_hour_of_the_day, number_of_vehicles_involved, 
    NULLIF(property_damage, '?'), bodily_injuries, witnesses, 
    NULLIF(police_report_available, '?'), lfd_id
FROM staging_raw;

INSERT INTO schaeden (unfall_id, gesamtschaden, personenschaden, sachschaden, fahrzeugschaden, betrug_gemeldet)
SELECT lfd_id, total_claim_amount, injury_claim, property_claim, vehicle_claim, 
       CASE WHEN fraud_reported = 'Y' THEN TRUE ELSE FALSE END
FROM staging_raw;

-- 5. VERIFY
SELECT k.alter_jahre, f.marke, u.schadensschwere, s.gesamtschaden
FROM schaeden s
JOIN unfaelle  u ON u.unfall_id  = s.unfall_id
JOIN policen   p ON p.police_id  = u.police_id
JOIN fahrzeuge f ON f.police_id  = p.police_id
JOIN kunden    k ON k.kunde_id   = p.kunde_id
LIMIT 3;
