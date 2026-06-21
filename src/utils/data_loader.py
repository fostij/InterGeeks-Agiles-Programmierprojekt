"""Laden der Datensätze aus der PostgreSQL-Datenbank.
 
Stellt die JOIN-Abfragen bereit, die die 5 normalisierten Tabellen
(kunden, policen, fahrzeuge, unfaelle, schaeden bzw. fahrzeug_katalog)
wieder zu den ursprünglichen englischen Spaltennamen zusammenführen,
damit der restliche Code (multi-head, regression) unverändert
funktioniert.
"""

import logging
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from src.db.connection import get_engine
from src.exceptions import DataPreparationError
 
logger = logging.getLogger(__name__)


def get_insurance_dataset() -> pd.DataFrame:
    """
    Lädt den vollständigen Versicherungsdatensatz per JOIN über alle
    5 normalisierten Tabellen (kunden, policen, fahrzeuge, unfaelle,
    schaeden) und bildet ihn auf die ursprünglichen englischen
    Spaltennamen ab, damit der restliche Code (multi-head, regression)
    unverändert funktioniert.
 
    Raises:
        DataPreparationError: Wenn die Datenbankverbindung oder die
            Abfrage fehlschlägt.
    """

    engine = get_engine()
    query = text("""
        SELECT
            k.alter_jahre              AS age,
            k.geschlecht                AS insured_sex,
            k.bildungsniveau             AS insured_education_level,
            k.beruf                       AS insured_occupation,
            k.hobbys                       AS insured_hobbies,
            k.familienverhaeltnis            AS insured_relationship,
            k.plz                              AS insured_zip,
            k.kapitalgewinne                    AS "capital-gains",
            k.kapitalverluste                    AS "capital-loss",
            k.kunde_seit_monaten                  AS months_as_customer,
 
            p.policen_nummer                       AS policy_number,
            p.vertragsbeginn                        AS policy_bind_date,
            p.bundesstaat                            AS policy_state,
            p.deckungsgrenze                          AS policy_csl,
            p.selbstbeteiligung                        AS policy_deductable,
            p.jahrespraemie                              AS policy_annual_premium,
            p.umbrella_limit                              AS umbrella_limit,
 
            f.marke                                        AS auto_make,
            f.modell                                         AS auto_model,
            f.baujahr                                         AS auto_year,
 
            u.unfall_datum                                     AS incident_date,
            u.unfall_typ                                        AS incident_type,
            u.kollisions_typ                                     AS collision_type,
            u.schadensschwere                                     AS incident_severity,
            u.behoerde                                             AS authorities_contacted,
            u.bundesstaat                                           AS incident_state,
            u.stadt                                                  AS incident_city,
            u.adresse                                                 AS incident_location,
            u.uhrzeit_stunde                                           AS incident_hour_of_the_day,
            u.anzahl_fahrzeuge                                          AS number_of_vehicles_involved,
            u.sachschaden_dritter                                        AS property_damage,
            u.anzahl_verletzte                                            AS bodily_injuries,
            u.anzahl_zeugen                                                AS witnesses,
            u.polizeibericht                                                AS police_report_available,
 
            s.gesamtschaden                                                  AS total_claim_amount,
            s.personenschaden                                                 AS injury_claim,
            s.sachschaden                                                      AS property_claim,
            s.fahrzeugschaden                                                   AS vehicle_claim,
            s.betrug_gemeldet                                                    AS fraud_reported
        FROM schaeden s
        JOIN unfaelle  u ON u.unfall_id  = s.unfall_id
        JOIN policen   p ON p.police_id  = u.police_id
        JOIN fahrzeuge f ON f.police_id  = p.police_id
        JOIN kunden    k ON k.kunde_id   = p.kunde_id
    """)
 
    with engine.connect() as conn:
        try:
            df = pd.read_sql(query, conn)
        except SQLAlchemyError as exc:
            raise DataPreparationError(f"Versicherungsdatensatz konnte nicht geladen werden: {exc}") from exc
        
    logger.info("Versicherungsdatensatz geladen: %d Zeilen.", len(df))
    return df
 
 
def get_vehicle_dataset() -> pd.DataFrame:
    """
    Lädt den Fahrzeugkatalog (Marke/Modell/Baujahr-Referenzdaten) aus
    fahrzeug_katalog und bildet ihn auf die ursprünglichen englischen
    Spaltennamen ab (year, model, make), damit Code wie
    validate_vehicle_with() in inference.py unverändert funktioniert.
 
    Raises:
        DataPreparationError: Wenn die Datenbankverbindung oder die
            Abfrage fehlschlägt.
    """

    engine = get_engine()
    query = text("""
        SELECT
            baujahr AS year,
            modell  AS model,
            marke   AS make
        FROM fahrzeug_katalog
    """)
 
    with engine.connect() as conn:
        try:
            df = pd.read_sql(query, conn)
        except SQLAlchemyError as exc:
            raise DataPreparationError(f"Fahrzeugkatalog konnte nicht geladen werden: {exc}") from exc

    logger.info("Fahrzeugkatalog geladen: %d Zeilen.", len(df))
    return df