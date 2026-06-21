"""Persistenz-Schicht für die Ergebnisse der Schadensschätzungs-Pipeline.
 
Kapselt sämtliche SQL-Zugriffe, die beim Durchlauf der Pipeline
(src.services.pipeline.Pipeline) anfallen: das Speichern der Rohanfrage
sowie der Ergebnisse jeder einzelnen Stufe (Multi-Head, CNN, Regression).
 
Diese Trennung folgt dem Single-Responsibility-Prinzip: Pipeline
orchestriert die fachliche Logik (welche Stufe wann läuft), während
PipelineResultRepository ausschließlich für die Persistenz zuständig
ist (wie und wo die Ergebnisse gespeichert werden). Pipeline hängt dabei
nur von dieser Abstraktion ab, nicht von SQLAlchemy direkt
(Dependency-Inversion-Prinzip) — ein Austausch der Speicherung (z. B.
gegen eine andere Datenbank oder einen Mock in Tests) erfordert keine
Änderung an Pipeline.
"""

import logging
from typing import Any
 
from sqlalchemy import text as sql_text
from sqlalchemy.exc import SQLAlchemyError
 
from src.db.connection import get_engine
from src.exceptions import PersistenceError
 
logger = logging.getLogger(__name__)

class PipelineResultRepository:
    """Speichert Rohanfragen und Stufenergebnisse der Pipeline in PostgreSQL."""

    def save_request(self, text: str, photo_path: str | None, source: str) -> int:
        """Speichert die eingehende Rohanfrage und liefert die erzeugte anfrage_id.
 
        Args:
            text: Unfallbeschreibung als Rohtext.
            photo_path: Pfad zu einem Foto des Schadens, falls vorhanden.
            source: Herkunft der Anfrage (z. B. "dashboard", "email").
 
        Returns:
            Die von der Datenbank erzeugte anfrage_id.
 
        Raises:
            PersistenceError: Wenn der Insert fehlschlägt.
        """
        engine = get_engine()
        try:
            with engine.begin() as conn:
                row = conn.execute(
                    sql_text("""
                        INSERT INTO anfragen (quelle, rohtext, foto_pfad)
                        VALUES (:quelle, :text, :foto)
                        RETURNING anfrage_id
                    """),
                    {"quelle": source, "text": text, "foto": photo_path},
                ).fetchone()
        except SQLAlchemyError as exc:
            raise PersistenceError(f"Anfrage konnte nicht gespeichert werden: {exc}") from exc
 
        request_id = row[0]
        logger.info("Anfrage gespeichert (anfrage_id=%d, source=%s).", request_id, source)
        return request_id
 
    def save_multihead_result(
        self, request_id: int, row: dict[str, Any], conf_row: dict[str, Any], missing_fields: list[str]
    ) -> None:
        """Speichert das Ergebnis der Multi-Head-Textanalyse.
 
        Args:
            request_id: anfrage_id, zu der das Ergebnis gehört.
            row: Vorhergesagte Werte je Zielfeld.
            conf_row: Konfidenzwerte je Zielfeld.
            missing_fields: Liste der Felder, die nicht sicher genug
                vorhergesagt werden konnten.
 
        Raises:
            PersistenceError: Wenn der Insert fehlschlägt.
        """
        engine = get_engine()
        try:
            with engine.begin() as conn:
                conn.execute(
                    sql_text("""
                        INSERT INTO multihead_ergebnisse
                            (anfrage_id,
                            incident_severity, incident_severity_conf,
                            incident_type, incident_type_conf,
                            collision_type, collision_type_conf,
                            number_of_vehicles, number_of_vehicles_conf,
                            bodily_injuries, bodily_injuries_conf,
                            witnesses, witnesses_conf,
                            police_report, police_report_conf,
                            property_damage, property_damage_conf,
                            auto_make, auto_make_conf,
                            auto_year, auto_year_conf,
                            fehlende_felder)
                        VALUES
                            (:anfrage_id,
                            :severity, :severity_conf,
                            :itype, :itype_conf,
                            :ctype, :ctype_conf,
                            :vehicles, :vehicles_conf,
                            :injuries, :injuries_conf,
                            :witnesses, :witnesses_conf,
                            :police, :police_conf,
                            :damage, :damage_conf,
                            :make, :make_conf,
                            :year, :year_conf,
                            :missing)
                    """),
                    {
                        "anfrage_id": request_id,
                        "severity": row.get("incident_severity"),
                        "severity_conf": conf_row.get("incident_severity"),
                        "itype": row.get("incident_type"),
                        "itype_conf": conf_row.get("incident_type"),
                        "ctype": row.get("collision_type"),
                        "ctype_conf": conf_row.get("collision_type"),
                        "vehicles": row.get("number_of_vehicles_involved"),
                        "vehicles_conf": conf_row.get("number_of_vehicles_involved"),
                        "injuries": row.get("bodily_injuries"),
                        "injuries_conf": conf_row.get("bodily_injuries"),
                        "witnesses": row.get("witnesses"),
                        "witnesses_conf": conf_row.get("witnesses"),
                        "police": row.get("police_report_available"),
                        "police_conf": conf_row.get("police_report_available"),
                        "damage": row.get("property_damage"),
                        "damage_conf": conf_row.get("property_damage"),
                        "make": row.get("auto_make"),
                        "make_conf": conf_row.get("auto_make"),
                        "year": row.get("auto_year"),
                        "year_conf": conf_row.get("auto_year"),
                        "missing": ",".join(missing_fields),
                    },
                )
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Multi-Head-Ergebnis konnte nicht gespeichert werden (anfrage_id={request_id}): {exc}"
            ) from exc
 
    def save_cnn_result(
        self, request_id: int, label_de: str, severity: str, confidence: float
    ) -> None:
        """Speichert das Ergebnis der CNN-Bildanalyse.
 
        Args:
            request_id: anfrage_id, zu der das Ergebnis gehört.
            label_de: Vom CNN vorhergesagtes deutsches Label.
            severity: Auf die englischen severity-Klassen gemapptes Label.
            confidence: Konfidenz der CNN-Vorhersage.
 
        Raises:
            PersistenceError: Wenn der Insert fehlschlägt.
        """
        engine = get_engine()
        try:
            with engine.begin() as conn:
                conn.execute(
                    sql_text("""
                        INSERT INTO cnn_ergebnisse (anfrage_id, severity, severity_de, confidence)
                        VALUES (:anfrage_id, :severity, :severity_de, :confidence)
                    """),
                    {
                        "anfrage_id": request_id,
                        "severity": severity,
                        "severity_de": label_de,
                        "confidence": confidence,
                    },
                )
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"CNN-Ergebnis konnte nicht gespeichert werden (anfrage_id={request_id}): {exc}"
            ) from exc
 
    def save_regression_result(
        self, request_id: int, predicted_amount: float, severity_source: str
    ) -> None:
        """Speichert das Ergebnis der Regressions-Kostenschätzung.
 
        Args:
            request_id: anfrage_id, zu der das Ergebnis gehört.
            predicted_amount: Geschätzter vehicle_claim-Betrag.
            severity_source: Quelle der zugrunde liegenden Schadensschwere
                ("text" oder "photo").
 
        Raises:
            PersistenceError: Wenn der Insert fehlschlägt.
        """
        engine = get_engine()
        try:
            with engine.begin() as conn:
                conn.execute(
                    sql_text("""
                        INSERT INTO regression_ergebnisse (anfrage_id, vehicle_claim, severity_quelle)
                        VALUES (:anfrage_id, :claim, :quelle)
                    """),
                    {
                        "anfrage_id": request_id,
                        "claim": predicted_amount,
                        "quelle": severity_source,
                    },
                )
        except SQLAlchemyError as exc:
            raise PersistenceError(
                f"Regressionsergebnis konnte nicht gespeichert werden (anfrage_id={request_id}): {exc}"
            ) from exc