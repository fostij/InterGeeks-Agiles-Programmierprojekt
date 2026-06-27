"""Zentrale Backend-Pipeline für die Schadensschätzung.
 
Einziger Einstiegspunkt für jeden Client (Dashboard, E-Mail-Worker,
Chatbot, ...). Orchestriert die Stufen: Anfrage speichern, Multi-Head-
Textanalyse, optionale CNN-Bildanalyse, Bestimmung der finalen
Schadensschwere und abschließende Regressions-Kostenschätzung. Jede
Stufe persistiert ihr eigenes Ergebnis über das übergebene
PipelineResultRepository (siehe src.db.pipeline_repository); Pipeline
selbst enthält keinen SQL-Code mehr (Single-Responsibility- und
Dependency-Inversion-Prinzip).
"""

import logging
from dataclasses import dataclass, field
import os
import tempfile
from typing import Any
import joblib
import pandas as pd
import torch
from src.db.pipeline_repository import PipelineResultRepository
from src.exceptions import InferenceError, ModelLoadError
from src.constants_paths import (
    MULTI_HEAD_MODEL_PATH,
    REGRESSION_MODEL_PATH,
    TARGET_FIELDS,
)
from src.models.multi_head_insurance.inference import (
    load_model as load_mh_model,
    predict as mh_predict,
)

logger = logging.getLogger(__name__)

# CNN gibt deutsche Labels zurück (3 Klassen), multi-head/Regression nutzt
# englische severity-Klassen (4 Klassen). "Mittlerer Schaden" wird auf
# "Major Damage" gemappt, da das CNN visuell nicht zuverlässig zwischen
# Major Damage und Total Loss unterscheiden kann (kein Signal wie
# "nicht mehr fahrbereit" wie im Text).
CNN_LABEL_TO_SEVERITY = {
    "Leichter Schaden": "Minor Damage",
    "Mittlerer Schaden": "Major Damage",
    "Erheblicher Schaden": "Total Loss",
}

# multi-head liefert "YES"/"NO" Strings; das Regressionsmodell wurde mit
# 1/0 trainiert (siehe regression/prepare_data.py).
YES_NO_TO_INT_FIELDS = ("police_report_available", "property_damage")


@dataclass
class PredictionResult:
    """Ergebnis eines vollständigen Pipeline-Durchlaufs für eine Anfrage.
 
    Sammelt die Ergebnisse aller Stufen (Multi-Head, CNN, Regression)
    sowie die Information, welche Schadensschwere-Quelle sich am Ende
    durchgesetzt hat.
    """

    request_id: int
    fields: dict[str, Any]

    missing_fields: list[str] = field(default_factory=list)

    severity_from_text: str | None = None
    text_confidence: dict | None = None

    severity_from_photo: str | None = None
    photo_confidence: float | None = None

    final_severity: str | None = None
    severity_source: str = "text"  # "text" | "photo"

    predicted_amount: float | None = None  # vehicle_claim


class Pipeline:
    """Einziger Backend-Einstiegspunkt für jeden Client (Dashboard, E-Mail-Worker, Chatbot, ...).
 
    Enthält keine UI-Importe und keinen SQL-Code. Jede Stufe delegiert
    das Speichern ihres Ergebnisses an das übergebene
    PipelineResultRepository, referenziert über anfrage_id.
    """

    def __init__(
        self,
        repository: PipelineResultRepository,
        mh_checkpoint: str = str(MULTI_HEAD_MODEL_PATH),
        regression_model_path: str | None = str(REGRESSION_MODEL_PATH),
        device: str = "cpu",
    ) -> None:
        """Lädt die Multi-Head- und (optional) Regressions-Modelle.
 
        Args:
            repository: Repository für die Persistenz der Stufenergebnisse
                (siehe src.db.pipeline_repository.PipelineResultRepository).
                Pipeline kennt keine Details der Speicherung, sondern ruft
                nur dessen Methoden auf (Dependency-Inversion-Prinzip).
            mh_checkpoint: Pfad zum Multi-Head-Checkpoint (Pflichtstufe).
            regression_model_path: Pfad zum Regressionsmodell. Falls None
                oder leer, wird die Regressionsstufe übersprungen.
            device: Zielgerät für die Multi-Head-Inferenz ("cpu"/"cuda").
 
        Raises:
            ModelLoadError: Wenn das Multi-Head- oder das Regressionsmodell
                nicht geladen werden kann.
        """

        self.repository = repository
        self.device = torch.device(device)

        self.mh_model, self.mh_network_config, self.mh_label_encoders = load_mh_model(
            mh_checkpoint, self.device
        )

        self.regression_model = None
        if regression_model_path:
            try:
                self.regression_model = joblib.load(regression_model_path)
            except Exception as exc:
                raise ModelLoadError(f"Regressionsmodell konnte nicht geladen werden: {exc}") from exc

        logger.info("Pipeline initialisiert (device=%s).", self.device)

    # ── Public API ───────────────────────────────────────────────────

    def run(
        self,
        text: str = "",
        photo_path: str | None = None,
        photo_bytes: bytes | None = None,
        source: str = "dashboard",
    ) -> PredictionResult:
        """Führt einen vollständigen Pipeline-Durchlauf für eine Anfrage aus.
 
        Stufen: Anfrage speichern, Multi-Head-Textanalyse (falls Text
        vorhanden), CNN-Bildanalyse (falls Foto vorhanden), Bestimmung
        der finalen Schadensschwere, abschließend Regressions-
        Kostenschätzung (falls Regressionsmodell verfügbar und Schwere
        bestimmt wurde).
 
        Args:
            text: Unfallbeschreibung als Rohtext. Leer, falls nicht vorhanden.
            photo_path: Pfad zu einem Foto des Schadens, falls vorhanden.
            photo_bytes: Bytes des Fotos, falls vorhanden.
            source: Herkunft der Anfrage (z. B. "dashboard", "email").
 
        Returns:
            PredictionResult mit den Ergebnissen aller durchlaufenen Stufen.
 
        Raises:
            PersistenceError: Wenn die Anfrage nicht in der Datenbank
                gespeichert werden kann.
            InferenceError: Wenn die Multi-Head-Textanalyse fehlschlägt.
        """

        tmp_file_path: str | None = None
        if photo_bytes is not None and photo_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(photo_bytes)
                tmp_file_path = tmp.name
            photo_path = tmp_file_path
            logger.debug("Foto-Bytes in temporäre Datei geschrieben: %s", tmp_file_path)

        hat_foto = photo_path is not None

        try:
            request_id = self._save_request(text, hat_foto, source)
            result = PredictionResult(request_id=request_id, fields={})
 
            if text and text.strip():
                self._run_multihead(text, result)
 
            if photo_path:
                self._run_cnn(photo_path, result)
 
            self._resolve_severity(result)
 
            if self.regression_model is not None and result.final_severity is not None:
                self._run_regression(result)
 
        finally:
            if tmp_file_path is not None:
                try:
                    os.unlink(tmp_file_path)
                    logger.debug("Temporäre Foto-Datei gelöscht: %s", tmp_file_path)
                except OSError as exc:
                    logger.warning("Temporäre Datei konnte nicht gelöscht werden: %s", exc)
 
        return result

    # ── Stage 1: persist the raw incoming request ───────────────────

    def _save_request(self, text: str, hat_foto: bool, source: str) -> int:
        """Speichert die eingehende Rohanfrage über das Repository und liefert die anfrage_id.
 
        Raises:
            PersistenceError: Wenn der Insert fehlschlägt (wird aus
                repository.save_request() durchgereicht).
        """

        return self.repository.save_request(text, hat_foto, source)

    # ── Stage 2: multi-head text -> fields (+ severity confidence) ──

    def _run_multihead(self, text: str, result: PredictionResult) -> None:
        """Führt die Multi-Head-Textanalyse aus und speichert das Ergebnis.
 
        Raises:
            InferenceError: Wenn die Multi-Head-Inferenz fehlschlägt
                (wird aus mh_predict() durchgereicht).
            PersistenceError: Wenn das Ergebnis nicht gespeichert werden kann.
        """

        df, confidences = mh_predict(
            texts=[text],
            model=self.mh_model,
            label_encoders=self.mh_label_encoders,
            device=self.device,
        )
        row = df.iloc[0].to_dict()
        conf_row = confidences.iloc[0].to_dict()

        result.fields = row
        result.missing_fields = [k for k, v in row.items() if v is None]
        result.severity_from_text = row.get("incident_severity")
        result.text_confidence = conf_row.get("incident_severity")

        logger.info("Multi-Head-Analyse abgeschlossen (anfrage_id=%d).", result.request_id)
        self.repository.save_multihead_result(result.request_id, row, conf_row, result.missing_fields)

    # ── Stage 3: photo -> severity (CNN), independent of text ───────

    def _run_cnn(self, photo_path: str, result: PredictionResult) -> None:
        """Führt die optionale CNN-Bildanalyse aus und speichert das Ergebnis.
 
        CNN ist eine optionale Stufe: Sind die CNN-Abhängigkeiten nicht
        installiert (ImportError), wird das als erwarteter Fall behandelt
        und die Stufe übersprungen. Schlägt die Inferenz selbst fehl
        (z. B. beschädigtes Bild, Modellfehler), wird das als Fehler
        geloggt, die Pipeline läuft aber ohne Foto-Schwere weiter, statt
        komplett abzubrechen.
        """

        try:
            from src.models.cnn.predict_image import predict_severity
        except ImportError:
            logger.info("CNN-Abhängigkeiten nicht verfügbar, Bildanalyse wird übersprungen.")
            return

        try:
            label_de, confidence = predict_severity(photo_path)
        except Exception as exc:
            logger.error("CNN-Inferenz fehlgeschlagen (anfrage_id=%d): %s", result.request_id, exc)
            return

        severity = CNN_LABEL_TO_SEVERITY.get(label_de)
        if severity is None:
            logger.warning("Unbekanntes CNN-Label '%s', Foto-Schwere wird ignoriert.", label_de)
            return

        result.severity_from_photo = severity
        result.photo_confidence = confidence

        logger.info("CNN-Analyse abgeschlossen (anfrage_id=%d, severity=%s).", result.request_id, severity)
        self.repository.save_cnn_result(result.request_id, label_de, severity, confidence)

    # ── Stage 4: decide which severity wins ──────────────────────────

    def _resolve_severity(self, result: PredictionResult) -> None:
        """Bestimmt die finale Schadensschwere aus Text- und Foto-Schätzung.
 
        CNN und Multi-Head können unabhängig voneinander eine Schwere
        schätzen. Die Schätzung mit der höheren Konfidenz gewinnt. Ist
        nur eine der beiden Quellen vorhanden, gewinnt diese automatisch.
        """

        text_conf = result.text_confidence or 0.0
        photo_conf = result.photo_confidence or 0.0

        if result.severity_from_photo is None:
            result.final_severity = result.severity_from_text
            result.severity_source = "text"
        elif result.severity_from_text is None:
            result.final_severity = result.severity_from_photo
            result.severity_source = "photo"
        elif photo_conf >= text_conf:
            result.final_severity = result.severity_from_photo
            result.severity_source = "photo"
        else:
            result.final_severity = result.severity_from_text
            result.severity_source = "text"

        result.fields["incident_severity"] = result.final_severity

    # ── Stage 5: fields -> vehicle_claim (regression) ────────────────

    def _run_regression(self, result: PredictionResult) -> None:
        """Schätzt vehicle_claim aus den bisher ermittelten Feldern und speichert das Ergebnis.
 
        Raises:
            InferenceError: Wenn die Regressions-Vorhersage fehlschlägt.
            PersistenceError: Wenn das Ergebnis nicht gespeichert werden kann.
        """

        feature_vector = self._fields_to_regression_input(result.fields)

        try:
            prediction = self.regression_model.predict(feature_vector)
        except Exception as exc:
            raise InferenceError(f"Regressions-Vorhersage fehlgeschlagen (anfrage_id={result.request_id}): {exc}") from exc

        result.predicted_amount = float(prediction[0])

        logger.info(
            "Regressions-Schätzung abgeschlossen (anfrage_id=%d, vehicle_claim=%.2f).",
            result.request_id, result.predicted_amount,
        )

        self.repository.save_regression_result(result.request_id, result.predicted_amount, result.severity_source)

    def _fields_to_regression_input(self, fields: dict[str, Any]) -> pd.DataFrame:
        """Bildet die Pipeline-Felder auf das vom Regressionsmodell erwartete Format ab.
 
        Konvertiert insbesondere die "YES"/"NO"-Strings aus der Multi-Head-
        Vorhersage in 1/0-Werte, wie sie das Regressionsmodell beim
        Training gesehen hat (siehe regression/prepare_data.py).
        """

        regression_input = {}
        for f in TARGET_FIELDS:
            val = fields.get(f)
            if f in YES_NO_TO_INT_FIELDS:
                val = self._yes_no_to_int(val)
            regression_input[f] = val
        return pd.DataFrame([regression_input])

    @staticmethod
    def _yes_no_to_int(value: Any) -> int:
        """Konvertiert "YES"/"NO" (oder None) in 1/0."""

        if value is None:
            return 0
        return 1 if str(value).upper() == "YES" else 0