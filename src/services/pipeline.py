from dataclasses import dataclass, field

import joblib
import pandas as pd
import torch
from sqlalchemy import text as sql_text

from src.db.connection import get_engine
from src.ml_config import (
    MULTI_HEAD_MODEL_PATH,
    REGRESSION_MODEL_PATH,
    TARGET_FIELDS,
)
from src.models.multi_head_insurance.inference import (
    load_model as load_mh_model,
    predict as mh_predict,
)

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
    request_id: int
    fields: dict

    missing_fields: list[str] = field(default_factory=list)

    severity_from_text: str | None = None
    text_confidence: float | None = None

    severity_from_photo: str | None = None
    photo_confidence: float | None = None

    final_severity: str | None = None
    severity_source: str = "text"  # "text" | "photo"

    predicted_amount: float | None = None  # vehicle_claim


class Pipeline:
    """
    Single backend entry point for any client (dashboard, email worker,
    chatbot, ...). No UI imports here. Every stage persists its own
    result to the database, keyed by anfrage_id.
    """

    def __init__(
        self,
        mh_checkpoint: str = str(MULTI_HEAD_MODEL_PATH),
        regression_model_path: str | None = str(REGRESSION_MODEL_PATH),
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.mh_model, self.mh_network_config, self.mh_label_encoders = load_mh_model(
            mh_checkpoint, self.device
        )

        self.regression_model = None
        if regression_model_path:
            self.regression_model = joblib.load(regression_model_path)

    # ── Public API ───────────────────────────────────────────────────

    def run(
        self,
        text: str = "",
        photo_path: str | None = None,
        source: str = "dashboard",
    ) -> PredictionResult:
        #request_id = self._save_request(text, photo_path, source)
        request_id = 0  # Dummy value since _save_request is disabled
        result = PredictionResult(request_id=request_id, fields={})

        if text and text.strip():
            self._run_multihead(text, result)

        if photo_path:
            self._run_cnn(photo_path, result)

        self._resolve_severity(result)

        if self.regression_model is not None and result.final_severity is not None:
            self._run_regression(result)

        return result

    # ── Stage 1: persist the raw incoming request ───────────────────

    def _save_request(self, text: str, photo_path: str | None, source: str) -> int:
        engine = get_engine()
        with engine.begin() as conn:
            row = conn.execute(
                sql_text("""
                    INSERT INTO anfragen (quelle, rohtext, foto_pfad)
                    VALUES (:quelle, :text, :foto)
                    RETURNING anfrage_id
                """),
                {"quelle": source, "text": text, "foto": photo_path},
            ).fetchone()
        return row[0]

    # ── Stage 2: multi-head text -> fields (+ severity confidence) ──

    def _run_multihead(self, text: str, result: PredictionResult):
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

        #self._save_multihead_result(result, row)

    def _save_multihead_result(self, result: PredictionResult, row: dict):
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                sql_text("""
                    INSERT INTO multihead_ergebnisse
                        (anfrage_id, incident_severity, severity_confidence, incident_type,
                         collision_type, number_of_vehicles, bodily_injuries, witnesses,
                         police_report, property_damage, auto_make, auto_year, fehlende_felder)
                    VALUES
                        (:anfrage_id, :severity, :conf, :itype, :ctype, :vehicles, :injuries,
                         :witnesses, :police, :damage, :make, :year, :missing)
                """),
                {
                    "anfrage_id": result.request_id,
                    "severity": row.get("incident_severity"),
                    "conf": result.text_confidence,
                    "itype": row.get("incident_type"),
                    "ctype": row.get("collision_type"),
                    "vehicles": row.get("number_of_vehicles_involved"),
                    "injuries": row.get("bodily_injuries"),
                    "witnesses": row.get("witnesses"),
                    "police": row.get("police_report_available"),
                    "damage": row.get("property_damage"),
                    "make": row.get("auto_make"),
                    "year": row.get("auto_year"),
                    "missing": ",".join(result.missing_fields),
                },
            )

    # ── Stage 3: photo -> severity (CNN), independent of text ───────

    def _run_cnn(self, photo_path: str, result: PredictionResult):
        try:
            from src.models.cnn.predict_image import predict_severity
        except Exception:
            return

        label_de, confidence = predict_severity(photo_path)
        severity = CNN_LABEL_TO_SEVERITY.get(label_de)
        if severity is None:
            return

        result.severity_from_photo = severity
        result.photo_confidence = confidence

        #self._save_cnn_result(result, label_de, severity, confidence)

    def _save_cnn_result(
        self, result: PredictionResult, label_de: str, severity: str, confidence: float
    ):
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                sql_text("""
                    INSERT INTO cnn_ergebnisse (anfrage_id, severity, severity_de, confidence)
                    VALUES (:anfrage_id, :severity, :severity_de, :confidence)
                """),
                {
                    "anfrage_id": result.request_id,
                    "severity": severity,
                    "severity_de": label_de,
                    "confidence": confidence,
                },
            )

    # ── Stage 4: decide which severity wins ──────────────────────────

    def _resolve_severity(self, result: PredictionResult):
        """
        Both CNN and multi-head can independently estimate severity.
        Whichever has higher confidence wins. If only one ran, it wins
        by default.
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

    def _run_regression(self, result: PredictionResult):
        feature_vector = self._fields_to_regression_input(result.fields)
        prediction = self.regression_model.predict(feature_vector)
        result.predicted_amount = float(prediction[0])

        #self._save_regression_result(result)

    def _save_regression_result(self, result: PredictionResult):
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(
                sql_text("""
                    INSERT INTO regression_ergebnisse (anfrage_id, vehicle_claim, severity_quelle)
                    VALUES (:anfrage_id, :claim, :quelle)
                """),
                {
                    "anfrage_id": result.request_id,
                    "claim": result.predicted_amount,
                    "quelle": result.severity_source,
                },
            )

    def _fields_to_regression_input(self, fields: dict) -> pd.DataFrame:
        regression_input = {}
        for f in TARGET_FIELDS:
            val = fields.get(f)
            if f in YES_NO_TO_INT_FIELDS:
                val = self._yes_no_to_int(val)
            regression_input[f] = val
        return pd.DataFrame([regression_input])

    @staticmethod
    def _yes_no_to_int(value) -> int:
        if value is None:
            return 0
        return 1 if str(value).upper() == "YES" else 0