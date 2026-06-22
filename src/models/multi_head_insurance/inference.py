"""Inferenz für das Multi-Head-Klassifikationsmodell.

Lädt einen trainierten Checkpoint und sagt für Unfallbeschreibungen
mehrere kategoriale Felder gleichzeitig voraus (z. B. auto_make,
auto_year, incident_severity). Zusätzlich wird die vorhergesagte
Fahrzeugkombination (Marke + Baujahr) gegen den Fahrzeugkatalog validiert.
"""

import logging

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer

from src.exceptions import InferenceError, ModelLoadError
from src.ml_config import INTEGER_FIELDS
from src.utils.data_loader import get_vehicle_dataset
from src.models.multi_head_insurance.models import GermanInsuranceClassifier

logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str, device: torch.device | str
) -> tuple[GermanInsuranceClassifier, dict, dict[str, dict[int, str]]]:
    """Lädt einen trainierten Checkpoint und baut das zugehörige Modell auf.

    Args:
        checkpoint_path: Pfad zur gespeicherten .pt-Checkpoint-Datei.
        device: Zielgerät (z. B. "cpu" oder "cuda"), auf das das Modell
            geladen wird.

    Returns:
        Tupel aus (Modell im eval-Modus, network_config, label_encoders).

    Raises:
        ModelLoadError: Wenn der Checkpoint nicht gefunden oder nicht
            geladen werden kann.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as exc:
        raise ModelLoadError(f"Checkpoint konnte nicht geladen werden von {checkpoint_path}: {exc}") from exc

    network_config = checkpoint["network_config"]

    label_encoders = {
        field: {i: val for i, val in enumerate(classes)}
        for field, classes in checkpoint["label_encoders"].items()
    }

    model = GermanInsuranceClassifier(network_config=network_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Multi-Head-Modell geladen von: %s", checkpoint_path)
    return model, network_config, label_encoders


def predict(
    texts: list[str],
    model: GermanInsuranceClassifier,
    label_encoders: dict[str, dict[int, str]],       # {field: {int_index: original_value}}
    tokenizer_name: str = "uklfr/gottbert-base",
    clf_threshold: float = 0.6,
    max_len: int = 256,
    device: torch.device | str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sagt für eine Liste von Unfallbeschreibungen mehrere Felder voraus.

    Args:
        texts: Liste von Unfallbeschreibungen (Rohtext).
        model: Geladenes GermanInsuranceClassifier-Modell (eval-Modus).
        label_encoders: Mapping {Feldname: {Klassenindex: Originalwert}}.
        tokenizer_name: Name/Pfad des HuggingFace-Tokenizers.
        clf_threshold: Mindest-Konfidenz, unterhalb der ein Feld als
            unsicher gilt und auf None gesetzt wird.
        max_len: Maximale Token-Länge nach dem Tokenisieren.
        device: Zielgerät für die Inferenz.

    Returns:
        Tupel (df, conf_df): df enthält die vorhergesagten Werte je Feld,
        conf_df die zugehörigen Konfidenzwerte.

    Raises:
        InferenceError: Wenn die Tokenisierung, die Modell-Inferenz oder
            der Aufbau des Ergebnis-DataFrames fehlschlägt.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        encoding = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            predictions = model(input_ids, attention_mask)
    except Exception as exc:
        raise InferenceError(f"Modell-Inferenz fehlgeschlagen: {exc}") from exc

    rows = [{} for _ in texts]
    conf_rows = [{} for _ in texts]

    for field, pred in predictions.items():
        probs = F.softmax(pred, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

        encoder = label_encoders.get(field, {})

        for i in range(len(texts)):
            conf_value = confidence[i].item()
            conf_rows[i][field] = conf_value

            if confidence[i].item() < clf_threshold:
                rows[i][field] = None
            else:
                idx = pred_idx[i].item()
                value = encoder.get(idx, None)

                if field in INTEGER_FIELDS and value is not None:
                    try:
                        value = int(str(value))
                    except (ValueError, TypeError):
                        logger.warning("Konvertierung zu int fehlgeschlagen für Feld '%s', Wert '%s' — wird auf None gesetzt.", field, value)
                        value = None

                rows[i][field] = value

    catalog = get_vehicle_dataset()

    for i in range(len(texts)):
        make = rows[i].get("auto_make")
        year = rows[i].get("auto_year")

        if all([make, year]):
            rows[i]["auto_make"], rows[i]["auto_year"] = \
                validate_vehicle_with(make, int(year), catalog)

    try:
        df = pd.DataFrame(rows)
    except Exception as exc:
        raise InferenceError(f"Erstellung des Ergebnis-DataFrames fehlgeschlagen: {exc}") from exc

    conf_df = pd.DataFrame(conf_rows)
    return df, conf_df

def validate_vehicle_with(make: str, year: int, catalog: pd.DataFrame) -> tuple[str, int | None]:
    """Validiert eine vorhergesagte Marke/Baujahr-Kombination gegen den Fahrzeugkatalog.

    Gibt es im Katalog keinen exakten Treffer für (make, year), wird das
    nächstgelegene verfügbare Baujahr derselben Marke verwendet. Gibt es
    die Marke gar nicht im Katalog, bleibt das Baujahr unbestimmt (None).

    Args:
        make: Vorhergesagte Fahrzeugmarke.
        year: Vorhergesagtes Baujahr.
        catalog: Fahrzeugkatalog mit den Spalten "make" und "year".

    Returns:
        Tupel (make, year), wobei year ggf. auf das nächstgelegene
        bekannte Baujahr korrigiert wurde, oder None, falls die Marke
        im Katalog nicht vorkommt.
    """
    match = catalog[
        (catalog["make"] == make) &
        (catalog["year"] == year)
    ]

    if not match.empty:
        return make, year

    same_model = catalog[
        (catalog["make"] == make)
    ]

    if not same_model.empty:
        closest_year = same_model["year"].values
        nearest = closest_year[abs(closest_year - year).argmin()]
        logger.debug(
            "Kein exakter Katalogtreffer für %s %d, nutze nächstgelegenes Baujahr %d.",
            make, year, int(nearest),
        )
        return make, int(nearest)

    logger.warning("Marke '%s' nicht im Fahrzeugkatalog gefunden, Baujahr bleibt unbestimmt.", make)
    return make, None