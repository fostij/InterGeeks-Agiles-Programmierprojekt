import sys
import traceback
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
from ml_config import INTEGER_FIELDS
from src.utils.data_loader import get_vehicle_dataset
from src.models.multi_head_insurance.models import GermanInsuranceClassifier

def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    network_config = checkpoint["network_config"]

    label_encoders = {
        field: {i: val for i, val in enumerate(classes)}
        for field, classes in checkpoint["label_encoders"].items()
    }

    model = GermanInsuranceClassifier(network_config=network_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, network_config, label_encoders


def predict(
    texts: list[str],
    model,
    label_encoders: dict,       # {field: {int_index: original_value}}
    tokenizer_name: str = "uklfr/gottbert-base",
    clf_threshold: float = 0.6,
    max_len: int = 256,
    device = "cpu",
) -> pd.DataFrame:

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

    rows = [{} for _ in texts]

    for field, pred in predictions.items():
        probs = F.softmax(pred, dim=-1)
        confidence, pred_idx = probs.max(dim=-1)

        encoder = label_encoders.get(field, {})

        for i in range(len(texts)):
            if confidence[i].item() < clf_threshold:
                rows[i][field] = None

            else:
                idx = pred_idx[i].item()
                value = encoder.get(idx, None)

                if field in INTEGER_FIELDS:
                    value = int(str(value)) if value is not None else None

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
    except Exception as e:
        print(f"DataFrame creation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    return df

def validate_vehicle_with(make: str, year: int, catalog: pd.DataFrame):
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
        return make, int(nearest)
    
    return make, None