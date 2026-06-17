import torch
import json
import pandas as pd
from transformers import AutoTokenizer
from src.models.multi_head_insurance.models import GermanInsuranceClassifier

def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    network_config = checkpoint["network_config"]
    numeric_stats = checkpoint["numeric_stats"]

    label_encoders = {
        field: {i: val for i, val in enumerate(classes)}
        for field, classes in checkpoint["label_encoders"].items()
    }

    model = GermanInsuranceClassifier(network_config=network_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, network_config, numeric_stats, label_encoders


def predict(
    texts: list[str],
    model,
    network_config: dict,
    numeric_stats: dict,
    label_encoders: dict,       # {field: {int_index: original_value}}
    tokenizer_name: str = "uklfr/gottbert-base",
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
        cfg = network_config[field]

        if cfg["type"] == "regression":
            pred_val = pred.squeeze(-1).cpu()
            mean = numeric_stats[field]["mean"]
            std = numeric_stats[field]["std"]
            pred_denorm = pred_val * std + mean

            for i, val in enumerate(pred_denorm.tolist()):
                rows[i][field] = round(val, 4)

        else:
            pred_idx = pred.argmax(-1).cpu().tolist()
            encoder = label_encoders.get(field, {})

            for i, idx in enumerate(pred_idx):
                rows[i][field] = encoder.get(idx, idx)

    return pd.DataFrame(rows)