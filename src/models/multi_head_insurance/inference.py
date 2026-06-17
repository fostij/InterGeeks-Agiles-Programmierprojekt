import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
from ml_config import INTEGER_FIELDS
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
                    value = int(value) if value is not None else None

                rows[i][field] = value

    return pd.DataFrame(rows)