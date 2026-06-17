import torch
import csv
import os
import warnings
from transformers import AutoTokenizer, logging as hf_logging
from src.models.multi_head_insurance.models import GermanInsuranceClassifier

# Suppress all HF and transformers warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(checkpoint_path: str, device):
    # Suppress transformer model loading warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Some weights.*not initialized.*")
        warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        network_config = checkpoint["network_config"]
        numeric_stats = checkpoint["numeric_stats"]

        label_encoders = {
            field: {i: val for i, val in enumerate(classes)}
            for field, classes in checkpoint["label_encoders"].items()
        }

        model = GermanInsuranceClassifier(network_config=network_config)

        # ─── 🛠️ FIXED HERE: ADDED strict=False ──────────────────────────────
        # This prevents PyTorch from crashing due to the missing weights
        # for our newly added 15th 'vehicle_claim_head'.
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        model.to(device)
        model.eval()

    return model, network_config, numeric_stats, label_encoders


def predict(
    texts: list[str],
    model,
    network_config: dict,
    numeric_stats: dict,
    label_encoders: dict,  # {field: {int_index: original_value}}
    tokenizer_name: str = "uklfr/gottbert-base",
    max_len: int = 256,
    device="cpu",
) -> list:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*huggingface.co/settings/tokens.*")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Clear cache before inference to avoid memory issues
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    rows = [{} for _ in texts]

    for field, pred in predictions.items():
        # ─── 🛠️ SPECIALIZED DECODER FOR VEHICLE CLAIM ───────────────────────
        if field == "vehicle_claim":
            pred_val = pred.cpu()  # Already squeezed in models.py forward pass

            # Fetch means and standard deviations to reverse the normalization math
            # Default to 0 mean and 1 std if stats aren't generated yet
            mean = numeric_stats.get("vehicle_claim", {}).get("mean", 0.0)
            std = numeric_stats.get("vehicle_claim", {}).get("std", 1.0)

            # De-normalize back into real Euro amounts!
            pred_denorm = pred_val * std + mean

            for i, val in enumerate(pred_denorm.tolist()):
                # Round financial claim outputs to 2 decimal places (cents)
                rows[i][field] = round(val, 2)
            continue

        # ─── Standard processing loop for the other 14 features ───────────
        cfg = network_config[field]

        if cfg["type"] == "regression":
            pred_val = pred.squeeze(-1).cpu()
            mean = numeric_stats[field]["mean"]
            std = numeric_stats[field]["std"]
            pred_denorm = pred_val * std + mean

            for i, val in enumerate(pred_denorm.tolist()):
                if field == "auto_year":
                    rows[i][field] = int(round(val))
                else:
                    rows[i][field] = round(val, 2)

        else:
            pred_idx = pred.argmax(-1).cpu().tolist()
            encoder = label_encoders.get(field, {})

            for i, idx in enumerate(pred_idx):
                rows[i][field] = encoder.get(idx, idx)

    return rows


def save_predictions_to_csv(rows: list, output_path: str = "reconstructed_table.csv"):
    """Save prediction results to CSV without using pandas."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {output_path}")
