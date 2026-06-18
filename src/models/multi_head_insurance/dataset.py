import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class GermanInsuranceDataset(Dataset):
    def __init__(self, jsonl_path: str,
                 network_config,
                 tokenizer_name: str = "uklfr/gottbert-base", 
                 max_len: int = 256,
                 ):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.network_config = network_config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        labels = sample["labels"]
        target_price = sample["target_price"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            # Keep original target_price intact just in case other evaluation metrics need it
            "target_price": torch.tensor(target_price, dtype=torch.float32),
        }

        # ─── 🛠️ ADD THIS: NORMALIZE THE VEHICLE CLAIM FINANCIAL VALUE ────────
        # This reads the mean and standard deviation for the cash amounts,
        # converting raw Euros (e.g. 15000) into optimized neural network scales.
        if "vehicle_claim" in self.numeric_stats and target_price != -100:
            claim_mean = self.numeric_stats["vehicle_claim"]["mean"]
            claim_std = self.numeric_stats["vehicle_claim"]["std"]
            normalized_claim = (target_price - claim_mean) / claim_std
            item["label_vehicle_claim"] = torch.tensor(
                normalized_claim, dtype=torch.float32
            )
        else:
            # If data is missing or stats aren't generated yet, pass -100 to mask it out
            item["label_vehicle_claim"] = torch.tensor(
                target_price, dtype=torch.float32
            )

        # Process your remaining 14 categorical and numeric targets
        for field, val in labels.items():
            item[f"label_{field}"] = torch.tensor(val, dtype=torch.long)

        return item
