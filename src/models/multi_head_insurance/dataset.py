import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class GermanInsuranceDataset(Dataset):
    def __init__(self, jsonl_path: str,
                 numeric_stats: dict,
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
        self.numeric_stats = numeric_stats
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
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_price": torch.tensor(target_price, dtype=torch.float32)
        }

        for field, val in labels.items():
            if self.network_config[field]["type"] == "regression":
                if val == -100:
                    item[f"label_{field}"] = torch.tensor(-100.0)
                else:
                    mean = self.numeric_stats[field]["mean"]
                    std = self.numeric_stats[field]["std"]
                    val = (val - mean) / std
                    item[f"label_{field}"] = torch.tensor(val, dtype=torch.float32)
            else:
                item[f"label_{field}"] = torch.tensor(val, dtype=torch.long)

        return item