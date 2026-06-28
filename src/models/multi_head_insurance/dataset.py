"""PyTorch-Dataset für das Multi-Head-Klassifikationsmodell.
 
Liest die von src.utils.labels_encoder erzeugte JSONL-Datei ein, in der
jede Zeile ein Trainingsbeispiel (Text, Labels je Feld, Zielpreis) ist,
und tokenisiert den Text bei Bedarf (__getitem__).
"""

import logging
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.exceptions import DataPreparationError

logger = logging.getLogger(__name__)

class GermanInsuranceDataset(Dataset):
    """Dataset, das Unfallbeschreibungen samt Multi-Head-Labels bereitstellt."""
    
    def __init__(self, jsonl_path: str,
                 network_config: dict[str, dict],
                 tokenizer_name: str = "uklfr/gottbert-base", 
                 max_len: int = 256,
                 ) -> None:
        """Lädt die Trainingsbeispiele aus einer JSONL-Datei.
 
        Args:
            jsonl_path: Pfad zur JSONL-Datei mit den Trainingsbeispielen.
            network_config: Head-Konfiguration (Feldnamen und Klassenanzahl).
            tokenizer_name: Name/Pfad des HuggingFace-Tokenizers.
            max_len: Maximale Token-Länge nach dem Tokenisieren.
 
        Raises:
            DataPreparationError: Wenn die Datei nicht gefunden wird oder
                eine Zeile kein gültiges JSON enthält.
        """

        self.samples = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if line.strip():
                        try:
                            self.samples.append(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise DataPreparationError(
                                f"Ungültiges JSON in {jsonl_path}, Zeile {line_number}: {exc}"
                            ) from exc
        except OSError as exc:
            raise DataPreparationError(f"JSONL-Datei konnte nicht gelesen werden: {jsonl_path}: {exc}") from exc
 
        if not self.samples:
            raise DataPreparationError(f"JSONL-Datei enthält keine Trainingsbeispiele: {jsonl_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.network_config = network_config
        logger.info("Dataset geladen: %d Beispiele aus %s.", len(self.samples), jsonl_path)
        
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        labels = sample["labels"]

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
        }

        for field, val in labels.items():
            item[f"label_{field}"] = torch.tensor(val, dtype=torch.long)

        return item