import json
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class GermanInsuranceDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer_name: str = "uklfr/gottbert-base", max_len: int = 256):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

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

        # Build the return package format for the batch iterator
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "target_price": torch.tensor(target_price, dtype=torch.float32)
        }

        # Dynamic multi-head labels extraction
        for field, val in labels.items():
            # Keep numeric columns as float values, classification values as long index IDs
            if field in ["auto_year", "number_of_vehicles_involved", "witnesses"]:
                item[f"label_{field}"] = torch.tensor(val, dtype=torch.float32)
            else:
                item[f"label_{field}"] = torch.tensor(val, dtype=torch.long)

        return item


class GermanInsuranceClassifier(nn.Module):
    def __init__(self, model_name="uklfr/gottbert-base", num_classes_dict=None):
        super(GermanInsuranceClassifier, self).__init__()
        
        # 1. Load the core German BERT backbone
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size # 768 for gottbert-base
        
        self.heads = nn.ModuleDict()
        self.num_classes_dict = num_classes_dict
        
        # 2. Dynamically build output heads based on your field types
        for field, num_classes in num_classes_dict.items():
            if num_classes == 1:
                # Regression heads (for auto_year, number_of_vehicles_involved, witnesses)
                self.heads[field] = nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            else:
                # Classification heads (for text fields like auto_make, weather, etc.)
                self.heads[field] = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        # Pass the German tokens through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # Mean Pooling: dynamic averaging over all word tokens using the attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Compute predictions across all 12 heads independently
        outputs_dict = {}
        for field, head in self.heads.items():
            outputs_dict[field] = head(pooled_output)
            
        return outputs_dict

class DynamicMultiHeadLoss(nn.Module):
    def __init__(self, target_fields):
        super(DynamicMultiHeadLoss, self).__init__()
        self.target_fields = target_fields
        
        # Standard classification loss (None is handled as a standard class ID, e.g. 0)
        self.clf_criterion = nn.CrossEntropyLoss()
        # Reduction='none' lets us manually filter numeric missing values row-by-row
        self.reg_criterion = nn.MSELoss(reduction='none')

    def forward(self, predictions, batch):
        total_loss = 0.0
        
        for field in self.target_fields:
            pred_logits = predictions[field]          # What the head predicted
            true_labels = batch[f"label_{field}"]      # Ground truth from JSONL
            
            # Numeric columns (Regression)
            if field in ["auto_year", "number_of_vehicles_involved", "witnesses"]:
                # Create a binary filter mask (1.0 for valid numbers, 0.0 for -100)
                mask = (true_labels != -100).float()
                
                raw_loss = self.reg_criterion(pred_logits.squeeze(-1), true_labels)
                masked_loss = raw_loss * mask
                
                # Only add numeric loss if there is at least one valid example in the batch
                if mask.sum() > 0:
                    total_loss += masked_loss.sum() / mask.sum()
            
            # Categorical columns (Classification)
            else:
                total_loss += self.clf_criterion(pred_logits, true_labels)
                
        return total_loss
