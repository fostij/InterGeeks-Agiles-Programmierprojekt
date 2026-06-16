import json
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from ml_config import NUMERIC_FIELDS

class GermanInsuranceDataset(Dataset):
    def __init__(self, jsonl_path: str,
                 numeric_stats: dict,
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
            if field in NUMERIC_FIELDS:
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


class GermanInsuranceClassifier(nn.Module):
    def __init__(self, model_name="uklfr/gottbert-base", network_config=None):
        super(GermanInsuranceClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.heads = nn.ModuleDict()
        self.network_config = network_config
        
        for field, out_dim in network_config.items():
            if out_dim == 1:
                # Regression heads (for auto_year, number_of_vehicles_involved, witnesses)
                self.heads[field] = nn.Linear(hidden_size, 1)
            else:
                # Classification heads (for text fields like auto_make, weather, etc.)
                self.heads[field] = nn.Linear(hidden_size, out_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        outputs_dict = {}
        for field, head in self.heads.items():
            outputs_dict[field] = head(pooled_output)
            
        return outputs_dict

class DynamicMultiHeadLoss(nn.Module):
    def __init__(self, target_fields):
        super(DynamicMultiHeadLoss, self).__init__()
        self.target_fields = target_fields
        
        self.clf_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss(reduction='none')

    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}
        for field in self.target_fields:
            pred_logits = predictions[field]
            true_labels = batch[f"label_{field}"]

            if field in NUMERIC_FIELDS:
                mask = (true_labels != -100).float()
                
                raw_loss = self.reg_criterion(pred_logits.squeeze(-1), true_labels)
                masked_loss = raw_loss * mask
                
                if mask.sum().item() > 0:
                    loss_value = masked_loss.sum() / mask.sum()
                else:
                    loss_value = torch.tensor(0.0, device=pred_logits.device)

                total_loss += loss_value
                loss_dict[field] = loss_value.item()

            else:
                loss_value = self.clf_criterion(pred_logits, true_labels)
                total_loss += loss_value
                loss_dict[field] = loss_value.item()
            
                
        return total_loss, loss_dict
