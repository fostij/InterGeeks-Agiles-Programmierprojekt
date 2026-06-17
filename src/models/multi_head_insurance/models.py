import torch
import torch.nn as nn
from transformers import AutoModel, logging as hf_logging

# Suppress model loading warnings
hf_logging.set_verbosity_error()


class GermanInsuranceClassifier(nn.Module):
    def __init__(
        self, model_name="uklfr/gottbert-base", network_config=None, dropout=0.1
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size
        self.network_config = network_config
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        for field, cfg in network_config.items():
            self.heads[field] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, cfg["size"]),
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        pooled = torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / input_mask_expanded.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)

        return {field: head(pooled) for field, head in self.heads.items()}
