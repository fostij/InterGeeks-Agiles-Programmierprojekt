import torch
import torch.nn as nn
from transformers import AutoModel

class GermanInsuranceClassifier(nn.Module):
    def __init__(self, model_name="uklfr/gottbert-base", network_config=None):
        super(GermanInsuranceClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.heads = nn.ModuleDict()
        self.network_config = network_config
        
        for field, cfg in network_config.items():
            self.heads[field] = nn.Linear(hidden_size, cfg["size"])

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