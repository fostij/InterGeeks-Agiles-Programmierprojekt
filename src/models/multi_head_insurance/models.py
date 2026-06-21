"""Modellarchitektur für die Multi-Head-Klassifikation von Unfallbeschreibungen.
 
Definiert ein Transformer-basiertes Modell (GottBERT) mit einem
gemeinsamen Encoder und mehreren unabhängigen Klassifikationsköpfen,
einem pro Zielfeld (z. B. auto_make, incident_severity).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

class GermanInsuranceClassifier(nn.Module):
    """Transformer-Encoder mit mehreren parallelen Klassifikationsköpfen.
 
    Der Encoder (GottBERT) erzeugt Token-Embeddings, die per
    Attention-Mask-gewichtetem Mean-Pooling zu einem einzigen
    Satz-Embedding zusammengefasst werden. Auf diesem Embedding setzt
    pro Zielfeld ein eigener kleiner MLP-Kopf an.
    """

    def __init__(self, model_name: str = "uklfr/gottbert-base", network_config: dict[str, dict] = None, dropout: float = 0.1) -> None:
        """Baut den Encoder und die Klassifikationsköpfe auf.
 
        Args:
            model_name: Name/Pfad des HuggingFace-Encoder-Modells.
            network_config: Mapping {Feldname: {"size": Klassenanzahl, ...}},
                bestimmt Anzahl und Ausgabegröße der Köpfe.
            dropout: Dropout-Rate für Pooling und Köpfe.
        """

        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.network_config = network_config
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        for field, cfg in network_config.items():
            self.heads[field] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, cfg["size"])
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Berechnet die Logits aller Köpfe für einen Batch von Eingaben.
 
        Args:
            input_ids: Tokenisierte Eingabe-IDs, Shape (batch, seq_len).
            attention_mask: Attention-Maske, Shape (batch, seq_len).
 
        Returns:
            Dictionary {Feldname: Logits-Tensor} für jeden konfigurierten Kopf.
        """

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                 input_mask_expanded.sum(1).clamp(min=1e-9)

        pooled = self.dropout(pooled)
        
        return {field: head(pooled) for field, head in self.heads.items()}