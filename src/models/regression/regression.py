# src/models/regression/regression.py

import torch
import torch.nn as nn

class DamageRegressionHead(nn.Module):
    """
    Independent Regression Head for Insurance Claim Prediction.
    Takes BERT pooled output (hidden_size=768) as input.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
        # Weight initialization for better convergence
        self._init_weights()

    def _init_weights(self):
        for m in self.regressor:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, pooled_features):
        return self.regressor(pooled_features).squeeze(-1)