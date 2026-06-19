# src/models/regression.py

import torch
import torch.nn as nn

class DamageRegressionHead(nn.Module):
    """
    Iyi ni model yigenga (standalone) yo gupredikta Schadenhöhe.
    Iyi model ntabwo ibangamira GermanInsuranceClassifier y'abandi.
    """
    def __init__(self, hidden_size=768):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.regressor(x).squeeze(-1)