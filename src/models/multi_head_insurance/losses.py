"""Verlustfunktion für das Multi-Head-Klassifikationsmodell.
 
Kombiniert die Cross-Entropy-Verluste mehrerer Klassifikationsköpfe zu
einem gewichteten Gesamtverlust.
"""
 
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class DynamicMultiHeadLoss(nn.Module):
    """Gewichtete Summe der Cross-Entropy-Verluste über alle Zielfelder.
 
    Padding-Labels (-100) werden bei der Verlustberechnung ignoriert.
    Tritt für ein Feld ein NaN-Verlust auf (z. B. weil ein Batch keine
    gültigen Labels für dieses Feld enthält), wird der Verlust für
    dieses Feld auf 0.0 gesetzt, damit das Training nicht abbricht.
    """

    def __init__(self, target_fields: list[str], network_config: dict[str, dict]) -> None:
        """Initialisiert die Verlustfunktion mit den Köpfen und ihren Gewichten.
 
        Args:
            target_fields: Liste der Feldnamen, die einen eigenen Kopf haben.
            network_config: Konfiguration je Feld, u. a. optionales
                "loss_weight" (Standard: 1.0).
        """

        super().__init__()
        self.target_fields = target_fields
        self.network_config = network_config
        self.clf_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.weights = {
            f: cfg.get("loss_weight", 1.0) 
            for f, cfg in network_config.items()
        }

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Berechnet den gewichteten Gesamtverlust über alle Köpfe.
 
        Returns:
            Tupel (total_loss, loss_dict), wobei loss_dict den
            Verlust je Feld als float enthält.
        """

        total_loss = 0.0
        loss_dict = {}
        for field in self.target_fields:
            pred_logits = predictions[field]
            true_labels = batch[f"label_{field}"]
            w = self.weights[field]

            loss_value = self.clf_criterion(pred_logits, true_labels)
            if torch.isnan(loss_value):
                logger.warning("NaN-Verlust für Feld '%s' erkannt, wird auf 0.0 gesetzt.", field)
                loss_value = torch.tensor(0.0, device=pred_logits.device)
            total_loss += w * loss_value
            loss_dict[field] = loss_value.item()            
                
        return total_loss, loss_dict