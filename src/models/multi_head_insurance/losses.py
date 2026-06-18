import torch
import torch.nn as nn


class DynamicMultiHeadLoss(nn.Module):
    def __init__(self, target_fields, network_config):
        super().__init__()
        self.target_fields = target_fields
        self.network_config = network_config
        self.clf_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.weights = {
            f: cfg.get("loss_weight", 1.0) for f, cfg in network_config.items()
        }

        # Financial regression values can generate massive errors early in training.
        # Setting a small starting weight (like 0.5 or 1.0) balances it out.
        self.vehicle_claim_weight = 1.0

    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}

        # 1. Process your original 14 target fields
        for field in self.target_fields:
            # Skip vehicle_claim if it accidentally bypassed checks into target_fields
            if field == "vehicle_claim":
                continue

            pred_logits = predictions[field]
            true_labels = batch[f"label_{field}"]
            w = self.weights[field]

            loss_value = self.clf_criterion(pred_logits, true_labels)
            if torch.isnan(loss_value):
                loss_value = torch.tensor(0.0, device=pred_logits.device)
            total_loss += w * loss_value
            loss_dict[field] = loss_value.item()

        if "vehicle_claim" in predictions and "label_vehicle_claim" in batch:
            pred_claim = predictions["vehicle_claim"]
            true_claim = batch["label_vehicle_claim"]

            # Handle possible missing data values (-100 mask)
            claim_mask = (true_claim != -100).float()

            raw_claim_loss = self.reg_criterion(pred_claim, true_claim)
            masked_claim_loss = raw_claim_loss * claim_mask

            # Standardized Mean Squared Error for the financial value
            claim_loss_value = masked_claim_loss.sum() / claim_mask.sum().clamp(min=1)

            # Add it to the main training calculations
            total_loss += self.vehicle_claim_weight * claim_loss_value
            loss_dict["vehicle_claim"] = claim_loss_value.item()

        return total_loss, loss_dict
