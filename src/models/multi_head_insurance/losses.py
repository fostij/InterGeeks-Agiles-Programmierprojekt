import torch.nn as nn

class DynamicMultiHeadLoss(nn.Module):
    def __init__(self, target_fields, network_config):
        super().__init__()
        self.target_fields = target_fields
        self.network_config = network_config
        self.clf_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.reg_criterion = nn.MSELoss(reduction='none')

        self.weights = {
            f: cfg.get("loss_weight", 1.0) 
            for f, cfg in network_config.items()
        }

    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}
        for field in self.target_fields:
            pred_logits = predictions[field]
            true_labels = batch[f"label_{field}"]
            cfg = self.network_config[field]
            w = self.weights[field]

            if cfg["type"] == "regression":
                mask = (true_labels != -100).float()
                
                raw_loss = self.reg_criterion(pred_logits.squeeze(-1), true_labels)
                masked_loss = raw_loss * mask
                
                loss_value = masked_loss.sum() / mask.sum().clamp(min=1)
            else:
                loss_value = self.clf_criterion(pred_logits, true_labels)

            total_loss += w * loss_value
            loss_dict[field] = loss_value.item()            
                
        return total_loss, loss_dict