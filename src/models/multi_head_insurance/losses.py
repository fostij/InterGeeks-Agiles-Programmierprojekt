import torch
import torch.nn as nn

class DynamicMultiHeadLoss(nn.Module):
    def __init__(self, target_fields, network_config):
        super(DynamicMultiHeadLoss, self).__init__()
        self.target_fields = target_fields
        self.network_config = network_config
        
        self.clf_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss(reduction='none')

    def forward(self, predictions, batch):
        total_loss = 0.0
        loss_dict = {}
        for field in self.target_fields:
            pred_logits = predictions[field]
            true_labels = batch[f"label_{field}"]
            cfg = self.network_config[field]
            if cfg["type"] == "regression":
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