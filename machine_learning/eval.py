import torch

class MultiHeadEvaluator:
    def __init__(self, network_config):
        self.network_config = network_config
        self.reset()

    def update_batch(self, predictions, batch_targets, loss, loss_dict, batch_size):
        self.total_loss += loss.item() * batch_size
        self.total_count += batch_size

        # loss per head
        for k, v in loss_dict.items():
            self.loss_sum[k] = self.loss_sum.get(k, 0.0) + v * batch_size
            self.loss_count[k] = self.loss_count.get(k, 0) + batch_size

        # metrics per field
        for field in self.network_config.keys():
            pred = predictions[field]
            true = batch_targets[f"label_{field}"]

            if self.network_config[field] == 1:
                mask = (true != -100)

                if mask.any():
                    pred_val = pred[..., 0] if pred.ndim > 1 else pred
                    diff = (pred_val[mask] - true[mask]).abs()

                    self.mae_sum[field] = self.mae_sum.get(field, 0.0) + diff.sum().item()
                    self.mae_count[field] = self.mae_count.get(field, 0) + mask.sum().item()

            else:
                pred_class = pred.argmax(dim=-1)

                self.acc_correct[field] = self.acc_correct.get(field, 0) + (pred_class == true).sum().item()
                self.acc_total[field] = self.acc_total.get(field, 0) + true.numel()

    def compute(self):
        avg_loss = self.total_loss / max(self.total_count, 1)

        loss_dict = {
            k: self.loss_sum[k] / max(self.loss_count[k], 1)
            for k in self.loss_sum
        }

        mae_dict = {
            k: self.mae_sum[k] / max(self.mae_count[k], 1)
            for k in self.mae_sum
        }

        acc_dict = {
            k: self.acc_correct[k] / max(self.acc_total[k], 1)
            for k in self.acc_correct
        }

        return avg_loss, loss_dict, mae_dict, acc_dict

    def reset(self):
        self.total_loss = 0.0
        self.total_count = 0

        self.loss_sum = {}
        self.loss_count = {}

        self.mae_sum = {}
        self.mae_count = {}

        self.acc_correct = {}
        self.acc_total = {}

def evaluate(model, dataloader, criterion, device, network_config, evaluator: MultiHeadEvaluator):
    model.eval()
    evaluator.reset()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            batch_targets = {
                k: v.to(device)
                for k, v in batch.items()
                if k.startswith("label_")
            }

            predictions = model(input_ids, attention_mask)
            loss, loss_dict = criterion(predictions, batch_targets)

            bs = input_ids.size(0)

            evaluator.update_batch(predictions, batch_targets, loss, loss_dict, bs)

    return evaluator.compute()