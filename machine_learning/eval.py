import torch

def evaluate(model, dataloader, criterion, device, network_config):
    model.eval()

    total_loss = 0.0
    total_count = 0

    loss_dict_total = {}
    loss_dict_count = {}

    mae_sum = {}
    mae_count = {}

    acc_correct = {}
    acc_total = {}

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

            total_loss += loss.item() * bs
            total_count += bs

            # ---- loss per head ----
            for k, v in loss_dict.items():
                loss_dict_total[k] = loss_dict_total.get(k, 0.0) + v * bs
                loss_dict_count[k] = loss_dict_count.get(k, 0) + bs

            # ---- metrics per field ----
            for field in network_config.keys():
                pred = predictions[field]
                true = batch_targets[f"label_{field}"]

                # ================= REGRESSION =================
                if network_config[field] == 1:
                    mask = (true != -100)

                    if mask.any():
                        pred_val = pred[..., 0] if pred.ndim > 1 else pred

                        diff = (pred_val[mask] - true[mask]).abs()

                        mae_sum[field] = mae_sum.get(field, 0.0) + diff.sum().item()
                        mae_count[field] = mae_count.get(field, 0) + mask.sum().item()

                # ================= CLASSIFICATION =================
                else:
                    pred_class = pred.argmax(dim=-1)

                    correct = (pred_class == true).sum().item()
                    total = true.numel()

                    acc_correct[field] = acc_correct.get(field, 0) + correct
                    acc_total[field] = acc_total.get(field, 0) + total

    # ---- safe averaging ----
    avg_loss = total_loss / max(total_count, 1)

    loss_dict = {
        k: loss_dict_total[k] / max(loss_dict_count[k], 1)
        for k in loss_dict_total
    }

    mae_dict = {
        k: mae_sum[k] / max(mae_count[k], 1)
        for k in mae_sum
    }

    acc_dict = {
        k: acc_correct[k] / max(acc_total[k], 1)
        for k in acc_correct
    }

    return avg_loss, loss_dict, mae_dict, acc_dict