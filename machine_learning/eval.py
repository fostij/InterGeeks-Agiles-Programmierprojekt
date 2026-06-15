import torch
import pandas as pd

def evaluate(model, dataloader, criterion, device, numeric_fields, target_fields):
    model.eval()

    total_loss = 0.0
    total_count = 0

    loss_dict_total = {}
    loss_dict_count = {}

    # --- metrics ---
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

            batch_size = input_ids.size(0)

            total_loss += loss.item() * batch_size
            total_count += batch_size

            # --- loss dict ---
            for k, v in loss_dict.items():
                loss_dict_total[k] = loss_dict_total.get(k, 0.0) + v * batch_size
                loss_dict_count[k] = loss_dict_count.get(k, 0) + batch_size

            # --- metrics per field ---
            for field in target_fields:
                pred = predictions[field]
                true = batch_targets[f"label_{field}"]

                # ================= REGRESSION =================
                if field in numeric_fields:
                    mask = (true != -100)

                    if mask.sum().item() > 0:
                        pred_val = pred.squeeze(-1)

                        diff = (pred_val[mask] - true[mask]).abs()

                        mae_sum[field] = mae_sum.get(field, 0.0) + diff.sum().item()
                        mae_count[field] = mae_count.get(field, 0) + mask.sum().item()

                # ================= CLASSIFICATION =================
                else:
                    pred_class = pred.argmax(dim=-1)

                    correct = (pred_class == true).sum().item()

                    acc_correct[field] = acc_correct.get(field, 0) + correct
                    acc_total[field] = acc_total.get(field, 0) + true.numel()

    # --- averages ---
    avg_loss = total_loss / total_count

    loss_dict = {
        k: loss_dict_total[k] / loss_dict_count[k]
        for k in loss_dict_total
    }

    mae_dict = {
        k: mae_sum[k] / mae_count[k]
        for k in mae_sum
    }

    acc_dict = {
        k: acc_correct[k] / acc_total[k]
        for k in acc_correct
    }

    return avg_loss, loss_dict, mae_dict, acc_dict