import torch
from collections import deque

class ErrorAnalyzer:
    def __init__(self):
        self.errors = []
    
    def update(self, field, pred, true, input_text=None):
        self.errors.append({
            "field": field,
            "pred": pred.detach().cpu(),
            "true": true.detach().cpu(),
            "input": input_text
        })

    def report(self, top_n=5):
        if not self.errors:
            print("No errors recorded.")
            return

        from collections import defaultdict
        by_field = defaultdict(list)
        for e in self.errors:
            by_field[e["field"]].append(e)

        print("\n===== ERROR ANALYSIS =====")
        for field, errs in by_field.items():
            print(f"\n[{field}] — {len(errs)} misclassified")
            for e in errs[:top_n]:
                preds = e["pred"].tolist()
                trues = e["true"].tolist()
                for p, t in zip(preds, trues):
                    print(f"  pred={p}  true={t}")

class FieldEvaluator:
    def __init__(self, field_type):
        self.field_type = field_type
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

        self.mae_sum = 0
        self.mae_count = 0
        
        self.pred_cache = deque(maxlen=1000)
        self.true_cache = deque(maxlen=1000)

    def update(self, pred, true, mask=None):
        if self.field_type == "regression":
            pred_val = pred.squeeze(-1)
            diff = (pred_val - true).abs()

            if mask is not None:
                if mask.sum() == 0:
                    return
                diff = diff[mask]

            self.mae_sum += diff.sum().item()
            self.mae_count += diff.numel()

        else:
            if mask is not None and mask.sum() == 0:
                return
            pred_cls = pred.argmax(-1)

            if mask is not None:
                pred_cls = pred_cls[mask]
                true = true[mask]

            self.correct += (pred_cls == true).sum().item()
            self.total += true.numel()

            wrong = pred_cls != true
            if wrong.any():
                    self.pred_cache.append(pred_cls[wrong].detach().cpu())
                    self.true_cache.append(true[wrong].detach().cpu())

    def compute(self):
        return {
            "acc": self.correct / max(self.total, 1),
            "mae": self.mae_sum / max(self.mae_count, 1),
        }

class MultiHeadEvaluator:
    def __init__(self, network_config, numeric_stats=None):
        self.fields = {
            k: FieldEvaluator(v["type"])
            for k, v in network_config.items()
        }
        self.error_analyzer = ErrorAnalyzer()
        self.numeric_stats = numeric_stats or {}
        self.total_loss = 0.0
        self.total_count = 0

    def update(self, predictions, targets, loss, bs, input_text=None):
        self.total_loss += loss.item() * bs
        self.total_count += bs

        for field, evaluator in self.fields.items():
            pred = predictions[field]
            true = targets[f"label_{field}"]

            mask = (true != -100)

            if mask.sum() == 0:
                continue

            evaluator.update(pred, true, mask)

            if evaluator.field_type == "classification":
                pred_cls = pred.argmax(-1)

                pred_cls = pred_cls[mask]
                true_masked = true[mask]

                wrong = pred_cls != true_masked

                if wrong.any():
                    self.error_analyzer.update(
                        field=field,
                        pred=pred_cls[wrong].detach().cpu(),
                        true=true_masked[wrong].detach().cpu(),
                        input_text=input_text
                    )

    def compute(self):
        return {
            field: ev.compute()
            for field, ev in self.fields.items()
        }
    
    def reset(self):
        self.total_loss = 0.0
        self.total_count = 0
        self.error_analyzer = ErrorAnalyzer()

        for ev in self.fields.values():
            ev.reset()

    def compute_denorm_mae(self, predictions, targets):
        result = {}
        for field, ev in self.fields.items():
            if ev.field_type != "regression":
                continue
            pred = predictions[field].squeeze(-1)
            true = targets[f"label_{field}"]
            mask = (true != -100)
            if mask.sum() == 0:
                continue
            
            stats = self.numeric_stats.get(field, {})
            mean, std = stats.get("mean", 0), stats.get("std", 1)
            
            pred_denorm = pred[mask] * std + mean
            true_denorm = true[mask] * std + mean
            result[field] = (pred_denorm - true_denorm).abs().mean().item()
        return result

    def exact_match(self, predictions, targets, reg_rel_threshold=0.10):
        n = len(next(iter(targets.values())))
        ok = 0
        total = 0

        for i in range(n):
            sample_match = True
            sample_has_fields = False

            for field, ev in self.fields.items():
                p = predictions[field][i]
                t = targets[f"label_{field}"][i]

                if ev.field_type == "regression":
                    valid = (t != -100)

                    if valid.sum() == 0:
                        continue

                    sample_has_fields = True

                    stats = self.numeric_stats.get(field, {})
                    mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)

                    p_v = p.squeeze(-1)[valid] * std + mean
                    t_v = t[valid] * std + mean
                    
                    rel_err = (p_v - t_v).abs() / t_v.abs().clamp(min=1e-6)
                    sample_match &= (rel_err < reg_rel_threshold).all().item()

                else:
                    valid = (t != -100)

                    if valid.sum() == 0:
                        continue

                    sample_has_fields = True

                    p_cls = p.argmax(-1)
                    sample_match &= (p_cls[valid] == t[valid]).all().item()

                if not sample_match:
                    break

            if sample_has_fields:
                ok += int(sample_match)
                total += 1

        return ok / max(total, 1)
    
    def partial_score(self, predictions, targets, reg_rel_threshold=0.10):
        n = len(next(iter(targets.values())))
        clf_scores = []
        reg_scores = []

        for i in range(n):
            clf_total, clf_correct = 0, 0
            reg_total, reg_correct = 0, 0

            for field, ev in self.fields.items():
                p = predictions[field][i]
                t = targets[f"label_{field}"][i]

                if ev.field_type == "regression":
                    t_valid = (t != -100)

                    if t_valid.sum() == 0:
                        continue
                    
                    stats = self.numeric_stats.get(field, {})
                    mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)
                    p_v = p.squeeze(-1)[t_valid] * std + mean
                    t_v = t[t_valid] * std + mean
                    rel_err = (p_v - t_v).abs() / (t_v.abs().clamp(min=1e-6))
                    reg_correct += int((rel_err < reg_rel_threshold).all().item())
                    reg_total += 1
                else:
                    t_valid = (t != -100)

                    if t_valid.sum() == 0:
                        continue
                    
                    clf_correct += int((p.argmax(-1)[t_valid] == t[t_valid]).all().item())
                    clf_total += 1

            if clf_total > 0:
                clf_scores.append(clf_correct / clf_total)
            if reg_total > 0:
                reg_scores.append(reg_correct / reg_total)

        return {
            "clf_partial": sum(clf_scores) / max(len(clf_scores), 1),
            "reg_partial": sum(reg_scores) / max(len(reg_scores), 1),
        }

def evaluate(model, dataloader, criterion, device, evaluator: MultiHeadEvaluator):
    model.eval()
    evaluator.reset()

    all_predictions = {}
    all_targets = {}

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

            evaluator.update(predictions, batch_targets, loss, bs)

            for field, pred in predictions.items():
                all_predictions.setdefault(field, []).append(pred.cpu())
            for key, val in batch_targets.items():
                all_targets.setdefault(key, []).append(val.cpu())

    all_predictions = {f: torch.cat(v) for f, v in all_predictions.items()}
    all_targets = {f: torch.cat(v) for f, v in all_targets.items()}

    metrics = evaluator.compute()
    avg_loss = evaluator.total_loss / max(evaluator.total_count, 1)

    denorm_mae = {}
    for field, ev in evaluator.fields.items():
        if ev.field_type != "regression":
            continue
        pred = all_predictions[field].squeeze(-1)
        true = all_targets[f"label_{field}"]
        mask = (true != -100)
        if mask.sum() == 0:
            continue
        stats = evaluator.numeric_stats.get(field, {})
        mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)
        pred_denorm = pred[mask] * std + mean
        true_denorm = true[mask] * std + mean
        denorm_mae[field] = (pred_denorm - true_denorm).abs().mean().item()

    mae_dict = {f: v["mae"] for f, v in metrics.items()}
    acc_dict = {f: v["acc"] for f, v in metrics.items()}

    exact = evaluator.exact_match(all_predictions, all_targets)
    partial = evaluator.partial_score(all_predictions, all_targets)

    return avg_loss, loss_dict, mae_dict, acc_dict, exact, partial, denorm_mae