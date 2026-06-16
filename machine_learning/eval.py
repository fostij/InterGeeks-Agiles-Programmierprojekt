import torch

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

class FieldEvaluator:
    def __init__(self, field_type):
        self.field_type = field_type
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

        self.mae_sum = 0
        self.mae_count = 0

        from collections import deque
        self.pred_cache = deque(maxlen=1000)
        self.true_cache = deque(maxlen=1000)

    def update(self, pred, true, mask=None):
        if self.field_type == "regression":
            pred_val = pred.squeeze(-1)
            diff = (pred_val - true).abs()

            if mask is not None:
                diff = diff[mask]

            self.mae_sum += diff.sum().item()
            self.mae_count += diff.numel()

        else:
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
    def __init__(self, network_config):
        self.fields = {
            k: FieldEvaluator("regression" if v == 1 else "classification")
            for k, v in network_config.items()
        }
        self.error_analyzer = ErrorAnalyzer()
        self.total_loss = 0.0
        self.total_count = 0

    def update(self, predictions, targets, loss, bs, input_text=None):
        self.total_loss += loss.item() * bs
        self.total_count += bs

        for field, evaluator in self.fields.items():
            pred = predictions[field]
            true = targets[f"label_{field}"]

            mask = (true != -100) if evaluator.field_type == "regression" else None

            evaluator.update(pred, true, mask)

            if evaluator.field_type == "classification":
                pred_cls = pred.argmax(-1)

                if mask is not None:
                    pred_cls = pred_cls[mask]
                    true_masked = true[mask]
                else:
                    true_masked = true

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

    def exact_match(self, predictions, targets):
        n = len(next(iter(targets.values())))
        ok = 0

        for i in range(n):
            match = True

            for field, ev in self.fields.items():
                p = predictions[field][i]
                t = targets[f"label_{field}"][i]

                if ev.field_type == "regression":
                    p = p.squeeze(-1)
                    match &= (abs(p - t) < 1e-3).all().item()
                else:
                    match &= ((p.argmax(-1) == t).all().item())

                if not match:
                    break

            ok += int(match)

        return ok / max(n, 1)
    
    def partial_score(self, predictions, targets):
        n = len(next(iter(targets.values())))
        scores = []

        for i in range(n):
            correct = 0
            total = len(self.fields)

            for field, ev in self.fields.items():
                p = predictions[field][i]
                t = targets[f"label_{field}"][i]

                if ev.field_type == "regression":
                    p = p.squeeze(-1)
                    correct += int((abs(p - t) < 1e-3).all().item())
                else:
                    correct += int((p.argmax(-1) == t).all().item())

            scores.append(correct / total)

        return sum(scores) / max(len(scores), 1)

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

            evaluator.update(predictions, batch_targets, loss, bs)

    return evaluator.compute()