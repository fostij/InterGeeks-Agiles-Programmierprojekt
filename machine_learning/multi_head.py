import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from dataclasses import dataclass
from machine_learning.german_insurance_classifier import GermanInsuranceDataset, GermanInsuranceClassifier, DynamicMultiHeadLoss
from utils.labels_encoder import prepare_pipeline_and_save_jsonl
from utils.discreptive_statistic import get_descriptive_statistics_for_numeric
import pandas as pd
import json
from machine_learning.eval import MultiHeadEvaluator, evaluate
from utils.orchester_data import get_cleaned_dataset

@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 2e-5
    epochs: int = 1
    max_grad_norm: float = 1.0
    train_split: float = 0.8
    dataset_count: int = 1000
    model_name: str = "uklfr/gottbert-base"

    dataset_path: str = None
    input_file: str = None
    network_config_path: str = None
    output_checkpoint: str = None
    best_checkpoint: str = None

    network_config: dict = None
    
@dataclass
class BestModelTracker:
    output_checkpoint: str
    metric: str = "partial_score"  # "partial_score" | "exact_match" | "val_loss"
    mode: str = "max"              # "max" for accuracy, "min" for loss

    best_value: float = None
    best_epoch: int = -1

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def update(self, value: float, epoch: int, model, optimizer, cfg, num_stats) -> bool:
        if self.is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            save_checkpoint(cfg, model, optimizer, epoch, value, num_stats)
            print(f"  ✓ New best {self.metric}={value:.4f} — checkpoint saved")
            return True
        return False

    def summary(self):
        print(f"\n Best {self.metric}={self.best_value:.4f} at epoch {self.best_epoch + 1}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(f"Using processing engine: {device}")
    return device

def build_save_set_network_config(cfg: TrainConfig, dataset: pd.DataFrame):
    network_config = prepare_pipeline_and_save_jsonl(cfg.dataset_count, cfg.dataset_path, dataset)

    with open(cfg.network_config_path, "w", encoding="utf-8") as f:
        json.dump(network_config, f, indent=4)

    with open(cfg.network_config_path, "r", encoding="utf-8") as f:
        final_network_config = json.load(f)

    cfg.network_config = final_network_config
    print("✓ The head size configuration has been successfully saved to disk.")
    return final_network_config

def build_dataloaders(cfg: TrainConfig, dataset):
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader

def build_model(cfg: TrainConfig, device):
    model = GermanInsuranceClassifier(model_name=cfg.model_name, network_config=cfg.network_config)
    model.to(device)

    criterion = DynamicMultiHeadLoss(target_fields=list(cfg.network_config.keys()), network_config=cfg.network_config)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    return model, criterion, optimizer

def train_step(batch, model, criterion, optimizer, device, max_grad_norm, network_config):
    optimizer.zero_grad()
            
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
            
    batch_targets = {}
    for field in network_config.keys():
        batch_targets[f"label_{field}"] = batch[f"label_{field}"].to(device)
                
    predictions = model(input_ids, attention_mask)
            
    loss, loss_dict = criterion(predictions, batch_targets)
            
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Gradient clipping stabilizer
    optimizer.step()
    return loss.item(), loss_dict
    

def train_epoch(train_loader, model, criterion, optimizer, device, epoch, cfg: TrainConfig):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        loss, loss_dict = train_step(batch, model, criterion, optimizer, device, cfg.max_grad_norm, cfg.network_config)
        running_loss += loss

        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch}/{cfg.epochs}] | Step [{batch_idx}/{len(train_loader)}] | Loss: {loss:.4f}")
        # if batch_idx in [0,5,10]:
        #    for field, field_loss in loss_dict.items():
        #        print(f"    {field}: {field_loss:.4f}")
    
    return running_loss / len(train_loader)

def evaluate_epoch(val_loader, model, criterion, device, epoch, evaluator: MultiHeadEvaluator):
    avg_loss, loss_dict, mae_dict, acc_dict, exact, partial = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        evaluator=evaluator
    )

    print(f"\n📊 Epoch {epoch+1} | val_loss: {avg_loss:.4f} | exact_match: {exact:.4f} | partial_score: {partial:.4f}")
    return avg_loss, loss_dict, mae_dict, acc_dict, exact, partial

def save_checkpoint(cfg, model, optimizer, epoch, loss, num_stats):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "network_config": cfg.network_config,
        "numeric_stats": num_stats
    }

    torch.save(checkpoint, cfg.output_checkpoint)

def build_report(network_config, loss_dict, mae_dict, acc_dict):
    print("\n===== EVALUATION REPORT =====")

    report = []

    for field in network_config.keys():
        row = {"field": field}

        if network_config[field]["type"] == "regression":
            row["MAE"] = mae_dict.get(field, None)
            row["accuracy"] = None
        else:
            row["MAE"] = None
            row["accuracy"] = acc_dict.get(field, None)

        row["loss"] = loss_dict.get(field, None)

        report.append(row)

    df = pd.DataFrame(report)

    return df

def run_training(cfg: TrainConfig):
    device = get_device()

    dataset = get_cleaned_dataset()
    _ = build_save_set_network_config(cfg, dataset)
    
    num_stats = get_descriptive_statistics_for_numeric(dataset)

    dataset = GermanInsuranceDataset(jsonl_path=cfg.dataset_path, numeric_stats=num_stats, network_config=cfg.network_config)
    
    train_loader, val_loader = build_dataloaders(cfg, dataset)

    model, criterion, optimizer = build_model(cfg, device)

    print("\nTraining started")
    
    evaluator = MultiHeadEvaluator(cfg.network_config)

    tracker = BestModelTracker(
        output_checkpoint=cfg.best_checkpoint,
        metric="partial_score",
        mode="max"
    )

    history = []
    
    for epoch in range(cfg.epochs):
        avg_loss = train_epoch(train_loader, model, criterion, optimizer, device, epoch, cfg)

        val_loss, loss_dict, mae_dict, acc_dict, exact, partial = evaluate_epoch(val_loader, model, criterion, device, epoch, evaluator)

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "exact": exact,
            "partial": partial,
        }

        for field, value in loss_dict.items():
            row[f"{field}_loss"] = value

        for field, value in mae_dict.items():
            row[f"{field}_mae"] = value

        for field, value in acc_dict.items():
            row[f"{field}_acc"] = value

        history.append(row)

        report = build_report(cfg.network_config, loss_dict, mae_dict, acc_dict)
        print(report)
        evaluator.error_analyzer.report(top_n=3)

        tracker.update(
            value=partial,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            num_stats=num_stats
        )

    tracker.summary()
    print(f"\n✓ Training completed. Best checkpoint saved to: {cfg.best_checkpoint}")
    pd.DataFrame(history).to_csv("training_history.csv", index=False)
    
    