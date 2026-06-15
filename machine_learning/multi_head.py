from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from transformers import dataclass
from machine_learning.german_insurance_classifier import GermanInsuranceDataset, GermanInsuranceClassifier, DynamicMultiHeadLoss
from utils.labels_encoder import prepare_pipeline_and_save_jsonl
from utils.discreptive_statistic import get_descriptive_statistics_for_numeric
import pandas as pd
import json
from ml_config import DATASET_PATH, NETWORK_CONFIG_PATH, INPUT_FILE, OUTPUT_CHECKPOINT
from machine_learning.eval import evaluate

@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 2e-5
    epochs: int = 1
    max_grad_norm: float = 1.0
    train_split: float = 0.8
    dataset_count: int = 1000
    model_name: str = "uklfr/gottbert-base"

    dataset_path: str = DATASET_PATH
    input_file: str = INPUT_FILE
    network_config_path: str = NETWORK_CONFIG_PATH
    output_checkpoint: str = OUTPUT_CHECKPOINT

    network_config: dict = None

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using processing engine: {device}")
    return device

def build_save_set_network_config(cfg: TrainConfig):
    num_classes_dict = prepare_pipeline_and_save_jsonl(cfg.dataset_count, cfg.dataset_path)

    with open(cfg.network_config_path, "w", encoding="utf-8") as f:
        json.dump(num_classes_dict, f, indent=4)

    with open(cfg.network_config_path, "r", encoding="utf-8") as f:
        final_network_config = json.load(f)

    cfg.network_config = final_network_config
    print("✓ The head size configuration has been successfully saved to disk.")
    return final_network_config, num_classes_dict

def build_dataloaders(cfg: TrainConfig, num_stats, dataset):
    train_size = int(cfg.train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader

def build_model(cfg: TrainConfig, device):
    model = GermanInsuranceClassifier(model_name=cfg.model_name, network_config=cfg.network_config)
    model.to(device)

    criterion = DynamicMultiHeadLoss(target_fields=list(cfg.network_config.keys()))
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    return model, criterion, optimizer

def train_step(batch, model, criterion, optimizer, device, max_grad_norm, network_config):
    optimizer.zero_grad()
            
    # Route core linguistic components to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
            
    # Group text extractors targets cleanly
    batch_targets = {}
    for field in network_config.keys():
        batch_targets[f"label_{field}"] = batch[f"label_{field}"].to(device)
                
    # 1. Direct Forward Pass (Compute text predictions across all 12 heads)
    predictions = model(input_ids, attention_mask)
            
    # 2. Extract Combined Structural Loss Matrix
    loss, loss_dict = criterion(predictions, batch_targets)
            
    # 3. Backpropagation Pass
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
        # Monitoring checkpoints
    
    return running_loss / len(train_loader)

def evaluate_epoch(val_loader, model, criterion, device, epoch, cfg: TrainConfig):
    avg_loss, loss_dict, mae_dict, acc_dict = evaluate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        target_fields=cfg.network_config
    )

    print(f"\n📊 Epoch {epoch+1} validation")

    print(f"val_loss: {avg_loss:.4f}")

    for field in cfg.network_config.keys():
        if cfg.network_config[field] == 1:
            print(f"{field}: MAE={mae_dict.get(field, 0.0):.4f}")
        else:
            print(f"{field}: ACC={acc_dict.get(field, 0.0):.4f}")

    return avg_loss, loss_dict, mae_dict, acc_dict

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

        if network_config[field] == 1:
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

    final_network_config, num_classes_dict = build_save_set_network_config(cfg)
    
    num_stats = get_descriptive_statistics_for_numeric(pd.read_csv(cfg.input_file))

    dataset = GermanInsuranceDataset(jsonl_path=cfg.dataset_path, numeric_stats=num_stats)
    
    train_loader, val_loader = build_dataloaders(cfg, dataset)

    model, criterion, optimizer = build_model(cfg, device)

    print("\nTraining started")

    for epoch in range(cfg.epochs):
        avg_loss = train_epoch(train_loader, model, criterion, optimizer, device, epoch, cfg)

        avg_loss, loss_dict, mae_dict, acc_dict = evaluate_epoch(val_loader, model, criterion, device, epoch, cfg)
        report = build_report(cfg.network_config, loss_dict, mae_dict, acc_dict)
        print(report)

    save_checkpoint(cfg, model, optimizer, cfg.epochs-1, avg_loss, num_stats)
    print(f"\n✓ Training completed. Model checkpoint saved to: {cfg.output_checkpoint}")
    
    
    