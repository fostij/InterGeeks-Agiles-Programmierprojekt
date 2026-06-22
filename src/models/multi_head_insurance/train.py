"""Training des Multi-Head-Klassifikationsmodells für Unfallbeschreibungen.
 
Orchestriert den kompletten Trainingsablauf: Datenaufbereitung, Aufbau
der DataLoader, Trainings-/Evaluierungsschleife über mehrere Epochen,
Tracking des besten Checkpoints anhand der partial_score-Metrik sowie
abschließende Evaluierung auf dem Testdatensatz.
"""

import logging
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
from src.exceptions import DataPreparationError, ModelLoadError, TrainingError
from src.models.multi_head_insurance.losses import DynamicMultiHeadLoss
from src.models.multi_head_insurance.models import GermanInsuranceClassifier
from src.models.multi_head_insurance.dataset import GermanInsuranceDataset
from src.utils.labels_encoder import prepare_pipeline_and_save_jsonl
from src.models.multi_head_insurance.engine import MultiHeadEvaluator, evaluate
from src.utils.data_orchestrator import get_cleaned_dataset
 
logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    """Konfiguration für einen vollständigen Trainingslauf.
 
    Bündelt Hyperparameter, Split-Verhältnisse und Dateipfade. Die Felder
    network_config und label_encoders werden erst zur Laufzeit befüllt
    (siehe build_save_set_network_config()).
    """

    batch_size: int = 4
    lr: float = 2e-5
    epochs: int = 1
    max_grad_norm: float = 1.0
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    split_seed: int = 42
    dataset_count: int = 1000
    model_name: str = "uklfr/gottbert-base"

    dataset_path: str = None
    input_file: str = None
    network_config_path: str = None
    output_checkpoint: str = None
    best_checkpoint: str = None

    network_config: dict[str, dict] = None
    label_encoders: dict[str, LabelEncoder] = None
    
@dataclass
class BestModelTracker:
    """Verfolgt den besten Checkpoint über alle Trainingsepochen hinweg.
 
    Speichert nach jeder Epoche automatisch den Checkpoint, falls sich
    die überwachte Metrik verbessert hat (siehe update()).
    """

    output_checkpoint: str
    metric: str = "partial_score"  # "partial_score" | "exact_match" | "val_loss"
    mode: str = "max"              # "max" for accuracy, "min" for loss

    best_value: float = None
    best_epoch: int = -1

    def is_better(self, value: float) -> bool:
        """Prüft, ob der übergebene Wert den bisher besten Wert übertrifft."""

        if self.best_value is None:
            return True
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def update(self, value: float, epoch: int, model: GermanInsuranceClassifier, optimizer: AdamW, cfg: TrainConfig) -> bool:
        """Aktualisiert den besten Wert und speichert ggf. einen neuen Checkpoint.
 
        Returns:
            True, wenn ein neuer bester Checkpoint gespeichert wurde.
        """

        if self.is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            save_checkpoint(cfg, model, optimizer, epoch, value, path=self.output_checkpoint)
            logger.info("Neuer bester %s=%.4f — Checkpoint gespeichert.", self.metric, value)
            return True
        return False

    def summary(self) -> None:
        """Loggt den besten erreichten Wert und die zugehörige Epoche."""

        logger.info("Bester %s=%.4f bei Epoche %d.", self.metric, self.best_value, self.best_epoch + 1)



def get_device() -> torch.device:
    """Wählt das verfügbare Rechengerät (GPU, falls vorhanden, sonst CPU)."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Verwende Recheneinheit: %s", device)
    return device

def build_save_set_network_config(cfg: TrainConfig, dataset: pd.DataFrame) -> tuple[dict[str, dict], dict[str, LabelEncoder]]:
    """Erstellt die Head-Konfiguration, speichert sie und hängt sie an cfg an.
 
    Args:
        cfg: Trainingskonfiguration; wird mit network_config und
            label_encoders befüllt.
        dataset: Bereinigter Datensatz, aus dem die JSONL-Trainingsdaten
            und die Label-Encoder erzeugt werden.
 
    Returns:
        Tupel (network_config, label_encoders).
 
    Raises:
        DataPreparationError: Wenn die Netzwerkkonfiguration nicht
            gespeichert oder wieder eingelesen werden kann.
    """

    network_config, label_encoders = prepare_pipeline_and_save_jsonl(cfg.dataset_count, cfg.dataset_path, dataset)
    try:
        with open(cfg.network_config_path, "w", encoding="utf-8") as f:
            json.dump(network_config, f, indent=4)

        with open(cfg.network_config_path, "r", encoding="utf-8") as f:
            final_network_config = json.load(f)

    except OSError as exc:
        raise DataPreparationError(f"Netzwerkkonfiguration konnte nicht gespeichert werden: {exc}") from exc

    cfg.network_config = final_network_config
    cfg.label_encoders = label_encoders
    logger.info("Head-Größen-Konfiguration erfolgreich gespeichert.")
    return final_network_config, label_encoders

def build_dataloaders(cfg: TrainConfig, dataset: GermanInsuranceDataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Teilt den Datensatz in Train/Val/Test auf und baut die DataLoader.
 
    Args:
        cfg: Trainingskonfiguration mit den Split-Verhältnissen.
        dataset: GermanInsuranceDataset-Instanz.
 
    Returns:
        Tupel (train_loader, val_loader, test_loader).
 
    Raises:
        ValueError: Wenn die Split-Verhältnisse nicht zu 1.0 aufsummieren
            oder eine der resultierenden Teilmengen leer wäre.
    """

    total = len(dataset)
    split_sum = cfg.train_split + cfg.val_split + cfg.test_split
    if abs(split_sum - 1.0) > 1e-8:
        raise ValueError(f"train/val/test splits must sum to 1.0, got {split_sum}")

    train_size = int(cfg.train_split * total)
    val_size = int(cfg.val_split * total)
    test_size = total - train_size - val_size

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            f"Invalid split sizes for dataset={total}: "
            f"train={train_size}, val={val_size}, test={test_size}"
        )

    generator = torch.Generator().manual_seed(cfg.split_seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def build_model(cfg: TrainConfig, device: torch.device) -> tuple[GermanInsuranceClassifier, DynamicMultiHeadLoss, AdamW]:
    """Baut Modell, Verlustfunktion und Optimierer für das Training auf.
 
    Args:
        cfg: Trainingskonfiguration mit bereits gesetztem network_config.
        device: Zielgerät, auf das das Modell geladen wird.
 
    Returns:
        Tupel (model, criterion, optimizer).
    """

    model = GermanInsuranceClassifier(model_name=cfg.model_name, network_config=cfg.network_config)
    model.to(device)

    criterion = DynamicMultiHeadLoss(target_fields=list(cfg.network_config.keys()), network_config=cfg.network_config)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)

    return model, criterion, optimizer

def train_step(
    batch: dict[str, torch.Tensor],
    model: GermanInsuranceClassifier,
    criterion: DynamicMultiHeadLoss,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    max_grad_norm: float,
    network_config: dict[str, dict],
) -> tuple[float, dict[str, float]]:
    """Führt einen einzelnen Trainingsschritt (Forward + Backward + Update) aus.
 
    Returns:
        Tupel (loss_value, loss_dict) für diesen Batch.
    """

    optimizer.zero_grad()
            
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
            
    batch_targets = {}
    for field in network_config.keys():
        batch_targets[f"label_{field}"] = batch[f"label_{field}"].to(device)
                
    predictions = model(input_ids, attention_mask)
    
    for field, tensor in batch_targets.items():
        max_val = tensor[tensor != -100].max().item() if (tensor != -100).any() else -1
        num_classes = network_config[field.replace("label_", "")]["size"]
        if max_val >= num_classes:
            logger.warning("Ungültiges Label: %s max=%d aber size=%d", field, max_val, num_classes)
    loss, loss_dict = criterion(predictions, batch_targets)
            
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Gradient clipping stabilizer
    optimizer.step()
    scheduler.step()
    return loss.item(), loss_dict
    

def train_epoch(
    train_loader: DataLoader,
    model: GermanInsuranceClassifier,
    criterion: DynamicMultiHeadLoss,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
) -> float:
    """Trainiert das Modell für eine vollständige Epoche.
 
    Returns:
        Durchschnittlicher Trainingsverlust über alle Batches der Epoche.
    """

    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        loss, loss_dict = train_step(batch, model, criterion, optimizer, scheduler, device, cfg.max_grad_norm, cfg.network_config)
        running_loss += loss

        if batch_idx % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                "Epoche [%d/%d] | Schritt [%d/%d] | Loss: %.4f | LR: %.2e",
                epoch + 1, cfg.epochs, batch_idx, len(train_loader), loss, current_lr,
            )
    
    return running_loss / len(train_loader)

def evaluate_epoch(
    loader: DataLoader,
    model: GermanInsuranceClassifier,
    criterion: DynamicMultiHeadLoss,
    device: torch.device,
    epoch: int,
    evaluator: MultiHeadEvaluator,
    split_name: str = "val",
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    """Evaluiert das Modell auf einem gegebenen DataLoader (val oder test).
 
    Returns:
        Tupel (avg_loss, loss_dict, acc_dict, exact_match, partial_score).
    """

    avg_loss, loss_dict, acc_dict, exact, partial = evaluate(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        evaluator=evaluator
    )

    logger.info(
        "Epoche %d | %s_loss: %.4f | exact: %.4f | partial: %.4f",
        epoch + 1, split_name, avg_loss, exact, partial,
    )

    return avg_loss, loss_dict, acc_dict, exact, partial

def save_checkpoint(
    cfg: TrainConfig,
    model: GermanInsuranceClassifier,
    optimizer: AdamW,
    epoch: int,
    loss: float,
    path: str | None = None,
) -> None:
    """Speichert Modell- und Optimierer-Zustand als Checkpoint.
 
    Args:
        cfg: Trainingskonfiguration (liefert network_config, label_encoders
            und ggf. den Standard-Speicherpfad output_checkpoint).
        model: Zu speicherndes Modell.
        optimizer: Zugehöriger Optimierer.
        epoch: Aktuelle Epoche.
        loss: Aktueller Verlustwert (zur Dokumentation im Checkpoint).
        path: Zielpfad. Falls None, wird cfg.output_checkpoint verwendet.
 
    Raises:
        TrainingError: Wenn der Checkpoint nicht geschrieben werden kann.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "network_config": cfg.network_config,
        "label_encoders": {
            field: le.classes_.tolist()
            for field, le in cfg.label_encoders.items()
        }
    }

    save_path = path or cfg.output_checkpoint

    try:
        torch.save(checkpoint, save_path)
    except OSError as exc:
        raise TrainingError(f"Checkpoint konnte nicht gespeichert werden unter {save_path}: {exc}") from exc


def build_report(
    network_config: dict[str, dict],
    loss_dict: dict[str, float],
    acc_dict: dict[str, float],
) -> pd.DataFrame:
    """Baut eine tabellarische Übersicht aus Accuracy und Loss je Feld.
 
    Returns:
        DataFrame mit den Spalten field, accuracy, loss.
    """

    logger.info("Evaluierungsbericht:")

    report = []

    for field in network_config.keys():
        row = {"field": field}
        row["accuracy"] = acc_dict.get(field)
        row["loss"] = loss_dict.get(field, None)

        report.append(row)

    df = pd.DataFrame(report)

    return df

def run_training(cfg: TrainConfig) -> None:
    """Führt den vollständigen Trainingsablauf für das Multi-Head-Modell aus.
 
    Schritte: Daten laden und aufbereiten, DataLoader bauen, Modell
    trainieren und je Epoche validieren, besten Checkpoint tracken,
    abschließend auf dem Testdatensatz evaluieren und den Trainingsverlauf
    als CSV speichern.
 
    Args:
        cfg: Trainingskonfiguration.
 
    Raises:
        DataPreparationError: Wenn die Daten nicht aufbereitet werden können.
        TrainingError: Wenn das Training oder das Speichern eines
            Checkpoints fehlschlägt.
        ModelLoadError: Wenn der beste Checkpoint nach dem Training nicht
            wieder geladen werden kann.
    """

    device = get_device()

    dataset = get_cleaned_dataset()
    _ = build_save_set_network_config(cfg, dataset)

    dataset = GermanInsuranceDataset(jsonl_path=cfg.dataset_path, network_config=cfg.network_config)
    
    train_loader, val_loader, test_loader = build_dataloaders(cfg, dataset)

    model, criterion, optimizer = build_model(cfg, device)

    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    logger.info("Training gestartet.")
    
    evaluator = MultiHeadEvaluator(cfg.network_config)

    tracker = BestModelTracker(
        output_checkpoint=cfg.best_checkpoint,
        metric="partial_score",
        mode="max"
    )

    history = []
    
    for epoch in range(cfg.epochs):
        avg_loss = train_epoch(train_loader, model, criterion, optimizer, scheduler, device, epoch, cfg)

        val_loss, loss_dict, acc_dict, exact, partial = evaluate_epoch(
            val_loader,
            model,
            criterion,
            device,
            epoch,
            evaluator,
            split_name="val",
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "exact": exact,
            "partial": partial,
        }

        for field, value in loss_dict.items():
            row[f"{field}_loss"] = value

        for field, value in acc_dict.items():
            row[f"{field}_acc"] = value

        history.append(row)

        report = build_report(cfg.network_config, loss_dict, acc_dict)
        logger.info("\n%s", report)
        evaluator.error_analyzer.report(top_n=3)

        tracker.update(
            value=partial,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            cfg=cfg,
        )

    tracker.summary()

    try:
        best_checkpoint = torch.load(cfg.best_checkpoint, map_location=device, weights_only=False)
    except Exception as exc:
        raise ModelLoadError(f"Bester Checkpoint konnte nicht geladen werden: {exc}") from exc

    model.load_state_dict(best_checkpoint["model_state_dict"])

    test_loss, test_loss_dict, test_acc_dict, test_exact, test_partial = evaluate_epoch(
        test_loader,
        model,
        criterion,
        device,
        cfg.epochs,
        evaluator,
        split_name="test",
    )

    if history:
        history[-1]["test_loss"] = test_loss
        history[-1]["test_exact"] = test_exact
        history[-1]["test_partial"] = test_partial

        for field, value in test_loss_dict.items():
            history[-1][f"test_{field}_loss"] = value

        for field, value in test_acc_dict.items():
            history[-1][f"test_{field}_acc"] = value

    try:
        pd.DataFrame(history).to_csv("training_history.csv", index=False)
    except OSError as exc:
        logger.error("Trainingsverlauf konnte nicht als CSV gespeichert werden: %s", exc)
 
    logger.info("Training abgeschlossen. Bester Checkpoint gespeichert unter: %s", cfg.best_checkpoint)