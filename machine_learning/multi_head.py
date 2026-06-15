import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import AdamW
from machine_learning.german_insurance_classifier import GermanInsuranceDataset, GermanInsuranceClassifier, DynamicMultiHeadLoss
from utils.labels_encoder import prepare_pipeline_and_save_jsonl
from utils.discreptive_statistic import get_descriptive_statistics_for_numeric
import pandas as pd
import json
from ml_config import TARGET_FIELDS, DATASET_PATH, NETWORK_CONFIG_PATH, INPUT_FILE, NUMERIC_FIELDS
from machine_learning.eval import evaluate

def run_training():
    # --- 1. Execution Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using processing engine: {device}")

    # Pass the exact num_classes_dict that your data preparation script printed out
    # Example shape mapping:
    num_classes_dict = prepare_pipeline_and_save_jsonl(count=1000, output_file=DATASET_PATH)

    with open(NETWORK_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(num_classes_dict, f, indent=4)
    print("✓ Конфигурация размеров голов успешно сохранена на диск.")

    with open(NETWORK_CONFIG_PATH, "r", encoding="utf-8") as f:
        final_network_config = json.load(f)

    # --- 2. Initialize Data Iterators ---
    num_stats = get_descriptive_statistics_for_numeric(pd.read_csv(INPUT_FILE))
    dataset = GermanInsuranceDataset(jsonl_path=DATASET_PATH, numeric_stats=num_stats)
    # Start with a conservative batch size (e.g. 4 or 8) to avoid memory crashes

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # --- 3. Construct the Network ---
    # The model setup we built earlier uses Mean Pooling over GottBERT's last hidden state
    model = GermanInsuranceClassifier(model_name="uklfr/gottbert-base", num_classes_dict=final_network_config)
    model.to(device)

    # Set up losses: Classification ignores nothing (None is an active class index), 
    # Numeric targets ignore -100 using manual mask logic inside our loss configuration
    criterion = DynamicMultiHeadLoss(target_fields=TARGET_FIELDS)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # --- 4. The Active Pipeline Loop ---
    epochs = 1
    model.train()

    print("\n🚀 Commencing training loop activation...")
    for epoch in range(epochs):
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Route core linguistic components to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Group text extractors targets cleanly
            batch_targets = {}
            for field in TARGET_FIELDS:
                batch_targets[f"label_{field}"] = batch[f"label_{field}"].to(device)
                
            # 1. Direct Forward Pass (Compute text predictions across all 12 heads)
            predictions = model(input_ids, attention_mask)
            
            # 2. Extract Combined Structural Loss Matrix
            loss, loss_dict = criterion(predictions, batch_targets)
            
            # 3. Backpropagation Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping stabilizer
            optimizer.step()
            
            running_loss += loss.item()
           # if batch_idx in [0,5,10]:
            #    for field, field_loss in loss_dict.items():
            #        print(f"    {field}: {field_loss:.4f}")
            # Monitoring checkpoints
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")
                
        print(f"🏁 Epoch {epoch+1} Complete. Average running loss parameter: {running_loss / len(train_loader):.4f}\n")

    print("Training cycle finalized successfully. Your multi-head extractor is now optimized.")
    
    val_loss, loss_dict, mae_dict, acc_dict = evaluate(
        model,
        val_loader,
        criterion,
        device,
        numeric_fields=NUMERIC_FIELDS,
        target_fields=TARGET_FIELDS
    )

    print("\n===== EVALUATION REPORT =====")

    report = []

    for field in TARGET_FIELDS:
        row = {"field": field}

        if field in NUMERIC_FIELDS:
            row["MAE"] = mae_dict.get(field, None)
            row["accuracy"] = None
        else:
            row["MAE"] = None
            row["accuracy"] = acc_dict.get(field, None)

        row["loss"] = loss_dict.get(field, None)

        report.append(row)

    df = pd.DataFrame(report)

    print(df)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": running_loss / len(train_loader),
        "num_classes_dict": num_classes_dict,
        "target_fields": TARGET_FIELDS,
        "numeric_stats": num_stats
    }

    torch.save(checkpoint, "data/output/model.pt")
