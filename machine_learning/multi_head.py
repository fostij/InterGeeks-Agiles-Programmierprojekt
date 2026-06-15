import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from machine_learning.german_insurance_classifier import GermanInsuranceDataset, GermanInsuranceClassifier, DynamicMultiHeadLoss
from utils.labels_encoder import prepare_pipeline_and_save_jsonl
import json

def run_training():
    # --- 1. Execution Setup ---
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using processing engine: {device}")

    # TARGET_FIELDS list as defined in your prompt layout
    TARGET_FIELDS = [
        "auto_year", "auto_make", "auto_model", "incident_type", 
        "incident_severity", "incident_city", "incident_state",
        "collision_type", "property_damage", "witnesses",
        "authorities_contacted", "number_of_vehicles_involved"
    ]

    # Provide your generated file paths here
    DATASET_PATH = "./data/output/output.jsonl" 

    # Pass the exact num_classes_dict that your data preparation script printed out
    # Example shape mapping:
    num_classes_dict = prepare_pipeline_and_save_jsonl(count=1000, output_file=DATASET_PATH)

    with open("network_config.json", "w", encoding="utf-8") as f:
        json.dump(num_classes_dict, f, indent=4)
    print("✓ Конфигурация размеров голов успешно сохранена на диск.")

    with open("network_config.json", "r", encoding="utf-8") as f:
        final_network_config = json.load(f)

    # --- 2. Initialize Data Iterators ---
    dataset = GermanInsuranceDataset(jsonl_path=DATASET_PATH)
    # Start with a conservative batch size (e.g. 4 or 8) to avoid memory crashes
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # --- 3. Construct the Network ---
    # The model setup we built earlier uses Mean Pooling over GottBERT's last hidden state
    model = GermanInsuranceClassifier(model_name="uklfr/gottbert-base", num_classes_dict=final_network_config)
    model.to(device)

    # Set up losses: Classification ignores nothing (None is an active class index), 
    # Numeric targets ignore -100 using manual mask logic inside our loss configuration
    criterion = DynamicMultiHeadLoss(target_fields=TARGET_FIELDS)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # --- 4. The Active Pipeline Loop ---
    epochs = 3
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
            loss = criterion(predictions, batch_targets)
            
            # 3. Backpropagation Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping stabilizer
            optimizer.step()
            
            running_loss += loss.item()
            
            # Monitoring checkpoints
            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx}/{len(train_loader)}] | Current Loss: {loss.item():.4f}")
                
        print(f"🏁 Epoch {epoch+1} Complete. Average running loss parameter: {running_loss / len(train_loader):.4f}\n")

    print("Training cycle finalized successfully. Your multi-head extractor is now optimized.")
