import torch
from transformers import AutoTokenizer
from src.models.multi_head_insurance.models import GermanInsuranceClassifier
from src.ml_config import BEST_CHECKPOINT

print("Step 1: Loading model...", flush=True)
device = torch.device("cpu")

checkpoint = torch.load(BEST_CHECKPOINT, map_location=device, weights_only=False)
network_config = checkpoint["network_config"]
numeric_stats = checkpoint["numeric_stats"]
label_encoders = {
    field: {i: val for i, val in enumerate(classes)}
    for field, classes in checkpoint["label_encoders"].items()
}

model = GermanInsuranceClassifier(network_config=network_config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print("Step 2: Model loaded successfully", flush=True)

print("Step 3: Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("uklfr/gottbert-base")
print("Step 4: Tokenizer loaded successfully", flush=True)

text = "Ich war gestern Abend mit meinem Ford Escape unterwegs."
print(f"Step 5: Tokenizing text: {text[:50]}...", flush=True)

encoding = tokenizer(
    [text], padding="max_length", truncation=True, max_length=256, return_tensors="pt"
)
print("Step 6: Tokenization complete", flush=True)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)
print(f"Step 7: Input shape: {input_ids.shape}", flush=True)

print("Step 8: Running model inference...", flush=True)
with torch.no_grad():
    predictions = model(input_ids, attention_mask)
print("Step 9: Inference complete", flush=True)

print(f"Step 10: Predictions keys: {predictions.keys()}", flush=True)
for field, pred in predictions.items():
    print(f"  {field}: shape={pred.shape}", flush=True)

print("DEBUG COMPLETE - All steps successful!")
