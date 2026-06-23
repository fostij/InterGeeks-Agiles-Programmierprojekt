# Multi-Head Insurance Classifier — Documentation

---

## 1. Purpose

A multi-head BERT-based model that reconstructs structured insurance claim fields from free-form German accident descriptions. The output table is intended as input for a downstream regression model predicting `vehicle_claim`.

---

## 2. Input Data

**Training:** JSONL file where each line contains:
- `text` — generated German accident description
- `labels` — dict of encoded field values (`-100` = field absent from text)

**Inference:** One or more raw German text strings (user-written or generated).

**Label encoding:** All fields are classification. Integer class indices are produced by `LabelEncoder` per field. Absent fields are encoded as `None` → stored as `-100` (ignored by `CrossEntropyLoss`).

---

## 3. Output Data

**Training:** PyTorch checkpoint (`.pt`) containing:
- `model_state_dict`
- `network_config` — field names + number of classes per head
- `label_encoders` — `{field: [class_0, class_1, ...]}` for decoding
- `epoch`, `loss`

**Inference:** `pd.DataFrame` with one row per input text. Each column is a predicted field value (decoded to original string/int). Fields where model confidence < `clf_threshold` are returned as `None`.

---

## 4. Architecture

Base model: `uklfr/gottbert-base` (German RoBERTa).

Pooling: mean pooling over token embeddings weighted by attention mask.

Each field has an independent head:
```
Linear(hidden, hidden//2) → GELU → Dropout → Linear(hidden//2, num_classes)
```

All heads share the same BERT encoder.

---

## 5. Training

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 16 | Use ≥16 for stable gradients across 14 heads |
| `lr` | 2e-5 | Standard for BERT fine-tuning |
| `epochs` | 15 | With linear warmup scheduler |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `train_split` | 0.85 | Train/val split |
| `dataset_count` | 5000+ | Minimum recommended |
| `warmup_steps` | 10% of total | Linear warmup then linear decay |

**Loss:** `CrossEntropyLoss(ignore_index=-100)` per head, summed with per-field weights (`loss_weight` in `network_config`, default 1.0).

**Optimizer:** AdamW with `weight_decay=0.01`.

**Checkpoint:** saved when `partial_score` improves.

---

## 6. Metrics

Computed on validation set after each epoch.

| Metric | Description |
|--------|-------------|
| `val_loss` | Weighted sum of per-field cross-entropy losses |
| `accuracy` per field | Fraction of correctly predicted classes (masked) |
| `exact_match` | Fraction of samples where ALL fields are correct |
| `partial_score` | Average fraction of correct fields per sample |

Fields with label `-100` are excluded from all metric computations.

---

## 7. Exact Match

A sample counts as exact match only if every present field is predicted correctly.

```
exact_match = |{samples where all fields correct}| / |{samples with ≥1 field}|
```

Strict metric — useful for tracking overall reconstruction quality. Typically lower than partial score.

---

## 8. Partial Score

Per-sample fraction of correctly predicted fields, averaged over all samples:

```
partial_score = mean over samples( correct_fields / total_present_fields )
```

More informative than exact match during early training. Used as the primary checkpoint criterion.

---

## 9. Error Analysis

`ErrorAnalyzer` collects misclassified samples during evaluation and prints the top-N errors per field:

```
===== ERROR ANALYSIS =====
[incident_severity] — 12 misclassified
  pred=2  true=0
  pred=1  true=3
```

Useful for identifying which fields the model struggles with and whether errors are systematic (e.g. always confusing two adjacent classes).

---

## 10. Usage

**Training:**
```python
from src.models.multi_head_insurance.interface import train
from src.models.multi_head_insurance.train import TrainConfig

cfg = TrainConfig(
    dataset_count=5000,
    batch_size=16,
    epochs=15,
    dataset_path="data/dataset.jsonl",
    network_config_path="data/network_config.json",
    output_checkpoint="checkpoints/last.pt",
    best_checkpoint="checkpoints/best.pt",
)
train(cfg)
```

**Inference:**
```python
from src.models.multi_head_insurance.interface import predict

df = predict(
    "Ich war mit meinem Ford Escape unterwegs. Ein anderes Fahrzeug fuhr von hinten auf.",
    "Mein BMW wurde bei starkem Regen seitlich gerammt.",
)
df.to_csv("reconstructed_table.csv", index=False)
```

**Confidence threshold:** `clf_threshold=0.6` by default. Lower it if too many `None` values appear; raise it if incorrect predictions are common. Inspect per-field confidence distribution on the validation set to calibrate.

---

## 11. Limitations

- **Unseen vehicle makes/models:** handled by generating training data from a vehicle catalog. New makes not in training data will be misclassified.
- **Absent fields → None:** when a field is not mentioned in the text and model confidence is below threshold, the field is `None`.
- **Year in text:** `auto_year` is a classification head over known years. Years outside the training distribution will map to the nearest known class.
- **Language:** model is trained on German text only. English or mixed-language input will degrade performance significantly.
- **Text length:** inputs are truncated to `max_len=256` tokens. Very long descriptions may lose tail information.
