Ок. Ниже единая нормальная документация всего пайплайна без раздувания по функциям.

---

# German Insurance Multi-Head Training Pipeline

## Overview

This project implements a multi-task learning pipeline for insurance-related structured prediction over German-language text.

The model jointly learns:

* classification tasks (categorical insurance attributes)
* regression tasks (numeric attributes)
* optional target price regression

The system is built around a shared Transformer encoder with multiple task-specific heads.

---

## Data pipeline

### Input data format

Raw dataset is a pandas DataFrame used to generate training samples.

Each sample is converted into:

```json id="1q9xq3"
{
  "text": "insurance description text",
  "labels": {
    "field_1": "value",
    "field_2": 3,
    ...
  },
  "target_price": 123.45
}
```

---

### Preprocessing

The pipeline performs:

#### 1. Sample generation

A subset of rows is converted into natural language training examples with labels.

#### 2. Label encoding

Each field is processed based on its type:

### Classification fields

* Extract unique values from dataset
* Add explicit `"None"` class
* Encode using `LabelEncoder`

Result:

* integer class IDs
* fixed vocabulary per field

### Regression fields

* values are kept as raw floats
* no encoding at this stage

---

### Output dataset format (JSONL)

Training data is stored as line-delimited JSON:

```json id="8h7q2x"
{
  "text": "...",
  "labels": {
    "field_1": 2,
    "field_2": 0
  },
  "target_price": 99.0
}
```

---

### Network configuration

During preprocessing, a `network_config` is generated:

```python id="k3v9pl"
{
  "field_name": {
    "type": "classification" | "regression",
    "size": int
  }
}
```

This config defines:

* number of output neurons per head
* task type per field

It is used consistently across:

* model architecture
* loss computation
* evaluation logic

---

## Model architecture

### Encoder

* Transformer backbone (`AutoModel`, e.g. GottBERT)
* masked mean pooling over token embeddings

### Multi-head output

For each field:

* separate linear layer (task-specific head)

Output:

```python id="m2x8aa"
{
  field_1: logits or regression value,
  field_2: logits or regression value
}
```

---

## Training objective

The model is trained using a dynamic multi-task loss.

### Classification

* CrossEntropyLoss

### Regression

* MSELoss with masking
* missing values encoded as `-100`

Loss is computed per field and summed:

```text id="p7q1kd"
total_loss = sum(field_losses)
```

---

## Evaluation system

The evaluation module supports multi-level analysis:

### Per-field metrics

* Accuracy (classification)
* MAE (regression)

### Global metrics

* Exact match (all fields correct simultaneously)
* Partial score (fraction of correctly predicted fields per sample)
* Validation loss

---

### Masking logic

Missing labels are represented as:

```text id="mask"
-100
```

These values are excluded from:

* loss computation
* metric calculation

---

## Training process

### Pipeline flow

1. Load raw dataset
2. Build label encoders + network config
3. Generate JSONL training dataset
4. Create Dataset + DataLoader
5. Initialize model, loss, optimizer
6. Train epoch loop
7. Validate after each epoch

---

### Optimization details

* Optimizer: AdamW
* Gradient clipping (stability control)
* Train/validation split
* Mixed task optimization (classification + regression)

---

## Checkpointing

The system tracks the best model based on a selected metric:

* partial_score (default)
* exact_match
* validation loss

Saved checkpoint includes:

* model weights
* optimizer state
* epoch
* network config
* numeric statistics

---

## Outputs

After training:

* best model checkpoint
* final checkpoint
* training history (CSV)
* evaluation report per field

---

## Key design characteristics

* Multi-task learning (shared encoder, separate heads)
* Dynamic architecture from dataset (`network_config`)
* Hybrid targets (classification + regression)
* Masked learning for missing labels
* Consistent encoding pipeline
* Structured evaluation beyond simple accuracy

---

## Pipeline summary

```text id="flow"
Raw DataFrame
   ↓
Preprocessing + encoding
   ↓
JSONL dataset + network_config
   ↓
Dataset + DataLoader
   ↓
Transformer encoder
   ↓
Multi-head outputs
   ↓
Multi-task loss
   ↓
Evaluation (exact / partial / per-field metrics)
   ↓
Best checkpoint saved
```

---

To start training, you need to create a `TrainConfig` instance with the required file paths and parameters, and then pass it to `run_training`.

Example:

```python
from src.models.multi_head_insurance.train import TrainConfig, run_training
from ml_config import BEST_CHECKPOINT, DATASET_PATH, NETWORK_CONFIG_PATH, INPUT_FILE, OUTPUT_CHECKPOINT

cfg = TrainConfig(
   dataset_path=DATASET_PATH,
   input_file=INPUT_FILE,
   network_config_path=NETWORK_CONFIG_PATH,
   output_checkpoint=OUTPUT_CHECKPOINT,
   best_checkpoint=BEST_CHECKPOINT,
   epochs=3,
   dataset_count=1000,
)
run_training(cfg)
```

This initializes the training pipeline with:

* dataset location
* network configuration path
* checkpoint outputs
* training hyperparameters

and then starts the full training process.