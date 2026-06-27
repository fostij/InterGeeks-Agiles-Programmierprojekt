"""
# Konfiguration für das Multi-Head-Klassifikationsmodell
"""

# Trainingsparameter
from ml_config import MULTI_HEAD_MODEL_PATH


BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
EPOCHS          = 10
MAX_GRAD_NORM   = 1.0
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1   # Anteil der Gesamtschritte für den Warmup (10%)

# Datensatz-Split
TRAIN_SPLIT     = 0.8
VAL_SPLIT       = 0.1
TEST_SPLIT      = 0.1
SPLIT_SEED      = 42

# Datengenerierung
DATASET_COUNT   = 20000

# Modell
MODEL_NAME      = "uklfr/gottbert-base"

# Checkpoint-Tracking
TRACKER_METRIC  = "partial_score"   # "partial_score" | "exact_match" | "val_loss"
TRACKER_MODE    = "max"             # "max" für Accuracy-Metriken, "min" für Loss


DEFAULT_DATASET_PATH        = "data/processed/train_dataset.jsonl"
DEFAULT_NETWORK_CONFIG_PATH = "data/processed/network_config.json"
DEFAULT_CHECKPOINT_PATH     = str(MULTI_HEAD_MODEL_PATH)
