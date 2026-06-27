from pathlib import Path

def _find_project_root(start: Path) -> Path:
    current = start.resolve()

    while current != current.parent:
        if (current / "data").exists() and (current / "src").exists():
            return current
        current = current.parent

    raise RuntimeError("Project root not found")

_BASE_DIR = _find_project_root(Path(__file__).parent)

INSURANCE_DATASET_PATH = _BASE_DIR / "data" / "raw" / "dataset.csv"
VEHICLE_DATASET_PATH = _BASE_DIR / "data" / "raw" / "vehicle_dataset.csv"

MULTI_HEAD_MODEL_PATH = _BASE_DIR / "src" / "models" / "multi_head_insurance" / "multi_head_model.pt"
CNN_MODEL_PATH = _BASE_DIR / "src" / "models" / "cnn" / "cnn_model.keras"
REGRESSION_MODEL_PATH = _BASE_DIR / "src" / "models" / "regression" / "regression_model.pkl"

SCHEMA_PATH = _BASE_DIR / "sql" / "schema.sql"