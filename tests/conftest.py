import pytest
import torch
from sklearn.preprocessing import LabelEncoder


@pytest.fixture
def network_config():
    return {
        "incident_severity": {"type": "classification", "size": 4},
        "auto_make": {"type": "classification", "size": 3},
        "witnesses": {"type": "classification", "size": 4},
        "property_damage": {"type": "classification", "size": 2},
        "collision_type": {"type": "classification", "size": 3},
    }


@pytest.fixture
def sklearn_label_encoders():
    """
    Raw sklearn LabelEncoder objects, as produced inside prepare_pipeline_and_save_jsonl
    (fit on training data, used via .transform()). Used by test_labels_encoder.py.
    """
    encoders = {}

    le = LabelEncoder()
    le.fit(["Total Loss", "Major Damage", "Minor Damage", "Trivial Damage"])
    encoders["incident_severity"] = le

    le = LabelEncoder()
    le.fit(["Ford", "Toyota", "BMW"])
    encoders["auto_make"] = le

    le = LabelEncoder()
    le.fit(["0", "1", "2", "3"])
    encoders["witnesses"] = le

    le = LabelEncoder()
    le.fit(["NO", "YES"])
    encoders["property_damage"] = le

    le = LabelEncoder()
    le.fit(["Front Collision", "Rear Collision", "Side Collision"])
    encoders["collision_type"] = le

    return encoders


@pytest.fixture
def label_encoders(sklearn_label_encoders):
    """
    Decoded {field: {int_index: original_value}} dicts, matching exactly what
    load_model() reconstructs from a checkpoint's saved `le.classes_.tolist()`.
    Used by test_inference.py and anywhere predict() is called directly.
    """
    return {
        field: {i: val for i, val in enumerate(le.classes_)}
        for field, le in sklearn_label_encoders.items()
    }


@pytest.fixture
def sample_row():
    """A minimal insurance row used to build a GenerationContext in generator tests."""
    return {
        "incident_type": "Rear Collision",
        "collision_type": "Rear Collision",
        "incident_severity": "Total Loss",
        "number_of_vehicles_involved": 2,
        "witnesses": 2,
        "bodily_injuries": 0,
        "property_damage": "NO",
        "police_report_available": "YES",
        "auto_make": "",
        "auto_model": "",
        "auto_year": "",
    }


@pytest.fixture
def make_logits():
    """Factory to build a one-hot-ish logits tensor for a given predicted class index."""

    def _make(predicted_idx: int, num_classes: int, batch_size: int = 1):
        logits = torch.zeros(batch_size, num_classes)
        logits[:, predicted_idx] = 10.0
        return logits

    return _make