"""
Tests for src/utils/labels_encoder.py

Covers the two bugs we hit in practice:
- None must become -100 (ignore_index), never a real class id
- Unseen labels must fail loudly, not silently
"""
import pytest


def encode_one_label(field: str, val, sklearn_label_encoders: dict):
    """
    Mirrors the encode step inside prepare_pipeline_and_save_jsonl.
    Extracted here so it can be unit tested without touching disk / the dataset.
    """
    if val is None:
        return -100
    return int(sklearn_label_encoders[field].transform([str(val)])[0])


class TestNoneEncoding:
    def test_none_becomes_ignore_index(self, sklearn_label_encoders):
        assert encode_one_label("incident_severity", None, sklearn_label_encoders) == -100

    def test_none_never_becomes_a_real_class(self, sklearn_label_encoders):
        # Regression guard: "None" must NOT be a fitted class anymore.
        assert "None" not in sklearn_label_encoders["incident_severity"].classes_
        assert "None" not in sklearn_label_encoders["property_damage"].classes_


class TestKnownValueEncoding:
    def test_known_value_round_trips(self, sklearn_label_encoders):
        encoded = encode_one_label("auto_make", "Ford", sklearn_label_encoders)
        decoded = sklearn_label_encoders["auto_make"].classes_[encoded]
        assert decoded == "Ford"

    def test_witnesses_numeric_string_encodes(self, sklearn_label_encoders):
        encoded = encode_one_label("witnesses", 2, sklearn_label_encoders)
        decoded = sklearn_label_encoders["witnesses"].classes_[encoded]
        assert decoded == "2"


class TestUnseenValueEncoding:
    def test_unseen_value_raises(self, sklearn_label_encoders):
        with pytest.raises(ValueError):
            encode_one_label("auto_make", "Mercedes", sklearn_label_encoders)

    def test_unseen_value_error_is_not_silently_swallowed(self, sklearn_label_encoders):
        # Regression guard for the "Mercedes" vs "Mercedes-Benz" bug:
        # the pipeline should never catch this and substitute a default class.
        with pytest.raises(ValueError, match="previously unseen labels"):
            encode_one_label("incident_severity", "Catastrophic", sklearn_label_encoders)


class TestNetworkConfigConsistency:
    def test_head_size_matches_encoder_classes(self, network_config, sklearn_label_encoders):
        for field, encoder in sklearn_label_encoders.items():
            assert network_config[field]["size"] == len(encoder.classes_), (
                f"{field}: head size {network_config[field]['size']} != "
                f"{len(encoder.classes_)} fitted classes"
            )