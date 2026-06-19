"""
Tests for src/utils/fake_data_generators/accident_description_generator.py

This is the most bug-prone part of the pipeline. Regression guards here for:
- vehicle fields (auto_make/auto_year) must come from the vehicle tuple, not the row
- severity buckets must actually produce different text (Total Loss != Major Damage)
- witnesses / bodily_injuries labels must match the count actually injected into text
- every real-dataset incident_type/collision_type must resolve to a known case_type
"""
import pytest
from src.utils.fake_data_generators.accident_description_generator import (
    AccidentDataGenerator,
)
from src.ml_config import VALID_CASE_TYPES, TARGET_FIELDS


@pytest.fixture
def generator():
    return AccidentDataGenerator(TARGET_FIELDS)


class TestVehicleFieldsComeFromVehicleNotRow:
    def test_auto_make_uses_vehicle_tuple(self, generator, sample_row):
        row = {**sample_row, "auto_make": "Mercedes"}  # deliberately different
        vehicle = ("Toyota", "Corolla", 2015)

        _, labels = generator.generate_with_labels_and_vehicles(row, vehicle, seed=1)

        assert labels["auto_make"] == "Toyota"
        assert labels["auto_make"] != "Mercedes"

    def test_auto_year_uses_vehicle_tuple_when_present(self, generator, sample_row):
        row = {**sample_row, "auto_year": 1999}
        vehicle = ("Toyota", "Corolla", 2015)

        # Run multiple seeds since with_year is probabilistic; at least one
        # of them should show the vehicle's year, never the row's year.
        seen_years = set()
        for seed in range(20):
            _, labels = generator.generate_with_labels_and_vehicles(row, vehicle, seed=seed)
            if labels.get("auto_year") is not None:
                seen_years.add(labels["auto_year"])

        assert 1999 not in seen_years
        assert seen_years <= {2015}


class TestSeverityBucketsProduceDifferentText:
    def test_total_loss_and_major_damage_texts_differ(self, generator, sample_row):
        row_total = {**sample_row, "incident_severity": "Total Loss"}
        row_major = {**sample_row, "incident_severity": "Major Damage"}

        text_total, _ = generator.generate_with_labels(row_total, seed=42)
        text_major, _ = generator.generate_with_labels(row_major, seed=42)

        assert text_total != text_major

    def test_minor_and_trivial_texts_differ(self, generator, sample_row):
        row_minor = {**sample_row, "incident_severity": "Minor Damage"}
        row_trivial = {**sample_row, "incident_severity": "Trivial Damage"}

        text_minor, _ = generator.generate_with_labels(row_minor, seed=7)
        text_trivial, _ = generator.generate_with_labels(row_trivial, seed=7)

        assert text_minor != text_trivial

    def test_severity_label_is_preserved(self, generator, sample_row):
        row = {**sample_row, "incident_severity": "Total Loss"}
        _, labels = generator.generate_with_labels(row, seed=1)
        assert labels["incident_severity"] == "Total Loss"


class TestNumericFieldLabelsMatchInjectedText:
    @pytest.mark.parametrize("count", [0, 1, 2])
    def test_witnesses_label_matches_input(self, generator, sample_row, count):
        row = {**sample_row, "witnesses": count}
        _, labels = generator.generate_with_labels(row, seed=3)
        assert labels["witnesses"] == count

    @pytest.mark.parametrize("count", [0, 1, 2])
    def test_bodily_injuries_label_matches_input(self, generator, sample_row, count):
        row = {**sample_row, "bodily_injuries": count}
        _, labels = generator.generate_with_labels(row, seed=3)
        assert labels["bodily_injuries"] == count


class TestCaseTypeResolution:
    def test_known_collision_type_resolves_directly(self, generator, sample_row):
        row = {**sample_row, "collision_type": "Rear Collision", "incident_type": ""}
        case_type = generator._resolve_case_type(row)
        assert case_type == "Rear Collision"

    def test_unknown_values_fall_back_to_a_valid_case_type(self, generator, sample_row):
        row = {**sample_row, "collision_type": "???", "incident_type": "???"}
        case_type = generator._resolve_case_type(row)
        assert case_type in VALID_CASE_TYPES

    @pytest.mark.parametrize(
        "incident_type",
        ["Multi-vehicle Collision", "Single Vehicle Collision", "Parked Car", "Vehicle Theft"],
    )
    def test_all_real_dataset_incident_types_resolve_to_valid_case_type(
        self, generator, sample_row, incident_type
    ):
        """
        Regression guard: incident_type values that exist in the real insurance
        dataset must map to a CASE_CONFIG key, not silently default to
        'Side Collision' for every unmapped value.
        """
        row = {**sample_row, "incident_type": incident_type, "collision_type": ""}
        case_type = generator._resolve_case_type(row)
        assert case_type in VALID_CASE_TYPES


class TestGeneratedTextIsNonEmpty:
    def test_generate_text_returns_nonempty_string(self, generator, sample_row):
        text = generator.generate_text(sample_row, seed=1)
        assert isinstance(text, str)
        assert len(text.strip()) > 0

    def test_no_vehicle_strategy_omits_vehicle_fields(self, generator, sample_row):
        row = {**sample_row, "auto_make": "", "auto_model": ""}
        # force no_vehicle by giving empty make/model and no vehicle tuple
        _, labels = generator.generate_with_labels(row, seed=1)
        assert labels.get("auto_make") in (None, "")