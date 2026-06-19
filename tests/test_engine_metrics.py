"""
Tests for src/models/multi_head_insurance/engine.py

Covers:
- partial_score / exact_match correctness on known inputs
- -100 fields are excluded from both numerator and denominator
- a fully-correct sample scores 1.0, a fully-wrong sample scores 0.0
"""
import torch
import pytest
from src.models.multi_head_insurance.engine import MultiHeadEvaluator


@pytest.fixture
def evaluator(network_config):
    return MultiHeadEvaluator(network_config)


class TestPartialScore:
    def test_all_correct_scores_one(self, evaluator, network_config, make_logits):
        predictions = {}
        targets = {}
        for field, cfg in network_config.items():
            predictions[field] = make_logits(predicted_idx=0, num_classes=cfg["size"])
            targets[f"label_{field}"] = torch.tensor([0])

        score = evaluator.partial_score(predictions, targets)
        assert score == pytest.approx(1.0)

    def test_all_wrong_scores_zero(self, evaluator, network_config, make_logits):
        predictions = {}
        targets = {}
        for field, cfg in network_config.items():
            predictions[field] = make_logits(predicted_idx=0, num_classes=cfg["size"])
            targets[f"label_{field}"] = torch.tensor([1])  # never matches predicted 0

        score = evaluator.partial_score(predictions, targets)
        assert score == pytest.approx(0.0)

    def test_masked_field_excluded_from_denominator(self, evaluator, network_config, make_logits):
        # one correct field, one masked field -> score should still be 1.0,
        # not 0.5 (which would happen if -100 were counted as wrong)
        fields = list(network_config.keys())
        f0, f1 = fields[0], fields[1]

        predictions = {f0: make_logits(0, network_config[f0]["size"])}
        targets = {f"label_{f0}": torch.tensor([0])}

        for field in fields[1:]:
            predictions[field] = make_logits(0, network_config[field]["size"])
            targets[f"label_{field}"] = torch.tensor([-100])

        score = evaluator.partial_score(predictions, targets)
        assert score == pytest.approx(1.0)


class TestExactMatch:
    def test_sample_with_one_wrong_field_fails_exact_match(
        self, evaluator, network_config, make_logits
    ):
        predictions = {}
        targets = {}
        fields = list(network_config.keys())

        for field in fields:
            predictions[field] = make_logits(0, network_config[field]["size"])
            targets[f"label_{field}"] = torch.tensor([0])

        # break just one field
        targets[f"label_{fields[0]}"] = torch.tensor([1])

        exact = evaluator.exact_match(predictions, targets)
        assert exact == pytest.approx(0.0)

    def test_sample_fully_correct_counts_as_exact_match(
        self, evaluator, network_config, make_logits
    ):
        predictions = {}
        targets = {}
        for field, cfg in network_config.items():
            predictions[field] = make_logits(0, cfg["size"])
            targets[f"label_{field}"] = torch.tensor([0])

        exact = evaluator.exact_match(predictions, targets)
        assert exact == pytest.approx(1.0)

    def test_sample_with_no_present_fields_is_excluded(
        self, evaluator, network_config, make_logits
    ):
        predictions = {
            field: make_logits(0, cfg["size"]) for field, cfg in network_config.items()
        }
        targets = {f"label_{field}": torch.tensor([-100]) for field in network_config}

        # No samples have any present field -> denominator should be 0,
        # the function must not divide by zero / must return a sane default.
        exact = evaluator.exact_match(predictions, targets)
        assert exact == pytest.approx(0.0)


class TestFieldEvaluatorAccuracy:
    def test_accuracy_computed_correctly(self, network_config, make_logits):
        evaluator = MultiHeadEvaluator(network_config)
        field = "incident_severity"
        size = network_config[field]["size"]

        pred = make_logits(0, size, batch_size=4)
        true = torch.tensor([0, 0, 1, 1])  # 2/4 correct
        mask = torch.tensor([True, True, True, True])

        evaluator.fields[field].update(pred, true, mask)
        result = evaluator.fields[field].compute()
        assert result == pytest.approx(0.5)

    def test_reset_clears_state(self, network_config, make_logits):
        evaluator = MultiHeadEvaluator(network_config)
        field = "incident_severity"
        size = network_config[field]["size"]

        pred = make_logits(0, size, batch_size=2)
        true = torch.tensor([0, 1])
        evaluator.fields[field].update(pred, true, torch.tensor([True, True]))

        evaluator.fields[field].reset()
        result = evaluator.fields[field].compute()
        assert result == 0.0