"""
Tests for src/models/multi_head_insurance/losses.py

Covers:
- ignore_index=-100 correctly excludes absent fields from the loss
- an all -100 batch must not produce nan (regression guard)
- per-field weights are applied
"""
import torch
import pytest
from src.models.multi_head_insurance.losses import DynamicMultiHeadLoss


@pytest.fixture
def criterion(network_config):
    return DynamicMultiHeadLoss(
        target_fields=list(network_config.keys()), network_config=network_config
    )


class TestIgnoreIndex:
    def test_all_masked_batch_does_not_nan(self, criterion, network_config):
        predictions = {
            field: torch.randn(4, cfg["size"]) for field, cfg in network_config.items()
        }
        targets = {
            f"label_{field}": torch.full((4,), -100, dtype=torch.long)
            for field in network_config
        }

        loss, loss_dict = criterion(predictions, targets)

        assert not torch.isnan(loss), "Loss is nan when an entire batch is -100"
        for field, value in loss_dict.items():
            assert value == 0.0 or not torch.isnan(torch.tensor(value))

    def test_masked_samples_do_not_affect_gradient(self, criterion, network_config):
        field = "incident_severity"
        size = network_config[field]["size"]

        predictions = {field: torch.randn(2, size, requires_grad=True)}
        # one real label, one masked
        targets_with_mask = {f"label_{field}": torch.tensor([0, -100])}
        targets_without_mask = {f"label_{field}": torch.tensor([0])}

        criterion_single_field = DynamicMultiHeadLoss(
            target_fields=[field], network_config={field: network_config[field]}
        )

        loss_masked, _ = criterion_single_field(
            {field: predictions[field]}, targets_with_mask
        )
        loss_unmasked, _ = criterion_single_field(
            {field: predictions[field][:1]}, targets_without_mask
        )

        assert torch.isclose(loss_masked, loss_unmasked, atol=1e-5)


class TestLossWeights:
    def test_default_weight_is_one(self, network_config):
        criterion = DynamicMultiHeadLoss(
            target_fields=list(network_config.keys()), network_config=network_config
        )
        for field in network_config:
            assert criterion.weights[field] == 1.0

    def test_custom_weight_is_applied(self, network_config):
        cfg = {**network_config}
        cfg["incident_severity"] = {**cfg["incident_severity"], "loss_weight": 5.0}

        criterion = DynamicMultiHeadLoss(target_fields=list(cfg.keys()), network_config=cfg)
        assert criterion.weights["incident_severity"] == 5.0
        assert criterion.weights["auto_make"] == 1.0


class TestLossOutputShape:
    def test_loss_dict_has_one_entry_per_field(self, criterion, network_config):
        predictions = {
            field: torch.randn(2, cfg["size"]) for field, cfg in network_config.items()
        }
        targets = {
            f"label_{field}": torch.zeros(2, dtype=torch.long) for field in network_config
        }

        _, loss_dict = criterion(predictions, targets)
        assert set(loss_dict.keys()) == set(network_config.keys())

    def test_total_loss_is_scalar(self, criterion, network_config):
        predictions = {
            field: torch.randn(2, cfg["size"]) for field, cfg in network_config.items()
        }
        targets = {
            f"label_{field}": torch.zeros(2, dtype=torch.long) for field in network_config
        }

        loss, _ = criterion(predictions, targets)
        assert loss.dim() == 0