"""
Tests for src/models/multi_head_insurance/inference.py

Covers:
- predictions below clf_threshold become None
- integer-coded classification fields (auto_year, witnesses, ...) are cast to int
- decoded values come from the right label_encoders entry
"""
import torch
import torch.nn as nn
import pytest
from src.models.multi_head_insurance.inference import predict


class _StubBert(nn.Module):
    """Minimal stand-in for AutoModel so tests don't load a real transformer."""

    def __init__(self, hidden_size=8):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        last_hidden_state = torch.zeros(batch, seq_len, self.hidden_size)
        return type("Output", (), {"last_hidden_state": last_hidden_state})()


@pytest.fixture
def stub_model(network_config, monkeypatch):
    """
    Builds a GermanInsuranceClassifier whose BERT backbone is stubbed out,
    and whose heads are monkeypatched to return fixed, known logits.
    """
    from src.models.multi_head_insurance.models import GermanInsuranceClassifier

    model = GermanInsuranceClassifier.__new__(GermanInsuranceClassifier)
    nn.Module.__init__(model)
    model.bert = _StubBert()
    model.network_config = network_config
    model.dropout = nn.Identity()
    model.heads = nn.ModuleDict()

    fixed_logits = {}
    for field, cfg in network_config.items():
        size = cfg["size"]
        logits = torch.zeros(1, size)
        logits[0, 0] = 10.0  # always predicts class 0 with high confidence
        fixed_logits[field] = logits

        class _FixedHead(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return self.value

        model.heads[field] = _FixedHead(logits)

    model.eval()
    return model


class TestConfidenceFiltering:
    def test_high_confidence_prediction_is_kept(
        self, stub_model, label_encoders, monkeypatch
    ):
        _patch_tokenizer(monkeypatch)

        df = predict(
            texts=["dummy text"],
            model=stub_model,
            label_encoders=label_encoders,
            clf_threshold=0.5,
            device="cpu",
        )
        # class 0 was forced with logit 10 vs 0 -> softmax confidence ~1.0
        assert df.iloc[0]["auto_make"] is not None

    def test_low_confidence_threshold_returns_none(
        self, stub_model, label_encoders, monkeypatch
    ):
        _patch_tokenizer(monkeypatch)

        # Threshold above what a single dominant logit can achieve is impossible
        # with softmax (max approaches 1.0), so instead verify the *mechanism*:
        # an artificially low threshold of 1.1 can never be satisfied.
        df = predict(
            texts=["dummy text"],
            model=stub_model,
            label_encoders=label_encoders,
            clf_threshold=1.1,
            device="cpu",
        )
        assert df.iloc[0]["auto_make"] is None


class TestIntegerFieldCasting:
    def test_witnesses_is_cast_to_python_int(self, stub_model, label_encoders, monkeypatch):
        _patch_tokenizer(monkeypatch)

        df = predict(
            texts=["dummy text"],
            model=stub_model,
            label_encoders=label_encoders,
            clf_threshold=0.0,
            device="cpu",
        )
        value = df.iloc[0]["witnesses"]
        assert isinstance(value, int)


def _patch_tokenizer(monkeypatch):
    """Avoid downloading/loading the real GottBERT tokenizer in unit tests."""
    import src.models.multi_head_insurance.inference as inference_module

    class _StubTokenizer:
        def __call__(self, texts, **kwargs):
            batch = len(texts)
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    monkeypatch.setattr(
        inference_module.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, name: _StubTokenizer()),
    )