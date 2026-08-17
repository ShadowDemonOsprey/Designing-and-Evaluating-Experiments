"""Unit tests for configuration module."""

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import ExperimentConfig, MetricWeights, ModelConfig


class TestMetricWeights:
    def test_valid_weights(self):
        w = MetricWeights(bleu=0.5, rouge_l=0.3, bert_score=0.1, faithfulness=0.1)
        assert w.bleu == 0.5

    def test_invalid_weights(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            MetricWeights(bleu=0.5, rouge_l=0.5, bert_score=0.5, faithfulness=0.5)

    def test_as_dict(self):
        w = MetricWeights()
        d = w.as_dict()
        assert "bleu" in d
        assert abs(sum(d.values()) - 1.0) < 1e-6


class TestModelConfig:
    def test_defaults(self):
        m = ModelConfig(name="test", model_id="test-model")
        assert m.temperature == 0.7
        assert m.max_tokens == 1024


class TestExperimentConfig:
    def test_creation(self):
        cfg = ExperimentConfig(
            experiment_name="test",
            models=[ModelConfig(name="m", model_id="m")],
        )
        assert cfg.experiment_name == "test"
        assert cfg.seed == 42

    def test_save_load_json(self):
        cfg = ExperimentConfig(experiment_name="json_test")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cfg.save_json(path)
            loaded = ExperimentConfig.from_json(path)
            assert loaded.experiment_name == "json_test"
        finally:
            os.unlink(path)

    def test_to_dict(self):
        cfg = ExperimentConfig(experiment_name="dict_test")
        d = cfg.to_dict()
        assert d["experiment_name"] == "dict_test"
        assert isinstance(d, dict)
