"""Experiment configuration management."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95


@dataclass
class MetricWeights:
    """Weights for the composite evaluation score.

    S = alpha*BLEU + beta*ROUGE_L + gamma*F1 + delta*Faithfulness
    Subject to: alpha + beta + gamma + delta = 1
    """

    bleu: float = 0.25
    rouge_l: float = 0.25
    f1: float = 0.25
    faithfulness: float = 0.25

    def __post_init__(self) -> None:
        total = self.bleu + self.rouge_l + self.f1 + self.faithfulness
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total}"
            )

    def as_dict(self) -> dict[str, float]:
        return {
            "bleu": self.bleu,
            "rouge_l": self.rouge_l,
            "f1": self.f1,
            "faithfulness": self.faithfulness,
        }


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    experiment_name: str
    description: str = ""
    seed: int = 42
    num_runs: int = 5

    models: list[ModelConfig] = field(default_factory=list)
    metric_weights: MetricWeights = field(default_factory=MetricWeights)

    prompt_templates: list[str] = field(default_factory=list)
    data_path: str = "data/processed"
    results_path: str = "experiments/results"
    log_path: str = "experiments/logs"

    significance_level: float = 0.05
    confidence_level: float = 0.95

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def from_json(cls, path: str) -> ExperimentConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        models = [ModelConfig(**m) for m in raw.get("models", [])]
        weights_raw = raw.get("metric_weights", {})
        metric_weights = MetricWeights(**weights_raw) if weights_raw else MetricWeights()
        return cls(
            experiment_name=raw["experiment_name"],
            description=raw.get("description", ""),
            seed=raw.get("seed", 42),
            num_runs=raw.get("num_runs", 5),
            models=models,
            metric_weights=metric_weights,
            prompt_templates=raw.get("prompt_templates", []),
            data_path=raw.get("data_path", "data/processed"),
            results_path=raw.get("results_path", "experiments/results"),
            log_path=raw.get("log_path", "experiments/logs"),
            significance_level=raw.get("significance_level", 0.05),
            confidence_level=raw.get("confidence_level", 0.95),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
