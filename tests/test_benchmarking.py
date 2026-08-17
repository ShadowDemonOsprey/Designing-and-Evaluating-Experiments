"""Unit tests for benchmarking module."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarking import BenchmarkRunner, BenchmarkResult, LatencyStats


def mock_inference(prompt: str, inp: str) -> str:
    """Mock inference function."""
    return f"Answer to: {inp}"


def mock_eval(pred: str, ref: str) -> float:
    """Mock evaluation: simple string similarity."""
    if pred == ref:
        return 1.0
    overlap = len(set(pred.split()) & set(ref.split()))
    total = len(set(pred.split()) | set(ref.split()))
    return overlap / total if total > 0 else 0.0


@pytest.fixture
def runner():
    return BenchmarkRunner(
        inference_fn=mock_inference,
        evaluation_fn=mock_eval,
        seed=42,
        n_bootstrap=100,
    )


class TestBenchmark:
    def test_basic_run(self, runner):
        result = runner.run(
            model_name="test_model",
            prompt_template="Answer: {input}",
            inputs=["q1", "q2"],
            references=["Answer to: q1", "Answer to: q2"],
            prompt_id="test_prompt",
        )
        assert isinstance(result, BenchmarkResult)
        assert result.n_samples == 2
        assert result.quality_score >= 0
        assert result.latency.mean >= 0
        assert result.throughput_tokens_per_sec >= 0

    def test_mismatched_lengths(self, runner):
        with pytest.raises(ValueError, match="equal length"):
            runner.run("m", "p", ["q1"], ["r1", "r2"])

    def test_latency_stats(self):
        stats = BenchmarkRunner._compute_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.mean == pytest.approx(3.0)
        assert stats.median == pytest.approx(3.0)
        assert stats.min_val == 1.0
        assert stats.max_val == 5.0

    def test_latency_stats_empty(self):
        stats = BenchmarkRunner._compute_latency_stats([])
        assert stats.mean == 0.0


class TestComparison:
    def test_compare(self, runner):
        baseline = runner.run(
            model_name="base", prompt_template="{input}",
            inputs=["q1"] * 5, references=["a1"] * 5, prompt_id="base",
        )
        treatment = runner.run(
            model_name="treat", prompt_template="Answer: {input}",
            inputs=["q1"] * 5, references=["Answer to: q1"] * 5, prompt_id="treat",
        )
        comp = runner.compare(baseline, treatment)
        assert hasattr(comp, "delta_quality")
        assert hasattr(comp, "p_value")
        assert hasattr(comp, "significant")
        assert hasattr(comp, "confidence_interval")
        assert hasattr(comp, "effect_size_cohens_d")

    def test_format_report(self, runner):
        result = runner.run(
            model_name="m", prompt_template="{input}",
            inputs=["q"], references=["a"], prompt_id="p",
        )
        report = BenchmarkRunner.format_report([result])
        assert "BENCHMARK REPORT" in report
        assert "test_model" in report or "m" in report
