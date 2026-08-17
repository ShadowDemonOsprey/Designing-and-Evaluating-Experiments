"""Unit tests for prompt optimizer."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.prompt_optimizer import PromptOptimizer, PromptVariant, OptimizationResult


def dummy_eval(template: str, inputs: list[str], outputs: list[str]) -> float:
    """Dummy evaluator: longer templates score slightly better."""
    return min(1.0, 0.5 + len(template) / 200.0)


class TestPromptOptimizer:
    def test_basic_optimization(self):
        optimizer = PromptOptimizer(
            evaluation_fn=dummy_eval,
            seed=42,
            max_iterations=3,
            patience=3,
        )
        result = optimizer.optimize(
            initial_templates=["{input}"],
            inputs=["q1", "q2"],
            expected_outputs=["a1", "a2"],
        )
        assert isinstance(result, OptimizationResult)
        assert result.iterations >= 0
        assert len(result.convergence_history) >= 1
        assert result.best_prompt is not None

    def test_improvement_detected(self):
        optimizer = PromptOptimizer(
            evaluation_fn=dummy_eval,
            seed=42,
            max_iterations=10,
            patience=3,
        )
        result = optimizer.optimize(
            initial_templates=["{input}"],
            inputs=["q1"],
            expected_outputs=["a1"],
        )
        assert result.improvement_pct >= 0

    def test_empty_templates_raises(self):
        optimizer = PromptOptimizer(evaluation_fn=dummy_eval)
        with pytest.raises(ValueError, match="At least one"):
            optimizer.optimize([], ["q"], ["a"])

    def test_mismatched_lengths_raises(self):
        optimizer = PromptOptimizer(evaluation_fn=dummy_eval)
        with pytest.raises(ValueError, match="equal length"):
            optimizer.optimize(["t"], ["q1", "q2"], ["a"])


class TestGridSearch:
    def test_grid_search(self):
        result = PromptOptimizer.grid_search(
            templates=["short", "a much longer template that should score higher"],
            evaluation_fn=dummy_eval,
            inputs=["q"],
            expected_outputs=["a"],
        )
        assert result.best_prompt.template == "a much longer template that should score higher"
        assert result.improvement_pct > 0

    def test_grid_search_all_none_raises(self):
        def none_fn(t, i, o): return None
        with pytest.raises(RuntimeError, match="failed evaluation"):
            PromptOptimizer.grid_search(["a", "b"], none_fn, ["q"], ["a"])
