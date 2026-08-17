"""Unit tests for model evaluator."""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_evaluator import ModelEvaluator, EvalResult


class TestBLEU:
    def test_identical(self):
        ev = ModelEvaluator(weights={"bleu": 1.0})
        result = ev.evaluate_single("hello world", "hello world")
        assert result.scores["bleu"] == pytest.approx(1.0, abs=1e-6)

    def test_empty_prediction(self):
        ev = ModelEvaluator(weights={"bleu": 1.0})
        result = ev.evaluate_single("", "hello world")
        assert result.scores["bleu"] == 0.0

    def test_partial_overlap(self):
        ev = ModelEvaluator(weights={"bleu": 1.0})
        result = ev.evaluate_single("the cat sat on the mat", "the cat sat on the big mat")
        assert 0.0 < result.scores["bleu"] < 1.0


class TestROUGEL:
    def test_identical(self):
        ev = ModelEvaluator(weights={"rouge_l": 1.0})
        result = ev.evaluate_single("the cat sat", "the cat sat")
        assert result.scores["rouge_l"] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        ev = ModelEvaluator(weights={"rouge_l": 1.0})
        result = ev.evaluate_single("abc", "xyz")
        assert result.scores["rouge_l"] == 0.0

    def test_partial(self):
        ev = ModelEvaluator(weights={"rouge_l": 1.0})
        result = ev.evaluate_single("the cat", "the cat sat on the mat")
        assert 0.0 < result.scores["rouge_l"] < 1.0


class TestTokenF1:
    def test_perfect(self):
        ev = ModelEvaluator(weights={"f1": 1.0})
        result = ev.evaluate_single("hello world", "hello world")
        assert result.scores["f1"] == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        ev = ModelEvaluator(weights={"f1": 1.0})
        result = ev.evaluate_single("abc def", "xyz uvw")
        assert result.scores["f1"] == 0.0

    def test_partial(self):
        ev = ModelEvaluator(weights={"f1": 1.0})
        result = ev.evaluate_single("a b c", "a b d")
        assert 0.0 < result.scores["f1"] < 1.0


class TestExactMatch:
    def test_match(self):
        ev = ModelEvaluator(weights={"em": 1.0})
        result = ev.evaluate_single("Hello", "hello")
        assert result.scores["em"] == 1.0

    def test_no_match(self):
        ev = ModelEvaluator(weights={"em": 1.0})
        result = ev.evaluate_single("hello", "world")
        assert result.scores["em"] == 0.0


class TestComposite:
    def test_equal_weights(self):
        ev = ModelEvaluator(weights={"bleu": 0.5, "f1": 0.5})
        result = ev.evaluate_single("hello world", "hello world")
        assert result.composite_score == pytest.approx(1.0, abs=1e-6)

    def test_weight_normalization(self):
        ev = ModelEvaluator(weights={"bleu": 2.0, "f1": 2.0})
        result = ev.evaluate_single("hello", "hello")
        assert result.composite_score == pytest.approx(1.0, abs=1e-6)


class TestBatch:
    def test_batch_size(self):
        ev = ModelEvaluator(weights={"em": 1.0})
        preds = ["a", "b", "c"]
        refs = ["a", "x", "c"]
        result = ev.evaluate_batch(preds, refs)
        assert result.n_samples == 3
        assert result.composite_mean == pytest.approx(2 / 3, abs=1e-6)

    def test_length_mismatch(self):
        ev = ModelEvaluator()
        with pytest.raises(ValueError, match="Length mismatch"):
            ev.evaluate_batch(["a", "b"], ["a"])


class TestLCS:
    def test_lcs_basic(self):
        assert ModelEvaluator._lcs_length(["a", "b", "c"], ["a", "c"]) == 2

    def test_lcs_empty(self):
        assert ModelEvaluator._lcs_length([], ["a"]) == 0

    def test_lcs_identical(self):
        x = ["a", "b", "c"]
        assert ModelEvaluator._lcs_length(x, x) == 3
