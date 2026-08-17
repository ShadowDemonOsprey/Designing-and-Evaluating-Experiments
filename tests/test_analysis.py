"""Unit tests for statistical analysis."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import StatisticalAnalyzer, TestResult


@pytest.fixture
def analyzer():
    return StatisticalAnalyzer(significance_level=0.05)


class TestPairedTTest:
    def test_identical_distributions(self, analyzer):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = analyzer.paired_t_test(a, a)
        assert result.p_value == pytest.approx(1.0, abs=1e-6)
        assert not result.significant

    def test_clear_difference(self, analyzer):
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = analyzer.paired_t_test(a, b)
        assert result.significant

    def test_size_mismatch(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.paired_t_test([1, 2], [1])

    def test_too_few_samples(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.paired_t_test([1.0], [2.0])


class TestWilcoxon:
    def test_identical(self, analyzer):
        a = [1.0, 2.0, 3.0]
        result = analyzer.wilcoxon_test(a, a)
        assert result.p_value == pytest.approx(1.0, abs=1e-6)

    def test_clear_difference(self, analyzer):
        a = [1.0, 1.0, 1.0, 1.0, 1.0]
        b = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = analyzer.wilcoxon_test(a, b)
        assert result.p_value < 0.05


class TestBootstrapCI:
    def test_narrow_ci(self, analyzer):
        scores = [1.0] * 100
        lo, hi = analyzer.bootstrap_ci(scores, n_bootstrap=500)
        assert lo == pytest.approx(1.0, abs=0.01)
        assert hi == pytest.approx(1.0, abs=0.01)

    def test_ci_contains_mean(self, analyzer):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        lo, hi = analyzer.bootstrap_ci(scores, n_bootstrap=1000)
        assert lo < 3.0 < hi


class TestHedgesG:
    def test_no_difference(self, analyzer):
        a = [1.0, 2.0, 3.0]
        g = analyzer.hedges_g(a, a)
        assert g == pytest.approx(0.0, abs=1e-6)

    def test_large_difference(self, analyzer):
        a = [10.0, 10.0, 10.0, 10.0]
        b = [1.0, 1.0, 1.0, 1.0]
        g = analyzer.hedges_g(a, b)
        assert abs(g) > 5.0


class TestReport:
    def test_report_generation(self, analyzer):
        results = {
            "experiment_name": "test",
            "elapsed_seconds": 1.5,
            "optimization": {"improvement_pct": 7.5, "iterations": 10},
            "comparison": {
                "delta_quality": 0.05,
                "delta_pct": 7.5,
                "p_value": 0.02,
                "significant": True,
                "confidence_interval": [0.01, 0.09],
                "effect_size_cohens_d": 0.8,
            },
        }
        report = analyzer.generate_report(results)
        assert "test" in report
        assert "7.50%" in report
        assert "Significant" in report
