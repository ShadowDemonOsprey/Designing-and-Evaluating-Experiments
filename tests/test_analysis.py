"""Unit tests for statistical analysis."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import StatisticalAnalyzer, StatisticalTestResult


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

    def test_tied_ranks(self, analyzer):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 6.0]
        result = analyzer.wilcoxon_test(a, b)
        assert result.p_value < 1.0

    def test_all_same_differences(self, analyzer):
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
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

    def test_empty_scores(self, analyzer):
        lo, hi = analyzer.bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0


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

    def test_report_none_ci(self, analyzer):
        results = {
            "experiment_name": "test",
            "comparison": {
                "delta_quality": 0,
                "delta_pct": 0,
                "p_value": 0.5,
                "significant": False,
                "confidence_interval": None,
                "effect_size_cohens_d": 0,
            },
        }
        report = analyzer.generate_report(results)
        assert "[0.0000" in report


class TestTTestCorrectness:
    """Validate t-test p-values against scipy for mathematical accuracy."""

    def test_t_test_matches_scipy(self, analyzer):
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            pytest.skip("scipy not installed")

        cases = [
            ([1.0, 2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5, 5.5]),
            ([0.8, 0.9, 0.7, 0.85, 0.75], [0.6, 0.5, 0.6, 0.55, 0.65]),
            ([10.0, 20.0, 30.0], [12.0, 22.0, 32.0]),
        ]
        for a, b in cases:
            ours = analyzer.paired_t_test(a, b)
            _, scipy_p = ttest_rel(a, b)
            assert ours.p_value == pytest.approx(scipy_p, abs=0.01), (
                f"Mismatch: ours={ours.p_value:.6f}, scipy={scipy_p:.6f}"
            )

    def test_incomplete_beta_correctness(self):
        try:
            from scipy.special import betainc
        except ImportError:
            pytest.skip("scipy not installed")

        # Non-singular cases (a >= 1 and b >= 1) should be accurate
        easy_cases = [(0.5, 1.0, 1.0), (0.3, 2.0, 3.0), (0.8, 5.0, 1.0)]
        for x, a, b in easy_cases:
            scipy_val = betainc(a, b, x)
            our_val = StatisticalAnalyzer._incomplete_beta_simpson(x, a, b)
            assert our_val == pytest.approx(scipy_val, abs=0.005), (
                f"I({x},{a},{b}): ours={our_val:.6f}, scipy={scipy_val:.6f}"
            )

    def test_incomplete_beta_singular_a_lt_1(self):
        try:
            from scipy.special import betainc
        except ImportError:
            pytest.skip("scipy not installed")

        singular_cases = [(0.5, 0.5, 0.5), (0.8, 0.5, 0.5), (0.3, 0.8, 0.5)]
        for x, a, b in singular_cases:
            scipy_val = betainc(a, b, x)
            our_val = StatisticalAnalyzer._incomplete_beta_simpson(x, a, b)
            assert our_val == pytest.approx(scipy_val, abs=0.02), (
                f"I({x},{a},{b}): ours={our_val:.6f}, scipy={scipy_val:.6f}"
            )


class TestSignificanceLevel:
    def test_summary_respects_custom_significance_level(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.1, 2.1, 3.1, 4.1, 5.1]
        analyzer = StatisticalAnalyzer(significance_level=0.20)
        result = analyzer.paired_t_test(a, b)
        if result.p_value < 0.20:
            assert "significant" in result.summary
        else:
            assert "not significant" in result.summary

    def test_strict_significance_level(self):
        a = [1.0, 1.5, 1.2, 1.4, 1.1]
        b = [1.05, 1.45, 1.25, 1.35, 1.15]
        analyzer = StatisticalAnalyzer(significance_level=0.001)
        result = analyzer.paired_t_test(a, b)
        assert not result.significant
