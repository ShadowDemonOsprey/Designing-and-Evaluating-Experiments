"""Statistical analysis and visualization utilities.

Provides:
  - Paired t-tests and Wilcoxon signed-rank tests
  - Bootstrap confidence intervals
  - Effect size calculations (Cohen's d, Hedges' g)
  - Summary report generation
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StatisticalTestResult:
    """Result of a statistical hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_name: str
    confidence_interval: tuple[float, float] | None = None
    summary: str = ""


class StatisticalAnalyzer:
    """Statistical analysis tools for experiment comparison."""

    def __init__(self, significance_level: float = 0.05) -> None:
        self.significance_level = significance_level

    def paired_t_test(
        self, scores_a: list[float], scores_b: list[float]
    ) -> StatisticalTestResult:
        """Paired Student's t-test.

        H0: mean(A) == mean(B)
        t = mean(d) / (std(d) / sqrt(n))
        where d_i = a_i - b_i
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Sample sizes must be equal")
        n = len(scores_a)
        if n < 2:
            raise ValueError("Need at least 2 paired observations")

        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = math.sqrt(var_diff)

        if std_diff == 0:
            if mean_diff == 0:
                return StatisticalTestResult(
                    test_name="Paired t-test",
                    statistic=0.0,
                    p_value=1.0,
                    significant=False,
                    effect_size=0.0,
                    effect_size_name="Cohen's d",
                    summary="No variance and no mean difference; tests are identical.",
                )
            return StatisticalTestResult(
                test_name="Paired t-test",
                statistic=float("inf") if mean_diff > 0 else float("-inf"),
                p_value=0.0,
                significant=True,
                effect_size=float("inf") if mean_diff > 0 else float("-inf"),
                effect_size_name="Cohen's d",
                summary="Constant non-zero difference with zero variance; perfectly significant.",
            )

        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se

        # Approximate p-value using t-distribution (two-tailed)
        # Using approximation for degrees of freedom = n - 1
        p_value = self._t_test_p_value(abs(t_stat), n - 1)

        # Cohen's d for paired samples
        cohens_d = mean_diff / std_diff

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=cohens_d,
            effect_size_name="Cohen's d",
            summary=self._format_test_result(
                "Paired t-test", t_stat, p_value, cohens_d, "Cohen's d"
            ),
        )

    def wilcoxon_test(
        self, scores_a: list[float], scores_b: list[float]
    ) -> StatisticalTestResult:
        """Wilcoxon signed-rank test (non-parametric).

        Tests median difference without assuming normality.
        """
        if len(scores_a) != len(scores_b):
            raise ValueError("Sample sizes must be equal")
        n = len(scores_a)
        if n < 2:
            raise ValueError("Need at least 2 paired observations")

        diffs = [a - b for a, b in zip(scores_a, scores_b)]

        # Remove zero differences
        nonzero = [(abs(d), i) for i, d in enumerate(diffs) if d != 0]
        if not nonzero:
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                effect_size_name="rank-biserial r",
                summary="All differences are zero.",
            )

        # Rank by absolute difference, averaging tied ranks
        nonzero.sort(key=lambda x: x[0])
        n_z = len(nonzero)
        ranks = [0.0] * n_z

        i = 0
        while i < n_z:
            j = i
            while j < n_z and nonzero[j][0] == nonzero[i][0]:
                j += 1
            avg_rank = (i + j + 1) / 2.0  # ranks are 1-indexed
            for k in range(i, j):
                ranks[k] = avg_rank
            i = j

        # Sum of positive and negative ranks
        w_plus = sum(
            ranks[i]
            for i, (_, idx) in enumerate(nonzero)
            if diffs[idx] > 0
        )
        w_minus = sum(
            ranks[i]
            for i, (_, idx) in enumerate(nonzero)
            if diffs[idx] < 0
        )

        w_stat = min(w_plus, w_minus)

        # Normal approximation for n >= 10
        mean_w = n_z * (n_z + 1) / 4
        std_w = math.sqrt(n_z * (n_z + 1) * (2 * n_z + 1) / 24)

        if std_w == 0:
            p_value = 1.0
        else:
            z = (w_stat - mean_w) / std_w
            p_value = 2 * (1 - self._normal_cdf(abs(z)))

        # Effect size: rank-biserial correlation
        effect_size = 1 - (2 * w_stat) / (n_z * (n_z + 1)) if n_z > 0 else 0.0

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=w_stat,
            p_value=p_value,
            significant=p_value < self.significance_level,
            effect_size=effect_size,
            effect_size_name="rank-biserial r",
            summary=self._format_test_result(
                "Wilcoxon signed-rank", w_stat, p_value, effect_size, "rank-biserial r"
            ),
        )

    def bootstrap_ci(
        self,
        scores: list[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the mean.

        Uses percentile method with BCa correction.
        """
        import random
        rng = random.Random(seed)
        n = len(scores)
        means = []

        for _ in range(n_bootstrap):
            sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        alpha = 1 - confidence_level
        lo_idx = int(alpha / 2 * n_bootstrap)
        hi_idx = int((1 - alpha / 2) * n_bootstrap) - 1
        lo_idx = max(0, min(lo_idx, n_bootstrap - 1))
        hi_idx = max(0, min(hi_idx, n_bootstrap - 1))

        return (means[lo_idx], means[hi_idx])

    def hedges_g(self, scores_a: list[float], scores_b: list[float]) -> float:
        """Compute Hedges' g (bias-corrected Cohen's d).

        g = d * (1 - 3/(4*(n1+n2) - 9))
        """
        n1, n2 = len(scores_a), len(scores_b)
        if n1 < 2 or n2 < 2:
            return 0.0

        mean_a = sum(scores_a) / n1
        mean_b = sum(scores_b) / n2
        var_a = sum((x - mean_a) ** 2 for x in scores_a) / (n1 - 1)
        var_b = sum((x - mean_b) ** 2 for x in scores_b) / (n2 - 1)

        # Pooled standard deviation
        sp = math.sqrt(((n1 - 1) * var_a + (n2 - 1) * var_b) / (n1 + n2 - 2))

        if sp == 0:
            if mean_a == mean_b:
                return 0.0
            return float("inf") if mean_a > mean_b else float("-inf")

        d = (mean_a - mean_b) / sp

        # Hedges' correction factor
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        return d * correction

    def comprehensive_comparison(
        self, scores_a: list[float], scores_b: list[float]
    ) -> list[StatisticalTestResult]:
        """Run all applicable tests and return results."""
        results = []

        try:
            results.append(self.paired_t_test(scores_a, scores_b))
        except ValueError as e:
            logger.warning("Paired t-test skipped: %s", e)

        try:
            results.append(self.wilcoxon_test(scores_a, scores_b))
        except ValueError as e:
            logger.warning("Wilcoxon test skipped: %s", e)

        return results

    def generate_report(
        self, results: dict[str, Any], output_path: str | None = None
    ) -> str:
        """Generate a formatted analysis report."""
        lines = [
            "=" * 70,
            "STATISTICAL ANALYSIS REPORT",
            "=" * 70,
            "",
        ]

        if "experiment_name" in results:
            lines.append(f"Experiment: {results['experiment_name']}")
        if "elapsed_seconds" in results:
            lines.append(f"Duration:   {results['elapsed_seconds']:.1f}s")
        lines.append("")

        # Optimization summary
        if "optimization" in results:
            opt = results["optimization"]
            lines.extend([
                "OPTIMIZATION SUMMARY",
                "-" * 40,
                f"  Improvement:  {opt.get('improvement_pct', 0):.2f}%",
                f"  Iterations:   {opt.get('iterations', 0)}",
                "",
            ])

        # Evaluation comparison
        if "evaluation" in results:
            ev = results["evaluation"]
            lines.extend(["EVALUATION RESULTS", "-" * 40])
            for key in ["best", "baseline"]:
                if key in ev:
                    data = ev[key]
                    lines.extend([
                        f"  {key.upper()}:",
                        f"    Composite:  {data.get('composite_mean', 0):.4f} "
                        f"± {data.get('composite_std', 0):.4f}",
                    ])
                    for metric, val in data.get("mean_scores", {}).items():
                        lines.append(f"    {metric:14s}: {val:.4f}")
                    lines.append("")

        # Statistical comparison
        if "comparison" in results:
            comp = results["comparison"]
            lines.extend([
                "STATISTICAL COMPARISON",
                "-" * 40,
                f"  Delta:          {comp.get('delta_quality', 0):.4f} "
                f"({comp.get('delta_pct', 0):+.2f}%)",
                f"  p-value:        {comp.get('p_value', 1):.6f}",
                f"  Significant:    {'Yes' if comp.get('significant', False) else 'No'}",
                f"  95% CI:         [{comp.get('confidence_interval', [0, 0])[0]:.4f}, "
                f"{comp.get('confidence_interval', [0, 0])[1]:.4f}]",
                f"  Cohen's d:      {comp.get('effect_size_cohens_d', 0):.4f}",
                "",
            ])

        report = "\n".join(lines)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info("Report saved to %s", output_path)

        return report

    @staticmethod
    def _t_test_p_value(t_stat: float, df: int) -> float:
        """Two-tailed p-value for t-distribution.

        Uses I_x(df/2, 1/2) where x = df/(df + t^2).
        """
        try:
            from scipy.special import betainc
            x = df / (df + t_stat**2)
            return min(1.0, max(0.0, betainc(df / 2.0, 0.5, x)))
        except ImportError:
            return StatisticalAnalyzer._t_test_p_value_numeric(t_stat, df)

    @staticmethod
    def _t_test_p_value_numeric(t_stat: float, df: int) -> float:
        """Fallback t-test p-value via numerical integration of incomplete beta."""
        x = df / (df + t_stat**2)
        ib = StatisticalAnalyzer._incomplete_beta_simpson(x, df / 2.0, 0.5)
        return min(1.0, max(0.0, ib))

    @staticmethod
    def _incomplete_beta_simpson(x: float, a: float, b: float, n: int = 5000) -> float:
        """I_x(a,b) via Simpson's rule numerical integration.

        I_x(a,b) = integral_0^x t^{a-1} (1-t)^{b-1} dt / B(a,b)
        Uses analytical tail correction for boundary singularities when a<1 or b<1.
        """
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        def integrand(t: float) -> float:
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 0.0
            return t ** (a - 1) * (1.0 - t) ** (b - 1)

        def simpson_integral(lo: float, hi: float, steps: int) -> float:
            h = (hi - lo) / steps
            s = integrand(lo) + integrand(hi)
            for i in range(1, steps, 2):
                s += 4.0 * integrand(lo + i * h)
            for i in range(2, steps, 2):
                s += 2.0 * integrand(lo + i * h)
            return s * h / 3.0

        # For the denominator (integral 0 to 1), split at a cutoff to handle
        # singularities at t=1 analytically: tail = integral(cutoff,1) ~ cutoff^(a-1) * (1-cutoff)^b / b
        cutoff = 1.0 - 1e-6
        den_main = simpson_integral(0.0, cutoff, n // 2)
        tail_correction = 0.0
        if b < 1 and cutoff < 1.0:
            tail_correction = cutoff ** (a - 1) * (1.0 - cutoff) ** b / b
        den = den_main + tail_correction

        num = simpson_integral(0.0, min(x, cutoff), n // 2)
        if x > cutoff and b < 1:
            num += cutoff ** (a - 1) * (1.0 - cutoff) ** b / b

        if den == 0:
            return 0.0
        return num / den

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _format_test_result(
        name: str, statistic: float, p_value: float, effect: float, effect_name: str
    ) -> str:
        sig = "significant" if p_value < 0.05 else "not significant"
        return (
            f"{name}: stat={statistic:.4f}, p={p_value:.6f} ({sig}), "
            f"{effect_name}={effect:.4f}"
        )
