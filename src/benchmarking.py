"""Performance benchmarking framework.

Benchmarks models across:
  - Latency (mean, p50, p95, p99)
  - Throughput (tokens/sec)
  - Quality (composite score)
  - Cost efficiency (quality per token)

Statistical comparison via paired bootstrap resampling.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class LatencyStats:
    """Latency statistics in milliseconds."""

    mean: float
    median: float
    p95: float
    p99: float
    std: float
    min_val: float
    max_val: float


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a model-prompt pair."""

    model_name: str
    prompt_id: str
    latency: LatencyStats
    throughput_tokens_per_sec: float
    quality_score: float
    total_tokens: int
    total_time_sec: float
    n_samples: int
    per_sample_scores: list[float] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Statistical comparison between two benchmarks."""

    delta_quality: float
    delta_pct: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    effect_size_cohens_d: float


class BenchmarkRunner:
    """Runs benchmarks and compares model/prompt performance."""

    def __init__(
        self,
        inference_fn: Callable[[str, str], str],
        evaluation_fn: Callable[[str, str], float],
        seed: int = 42,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Args:
            inference_fn: Takes (prompt, input) and returns generated text.
            evaluation_fn: Takes (prediction, reference) and returns score.
            seed: Random seed for bootstrap resampling.
            n_bootstrap: Number of bootstrap resamples.
            confidence_level: Confidence level for intervals.
        """
        self.inference_fn = inference_fn
        self.evaluation_fn = evaluation_fn
        self.seed = seed
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        random.seed(seed)

    def run(
        self,
        model_name: str,
        prompt_template: str,
        inputs: list[str],
        references: list[str],
        prompt_id: str = "default",
    ) -> BenchmarkResult:
        """Run a full benchmark for a model-prompt pair.

        Args:
            model_name: Identifier for the model.
            prompt_template: Prompt template with {input} placeholder.
            inputs: List of input texts.
            references: List of reference outputs.
            prompt_id: Identifier for this prompt variant.

        Returns:
            BenchmarkResult with latency, throughput, and quality metrics.
        """
        if len(inputs) != len(references):
            raise ValueError("inputs and references must have equal length")

        latencies: list[float] = []
        scores: list[float] = []
        total_tokens = 0

        for inp, ref in zip(inputs, references):
            prompt = prompt_template.replace("{input}", inp)

            start = time.perf_counter()
            prediction = self.inference_fn(prompt, inp)
            elapsed_ms = (time.perf_counter() - start) * 1000

            latencies.append(elapsed_ms)
            score = self.evaluation_fn(prediction, ref)
            scores.append(score)
            total_tokens += len(prediction.split())

        total_time_sec = sum(latencies) / 1000
        latency_stats = self._compute_latency_stats(latencies)
        throughput = total_tokens / max(total_time_sec, 1e-6)

        return BenchmarkResult(
            model_name=model_name,
            prompt_id=prompt_id,
            latency=latency_stats,
            throughput_tokens_per_sec=throughput,
            quality_score=sum(scores) / len(scores) if scores else 0.0,
            total_tokens=total_tokens,
            total_time_sec=total_time_sec,
            n_samples=len(inputs),
            per_sample_scores=scores,
        )

    def compare(
        self,
        baseline: BenchmarkResult,
        treatment: BenchmarkResult,
    ) -> ComparisonResult:
        """Compare two benchmarks using bootstrap resampling.

        Tests whether the treatment significantly improves over baseline
        using paired bootstrap hypothesis testing.
        """
        if not baseline.per_sample_scores or not treatment.per_sample_scores:
            raise ValueError("Both benchmarks must have per-sample scores")

        n = min(
            len(baseline.per_sample_scores), len(treatment.per_sample_scores)
        )
        base_scores = baseline.per_sample_scores[:n]
        treat_scores = treatment.per_sample_scores[:n]

        # Observed difference
        base_mean = sum(base_scores) / n
        treat_mean = sum(treat_scores) / n
        delta = treat_mean - base_mean
        delta_pct = (delta / base_mean * 100) if base_mean > 0 else 0.0

        # Bootstrap resampling for confidence interval
        rng = random.Random(self.seed)
        bootstrap_diffs: list[float] = []

        for _ in range(self.n_bootstrap):
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            boot_base = sum(base_scores[i] for i in indices) / n
            boot_treat = sum(treat_scores[i] for i in indices) / n
            bootstrap_diffs.append(boot_treat - boot_base)

        bootstrap_diffs.sort()

        # Confidence interval
        alpha = 1 - self.confidence_level
        lo_idx = int(alpha / 2 * self.n_bootstrap)
        hi_idx = int((1 - alpha / 2) * self.n_bootstrap) - 1
        lo_idx = max(0, min(lo_idx, self.n_bootstrap - 1))
        hi_idx = max(0, min(hi_idx, self.n_bootstrap - 1))
        ci = (bootstrap_diffs[lo_idx], bootstrap_diffs[hi_idx])

        # p-value: proportion of bootstrap diffs <= 0
        p_value = sum(1 for d in bootstrap_diffs if d <= 0) / self.n_bootstrap

        # Cohen's d effect size from original paired differences
        diffs = [t - b for t, b in zip(treat_scores, base_scores)]
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)
        pooled_std = var_diff**0.5 if var_diff > 0 else 1e-10
        cohens_d = delta / pooled_std

        return ComparisonResult(
            delta_quality=delta,
            delta_pct=delta_pct,
            p_value=p_value,
            significant=p_value < 0.05,
            confidence_interval=ci,
            effect_size_cohens_d=cohens_d,
        )

    @staticmethod
    def _compute_latency_stats(latencies: list[float]) -> LatencyStats:
        """Compute latency statistics from a list of measurements."""
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        mean_val = sum(sorted_lat) / n
        var_val = sum((x - mean_val) ** 2 for x in sorted_lat) / max(n - 1, 1)

        def percentile(data: list[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            return data[f] + (k - f) * (data[c] - data[f])

        return LatencyStats(
            mean=mean_val,
            median=percentile(sorted_lat, 0.5),
            p95=percentile(sorted_lat, 0.95),
            p99=percentile(sorted_lat, 0.99),
            std=var_val**0.5,
            min_val=sorted_lat[0],
            max_val=sorted_lat[-1],
        )

    @staticmethod
    def format_report(results: list[BenchmarkResult]) -> str:
        """Format benchmark results into a human-readable report."""
        lines = ["=" * 70, "BENCHMARK REPORT", "=" * 70, ""]

        for r in results:
            lines.extend([
                f"Model: {r.model_name} | Prompt: {r.prompt_id}",
                f"  Samples:     {r.n_samples}",
                f"  Quality:     {r.quality_score:.4f}",
                f"  Latency:     {r.latency.mean:.1f}ms (p95={r.latency.p95:.1f}ms)",
                f"  Throughput:  {r.throughput_tokens_per_sec:.1f} tok/s",
                f"  Total time:  {r.total_time_sec:.2f}s",
                "",
            ])

        return "\n".join(lines)
