"""Model evaluation metrics with mathematical rigor.

Composite scoring:
  S = alpha*BLEU + beta*ROUGE_L + gamma*F1 + delta*Faithfulness
  subject to: alpha + beta + gamma + delta = 1

Additional metrics:
  - Exact Match (EM)
  - Token-level F1
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single sample."""

    sample_id: str
    scores: dict[str, float]
    composite_score: float


@dataclass
class AggregateResult:
    """Aggregated evaluation results across all samples."""

    mean_scores: dict[str, float]
    std_scores: dict[str, float]
    composite_mean: float
    composite_std: float
    individual_results: list[EvalResult]
    n_samples: int


class ModelEvaluator:
    """Multi-metric evaluation framework.

    Computes a battery of metrics and a weighted composite score.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            weights: Dict mapping metric names to weights.
                     Keys: bleu, rouge_l, f1, em, faithfulness.
                     Weights are normalized to sum to 1.
        """
        self.weights = weights or {
            "bleu": 0.25,
            "rouge_l": 0.25,
            "f1": 0.25,
            "faithfulness": 0.25,
        }
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Sum of weights must be positive")
        self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate_batch(
        self,
        predictions: list[str],
        references: list[str],
        sample_ids: list[str] | None = None,
    ) -> AggregateResult:
        """Evaluate a batch of predictions against references.

        Args:
            predictions: Model-generated responses.
            references: Ground truth responses.
            sample_ids: Optional identifiers for each sample.

        Returns:
            AggregateResult with mean, std, and per-sample scores.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions vs "
                f"{len(references)} references"
            )

        n = len(predictions)
        if sample_ids is None:
            sample_ids = [str(i) for i in range(n)]

        results: list[EvalResult] = []
        for i in range(n):
            scores = self._compute_metrics(predictions[i], references[i])
            composite = self._weighted_composite(scores)
            results.append(
                EvalResult(
                    sample_id=sample_ids[i],
                    scores=scores,
                    composite_score=composite,
                )
            )

        return self._aggregate(results)

    def evaluate_single(
        self, prediction: str, reference: str, sample_id: str = "0"
    ) -> EvalResult:
        """Evaluate a single prediction."""
        scores = self._compute_metrics(prediction, reference)
        composite = self._weighted_composite(scores)
        return EvalResult(
            sample_id=sample_id,
            scores=scores,
            composite_score=composite,
        )

    def _compute_metrics(self, prediction: str, reference: str) -> dict[str, float]:
        """Compute all individual metrics."""
        metrics: dict[str, float] = {}
        for name in self.weights:
            if name == "bleu":
                metrics["bleu"] = self._bleu_score(prediction, reference)
            elif name == "rouge_l":
                metrics["rouge_l"] = self._rouge_l(prediction, reference)
            elif name == "f1":
                metrics["f1"] = self._token_f1(prediction, reference)
            elif name == "em":
                metrics["em"] = self._exact_match(prediction, reference)
            elif name == "faithfulness":
                metrics["faithfulness"] = self._faithfulness_score(
                    prediction, reference
                )
            else:
                raise ValueError(f"Unknown metric: {name}")
        return metrics

    def _weighted_composite(self, scores: dict[str, float]) -> float:
        """Compute weighted composite score."""
        return sum(
            self.weights.get(name, 0.0) * score for name, score in scores.items()
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenization."""
        return text.lower().split()

    def _bleu_score(self, prediction: str, reference: str, max_n: int = 4) -> float:
        """Compute modified BLEU score (simplified, corpus-level for single pair).

        BLEU = BP * exp(sum_n w_n * log(p_n))
        where p_n is modified n-gram precision and BP is brevity penalty.
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        if not pred_tokens:
            return 0.0

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))

        log_avg = 0.0
        weight = 1.0 / max_n

        for n in range(1, max_n + 1):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                continue

            clipped = sum(
                min(count, ref_ngrams.get(ng, 0))
                for ng, count in pred_ngrams.items()
            )
            total = sum(pred_ngrams.values())

            precision = clipped / total
            if precision > 0:
                log_avg += weight * math.log(precision)
            else:
                return 0.0

        return bp * math.exp(log_avg)

    @staticmethod
    def _get_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
        """Extract n-grams as a counter."""
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    def _rouge_l(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 score using LCS.

        R_lcs = LCS(X, Y) / m
        P_lcs = LCS(X, Y) / n
        F_lcs = (1 + beta^2) * R_lcs * P_lcs / (R_lcs + beta^2 * P_lcs)
        where beta = 1.2 (convention).
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)

        lcs_len = self._lcs_length(pred_tokens, ref_tokens)
        m = len(ref_tokens)
        n = len(pred_tokens)

        if m == 0 or n == 0:
            return 0.0

        beta = 1.2
        r_lcs = lcs_len / m
        p_lcs = lcs_len / n
        denominator = r_lcs + beta**2 * p_lcs

        if denominator == 0:
            return 0.0

        return (1 + beta**2) * r_lcs * p_lcs / denominator

    @staticmethod
    def _lcs_length(x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence via DP.

        O(mn) time, O(min(m,n)) space.
        """
        if len(x) < len(y):
            x, y = y, x
        prev = [0] * (len(y) + 1)
        curr = [0] * (len(y) + 1)

        for i in range(1, len(x) + 1):
            for j in range(1, len(y) + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(curr[j - 1], prev[j])
            prev, curr = curr, [0] * (len(y) + 1)

        return prev[len(y)]

    def _token_f1(self, prediction: str, reference: str) -> float:
        """Token-level F1 score.

        F1 = 2 * P * R / (P + R)
        P = |pred ∩ ref| / |pred|
        R = |pred ∩ ref| / |ref|
        """
        pred_tokens = Counter(self._tokenize(prediction))
        ref_tokens = Counter(self._tokenize(reference))

        overlap = sum((pred_tokens & ref_tokens).values())
        pred_total = sum(pred_tokens.values())
        ref_total = sum(ref_tokens.values())

        if pred_total == 0 or ref_total == 0:
            return 0.0

        precision = overlap / pred_total
        recall = overlap / ref_total

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _exact_match(prediction: str, reference: str) -> float:
        """Binary exact match after normalization."""
        norm_pred = prediction.lower().strip()
        norm_ref = reference.lower().strip()
        return 1.0 if norm_pred == norm_ref else 0.0

    @staticmethod
    def _faithfulness_score(prediction: str, reference: str) -> float:
        """Approximate faithfulness via token overlap ratio.

        faithfulness = |pred_tokens ∩ ref_tokens| / |ref_tokens|
        Measures how much of the reference is covered by the prediction.
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())

        if not ref_tokens:
            return 1.0

        return len(pred_tokens & ref_tokens) / len(ref_tokens)

    @staticmethod
    def _aggregate(results: list[EvalResult]) -> AggregateResult:
        """Compute mean and std of scores across results."""
        if not results:
            return AggregateResult(
                mean_scores={},
                std_scores={},
                composite_mean=0.0,
                composite_std=0.0,
                individual_results=[],
                n_samples=0,
            )

        n = len(results)

        # Collect all metric keys
        all_keys = set()
        for r in results:
            all_keys.update(r.scores.keys())

        mean_scores: dict[str, float] = {}
        std_scores: dict[str, float] = {}

        for key in all_keys:
            values = [r.scores.get(key, 0.0) for r in results]
            mean_val = sum(values) / n
            var_val = sum((v - mean_val) ** 2 for v in values) / max(n - 1, 1)
            mean_scores[key] = mean_val
            std_scores[key] = math.sqrt(var_val)

        composite_values = [r.composite_score for r in results]
        composite_mean = sum(composite_values) / n
        composite_var = (
            sum((v - composite_mean) ** 2 for v in composite_values) / max(n - 1, 1)
        )

        return AggregateResult(
            mean_scores=mean_scores,
            std_scores=std_scores,
            composite_mean=composite_mean,
            composite_std=math.sqrt(composite_var),
            individual_results=results,
            n_samples=n,
        )
