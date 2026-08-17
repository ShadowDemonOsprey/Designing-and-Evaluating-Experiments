"""Experiment orchestration and pipeline management.

Coordinates:
  1. Data loading
  2. Prompt optimization
  3. Model evaluation
  4. Benchmarking
  5. Statistical comparison
  6. Result persistence
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable

from .config import ExperimentConfig
from .data_loader import DataLoader, Sample
from .prompt_optimizer import PromptOptimizer
from .model_evaluator import ModelEvaluator
from .benchmarking import BenchmarkRunner

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates the full experiment lifecycle."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._log_handler: logging.FileHandler | None = None
        self._setup_logging()
        self.results: dict[str, Any] = {}

    def _setup_logging(self) -> None:
        os.makedirs(self.config.log_path, exist_ok=True)
        log_file = os.path.join(
            self.config.log_path,
            f"{self.config.experiment_name}_{int(time.time())}.log",
        )
        self._log_handler = logging.FileHandler(log_file)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                self._log_handler,
                logging.StreamHandler(),
            ],
            force=True,
        )

    def close(self) -> None:
        """Close file handler to release log file."""
        if self._log_handler is not None:
            logging.root.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None

    def run(
        self,
        inference_fn: Callable[[str, str], str],
    ) -> dict[str, Any]:
        """Execute the full experiment pipeline.

        Args:
            inference_fn: Model inference function (prompt, input) -> response.

        Returns:
            Dictionary of all experiment results.
        """
        logger.info("Starting experiment: %s", self.config.experiment_name)
        start_time = time.time()

        # 1. Load data
        samples = self._load_data()
        inputs = [s.input_text for s in samples]
        references = [s.expected_output for s in samples]
        logger.info("Loaded %d samples", len(samples))

        # 2. Create evaluator
        evaluator = ModelEvaluator(weights=self.config.metric_weights.as_dict())

        # 3. Define evaluation function for optimizer
        def eval_fn(
            template: str, ins: list[str], refs: list[str]
        ) -> float:
            predictions = []
            for inp in ins:
                prompt = template.replace("{input}", inp)
                pred = inference_fn(prompt, inp)
                predictions.append(pred)
            result = evaluator.evaluate_batch(predictions, refs)
            return result.composite_mean

        # 4. Prompt optimization
        optimizer = PromptOptimizer(
            evaluation_fn=eval_fn,
            seed=self.config.seed,
            max_iterations=15,
            patience=5,
        )

        initial_templates = self.config.prompt_templates or [
            "Answer the following question:\n{input}"
        ]
        opt_result = optimizer.optimize(
            initial_templates=initial_templates,
            inputs=inputs,
            expected_outputs=references,
        )
        logger.info(
            "Optimization complete: %.2f%% improvement over %d iterations",
            opt_result.improvement_pct,
            opt_result.iterations,
        )

        # 5. Evaluate best prompt vs baseline
        best_template = opt_result.best_prompt.template
        baseline_template = initial_templates[0]

        best_predictions = self._generate_batch(
            inference_fn, best_template, inputs
        )
        baseline_predictions = self._generate_batch(
            inference_fn, baseline_template, inputs
        )

        best_eval = evaluator.evaluate_batch(best_predictions, references)
        baseline_eval = evaluator.evaluate_batch(baseline_predictions, references)

        # 6. Run benchmarks
        runner = BenchmarkRunner(
            inference_fn=inference_fn,
            evaluation_fn=lambda p, r: evaluator.evaluate_single(p, r).composite_score,
            seed=self.config.seed,
        )

        best_benchmark = runner.run(
            model_name="optimized",
            prompt_template=best_template,
            inputs=inputs,
            references=references,
            prompt_id="best_optimized",
        )
        baseline_benchmark = runner.run(
            model_name="baseline",
            prompt_template=baseline_template,
            inputs=inputs,
            references=references,
            prompt_id="baseline",
        )

        comparison = runner.compare(baseline_benchmark, best_benchmark)

        # 7. Collect results
        elapsed = time.time() - start_time
        self.results = {
            "experiment_name": self.config.experiment_name,
            "elapsed_seconds": elapsed,
            "n_samples": len(samples),
            "optimization": {
                "improvement_pct": opt_result.improvement_pct,
                "iterations": opt_result.iterations,
                "best_prompt": best_template,
                "convergence_history": opt_result.convergence_history,
            },
            "evaluation": {
                "best": {
                    "composite_mean": best_eval.composite_mean,
                    "composite_std": best_eval.composite_std,
                    "mean_scores": best_eval.mean_scores,
                },
                "baseline": {
                    "composite_mean": baseline_eval.composite_mean,
                    "composite_std": baseline_eval.composite_std,
                    "mean_scores": baseline_eval.mean_scores,
                },
            },
            "comparison": {
                "delta_quality": comparison.delta_quality,
                "delta_pct": comparison.delta_pct,
                "p_value": comparison.p_value,
                "significant": comparison.significant,
                "confidence_interval": list(comparison.confidence_interval),
                "effect_size_cohens_d": comparison.effect_size_cohens_d,
            },
            "benchmarks": {
                "best": {
                    "latency_mean_ms": best_benchmark.latency.mean,
                    "throughput": best_benchmark.throughput_tokens_per_sec,
                    "quality": best_benchmark.quality_score,
                },
                "baseline": {
                    "latency_mean_ms": baseline_benchmark.latency.mean,
                    "throughput": baseline_benchmark.throughput_tokens_per_sec,
                    "quality": baseline_benchmark.quality_score,
                },
            },
        }

        self._save_results()
        logger.info("Experiment completed in %.2fs", elapsed)
        self.close()
        return self.results

    def _load_data(self) -> list[Sample]:
        """Load evaluation data."""
        loader = DataLoader(self.config.data_path)
        samples = loader.load_all()
        if not samples:
            raise FileNotFoundError(
                f"No data files found in {self.config.data_path}"
            )
        return samples

    @staticmethod
    def _generate_batch(
        inference_fn: Callable[[str, str], str],
        template: str,
        inputs: list[str],
    ) -> list[str]:
        """Generate predictions for a batch of inputs."""
        predictions = []
        for inp in inputs:
            prompt = template.replace("{input}", inp)
            pred = inference_fn(prompt, inp)
            predictions.append(pred)
        return predictions

    def _save_results(self) -> None:
        """Persist results to disk."""
        os.makedirs(self.config.results_path, exist_ok=True)
        path = os.path.join(
            self.config.results_path,
            f"{self.config.experiment_name}_results.json",
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Results saved to %s", path)
