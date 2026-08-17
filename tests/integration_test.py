"""Integration test: run the full experiment pipeline with a mock model."""

import sys
import json
sys.path.insert(0, ".")

from src.config import ExperimentConfig
from src.experiment_runner import ExperimentRunner


def mock_inference(prompt: str, inp: str) -> str:
    return f'The answer to "{inp}" is that it involves careful analysis and reasoning.'


def main() -> None:
    cfg = ExperimentConfig.from_yaml("configs/default.yaml")
    runner = ExperimentRunner(cfg)
    results = runner.run(inference_fn=mock_inference)

    print("=== EXPERIMENT RESULTS ===")
    print(f"Name: {results['experiment_name']}")
    print(f"Samples: {results['n_samples']}")
    print(f"Elapsed: {results['elapsed_seconds']:.2f}s")
    print(f"Optimization improvement: {results['optimization']['improvement_pct']:.2f}%")
    print(f"Optimization iterations: {results['optimization']['iterations']}")
    print(f"Best composite (optimized): {results['evaluation']['best']['composite_mean']:.4f}")
    print(f"Best composite (baseline): {results['evaluation']['baseline']['composite_mean']:.4f}")
    print(f"Delta: {results['comparison']['delta_pct']:.2f}%")
    print(f"Significant: {results['comparison']['significant']}")
    print(f"Cohen's d: {results['comparison']['effect_size_cohens_d']:.4f}")
    print()

    # Test report generation
    from src.analysis import StatisticalAnalyzer
    sa = StatisticalAnalyzer()
    report = sa.generate_report(results)
    print(report)

    print("=== INTEGRATION TEST PASSED ===")


if __name__ == "__main__":
    main()
