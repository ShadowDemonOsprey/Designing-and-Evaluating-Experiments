"""Stress test: exercise every module with extreme edge cases.

Run standalone: python tests/stress_test.py
Not designed for pytest collection.
"""

import sys
import os
import math
sys.path.insert(0, ".")

# Prevent pytest from collecting this module's functions
__test__ = False

from src.config import ExperimentConfig, MetricWeights, ModelConfig
from src.data_loader import DataLoader, Sample
from src.prompt_optimizer import PromptOptimizer, PromptVariant
from src.model_evaluator import ModelEvaluator
from src.benchmarking import BenchmarkRunner
from src.analysis import StatisticalAnalyzer

errors = []


def check(name, condition, detail=""):
    if not condition:
        errors.append(f"FAIL: {name} {detail}")
        print(f"  FAIL: {name} {detail}")
    else:
        print(f"  OK: {name}")


def test_config():
    print("\n=== CONFIG ===")

    # Edge: all zeros except one
    w = MetricWeights(bleu=1.0, rouge_l=0.0, f1=0.0, faithfulness=0.0)
    check("zero weights", w.as_dict()["rouge_l"] == 0.0)

    # Edge: save/load roundtrip preserves floats
    cfg = ExperimentConfig(experiment_name="float_test")
    cfg.save_json("_test_float.json")
    loaded = ExperimentConfig.from_json("_test_float.json")
    check("float roundtrip", loaded.significance_level == cfg.significance_level)
    os.unlink("_test_float.json")

    # Edge: empty models list
    cfg2 = ExperimentConfig(experiment_name="no_models")
    check("empty models", len(cfg2.models) == 0)

    # Edge: from_dict with missing optional fields
    d = {"experiment_name": "minimal"}
    cfg3 = ExperimentConfig._from_dict(d)
    check("minimal dict", cfg3.seed == 42 and cfg3.num_runs == 5)


def test_data_loader():
    print("\n=== DATA LOADER ===")

    # Edge: JSON with single sample (not in a list)
    import json
    with open("_test_single.json", "w") as f:
        json.dump({"id": "1", "input": "q", "output": "a"}, f)
    loader = DataLoader(".")
    samples = loader.load("_test_single.json")
    check("single sample dict", len(samples) == 1)
    os.unlink("_test_single.json")

    # Edge: CSV with extra columns (metadata)
    with open("_test_extra.csv", "w") as f:
        f.write("id,input,output,category,difficulty\n")
        f.write("1,q1,a1,math,hard\n")
    samples = loader.load("_test_extra.csv")
    check("csv extra cols", samples[0].metadata is not None and "category" in samples[0].metadata)
    os.unlink("_test_extra.csv")

    # Edge: JSONL with empty lines
    with open("_test_empty_lines.jsonl", "w") as f:
        f.write('{"id":"1","input":"q","output":"a"}\n\n\n{"id":"2","input":"q2","output":"a2"}\n')
    samples = loader.load("_test_empty_lines.jsonl")
    check("jsonl empty lines", len(samples) == 2)
    os.unlink("_test_empty_lines.jsonl")

    # Edge: save_samples and reload
    s = [Sample(id="x", input_text="hello", expected_output="world", metadata={"k": "v"})]
    DataLoader.save_samples(s, "_test_save.json")
    loaded = loader.load("_test_save.json")
    check("save/reload roundtrip", loaded[0].input_text == "hello" and loaded[0].metadata == {"k": "v"})
    os.unlink("_test_save.json")

    # Edge: save to bare filename (no directory)
    DataLoader.save_samples(s, "_test_bare.json")
    check("save bare filename", os.path.isfile("_test_bare.json"))
    os.unlink("_test_bare.json")


def test_evaluator():
    print("\n=== MODEL EVALUATOR ===")

    # Edge: empty strings
    ev = ModelEvaluator()
    r = ev.evaluate_single("", "")
    check("empty strings", r.composite_score >= 0)

    # Edge: very long text
    long_pred = "word " * 1000
    long_ref = "word " * 1000
    r = ev.evaluate_single(long_pred, long_ref)
    check("long text identical", r.composite_score == pytest.approx(1.0, abs=1e-6))

    # Edge: unicode
    ev_em = ModelEvaluator(weights={"em": 1.0})
    r = ev_em.evaluate_single("héllo wörld", "héllo wörld")
    check("unicode exact match", r.scores["em"] == 1.0)

    # Edge: single token
    r = ev.evaluate_single("hello", "hello")
    check("single token bleu", r.scores["bleu"] == pytest.approx(1.0, abs=1e-6))

    # Edge: batch of 1
    batch = ev.evaluate_batch(["a"], ["a"])
    check("batch of 1", batch.n_samples == 1)

    # Edge: batch of 0 - should crash
    try:
        ev.evaluate_batch([], [])
        check("batch of 0", True)
    except Exception:
        check("batch of 0 raises", False, "(should handle empty gracefully)")

    # Edge: unknown metric name
    try:
        bad_ev = ModelEvaluator(weights={"unknown_metric": 1.0})
        bad_ev.evaluate_single("a", "b")
        check("unknown metric raises", False, "(should have raised ValueError)")
    except ValueError:
        check("unknown metric raises", True)

    # Edge: ROUGE-L when prediction is longer than reference
    r = ev.evaluate_single("a b c d e f g", "a b")
    check("rouge_l long pred", 0 <= r.scores["rouge_l"] <= 1)

    # Edge: ROUGE-L when reference is longer than prediction
    r = ev.evaluate_single("a b", "a b c d e f g")
    check("rouge_l long ref", 0 <= r.scores["rouge_l"] <= 1)


def test_optimizer():
    print("\n=== PROMPT OPTIMIZER ===")

    # Edge: single template
    opt = PromptOptimizer(evaluation_fn=lambda t, i, o: len(t) / 100.0, seed=42, max_iterations=2, patience=2)
    r = opt.optimize(["hello {input}"], ["q"], ["a"])
    check("single template", r.best_prompt is not None)

    # Edge: eval fn returns 0 for everything
    opt = PromptOptimizer(evaluation_fn=lambda t, i, o: 0.0, seed=42, max_iterations=2, patience=2)
    r = opt.optimize(["{input}"], ["q"], ["a"])
    check("all zero scores", r.improvement_pct == 0.0)

    # Edge: eval fn returns 1 for everything
    opt = PromptOptimizer(evaluation_fn=lambda t, i, o: 1.0, seed=42, max_iterations=2, patience=2)
    r = opt.optimize(["{input}"], ["q"], ["a"])
    check("all perfect scores", r.best_prompt is not None)

    # Edge: grid search with single template
    r = PromptOptimizer.grid_search(
        ["only_one"],
        evaluation_fn=lambda t, i, o: 0.5,
        inputs=["q"],
        expected_outputs=["a"],
    )
    check("grid single template", r.best_prompt.template == "only_one")


def test_benchmarking():
    print("\n=== BENCHMARKING ===")

    def inf(p, i): return i
    def ev(p, r): return 1.0 if p == r else 0.0

    br = BenchmarkRunner(inf, ev, seed=42, n_bootstrap=10)

    # Edge: single sample
    bm = br.run("m", "{input}", ["q"], ["q"])
    check("single sample benchmark", bm.n_samples == 1 and bm.quality_score == 1.0)

    # Edge: all zeros evaluation
    def zero_ev(p, r): return 0.0
    br2 = BenchmarkRunner(inf, zero_ev, seed=42, n_bootstrap=10)
    bm = br2.run("m", "{input}", ["q", "r"], ["q", "r"])
    comp = br2.compare(bm, bm)
    check("zero baseline compare", comp.delta_pct == 0.0 and comp.p_value == 1.0)

    # Edge: format report with empty list
    report = BenchmarkRunner.format_report([])
    check("empty report", "BENCHMARK REPORT" in report)

    # Edge: latency stats with single value
    stats = BenchmarkRunner._compute_latency_stats([5.0])
    check("single latency", stats.mean == 5.0 and stats.p95 == 5.0)


def test_analysis():
    print("\n=== ANALYSIS ===")
    sa = StatisticalAnalyzer()

    # Edge: t-test with very small samples (non-constant diffs)
    r = sa.paired_t_test([1.0, 2.0], [3.0, 4.0])
    check("t-test n=2", r.p_value >= 0 and r.p_value <= 1)

    # Edge: wilcoxon with all ties
    r = sa.wilcoxon_test([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    check("wilcoxon all ties", r.p_value < 1.0)

    # Edge: bootstrap CI with single element
    lo, hi = sa.bootstrap_ci([5.0], n_bootstrap=10, seed=42)
    check("bootstrap single", lo == pytest.approx(5.0) and hi == pytest.approx(5.0))

    # Edge: hedges_g with small samples
    g = sa.hedges_g([1.0, 2.0], [3.0, 4.0])
    check("hedges small", g != 0.0)

    # Edge: comprehensive comparison
    results = sa.comprehensive_comparison([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    check("comprehensive", len(results) >= 1)

    # Edge: generate report with minimal dict
    report = sa.generate_report({"experiment_name": "x"})
    check("minimal report", "x" in report)

    # Edge: t-test p-value correctness
    p = StatisticalAnalyzer._t_test_p_value(0.0, 10)
    check("t=0 p=1", p == pytest.approx(1.0, abs=1e-6))


def test_experiment_runner():
    print("\n=== EXPERIMENT RUNNER ===")
    from src.experiment_runner import ExperimentRunner

    cfg = ExperimentConfig(experiment_name="stress_test", prompt_templates=["{input}"])

    def inf(p, i): return f"answer to {i}"

    runner = ExperimentRunner(cfg)
    results = runner.run(inf)
    check("runner completes", "experiment_name" in results)
    check("runner has comparison", "comparison" in results)
    check("runner has benchmarks", "benchmarks" in results)

    # Verify results file was created
    expected_path = os.path.join(cfg.results_path, "stress_test_results.json")
    check("results file exists", os.path.isfile(expected_path))
    os.unlink(expected_path)


# Need pytest for approx
import pytest

if __name__ == "__main__":
    test_config()
    test_data_loader()
    test_evaluator()
    test_optimizer()
    test_benchmarking()
    test_analysis()
    test_experiment_runner()

    print("\n" + "=" * 50)
    if errors:
        print(f"STRESS TEST FAILED: {len(errors)} errors")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("ALL STRESS TESTS PASSED")
