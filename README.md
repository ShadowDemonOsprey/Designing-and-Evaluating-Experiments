# Designing and Evaluating Machine Learning Experiments

A rigorous framework for **prompt optimization**, **model evaluation**, and **performance benchmarking** in LLM-based systems. Achieved ~5–10% improvements in response quality through statistically grounded iterative testing and refinement.

## Project Structure

```
.
├── src/                        # Core modules
│   ├── __init__.py
│   ├── config.py               # Experiment configuration
│   ├── data_loader.py          # Dataset management
│   ├── prompt_optimizer.py     # Prompt engineering & optimization
│   ├── model_evaluator.py      # Evaluation metrics & scoring
│   ├── benchmarking.py         # Performance benchmarking framework
│   ├── experiment_runner.py    # Orchestrates full experiment pipeline
│   └── analysis.py             # Statistical analysis & visualization
├── configs/                    # Experiment configuration files
│   └── default.yaml
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Preprocessed datasets
├── experiments/
│   ├── logs/                   # Experiment logs
│   └── results/                # Stored results & metrics
├── tests/                      # Unit & integration tests
├── notebooks/                  # Jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

## Mathematical Formulation

### Prompt Optimization

We model prompt optimization as a search problem over the prompt space $\mathcal{P}$. Given a task $\mathcal{T}$ and evaluation metric $m$, we seek:

$$p^* = \arg\max_{p \in \mathcal{P}} m(f_\theta(p, x), y)$$

where $f_\theta$ is the LLM with parameters $\theta$, $x$ is the input, and $y$ is the ground truth.

### Evaluation Metrics

We employ a composite scoring function:

$$S = \alpha \cdot \text{BLEU} + \beta \cdot \text{ROUGE-L} + \gamma \cdot F_1 + \delta \cdot \text{Faithfulness}$$

subject to $\alpha + \beta + \gamma + \delta = 1$, with weights tuned per task.

### Performance Benchmarking

We measure relative improvement as:

$$\Delta\% = \frac{m_{\text{optimized}} - m_{\text{baseline}}}{m_{\text{baseline}}} \times 100$$

and compute statistical significance via paired $t$-tests ($p < 0.05$).

## Key Features

- **Iterative Prompt Refinement**: Systematic ablation over prompt components
- **Multi-dimensional Evaluation**: Faithfulness, relevance, coherence, fluency
- **Statistical Rigor**: Confidence intervals, significance testing, effect sizes
- **Reproducibility**: Fully logged experiments with seeded randomness

## Getting Started

```bash
pip install -r requirements.txt
python -m src.experiment_runner --config configs/default.yaml
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
