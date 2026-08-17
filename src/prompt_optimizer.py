"""Prompt optimization via systematic ablation and search.

Models prompt optimization as a search over the prompt space P:
  p* = argmax_{p in P} m(f_theta(p, x), y)

where f_theta is the LLM, x is input, y is ground truth, and m is the metric.
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A single prompt variant with its template and metadata."""

    id: str
    template: str
    components: dict[str, str] = field(default_factory=dict)
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of a prompt optimization run."""

    best_prompt: PromptVariant
    all_variants: list[PromptVariant]
    improvement_pct: float
    iterations: int
    convergence_history: list[float]


class PromptOptimizer:
    """Iterative prompt optimizer using ablation and template mutation.

    Supports:
    - Component ablation (remove/replace parts)
    - Template mutation (rewrite via LLM or rules)
    - Grid search over prompt hyperparameters
    - Greedy hill-climbing with early stopping
    """

    def __init__(
        self,
        evaluation_fn: Callable[[str, list[str], list[str]], float],
        seed: int = 42,
        max_iterations: int = 20,
        patience: int = 5,
        improvement_threshold: float = 0.001,
    ) -> None:
        """
        Args:
            evaluation_fn: Takes (template, inputs, outputs) and returns score.
            seed: Random seed for reproducibility.
            max_iterations: Maximum optimization iterations.
            patience: Early stopping patience (iterations without improvement).
            improvement_threshold: Minimum improvement to reset patience counter.
        """
        self.evaluation_fn = evaluation_fn
        self.seed = seed
        self.max_iterations = max_iterations
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        random.seed(seed)

    def optimize(
        self,
        initial_templates: list[str],
        inputs: list[str],
        expected_outputs: list[str],
    ) -> OptimizationResult:
        """Run prompt optimization.

        Args:
            initial_templates: Starting prompt templates.
            inputs: Evaluation input texts.
            expected_outputs: Ground truth outputs.

        Returns:
            OptimizationResult with best prompt and convergence history.
        """
        if not initial_templates:
            raise ValueError("At least one initial template is required")
        if len(inputs) != len(expected_outputs):
            raise ValueError("inputs and expected_outputs must have equal length")

        variants = [
            PromptVariant(id=f"init_{i}", template=t)
            for i, t in enumerate(initial_templates)
        ]

        best_variant = self._evaluate_variants(variants, inputs, expected_outputs)
        best_score = best_variant.score or 0.0
        convergence_history = [best_score]
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            new_variants = self._generate_mutations(best_variant, iteration)

            if not new_variants:
                logger.info("No new mutations generated at iteration %d", iteration)
                break

            candidate = self._evaluate_variants(
                new_variants, inputs, expected_outputs
            )
            candidate_score = candidate.score or 0.0

            if candidate_score > best_score + self.improvement_threshold:
                improvement = candidate_score - best_score
                logger.info(
                    "Iteration %d: improvement %.4f (%.4f -> %.4f)",
                    iteration, improvement, best_score, candidate_score,
                )
                best_variant = candidate
                best_score = candidate_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            convergence_history.append(best_score)

            if no_improvement_count >= self.patience:
                logger.info("Early stopping at iteration %d (patience=%d)", iteration, self.patience)
                break

        initial_score = convergence_history[0] if convergence_history else 0.0
        improvement_pct = (
            ((best_score - initial_score) / initial_score * 100)
            if initial_score > 0
            else 0.0
        )

        return OptimizationResult(
            best_prompt=best_variant,
            all_variants=variants,
            improvement_pct=improvement_pct,
            iterations=len(convergence_history) - 1,
            convergence_history=convergence_history,
        )

    def _evaluate_variants(
        self,
        variants: list[PromptVariant],
        inputs: list[str],
        expected_outputs: list[str],
    ) -> PromptVariant:
        """Evaluate all variants and return the best one."""
        for v in variants:
            v.score = self.evaluation_fn(v.template, inputs, expected_outputs)

        scored = [v for v in variants if v.score is not None]
        if not scored:
            raise RuntimeError("All variants failed evaluation")

        return max(scored, key=lambda v: v.score or 0.0)  # type: ignore[arg-type]

    def _generate_mutations(
        self, base: PromptVariant, iteration: int
    ) -> list[PromptVariant]:
        """Generate candidate mutations from the best variant."""
        mutations: list[PromptVariant] = []
        template = base.template

        # Strategy 1: instruction variants
        instruction_prefixes = [
            "Answer the following question concisely.",
            "Provide a detailed and accurate response.",
            "Think step by step and then answer.",
            "You are an expert. Answer precisely.",
            "Consider the following carefully and respond.",
            "Based on your knowledge, answer the following.",
        ]
        for i, prefix in enumerate(instruction_prefixes):
            mutated = self._prepend_instruction(template, prefix)
            mutations.append(
                PromptVariant(
                    id=f"inst_{iteration}_{i}",
                    template=mutated,
                    components={"instruction": prefix},
                    metadata={"strategy": "instruction_swap"},
                )
            )

        # Strategy 2: formatting variants
        format_mutations = [
            template.replace("{input}", "Question: {input}\nAnswer:"),
            template.replace("{input}", "Context: {input}\nResponse:"),
            template.replace("{input}", "{input}\n\nDetailed explanation:"),
        ]
        for i, mutated in enumerate(format_mutations):
            if mutated != template:
                mutations.append(
                    PromptVariant(
                        id=f"fmt_{iteration}_{i}",
                        template=mutated,
                        metadata={"strategy": "format_mutation"},
                    )
                )

        # Strategy 3: few-shot injection
        few_shot_template = (
            "Here is an example:\n"
            "Input: What is 2+2?\n"
            "Output: 4\n\n"
            f"{template}"
        )
        mutations.append(
            PromptVariant(
                id=f"fewshot_{iteration}_0",
                template=few_shot_template,
                metadata={"strategy": "few_shot"},
            )
        )

        # Strategy 4: output format constraints
        constraint_suffixes = [
            "\nRespond in exactly one sentence.",
            "\nProvide a concise answer (under 50 words).",
            "\nFormat: [Answer] followed by [Explanation].",
        ]
        for i, suffix in enumerate(constraint_suffixes):
            mutations.append(
                PromptVariant(
                    id=f"constraint_{iteration}_{i}",
                    template=template + suffix,
                    metadata={"strategy": "output_constraint"},
                )
            )

        return mutations

    @staticmethod
    def _prepend_instruction(template: str, instruction: str) -> str:
        """Prepend an instruction before the existing template."""
        lines = template.strip().split("\n")
        if lines and lines[0].startswith("Instruction:"):
            lines[0] = f"Instruction: {instruction}"
            return "\n".join(lines)
        return f"Instruction: {instruction}\n\n{template}"

    @staticmethod
    def grid_search(
        templates: list[str],
        evaluation_fn: Callable[[str, list[str], list[str]], float],
        inputs: list[str],
        expected_outputs: list[str],
    ) -> OptimizationResult:
        """Exhaustive grid search over all templates.

        Useful for small template sets. Returns the best variant.
        """
        variants = [
            PromptVariant(id=f"grid_{i}", template=t)
            for i, t in enumerate(templates)
        ]

        for v in variants:
            v.score = evaluation_fn(v.template, inputs, expected_outputs)

        scored = [v for v in variants if v.score is not None]
        best = max(scored, key=lambda v: v.score or 0.0)  # type: ignore[arg-type]
        scores = [v.score or 0.0 for v in scored]
        initial_score = scores[0] if scores else 0.0
        best_score = best.score or 0.0
        improvement_pct = (
            ((best_score - initial_score) / initial_score * 100)
            if initial_score > 0
            else 0.0
        )

        return OptimizationResult(
            best_prompt=best,
            all_variants=variants,
            improvement_pct=improvement_pct,
            iterations=0,
            convergence_history=scores,
        )
