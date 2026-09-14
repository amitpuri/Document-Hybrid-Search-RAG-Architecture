"""
Evaluation package: metrics, benchmark dataset, and evaluation harness.
"""

from src.evaluation.metrics import evaluate_ranking, is_relevant
from src.evaluation.dataset import EVAL_DATASET, validate_ground_truth
from src.evaluation.harness import EvaluationHarness

__all__ = [
    "evaluate_ranking",
    "is_relevant",
    "EVAL_DATASET",
    "validate_ground_truth",
    "EvaluationHarness",
]
