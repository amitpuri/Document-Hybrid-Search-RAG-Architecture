"""
Linear Score Fusion with Min-Max Normalization.
Formula: S_hybrid = alpha * S_dense_norm + (1 - alpha) * S_sparse_norm
"""

from typing import List
import numpy as np


def min_max_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes score array to range [0, 1]."""
    s_min = scores.min()
    s_max = scores.max()
    denom = s_max - s_min
    if denom < eps:
        return np.zeros_like(scores)
    return (scores - s_min) / (denom + eps)


def linear_fusion(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    alpha: float = 0.3
) -> List[int]:
    """
    Fuses sparse and dense score vectors via convex combination:
    hybrid = (alpha * dense_norm) + ((1.0 - alpha) * sparse_norm)
    Returns indices sorted in descending order of hybrid score.
    """
    sparse_norm = min_max_normalize(sparse_scores)
    dense_norm = min_max_normalize(dense_scores)
    hybrid_scores = (alpha * dense_norm) + ((1.0 - alpha) * sparse_norm)
    return np.argsort(hybrid_scores)[::-1].tolist()
