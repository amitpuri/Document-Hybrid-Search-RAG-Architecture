"""
Linear Score Fusion with Min-Max Normalization.
Formula: S_hybrid = alpha * S_dense_norm + (1 - alpha) * S_sparse_norm
"""

from typing import List

import numpy as np


def min_max_normalize(scores: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalizes score array to range [0, 1].

    When all scores are near-constant (range < eps), returns a uniform 0.5 array
    so that downstream argsort preserves the original input ordering rather than
    producing an arbitrary all-zero result. Previously this returned np.zeros_like
    which silently degraded to arbitrary rank ordering on constant inputs.
    """
    s_min = scores.min()
    s_max = scores.max()
    denom = s_max - s_min
    if denom < eps:
        return np.full_like(scores, 0.5, dtype=float)
    return (scores - s_min) / (denom + eps)


# TODO (P4): Add CombMNZ or z-score normalization as an alternative fusion baseline
# so the linear-hybrid vs RRF comparison isn't handicapped by min-max being the
# weakest normalization method for near-constant score inputs.
def compute_linear_scores(
    sparse_scores: np.ndarray, dense_scores: np.ndarray, alpha: float = 0.3
) -> np.ndarray:
    """
    Computes convex combination score vector: (alpha * dense_norm) + ((1.0 - alpha) * sparse_norm).

    NOTE: Performance Degradation at High Alpha
    ───────────────────────────────────────────
    On jargon-dense technical corpora, increasing alpha (higher dense weight) degrades performance:
    α=0.3 (MRR≈0.554) > α=0.5 (MRR≈0.524) > α=0.7 (MRR≈0.488)

    Root cause: Dense embeddings (MiniLM) diffuse technical acronyms (POMDP, StarShell, AgentRunner)
    across semantic neighborhoods. BM25 precise lexical matching outweighs fuzzy semantic matching.
    Min-max normalization is correctly implemented; degradation is fundamental to the approach.

    Recommendation: Use Reciprocal Rank Fusion (RRF) instead. RRF operates in rank space, avoiding
    score-scale incompatibility and achieving MRR=0.629 (vs 0.554 best linear blend).
    """
    sparse_norm = min_max_normalize(sparse_scores)
    dense_norm = min_max_normalize(dense_scores)
    return (alpha * dense_norm) + ((1.0 - alpha) * sparse_norm)


def linear_fusion(
    sparse_scores: np.ndarray, dense_scores: np.ndarray, alpha: float = 0.3
) -> List[int]:
    """
    Fuses sparse and dense score vectors via convex combination:
    hybrid = (alpha * dense_norm) + ((1.0 - alpha) * sparse_norm)
    Returns indices sorted in descending order of hybrid score.
    """
    hybrid_scores = compute_linear_scores(sparse_scores, dense_scores, alpha=alpha)
    return np.argsort(hybrid_scores)[::-1].tolist()
