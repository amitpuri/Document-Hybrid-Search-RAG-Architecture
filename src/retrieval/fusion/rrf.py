"""
Reciprocal Rank Fusion (RRF).
Formula: RRF_score(d) = sum_m (1 / (k + rank_m(d) + 1))
"""

from typing import List, Dict, Tuple


def reciprocal_rank_fusion(
    ranking_a: List[int],
    ranking_b: List[int],
    k: int = 60
) -> Tuple[List[int], Dict[int, float]]:
    """
    Combines two ranked lists of document indices using Reciprocal Rank Fusion.

    Args:
        ranking_a: List of document indices sorted by retriever A.
        ranking_b: List of document indices sorted by retriever B.
        k: Smoothing constant (default: 60).

    Returns:
        Tuple: (fused_indices_sorted_descending, dict_of_fused_scores)
    """
    scores: Dict[int, float] = {}

    for rank, idx in enumerate(ranking_a):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    for rank, idx in enumerate(ranking_b):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_indices = [idx for idx, _ in sorted_items]
    fused_scores = dict(sorted_items)

    return fused_indices, fused_scores
