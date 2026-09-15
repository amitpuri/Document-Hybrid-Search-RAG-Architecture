"""
Reciprocal Rank Fusion (RRF).
Formula: RRF_score(d) = sum_m (1 / (k + rank_m(d) + 1))
"""

from typing import List, Dict, Tuple, Optional


def reciprocal_rank_fusion(
    ranking_a: List[int],
    ranking_b: List[int],
    k: int = 60,
    additional_rankings: Optional[List[List[int]]] = None,
    weights: Optional[List[float]] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """
    Combines two or more ranked lists of document indices using Reciprocal Rank Fusion.

    Args:
        ranking_a: List of document indices sorted by retriever A.
        ranking_b: List of document indices sorted by retriever B.
        k: Smoothing constant (default: 60).
        additional_rankings: Optional additional rankings to fuse.
        weights: Optional relative weighting factors per ranking list. Defaults to 1.0 each.

    Returns:
        Tuple: (fused_indices_sorted_descending, dict_of_fused_scores)
    """
    all_rankings = [ranking_a, ranking_b]
    if additional_rankings:
        all_rankings.extend(additional_rankings)

    if weights is None:
        weights = [1.0] * len(all_rankings)

    scores: Dict[int, float] = {}

    for w, ranking in zip(weights, all_rankings):
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_indices = [idx for idx, _ in sorted_items]
    fused_scores = dict(sorted_items)

    return fused_indices, fused_scores


def reciprocal_rank_fusion_multi(
    *rankings: List[int],
    k: int = 60
) -> Tuple[List[int], Dict[int, float]]:
    """
    Combines N ranked lists of document indices using Reciprocal Rank Fusion.
    """
    scores: Dict[int, float] = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fused_indices = [idx for idx, _ in sorted_items]
    fused_scores = dict(sorted_items)

    return fused_indices, fused_scores

