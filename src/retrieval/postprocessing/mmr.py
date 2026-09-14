r"""
Maximal Marginal Relevance (MMR) Diversity Re-Ranking.
Balances relevance to query with diversity among selected documents.
Formula: MMR = argmax_{d in R \ S} [ lambda * Sim(d, q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.config import DEFAULT_MMR_LAMBDA, DEFAULT_MMR_TOP_K


def maximal_marginal_relevance(
    query_vec: Any,
    candidate_indices: List[int],
    corpus_vecs: Any,
    relevance_scores: Dict[int, float],
    lambda_param: float = DEFAULT_MMR_LAMBDA,
    top_k: int = DEFAULT_MMR_TOP_K
) -> List[int]:
    """
    Applies MMR to balance relevance with diversity.

    Args:
        query_vec: Vector representation of the query (e.g. TF-IDF sparse vector).
        candidate_indices: Deduplicated pool of candidate indices.
        corpus_vecs: Matrix of vectors for all corpus documents.
        relevance_scores: Dict mapping chunk index -> relevance score (e.g. RRF score).
        lambda_param: Trade-off parameter (1.0 = pure relevance, 0.0 = pure diversity).
        top_k: Number of diverse documents to select.

    Returns:
        List of selected chunk indices.
    """
    if not candidate_indices:
        return []

    pool = list(candidate_indices[:min(len(candidate_indices), top_k * 4)])
    selected: List[int] = []

    pool_rel = np.array([relevance_scores.get(idx, 0.0) for idx in pool], dtype=np.float32)
    min_r, max_r = np.min(pool_rel), np.max(pool_rel)
    if max_r > min_r:
        pool_rel_norm = {idx: float((r - min_r) / (max_r - min_r)) for idx, r in zip(pool, pool_rel)}
    else:
        pool_rel_norm = {idx: 1.0 for idx in pool}

    while pool and len(selected) < top_k:
        if not selected:
            best_idx = max(pool, key=lambda idx: pool_rel_norm[idx])
            selected.append(best_idx)
            pool.remove(best_idx)
            continue

        best_score = -float("inf")
        best_candidate = None

        selected_vecs = corpus_vecs[selected]

        for cand in pool:
            cand_vec = corpus_vecs[cand]
            rel_score = pool_rel_norm[cand]

            sim_to_selected = float(np.max(cosine_similarity(cand_vec, selected_vecs)))
            mmr_score = (lambda_param * rel_score) - ((1.0 - lambda_param) * sim_to_selected)

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand

        if best_candidate is not None:
            selected.append(best_candidate)
            pool.remove(best_candidate)

    return selected
