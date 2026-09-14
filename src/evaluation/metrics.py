"""
Information Retrieval Evaluation Metrics.
Computes MRR, Recall@K, and NDCG@5 with fine-grained chunk-level relevance.
"""

import math
from typing import List, Optional, Set, Sequence
from src.common.types import MetricScores


def is_relevant(
    chunk_text: str,
    chunk_idx: int,
    target_doc: str,
    target_chunk_idx: Optional[int | Set[int] | List[int]] = None
) -> bool:
    """Returns True if the chunk matches target_chunk_idx (or doc prefix fallback)."""
    if target_chunk_idx is not None:
        if isinstance(target_chunk_idx, (set, list, tuple)):
            return chunk_idx in target_chunk_idx
        return chunk_idx == target_chunk_idx
    return target_doc.lower() in chunk_text[:120].lower()


def evaluate_ranking(
    ranked_indices: Sequence[int],
    corpus_texts: Sequence[str],
    target_doc: str,
    target_chunk_idx: Optional[int | Set[int] | List[int]] = None,
    k_list: Sequence[int] = (1, 3, 5)
) -> MetricScores:
    """
    Computes MRR, Recall@K, and NDCG@5 for a single query ranking.
    """
    relevance_flags = [
        is_relevant(corpus_texts[idx], idx, target_doc, target_chunk_idx)
        for idx in ranked_indices
    ]

    # MRR (Mean Reciprocal Rank)
    reciprocal_rank = 0.0
    for rank, rel in enumerate(relevance_flags, start=1):
        if rel:
            reciprocal_rank = 1.0 / rank
            break

    # Recall@K
    recalls = {k: 1.0 if any(relevance_flags[:k]) else 0.0 for k in k_list}

    # NDCG@5
    dcg = sum(
        (1.0 if rel else 0.0) / math.log2(rank + 1)
        for rank, rel in enumerate(relevance_flags[:5], start=1)
    )
    idcg = 1.0 / math.log2(2)  # Ideal: 1 relevant doc at rank 1
    ndcg_5 = dcg / idcg if idcg > 0 else 0.0

    return MetricScores(
        mrr=reciprocal_rank,
        recall_1=recalls.get(1, 0.0),
        recall_3=recalls.get(3, 0.0),
        recall_5=recalls.get(5, 0.0),
        ndcg_5=ndcg_5
    )
