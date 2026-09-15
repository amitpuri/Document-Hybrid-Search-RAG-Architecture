"""
Information Retrieval Evaluation Metrics.
Computes MRR, Recall@K, and NDCG@5 with fine-grained chunk-level relevance.
"""

import math
import re
from typing import List, Optional, Set, Sequence, Tuple
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


def evaluate_graph_coverage(
    context_text: str,
    target_entities: Optional[Sequence[str]] = None,
    target_relations: Optional[Sequence[Tuple[str, str, str]]] = None,
    chunk_texts: Optional[Sequence[str]] = None,
) -> Tuple[float, float]:
    """
    Computes deterministic entity and relationship coverage against ground-truth sets.
    Uses case-insensitive regex word boundary matching.

    For entity coverage: checks each entity appears anywhere in the concatenated context.
    For relation coverage: requires ALL THREE terms (source, predicate, target) to co-occur
    within the SAME individual chunk (per-chunk window), not just anywhere in the blob.
    This prevents false positives where terms span different chunks.
    """
    entity_cov = 0.0
    if target_entities:
        matched_entities = 0
        for ent in target_entities:
            pattern = r"\b" + re.escape(ent.strip()) + r"\b"
            if re.search(pattern, context_text, re.IGNORECASE):
                matched_entities += 1
        entity_cov = matched_entities / len(target_entities)

    rel_cov = 0.0
    if target_relations:
        matched_rels = 0
        # Use individual chunk texts for per-chunk co-occurrence check if provided,
        # otherwise fall back to checking whole concatenated context (legacy behaviour)
        windows: Sequence[str] = chunk_texts if chunk_texts else [context_text]
        for src, pred, tgt in target_relations:
            p_src = re.compile(r"\b" + re.escape(src.strip()) + r"\b", re.IGNORECASE)
            p_pred = re.compile(r"\b" + re.escape(pred.strip()) + r"\b", re.IGNORECASE)
            p_tgt = re.compile(r"\b" + re.escape(tgt.strip()) + r"\b", re.IGNORECASE)
            # A relation is covered only if all three terms appear in the SAME window
            covered = any(
                p_src.search(window) and p_pred.search(window) and p_tgt.search(window)
                for window in windows
            )
            if covered:
                matched_rels += 1
        rel_cov = matched_rels / len(target_relations)

    return entity_cov, rel_cov


def evaluate_ranking(
    ranked_indices: Sequence[int],
    corpus_texts: Sequence[str],
    target_doc: str,
    target_chunk_idx: Optional[int | Set[int] | List[int]] = None,
    k_list: Sequence[int] = (1, 3, 5),
    target_entities: Optional[Sequence[str]] = None,
    target_relations: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> MetricScores:
    """
    Computes MRR, Recall@K, NDCG@5, and Entity/Relation Graph Coverage for a single query ranking.
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

    # Top-5 Context Window for Entity/Relation Coverage
    top5_indices = ranked_indices[:5]
    top5_chunk_texts = [corpus_texts[idx] for idx in top5_indices if 0 <= idx < len(corpus_texts)]
    top5_context = " ".join(top5_chunk_texts)
    ent_cov, rel_cov = evaluate_graph_coverage(
        top5_context, target_entities, target_relations, chunk_texts=top5_chunk_texts
    )

    return MetricScores(
        mrr=reciprocal_rank,
        recall_1=recalls.get(1, 0.0),
        recall_3=recalls.get(3, 0.0),
        recall_5=recalls.get(5, 0.0),
        ndcg_5=ndcg_5,
        entity_coverage=ent_cov,
        relation_coverage=rel_cov,
    )

