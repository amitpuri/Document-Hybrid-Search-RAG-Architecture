"""
Result-Level Deduplication based on Jaccard Set Similarity.
Filters redundant adjacent chunks resulting from sliding-window overlap.
"""

from typing import List
from src.common.text import jaccard_similarity
from src.config import DEFAULT_DEDUP_THRESHOLD, DEFAULT_DEDUP_MAX_RESULTS


def deduplicate_results(
    candidate_indices: List[int],
    corpus_texts: List[str],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    max_results: int = DEFAULT_DEDUP_MAX_RESULTS
) -> List[int]:
    """
    Filters candidate indices to remove near-duplicate chunks.

    Args:
        candidate_indices: List of chunk indices sorted by relevance.
        corpus_texts: List of chunk text strings.
        threshold: Jaccard similarity threshold for discarding chunks (default: 0.65).
        max_results: Maximum number of deduplicated results to return.

    Returns:
        List of selected unique chunk indices.
    """
    selected_indices: List[int] = []
    selected_token_sets: List[List[str]] = []

    for idx in candidate_indices:
        chunk_text = corpus_texts[idx].lower()
        chunk_tokens = chunk_text.split()

        is_duplicate = False
        for accepted_tokens in selected_token_sets:
            sim = jaccard_similarity(chunk_tokens, accepted_tokens)
            if sim >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            selected_indices.append(idx)
            selected_token_sets.append(chunk_tokens)
            if len(selected_indices) >= max_results:
                break

    return selected_indices
