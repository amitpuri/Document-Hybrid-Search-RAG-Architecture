"""
Adaptive Hybrid Fusion: Dynamic query-intent alpha heuristic.
Selects alpha for linear fusion based on linguistic query characteristics.
"""

from typing import List, Set

import numpy as np

from src.retrieval.fusion.linear import compute_linear_scores, linear_fusion

NL_INDICATORS: Set[str] = {
    "what",
    "how",
    "why",
    "when",
    "where",
    "which",
    "does",
    "do",
    "can",
    "will",
    "should",
    "could",
    "would",
    "explain",
    "describe",
    "define",
    "compare",
    "summarize",
    "discuss",
}


# TODO (P4): compute_adaptive_alpha() is effectively a binary switch (0.3 / 0.7)
# driven by a hardcoded 20-word interrogative list and a <=4 token length threshold.
# Future: replace with a continuous signal, e.g.:
#   - BM25 score-distribution peakiness: top-1 vs top-10 score gap
#     (high peakiness -> BM25 is confident -> lower alpha; flat -> lean on dense)
#   - BM25 / TF-IDF top-k rank overlap (high overlap -> sparse is sufficient)
# This would make alpha smoothly and genuinely query-dependent rather than a
# discrete two-class classifier.
def compute_adaptive_alpha(query_text: str) -> float:
    """
    Computes optimal alpha weight:
    - Short / keyword queries (<= 4 tokens) or queries without NL interrogatives:
      benefit from exact sparse matching -> alpha = 0.3.
    - Longer queries with conversational / natural language question phrasing:
      benefit from semantic conceptual dense matching -> alpha = 0.7.
    """
    tokens = query_text.lower().split()
    if len(tokens) <= 4:
        return 0.3
    nl_hits = sum(1 for t in tokens if t in NL_INDICATORS)
    return 0.7 if nl_hits >= 1 else 0.3


def compute_adaptive_scores(
    query_text: str, sparse_scores: np.ndarray, dense_scores: np.ndarray
) -> np.ndarray:
    """Computes hybrid score vector using adaptively computed alpha."""
    alpha = compute_adaptive_alpha(query_text)
    return compute_linear_scores(sparse_scores, dense_scores, alpha=alpha)


def adaptive_hybrid_fusion(
    query_text: str, sparse_scores: np.ndarray, dense_scores: np.ndarray
) -> List[int]:
    """Applies linear fusion with an adaptively computed alpha."""
    alpha = compute_adaptive_alpha(query_text)
    return linear_fusion(sparse_scores, dense_scores, alpha=alpha)
