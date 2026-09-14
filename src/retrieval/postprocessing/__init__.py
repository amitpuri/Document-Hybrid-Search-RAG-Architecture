"""
Postprocessing package: result deduplication and MMR diversity re-ranking.
"""

from src.retrieval.postprocessing.deduplication import deduplicate_results
from src.retrieval.postprocessing.mmr import maximal_marginal_relevance

__all__ = [
    "deduplicate_results",
    "maximal_marginal_relevance",
]
