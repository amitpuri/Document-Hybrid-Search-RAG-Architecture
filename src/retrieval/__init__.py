"""
Retrieval pipeline package: retrievers, fusion, postprocessing, and retrieval pipeline.
"""

from src.retrieval.pipeline import RetrievalPipeline, STRATEGY_ALIASES
from src.retrieval.retrievers import (
    BaseRetriever,
    BM25Retriever,
    TfidfRetriever,
    PPMIRetriever,
    SentenceTransformerRetriever,
    CrossEncoderReranker,
)
from src.retrieval.fusion import (
    linear_fusion,
    reciprocal_rank_fusion,
    adaptive_hybrid_fusion,
)
from src.retrieval.postprocessing import (
    deduplicate_results,
    maximal_marginal_relevance,
)

__all__ = [
    "RetrievalPipeline",
    "STRATEGY_ALIASES",
    "BaseRetriever",
    "BM25Retriever",
    "TfidfRetriever",
    "PPMIRetriever",
    "SentenceTransformerRetriever",
    "CrossEncoderReranker",
    "linear_fusion",
    "reciprocal_rank_fusion",
    "adaptive_hybrid_fusion",
    "deduplicate_results",
    "maximal_marginal_relevance",
]
