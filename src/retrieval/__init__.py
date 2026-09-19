"""
Retrieval pipeline package: retrievers, fusion, postprocessing, and retrieval pipeline.
"""

from src.retrieval.fusion import (
    adaptive_hybrid_fusion,
    linear_fusion,
    reciprocal_rank_fusion,
)
from src.retrieval.pipeline import STRATEGY_ALIASES, RetrievalPipeline
from src.retrieval.postprocessing import (
    deduplicate_results,
    maximal_marginal_relevance,
)
from src.retrieval.retrievers import (
    BaseRetriever,
    BM25Retriever,
    CrossEncoderReranker,
    PPMIRetriever,
    SentenceTransformerRetriever,
    TfidfRetriever,
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
