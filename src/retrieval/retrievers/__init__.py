"""
Retrievers package: sparse, dense, semantic, and neural retrievers.
"""

from src.retrieval.retrievers.base import BaseRetriever
from src.retrieval.retrievers.bm25 import BM25Retriever
from src.retrieval.retrievers.neural import (
    HAS_NEURAL,
    BGERetriever,
    CrossEncoderReranker,
    E5Retriever,
    SentenceTransformerRetriever,
    SPECTER2Retriever,
)
from src.retrieval.retrievers.ppmi import (
    BM25FromScratch,
    PPMIEmbeddings,
    PPMIRetriever,
)
from src.retrieval.retrievers.tfidf import TfidfRetriever

try:
    from src.retrieval.retrievers.qdrant import HAS_QDRANT, QdrantRetriever
except ImportError:
    QdrantRetriever = None  # type: ignore[misc]
    HAS_QDRANT = False

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "TfidfRetriever",
    "PPMIRetriever",
    "PPMIEmbeddings",
    "BM25FromScratch",
    "SentenceTransformerRetriever",
    "CrossEncoderReranker",
    "BGERetriever",
    "E5Retriever",
    "SPECTER2Retriever",
    "QdrantRetriever",
    "HAS_NEURAL",
    "HAS_QDRANT",
]
