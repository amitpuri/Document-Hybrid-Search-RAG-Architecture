"""
Retrievers package: sparse, dense, semantic, and neural retrievers.
"""

from src.retrieval.retrievers.base import BaseRetriever
from src.retrieval.retrievers.bm25 import BM25Retriever
from src.retrieval.retrievers.tfidf import TfidfRetriever
from src.retrieval.retrievers.ppmi import PPMIRetriever, PPMIEmbeddings, BM25FromScratch
from src.retrieval.retrievers.neural import (
    SentenceTransformerRetriever,
    CrossEncoderReranker,
    HAS_NEURAL,
)

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "TfidfRetriever",
    "PPMIRetriever",
    "PPMIEmbeddings",
    "BM25FromScratch",
    "SentenceTransformerRetriever",
    "CrossEncoderReranker",
    "HAS_NEURAL",
]
