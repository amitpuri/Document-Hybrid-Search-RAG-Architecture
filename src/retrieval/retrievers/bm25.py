"""
BM25 Sparse Keyword Retriever.
"""

from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.retrieval.retrievers.base import BaseRetriever
from src.common.text import tokenize


class BM25Retriever(BaseRetriever):
    """BM25 Okapi sparse retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_size: int = 0

    def index(self, corpus_texts: List[str]) -> None:
        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self._corpus_size = len(corpus_texts)

    def score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever must be indexed before scoring.")
        q_tokens = query.lower().split()
        return np.array(self._bm25.get_scores(q_tokens), dtype=np.float32)
