"""
TF-IDF Vector Space Model Retriever.
"""

from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.retrievers.base import BaseRetriever


class TfidfRetriever(BaseRetriever):
    """Dense TF-IDF Vector Space Model retriever using cosine similarity."""

    def __init__(self, sublinear_tf: bool = True):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.corpus_vectors = None
        self._corpus_size: int = 0

    def index(self, corpus_texts: List[str]) -> None:
        self.corpus_vectors = self.vectorizer.fit_transform(corpus_texts)
        self._corpus_size = len(corpus_texts)

    def get_query_vector(self, query: str):
        """Returns the TF-IDF sparse matrix representation for a query."""
        return self.vectorizer.transform([query])

    def score(self, query: str) -> np.ndarray:
        if self.corpus_vectors is None:
            raise RuntimeError("TfidfRetriever must be indexed before scoring.")
        q_vec = self.get_query_vector(query)
        return cosine_similarity(q_vec, self.corpus_vectors).flatten().astype(np.float32)
