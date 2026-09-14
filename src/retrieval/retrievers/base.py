"""
Abstract Base Class for Retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseRetriever(ABC):
    """Abstract interface for all sparse, dense, and semantic retrievers."""

    @abstractmethod
    def index(self, corpus_texts: List[str]) -> None:
        """Indexes or fits the retriever model on a list of document strings."""
        pass

    @abstractmethod
    def score(self, query: str) -> np.ndarray:
        """
        Scores all indexed documents against the query.
        Returns a 1D numpy array of scores of length len(corpus_texts).
        """
        pass

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieves top_k document indices with their scores.
        Returns list of (doc_index, score) pairs sorted descending by score.
        """
        scores = self.score(query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in ranked_indices]
