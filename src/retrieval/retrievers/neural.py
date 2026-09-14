"""
Neural Retrievers & Rerankers: SentenceTransformers and CrossEncoder.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.retrievers.base import BaseRetriever
from src.ingestion.cache import compute_cache_key, load_cached_embeddings, save_cached_embeddings
from src.config import (
    DEFAULT_BI_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_MODEL,
    CACHE_DIR,
    DEFAULT_CROSS_ENCODER_POOL_SIZE
)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


class SentenceTransformerRetriever(BaseRetriever):
    """Dense vector retriever using pretrained SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER_MODEL,
        cache_dir: str | Path = CACHE_DIR
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.model: Optional[SentenceTransformer] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._corpus_size = 0

    def load_model(self) -> None:
        if not HAS_NEURAL:
            raise ImportError("sentence_transformers is not installed.")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def index(
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[str | Path] = None,
        force_rebuild: bool = False
    ) -> None:
        self.load_model()
        self._corpus_size = len(corpus_texts)

        # Check corpus-aware cache if corpus_dir is provided
        if corpus_dir and not force_rebuild:
            st_cache_key = compute_cache_key(corpus_dir, extra_tag="st_minilm")
            cache_file = self.cache_dir / f"st_minilm_{st_cache_key}.npy"
            cached = load_cached_embeddings(cache_file)
            if cached is not None and len(cached) == len(corpus_texts):
                self.corpus_embeddings = cached
                return

        # Encode and cache
        assert self.model is not None
        self.corpus_embeddings = self.model.encode(
            corpus_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        if corpus_dir:
            st_cache_key = compute_cache_key(corpus_dir, extra_tag="st_minilm")
            cache_file = self.cache_dir / f"st_minilm_{st_cache_key}.npy"
            save_cached_embeddings(cache_file, self.corpus_embeddings)

    def score(self, query: str) -> np.ndarray:
        self.load_model()
        if self.corpus_embeddings is None:
            raise RuntimeError("SentenceTransformerRetriever must be indexed before scoring.")
        assert self.model is not None
        q_vec = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, self.corpus_embeddings).flatten()
        return sims.astype(np.float32)


class CrossEncoderReranker:
    """Reranker using a pretrained CrossEncoder over candidate pairs."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None

    def load_model(self) -> None:
        if not HAS_NEURAL:
            raise ImportError("sentence_transformers is not installed.")
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank_pool(
        self,
        query: str,
        candidate_indices: List[int],
        corpus_texts: List[str],
        pool_size: int = DEFAULT_CROSS_ENCODER_POOL_SIZE
    ) -> Tuple[List[int], List[int]]:
        """
        Takes candidate indices (e.g. from RRF), extracts a top pool of pool_size,
        reranks the pool via cross-encoder, and returns (reranked_pool, remainder).
        """
        self.load_model()
        assert self.model is not None

        pool = candidate_indices[:pool_size]
        remainder = candidate_indices[pool_size:]

        pairs = [(query, corpus_texts[idx]) for idx in pool]
        scores = self.model.predict(pairs)
        reranked_pool = [pool[i] for i in np.argsort(scores)[::-1]]

        return reranked_pool, remainder
