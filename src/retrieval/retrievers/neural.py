"""
Neural Retrievers & Rerankers: SentenceTransformers, CrossEncoder, and SPECTER2.
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
    DEFAULT_SPECTER2_BASE_MODEL,
    DEFAULT_SPECTER2_ADAPTER,
    CACHE_DIR,
    DEFAULT_CROSS_ENCODER_POOL_SIZE
)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import adapters  # noqa: F401 — imported to verify installation
    HAS_SPECTER2 = True
except ImportError:
    HAS_SPECTER2 = False


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


class SPECTER2Retriever(BaseRetriever):
    """
    Dense vector retriever using SPECTER2 (AllenAI) with the proximity adapter.

    SPECTER2 is purpose-built for scientific paper embeddings and significantly
    outperforms generic models (MiniLM, etc.) on technical/academic corpora.
    The proximity adapter is the recommended task head for document retrieval.

    Requires:
        pip install transformers adapters torch
    Model:
        Base:    allenai/specter2_base
        Adapter: allenai/specter2_proximity  (set_active=True)
    """

    def __init__(
        self,
        base_model: str = DEFAULT_SPECTER2_BASE_MODEL,
        adapter_name: str = DEFAULT_SPECTER2_ADAPTER,
        cache_dir: str | Path = CACHE_DIR,
        batch_size: int = 16,
    ):
        self.base_model = base_model
        self.adapter_name = adapter_name
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None
        self.corpus_embeddings: Optional[np.ndarray] = None

    def _load_model(self) -> None:
        if not HAS_SPECTER2:
            raise ImportError(
                "SPECTER2 requires: pip install transformers adapters torch"
            )
        if self.model is not None:
            return
        import adapters as adapters_lib
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModel.from_pretrained(self.base_model)
        adapters_lib.init(model)
        model.load_adapter(self.adapter_name, source="hf", set_active=True)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

    def _mean_pool(self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor") -> np.ndarray:
        """Mean-pool token embeddings weighted by the attention mask."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).detach().cpu().numpy()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Tokenizes and encodes texts in batches, returns (N, D) float32 array."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            encoded = self.tokenizer(  # type: ignore[operator]
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)  # type: ignore[operator]
            embeddings = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(embeddings)
        return np.vstack(all_embeddings).astype(np.float32)

    def index(
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[str | Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Encodes all corpus texts into SPECTER2 embeddings, with disk caching."""
        self._load_model()

        if corpus_dir and not force_rebuild:
            cache_key = compute_cache_key(corpus_dir, extra_tag="specter2")
            cache_file = self.cache_dir / f"specter2_{cache_key}.npy"
            cached = load_cached_embeddings(cache_file)
            if cached is not None and len(cached) == len(corpus_texts):
                self.corpus_embeddings = cached
                return

        self.corpus_embeddings = self._encode_texts(corpus_texts)

        if corpus_dir:
            cache_key = compute_cache_key(corpus_dir, extra_tag="specter2")
            cache_file = self.cache_dir / f"specter2_{cache_key}.npy"
            save_cached_embeddings(cache_file, self.corpus_embeddings)

    def score(self, query: str) -> np.ndarray:
        """Returns cosine similarity of the query embedding against all corpus embeddings."""
        self._load_model()
        if self.corpus_embeddings is None:
            raise RuntimeError("SPECTER2Retriever must be indexed before scoring.")
        q_emb = self._encode_texts([query])
        sims = cosine_similarity(q_emb, self.corpus_embeddings).flatten()
        return sims.astype(np.float32)
