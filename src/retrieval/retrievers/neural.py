"""
Neural Retrievers & Rerankers: SentenceTransformers, CrossEncoder, SPECTER2.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CACHE_DIR,
    DEFAULT_BI_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_MODEL,
    DEFAULT_CROSS_ENCODER_POOL_SIZE,
    DEFAULT_SPECTER2_ADAPTER,
    DEFAULT_SPECTER2_BASE_MODEL,
    DEFAULT_SPECTER2_QUERY_ADAPTER,
)
from src.ingestion.cache import (
    compute_cache_key,
    load_cached_embeddings,
    save_cached_embeddings,
)
from src.retrieval.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

try:
    import adapters  # noqa: F401 — imported to verify installation
    import torch
    from transformers import AutoModel, AutoTokenizer

    HAS_SPECTER2 = True
except ImportError:
    HAS_SPECTER2 = False


class SentenceTransformerRetriever(BaseRetriever):
    """Dense vector retriever using pretrained SentenceTransformer embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER_MODEL,
        cache_dir: Union[str, Path] = CACHE_DIR,
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
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
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
            corpus_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
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


class BGERetriever(BaseRetriever):
    """
    Dense vector retriever using BGE (Billion-scale General Embeddings) models.

    BGE models are state-of-the-art dense retrievers, achieving top MTEB rankings.
    Available variants:
    - bge-small-en-v1.5 (384 dims): Fast, general-purpose English retrieval
    - bge-base-en-v1.5 (768 dims): Higher quality, more compute-intensive
    - bge-large-en-v1.5 (1024 dims): Highest quality for jargon-dense corpora

    Note: Requires pip install ".[modern-embeddings]" or manual sentence_transformers install.
    Benchmarking against corpus in Phase 3 (currently marked as "pending").
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: Union[str, Path] = CACHE_DIR,
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
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> None:
        self.load_model()
        self._corpus_size = len(corpus_texts)

        # Check corpus-aware cache if corpus_dir is provided
        if corpus_dir and not force_rebuild:
            bge_cache_key = compute_cache_key(corpus_dir, extra_tag="bge_small")
            cache_file = self.cache_dir / f"bge_small_{bge_cache_key}.npy"
            cached = load_cached_embeddings(cache_file)
            if cached is not None and len(cached) == len(corpus_texts):
                self.corpus_embeddings = cached
                return

        # Encode and cache
        assert self.model is not None
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if corpus_dir:
            bge_cache_key = compute_cache_key(corpus_dir, extra_tag="bge_small")
            cache_file = self.cache_dir / f"bge_small_{bge_cache_key}.npy"
            save_cached_embeddings(cache_file, self.corpus_embeddings)

    def score(self, query: str) -> np.ndarray:
        self.load_model()
        if self.corpus_embeddings is None:
            raise RuntimeError("BGERetriever must be indexed before scoring.")
        assert self.model is not None
        q_vec = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, self.corpus_embeddings).flatten()
        return sims.astype(np.float32)


class E5Retriever(BaseRetriever):
    """
    Dense vector retriever using E5 (Text Embeddings by Contrastive Learning) models.

    E5 models are trained via contrastive learning on diverse datasets and are known for
    strong generalization across domains and tasks.
    Available variants:
    - intfloat/e5-small-v2 (384 dims): Fast, good generalization
    - intfloat/e5-base-v2 (768 dims): Balance of speed and quality
    - intfloat/e5-large-v2 (1024 dims): Highest quality for specialized domains

    Note: E5 models benefit from instructional prefixes in queries:
    - Queries: prefix with "query: "
    - Documents: prefix with "passage: "
    This retriever applies prefixes automatically.

    Benchmarking against corpus in Phase 3 (currently marked as "pending").
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",
        cache_dir: Union[str, Path] = CACHE_DIR,
        use_instruction_prefix: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_instruction_prefix = use_instruction_prefix
        self.model: Optional[SentenceTransformer] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._corpus_size = 0

    def load_model(self) -> None:
        if not HAS_NEURAL:
            raise ImportError("sentence_transformers is not installed.")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _add_prefix(self, texts: List[str], prefix: str = "passage: ") -> List[str]:
        """Optionally adds E5 instruction prefix to texts."""
        if self.use_instruction_prefix:
            return [f"{prefix}{text}" for text in texts]
        return texts

    def index(  # noqa: F811
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> None:
        self.load_model()
        self._corpus_size = len(corpus_texts)

        # Check corpus-aware cache if corpus_dir is provided
        if corpus_dir and not force_rebuild:
            e5_cache_key = compute_cache_key(corpus_dir, extra_tag="e5_small")
            cache_file = self.cache_dir / f"e5_small_{e5_cache_key}.npy"
            cached = load_cached_embeddings(cache_file)
            if cached is not None and len(cached) == len(corpus_texts):
                self.corpus_embeddings = cached
                return

        # Encode with passage prefix and cache
        assert self.model is not None
        prefixed_texts = self._add_prefix(corpus_texts, prefix="passage: ")
        self.corpus_embeddings = self.model.encode(
            prefixed_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if corpus_dir:
            e5_cache_key = compute_cache_key(corpus_dir, extra_tag="e5_small")
            cache_file = self.cache_dir / f"e5_small_{e5_cache_key}.npy"
            save_cached_embeddings(cache_file, self.corpus_embeddings)

    def score(self, query: str) -> np.ndarray:
        self.load_model()
        if self.corpus_embeddings is None:
            raise RuntimeError("E5Retriever must be indexed before scoring.")
        assert self.model is not None
        prefixed_query = self._add_prefix([query], prefix="query: ")[0]
        q_vec = self.model.encode([prefixed_query], convert_to_numpy=True)
        sims = cosine_similarity(q_vec, self.corpus_embeddings).flatten()
        return sims.astype(np.float32)


class CrossEncoderReranker:
    """
    Reranker using a pretrained CrossEncoder over candidate pairs.

    NOTE: Domain Mismatch Caveat
    ────────────────────────────
    The default model (ms-marco-MiniLM-L-6-v2) is trained on MS MARCO web Q&A passages,
    not scientific/technical PDFs. On technical literature corpora, cross-encoder reranking
    may underperform the upstream RRF retriever due to domain mismatch. This is expected
    behavior, not a bug. See docs/researchpaper.md Section 6.6 for empirical evidence and
    recommended alternatives (domain-adapted scientific rerankers).
    """

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL):
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None

    def load_model(self) -> None:
        if not HAS_NEURAL:
            raise ImportError("sentence_transformers is not installed.")
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank_pool_with_scores(
        self,
        query: str,
        candidate_indices: List[int],
        corpus_texts: List[str],
        pool_size: int = DEFAULT_CROSS_ENCODER_POOL_SIZE,
    ) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Takes candidate indices (e.g. from RRF), extracts a top pool of pool_size,
        reranks the pool via cross-encoder, and returns reranked_pool, remainder,
        and pool_scores_dict.
        """
        self.load_model()
        assert self.model is not None

        pool = candidate_indices[:pool_size]
        remainder = candidate_indices[pool_size:]

        pairs = [(query, corpus_texts[idx]) for idx in pool]
        scores = self.model.predict(pairs)
        scores_dict = {pool[i]: float(scores[i]) for i in range(len(pool))}
        reranked_pool = [pool[i] for i in np.argsort(scores)[::-1]]

        return reranked_pool, remainder, scores_dict

    def rerank_pool(
        self,
        query: str,
        candidate_indices: List[int],
        corpus_texts: List[str],
        pool_size: int = DEFAULT_CROSS_ENCODER_POOL_SIZE,
    ) -> Tuple[List[int], List[int]]:
        """
        Takes candidate indices (e.g. from RRF), extracts a top pool of pool_size,
        reranks the pool via cross-encoder, and returns (reranked_pool, remainder).
        """
        reranked_pool, remainder, _ = self.rerank_pool_with_scores(
            query, candidate_indices, corpus_texts, pool_size=pool_size
        )
        return reranked_pool, remainder


class SPECTER2Retriever(BaseRetriever):
    """
    Dense vector retriever using SPECTER2 (AllenAI) with dual asymmetric task adapters.

    SPECTER2 is purpose-built for scientific paper embeddings. To support effective
    retrieval between short queries and long document passages:
      - Document passages are encoded using `allenai/specter2_proximity`
      - Search queries are encoded using `allenai/specter2_adhoc_query`
      - Embeddings are pooled from the [CLS] token (index 0), as trained.

    Requires:
        pip install transformers adapters torch
    Models:
        Base:          allenai/specter2_base
        Doc Adapter:   allenai/specter2_proximity
        Query Adapter: allenai/specter2_adhoc_query
    """

    def __init__(
        self,
        base_model: str = DEFAULT_SPECTER2_BASE_MODEL,
        doc_adapter: str = DEFAULT_SPECTER2_ADAPTER,
        query_adapter: str = DEFAULT_SPECTER2_QUERY_ADAPTER,
        cache_dir: Union[str, Path] = CACHE_DIR,
        batch_size: int = 16,
    ):
        self.base_model = base_model
        self.doc_adapter = doc_adapter
        self.query_adapter = query_adapter
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.tokenizer: Optional[object] = None
        self.model: Optional[object] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self._loaded_adapters: set = set()
        self._adapter_internal_names: dict = {}

    @property
    def adapter_name(self) -> str:
        """Backwards compatibility alias for default doc_adapter."""
        return self.doc_adapter

    def _load_model(self, adapter_to_activate: Optional[str] = None) -> None:
        if not HAS_SPECTER2:
            raise ImportError("SPECTER2 requires: pip install transformers adapters torch")
        import adapters as adapters_lib

        if self.model is None or self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            model = AutoModel.from_pretrained(self.base_model)
            adapters_lib.init(model)
            self.tokenizer = tokenizer
            self.model = model

        target_adapter = adapter_to_activate or self.doc_adapter
        if target_adapter not in self._loaded_adapters:
            try:
                internal_name = self.model.load_adapter(  # type: ignore[union-attr]
                    target_adapter, source="hf", set_active=True
                )
                self._adapter_internal_names[target_adapter] = internal_name or target_adapter
                self._loaded_adapters.add(target_adapter)
                logger.info(
                    "SPECTER2 successfully loaded adapter '%s' (internal name: '%s')",
                    target_adapter,
                    self._adapter_internal_names[target_adapter],
                )
            except Exception as e:
                logger.error(
                    "SPECTER2 failed to load adapter '%s': %s",
                    target_adapter,
                    e,
                )
                raise RuntimeError(
                    f"SPECTER2 failed to load required adapter '{target_adapter}' "
                    f"from HuggingFace Hub: {e}"
                ) from e
        else:
            internal_name = self._adapter_internal_names.get(target_adapter, target_adapter)
            self.model.set_active_adapters(internal_name)  # type: ignore[union-attr]

        self.model.eval()  # type: ignore[union-attr]

    def _encode_texts(self, texts: List[str], adapter_name: Optional[str] = None) -> np.ndarray:
        """
        Tokenizes and encodes texts in batches using the [CLS] token representation.
        Returns an (N, D) float32 numpy array.
        """
        self._load_model(adapter_to_activate=adapter_name)
        assert self.model is not None and self.tokenizer is not None

        active = getattr(self.model, "active_adapters", None)
        active_repr = active() if callable(active) else active
        logger.info(
            "SPECTER2 encoding %d texts with target adapter '%s' (active: %s)",
            len(texts),
            adapter_name,
            active_repr,
        )

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            encoded = self.tokenizer(  # type: ignore[operator]
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = self.model(**encoded)  # type: ignore[operator]
            # SPECTER2 uses CLS token (index 0) representation, NOT mean pooling
            cls_embeddings = output.last_hidden_state[:, 0, :].detach().cpu().numpy()
            all_embeddings.append(cls_embeddings)

        embeddings_arr = np.vstack(all_embeddings).astype(np.float32)
        return embeddings_arr

    def index(
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Encodes all corpus texts into SPECTER2 proximity embeddings, with disk caching."""
        if corpus_dir and not force_rebuild:
            cache_key = compute_cache_key(corpus_dir, extra_tag="specter2_cls")
            cache_file = self.cache_dir / f"specter2_{cache_key}.npy"
            cached = load_cached_embeddings(cache_file)
            if cached is not None and len(cached) == len(corpus_texts):
                self.corpus_embeddings = cached
                logger.info(
                    "SPECTER2 loaded cached embeddings from %s: shape=%s, "
                    "mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    cache_file.name,
                    self.corpus_embeddings.shape,
                    float(self.corpus_embeddings.mean()),
                    float(self.corpus_embeddings.std()),
                    float(self.corpus_embeddings.min()),
                    float(self.corpus_embeddings.max()),
                )
                return

        self.corpus_embeddings = self._encode_texts(corpus_texts, adapter_name=self.doc_adapter)
        logger.info(
            "SPECTER2 indexed %d texts: shape=%s, mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            len(corpus_texts),
            self.corpus_embeddings.shape,
            float(self.corpus_embeddings.mean()),
            float(self.corpus_embeddings.std()),
            float(self.corpus_embeddings.min()),
            float(self.corpus_embeddings.max()),
        )

        if corpus_dir:
            cache_key = compute_cache_key(corpus_dir, extra_tag="specter2_cls")
            cache_file = self.cache_dir / f"specter2_{cache_key}.npy"
            save_cached_embeddings(cache_file, self.corpus_embeddings)

    def score(self, query: str) -> np.ndarray:
        """Returns cosine similarity of query embedding (via adhoc_query adapter) against corpus."""
        if self.corpus_embeddings is None:
            raise RuntimeError("SPECTER2Retriever must be indexed before scoring.")
        q_emb = self._encode_texts([query], adapter_name=self.query_adapter)
        sims = cosine_similarity(q_emb, self.corpus_embeddings).flatten()
        return sims.astype(np.float32)
