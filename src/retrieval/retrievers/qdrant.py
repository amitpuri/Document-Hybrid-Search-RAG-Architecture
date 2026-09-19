"""
Qdrant Vector Retriever using Approximate Nearest Neighbor (ANN) HNSW search.
Integrates local or remote Qdrant vector database with dense sentence embeddings.
"""

import logging
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np

from src.common.types import DocumentChunk, SearchResult
from src.config import (
    CACHE_DIR,
    DEFAULT_BI_ENCODER_MODEL,
    QDRANT_COLLECTION_NAME,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_TIMEOUT,
    QDRANT_VECTOR_SIZE,
)
from src.ingestion.cache import (
    compute_cache_key,
    load_cached_embeddings,
    save_cached_embeddings,
)
from src.retrieval.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels

    HAS_QDRANT = True
except ImportError:
    QdrantClient = None
    qmodels = None
    HAS_QDRANT = False


class QdrantRetriever(BaseRetriever):
    """
    Dense vector retriever backed by Qdrant HNSW vector search.
    Provides scalable, low-latency approximate nearest neighbor (ANN) retrieval
    and optional payload-level predicate filtering.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER_MODEL,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        prefer_grpc: bool = QDRANT_PREFER_GRPC,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_size: int = QDRANT_VECTOR_SIZE,
        timeout: float = QDRANT_TIMEOUT,
        client: Optional[Any] = None,
        cache_dir: Union[str, Path] = CACHE_DIR,
    ):
        if not HAS_QDRANT:
            msg = (
                "qdrant-client is required to use QdrantRetriever. "
                "Install with: pip install qdrant-client"
            )
            raise ImportError(msg)

        self.model_name = model_name
        # Model-tag default collection to avoid cross-model vector contamination
        model_tag = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        if collection_name == QDRANT_COLLECTION_NAME:
            self.collection_name = f"{QDRANT_COLLECTION_NAME}_{model_tag}"
        else:
            self.collection_name = collection_name
        self.vector_size = vector_size
        self.cache_dir = Path(cache_dir)
        self.model: Optional[SentenceTransformer] = None
        self._corpus_size = 0

        if client is not None:
            self.client = client
        else:
            self.client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=timeout,
            )

    def load_model(self) -> None:
        """Lazily loads the dense sentence embedding model."""
        if not HAS_NEURAL:
            raise ImportError(
                "sentence_transformers is required for QdrantRetriever dense embeddings."
            )
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def index(
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Indexes corpus embeddings into Qdrant collection under the 'dense' vector name.
        Uses cached embeddings if available, or computes and caches them.
        """
        self.load_model()
        self._corpus_size = len(corpus_texts)

        # Check if collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=self.vector_size,
                        distance=qmodels.Distance.COSINE,
                    )
                },
            )

        # 1. Obtain or compute embeddings
        corpus_embeddings = embeddings
        if corpus_embeddings is None:
            if corpus_dir and not force_rebuild:
                st_cache_key = compute_cache_key(corpus_dir, extra_tag="st_minilm")
                cache_file = self.cache_dir / f"st_minilm_{st_cache_key}.npy"
                cached = load_cached_embeddings(cache_file)
                if cached is not None and len(cached) == len(corpus_texts):
                    corpus_embeddings = cached

            if corpus_embeddings is None:
                assert self.model is not None
                corpus_embeddings = self.model.encode(
                    corpus_texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                if corpus_dir:
                    st_cache_key = compute_cache_key(corpus_dir, extra_tag="st_minilm")
                    cache_file = self.cache_dir / f"st_minilm_{st_cache_key}.npy"
                    save_cached_embeddings(cache_file, corpus_embeddings)

        # 2. Upsert vectors in Qdrant (ensures points are created if not present)
        points: List[qmodels.PointStruct] = []
        for idx in range(len(corpus_texts)):
            vec = corpus_embeddings[idx]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            points.append(
                qmodels.PointStruct(
                    id=int(idx),
                    vector={"dense": vec},
                    payload={"chunk_id": int(idx)},
                )
            )

        batch_size = 250
        for offset in range(0, len(points), batch_size):
            batch = points[offset : offset + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
        logger.info(
            "QdrantRetriever indexed %d vectors into collection '%s'",
            len(points),
            self.collection_name,
        )

    def score(self, query: str) -> np.ndarray:
        """
        Scores all chunks in the corpus against query using Qdrant vector ANN search.
        Returns a float32 numpy array indexed by chunk_id.
        """
        self.load_model()
        assert self.model is not None

        q_vec = self.model.encode([query], convert_to_numpy=True)[0]
        if hasattr(q_vec, "tolist"):
            q_vec = q_vec.tolist()

        limit = max(self._corpus_size, 1)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            using="dense",
            limit=limit,
            with_payload=False,
            with_vectors=False,
        )

        scores = np.zeros(self._corpus_size, dtype=np.float32)
        for point in response.points:
            chunk_id = int(point.id)
            if 0 <= chunk_id < self._corpus_size:
                scores[chunk_id] = float(point.score)

        logger.debug(
            "Qdrant scored query points: received %d points for collection '%s'",
            len(response.points),
            self.collection_name,
        )
        return scores

    def search_top_k(
        self,
        query: str,
        top_k: int = 5,
        query_filter: Optional[Any] = None,
    ) -> List[SearchResult]:
        """
        Executes sub-millisecond approximate nearest neighbor search directly in Qdrant.
        Returns top_k SearchResults with reconstructed DocumentChunks from point payloads.
        """
        self.load_model()
        assert self.model is not None

        q_vec = self.model.encode([query], convert_to_numpy=True)[0]
        if hasattr(q_vec, "tolist"):
            q_vec = q_vec.tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            query_filter=query_filter,
            using="dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: List[SearchResult] = []
        for rank, p in enumerate(response.points, start=1):
            payload = p.payload or {}
            chunk = DocumentChunk(
                chunk_id=int(payload.get("chunk_id", p.id)),
                doc_name=str(payload.get("doc_name", "Unknown")),
                page_num=int(payload.get("page_num", 1)),
                section=str(payload.get("section", "")),
                text=str(payload.get("text", "")),
                metadata=payload.get("metadata", {}),
            )
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    chunk=chunk,
                    score=float(p.score),
                    rank=rank,
                    strategy="15. Qdrant Vector (ANN)",
                )
            )

        return results
