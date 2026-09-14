"""
Retrieval Pipeline coordinating indexing, scoring, fusion, and postprocessing.
Implements single dispatch across all 12 hybrid search strategies.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from pathlib import Path

from src.common.types import SearchResult, DocumentChunk
from src.ingestion.storage import BaseChunkStore
from src.retrieval.retrievers.bm25 import BM25Retriever
from src.retrieval.retrievers.tfidf import TfidfRetriever
from src.retrieval.retrievers.ppmi import PPMIRetriever
from src.retrieval.retrievers.neural import (
    SentenceTransformerRetriever,
    CrossEncoderReranker,
    SPECTER2Retriever,
    HAS_NEURAL,
    HAS_SPECTER2
)
from src.retrieval.fusion.linear import linear_fusion
from src.retrieval.fusion.rrf import reciprocal_rank_fusion
from src.retrieval.fusion.adaptive import adaptive_hybrid_fusion
from src.retrieval.postprocessing.deduplication import deduplicate_results
from src.retrieval.postprocessing.mmr import maximal_marginal_relevance
from src.config import (
    DEFAULT_RRF_K,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_DEDUP_MAX_RESULTS,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_MMR_TOP_K,
    DEFAULT_CROSS_ENCODER_POOL_SIZE,
    CORPUS_DIR,
    CACHE_DIR
)


STRATEGY_ALIASES = {
    "bm25": "1. Pure BM25 (Sparse)",
    "tfidf": "2. Pure TF-IDF (Dense)",
    "linear_0.3": "3. Linear Hybrid (a=0.3)",
    "linear_0.5": "4. Linear Hybrid (a=0.5)",
    "linear_0.7": "5. Linear Hybrid (a=0.7)",
    "rrf": "6. RRF (k=60)",
    "rrf_dedup": "7. RRF + Deduplication",
    "rrf_dedup_mmr": "8. RRF + Dedup + MMR",
    "ppmi": "9. PPMI Semantic + BM25 RRF",
    "cross_encoder": "10. Cross-Encoder Re-rank",
    "sentence_transformer": "11. Sentence-Transformer (MiniLM)",
    "adaptive": "12. Adaptive Hybrid",
    "specter2": "13. SPECTER2 (Scientific Bi-Encoder)",
}


class RetrievalPipeline:
    """
    Coordinates multi-strategy document retrieval over an ingested chunk store.
    """

    def __init__(
        self,
        chunk_store: BaseChunkStore,
        corpus_dir: Optional[str | Path] = CORPUS_DIR,
        cache_dir: Optional[str | Path] = CACHE_DIR
    ):
        self.chunk_store = chunk_store
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.corpus_texts = chunk_store.get_texts()

        # Retrievers
        self.bm25 = BM25Retriever()
        self.tfidf = TfidfRetriever()
        self.ppmi = PPMIRetriever()
        self.st_model: Optional[SentenceTransformerRetriever] = None
        self.cross_encoder: Optional[CrossEncoderReranker] = None
        self.specter2: Optional[SPECTER2Retriever] = None

        if HAS_NEURAL:
            try:
                self.st_model = SentenceTransformerRetriever(cache_dir=self.cache_dir)
                self.cross_encoder = CrossEncoderReranker()
            except Exception:
                self.st_model = None
                self.cross_encoder = None

        if HAS_SPECTER2:
            try:
                self.specter2 = SPECTER2Retriever(cache_dir=self.cache_dir)
            except Exception:
                self.specter2 = None

        self._indexed = False

    def index(self, include_neural: bool = True, include_ppmi: bool = True) -> None:
        """Indexes all underlying retriever models on the loaded corpus."""
        self.bm25.index(self.corpus_texts)
        self.tfidf.index(self.corpus_texts)

        if include_ppmi:
            self.ppmi.index(self.corpus_texts, corpus_dir=self.corpus_dir)

        if include_neural and self.st_model is not None:
            try:
                self.st_model.index(self.corpus_texts, corpus_dir=self.corpus_dir)
            except Exception:
                self.st_model = None

        if include_neural and self.cross_encoder is not None:
            try:
                self.cross_encoder.load_model()
            except Exception:
                self.cross_encoder = None

        if include_neural and self.specter2 is not None:
            try:
                self.specter2.index(self.corpus_texts, corpus_dir=self.corpus_dir)
            except Exception:
                self.specter2 = None

        self._indexed = True

    def get_strategy_rankings(self, query: str) -> Dict[str, List[int]]:
        """
        Executes query retrieval and returns full ranked chunk indices for all active strategies.
        This provides single dispatch path parity for evaluation benchmarking.
        """
        if not self._indexed:
            self.index()

        # 1. Base Sparse & Dense Scores
        b_scores = self.bm25.score(query)
        d_scores = self.tfidf.score(query)

        b_rank = np.argsort(b_scores)[::-1].tolist()
        d_rank = np.argsort(d_scores)[::-1].tolist()

        # 2. RRF wide candidate list
        rrf_wide, rrf_scores = reciprocal_rank_fusion(b_rank, d_rank, k=DEFAULT_RRF_K)

        # 3. Deduplicated candidates
        rrf_dedup_candidates = deduplicate_results(
            rrf_wide, self.corpus_texts, threshold=DEFAULT_DEDUP_THRESHOLD, max_results=DEFAULT_DEDUP_MAX_RESULTS
        )

        # 4. MMR candidates
        q_vec = self.tfidf.get_query_vector(query)
        mmr_pool = deduplicate_results(
            rrf_wide, self.corpus_texts, threshold=DEFAULT_DEDUP_THRESHOLD, max_results=20
        )
        mmr_results = maximal_marginal_relevance(
            q_vec, mmr_pool, self.tfidf.corpus_vectors, rrf_scores,
            lambda_param=DEFAULT_MMR_LAMBDA, top_k=DEFAULT_MMR_TOP_K
        )

        # 5. PPMI fusion
        ppmi_fused = self.ppmi.retrieve_ppmi_bm25_rrf(query, k=DEFAULT_RRF_K)

        rankings: Dict[str, List[int]] = {
            "1. Pure BM25 (Sparse)": b_rank,
            "2. Pure TF-IDF (Dense)": d_rank,
            "3. Linear Hybrid (a=0.3)": linear_fusion(b_scores, d_scores, alpha=0.3),
            "4. Linear Hybrid (a=0.5)": linear_fusion(b_scores, d_scores, alpha=0.5),
            "5. Linear Hybrid (a=0.7)": linear_fusion(b_scores, d_scores, alpha=0.7),
            "6. RRF (k=60)": rrf_wide,
            "7. RRF + Deduplication": rrf_dedup_candidates,
            "8. RRF + Dedup + MMR": mmr_results,
            "9. PPMI Semantic + BM25 RRF": ppmi_fused,
        }

        # 6. Neural strategies
        if self.cross_encoder is not None and self.cross_encoder.model is not None:
            ce_reranked, ce_remainder = self.cross_encoder.rerank_pool(
                query, rrf_wide, self.corpus_texts, pool_size=DEFAULT_CROSS_ENCODER_POOL_SIZE
            )
            ce_deduped = deduplicate_results(
                ce_reranked, self.corpus_texts, threshold=DEFAULT_DEDUP_THRESHOLD, max_results=DEFAULT_DEDUP_MAX_RESULTS
            )
            rankings["10. Cross-Encoder Re-rank"] = ce_deduped + ce_remainder

        if self.st_model is not None and self.st_model.corpus_embeddings is not None:
            st_scores = self.st_model.score(query)
            rankings["11. Sentence-Transformer (MiniLM)"] = np.argsort(st_scores)[::-1].tolist()

        rankings["12. Adaptive Hybrid"] = adaptive_hybrid_fusion(query, b_scores, d_scores)

        if self.specter2 is not None and self.specter2.corpus_embeddings is not None:
            sp2_scores = self.specter2.score(query)
            rankings["13. SPECTER2 (Scientific Bi-Encoder)"] = np.argsort(sp2_scores)[::-1].tolist()

        def _strategy_sort_key(name: str) -> int:
            try:
                return int(name.split(".")[0])
            except (ValueError, IndexError):
                return 999

        return dict(sorted(rankings.items(), key=lambda item: _strategy_sort_key(item[0])))

    def search(
        self,
        query: str,
        strategy: str = "rrf_dedup_mmr",
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Executes search using a specific strategy.

        Args:
            query: Search query string.
            strategy: Strategy key or full name.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects.
        """
        canonical_strategy = STRATEGY_ALIASES.get(strategy.lower(), strategy)
        all_rankings = self.get_strategy_rankings(query)

        if canonical_strategy not in all_rankings:
            # Fallback to strategy substring matching or default to RRF+Dedup+MMR
            matched = [k for k in all_rankings if strategy.lower() in k.lower()]
            canonical_strategy = matched[0] if matched else "8. RRF + Dedup + MMR"

        ranked_indices = all_rankings[canonical_strategy][:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunk_store.get_chunk(idx)
            if chunk is not None:
                results.append(SearchResult(
                    chunk_id=idx,
                    chunk=chunk,
                    score=1.0 / rank,  # relative rank score
                    rank=rank,
                    strategy=canonical_strategy
                ))

        return results
