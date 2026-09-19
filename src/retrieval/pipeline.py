"""
Retrieval Pipeline coordinating indexing, scoring, fusion, and postprocessing.
Implements single dispatch across all 15 hybrid search strategies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.common.types import SearchResult
from src.config import (
    CACHE_DIR,
    CORPUS_DIR,
    DEFAULT_CROSS_ENCODER_POOL_SIZE,
    DEFAULT_DEDUP_MAX_RESULTS,
    DEFAULT_DEDUP_THRESHOLD,
    DEFAULT_MMR_LAMBDA,
    DEFAULT_MMR_TOP_K,
    DEFAULT_RRF_K,
)
from src.ingestion.storage import BaseChunkStore
from src.retrieval.fusion.adaptive import compute_adaptive_scores
from src.retrieval.fusion.linear import compute_linear_scores
from src.retrieval.fusion.rrf import reciprocal_rank_fusion
from src.retrieval.postprocessing.deduplication import deduplicate_results
from src.retrieval.postprocessing.mmr import maximal_marginal_relevance
from src.retrieval.retrievers.bm25 import BM25Retriever
from src.retrieval.retrievers.graph import GraphRetriever
from src.retrieval.retrievers.neural import (
    HAS_NEURAL,
    HAS_SPECTER2,
    CrossEncoderReranker,
    SentenceTransformerRetriever,
    SPECTER2Retriever,
)
from src.retrieval.retrievers.ppmi import PPMIRetriever
from src.retrieval.retrievers.tfidf import TfidfRetriever

logger = logging.getLogger(__name__)

try:
    from src.retrieval.retrievers.qdrant import HAS_QDRANT, QdrantRetriever
except ImportError:
    QdrantRetriever = None  # type: ignore[misc]
    HAS_QDRANT = False

# TODO (P6): Strategy identity is stringly-typed and sort order is parsed from
# name.split(".")[0]. Future: refactor STRATEGY_ALIASES and the rankings dict into
# a small Strategy enum/dataclass with an explicit `order` field to eliminate
# rename/typo risk across pipeline.py, harness.py, and the README table.
STRATEGY_ALIASES = {
    "bm25": "1. Pure BM25 (Sparse)",
    "tfidf": "2. Pure TF-IDF (Sparse Vector Space)",
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
    "rrf_graph_dedup_mmr": "14. RRF + Graph + Dedup + MMR",
    "graph": "14. RRF + Graph + Dedup + MMR",
    "qdrant": "15. Qdrant Vector (ANN)",
    "qdrant_vector": "15. Qdrant Vector (ANN)",
    # Ablation cells (P0 - Factorial isolation of Graph, Dedup, MMR components)
    "graph_only": "Ablation: Graph only",
    "rrf_graph": "Ablation: RRF + Graph (no dedup/MMR)",
    "rrf_graph_dedup": "Ablation: RRF + Graph + Dedup (no MMR)",
}


class RetrievalPipeline:
    """
    Coordinates multi-strategy document retrieval over an ingested chunk store.
    """

    def __init__(
        self,
        chunk_store: BaseChunkStore,
        corpus_dir: Optional[Union[str, Path]] = CORPUS_DIR,
        cache_dir: Optional[Union[str, Path]] = CACHE_DIR,
    ):
        self.chunk_store = chunk_store
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.corpus_texts = chunk_store.get_texts()

        # Retrievers
        self.bm25 = BM25Retriever()
        self.tfidf = TfidfRetriever()
        self.ppmi = PPMIRetriever()
        self.graph_retriever = GraphRetriever(cache_dir=self.cache_dir)
        self.st_model: Optional[SentenceTransformerRetriever] = None
        self.cross_encoder: Optional[CrossEncoderReranker] = None
        self.specter2: Optional[SPECTER2Retriever] = None

        if HAS_NEURAL:
            try:
                self.st_model = SentenceTransformerRetriever(cache_dir=self.cache_dir)
                self.cross_encoder = CrossEncoderReranker()
            except Exception as e:
                logger.warning(
                    "Failed to initialise SentenceTransformerRetriever/"
                    "CrossEncoderReranker (Strategies 10 & 11 will be unavailable): %s",
                    e,
                )
                self.st_model = None
                self.cross_encoder = None

        if HAS_SPECTER2:
            try:
                self.specter2 = SPECTER2Retriever(cache_dir=self.cache_dir)
            except Exception as e:
                logger.warning(
                    "Failed to initialise SPECTER2Retriever "
                    "(Strategy 13 will be unavailable): %s",
                    e,
                )
                self.specter2 = None

        self.qdrant_retriever: Optional[QdrantRetriever] = None
        if HAS_QDRANT:
            try:
                client = getattr(chunk_store, "client", None)
                self.qdrant_retriever = QdrantRetriever(cache_dir=self.cache_dir, client=client)
            except Exception as e:
                logger.warning(
                    "Failed to initialise QdrantRetriever " "(Strategy 15 will be unavailable): %s",
                    e,
                )
                self.qdrant_retriever = None

        self._indexed = False

    def index(self, include_neural: bool = True, include_ppmi: bool = True) -> None:
        """Indexes all underlying retriever models on the loaded corpus."""
        self.bm25.index(self.corpus_texts)
        self.tfidf.index(self.corpus_texts)
        self.graph_retriever.index(
            self.corpus_texts,
            chunk_store=self.chunk_store,
            corpus_dir=self.corpus_dir,
        )

        if include_ppmi:
            self.ppmi.index(self.corpus_texts, corpus_dir=self.corpus_dir)

        if include_neural and self.st_model is not None:
            try:
                self.st_model.index(self.corpus_texts, corpus_dir=self.corpus_dir)
            except Exception as e:
                logger.warning(
                    "SentenceTransformerRetriever.index() failed (Strategy 11 disabled): %s",
                    e,
                )
                self.st_model = None

        if include_neural and self.qdrant_retriever is not None:
            try:
                st_embeddings = getattr(self.st_model, "corpus_embeddings", None)
                self.qdrant_retriever.index(
                    self.corpus_texts,
                    corpus_dir=self.corpus_dir,
                    embeddings=st_embeddings,
                )
            except Exception as e:
                logger.warning(
                    "QdrantRetriever.index() failed (Strategy 15 disabled): %s",
                    e,
                )
                self.qdrant_retriever = None

        if include_neural and self.cross_encoder is not None:
            try:
                self.cross_encoder.load_model()
            except Exception as e:
                logger.warning(
                    "CrossEncoderReranker.load_model() failed (Strategy 10 disabled): %s",
                    e,
                )
                self.cross_encoder = None

        if include_neural and self.specter2 is not None:
            try:
                self.specter2.index(self.corpus_texts, corpus_dir=self.corpus_dir)
            except Exception as e:
                logger.warning(
                    "SPECTER2Retriever.index() failed (Strategy 13 disabled): %s",
                    e,
                )
                self.specter2 = None

        self._indexed = True

    def get_strategy_rankings_with_scores(
        self, query: str
    ) -> Dict[str, Tuple[List[int], Dict[int, float]]]:
        """
        Execute query retrieval and return ranked_chunk_indices with scores.

        Provides genuine strategy scoring across all 14 retrieval and fusion
        strategies.
        """
        if not self._indexed:
            self.index()

        # 1. Base Sparse & Dense Scores
        b_scores = self.bm25.score(query)
        d_scores = self.tfidf.score(query)

        b_rank = np.argsort(b_scores)[::-1].tolist()
        d_rank = np.argsort(d_scores)[::-1].tolist()

        b_scores_dict: Dict[int, float] = {i: float(b_scores[i]) for i in range(len(b_scores))}
        d_scores_dict: Dict[int, float] = {i: float(d_scores[i]) for i in range(len(d_scores))}

        # 2. RRF wide candidate list
        rrf_wide, rrf_scores = reciprocal_rank_fusion(b_rank, d_rank, k=DEFAULT_RRF_K)

        # 3. Deduplicated candidates
        rrf_dedup_candidates = deduplicate_results(
            rrf_wide,
            self.corpus_texts,
            threshold=DEFAULT_DEDUP_THRESHOLD,
            max_results=DEFAULT_DEDUP_MAX_RESULTS,
        )

        # 4. MMR candidates
        q_vec = self.tfidf.get_query_vector(query)
        mmr_pool = deduplicate_results(
            rrf_wide,
            self.corpus_texts,
            threshold=DEFAULT_DEDUP_THRESHOLD,
            max_results=20,
        )
        mmr_results = maximal_marginal_relevance(
            q_vec,
            mmr_pool,
            self.tfidf.corpus_vectors,
            rrf_scores,
            lambda_param=DEFAULT_MMR_LAMBDA,
            top_k=DEFAULT_MMR_TOP_K,
        )

        # 5. PPMI fusion
        ppmi_fused, ppmi_fused_scores = self.ppmi.retrieve_ppmi_bm25_rrf_with_scores(
            query, k=DEFAULT_RRF_K
        )

        # Linear hybrid scores
        lin3_scores = compute_linear_scores(b_scores, d_scores, alpha=0.3)
        lin3_rank = np.argsort(lin3_scores)[::-1].tolist()
        lin3_dict: Dict[int, float] = {i: float(lin3_scores[i]) for i in range(len(lin3_scores))}

        lin5_scores = compute_linear_scores(b_scores, d_scores, alpha=0.5)
        lin5_rank = np.argsort(lin5_scores)[::-1].tolist()
        lin5_dict: Dict[int, float] = {i: float(lin5_scores[i]) for i in range(len(lin5_scores))}

        lin7_scores = compute_linear_scores(b_scores, d_scores, alpha=0.7)
        lin7_rank = np.argsort(lin7_scores)[::-1].tolist()
        lin7_dict: Dict[int, float] = {i: float(lin7_scores[i]) for i in range(len(lin7_scores))}

        rankings: Dict[str, Tuple[List[int], Dict[int, float]]] = {
            "1. Pure BM25 (Sparse)": (b_rank, b_scores_dict),
            "2. Pure TF-IDF (Sparse Vector Space)": (d_rank, d_scores_dict),
            "3. Linear Hybrid (a=0.3)": (lin3_rank, lin3_dict),
            "4. Linear Hybrid (a=0.5)": (lin5_rank, lin5_dict),
            "5. Linear Hybrid (a=0.7)": (lin7_rank, lin7_dict),
            "6. RRF (k=60)": (rrf_wide, rrf_scores),
            "7. RRF + Deduplication": (rrf_dedup_candidates, rrf_scores),
            "8. RRF + Dedup + MMR": (mmr_results, rrf_scores),
            "9. PPMI Semantic + BM25 RRF": (ppmi_fused, ppmi_fused_scores),
        }

        # 6. Neural strategies
        # TODO (P5): Experiment with an alternative cascade ordering:
        #   RRF fuse -> Jaccard dedup -> cross-encoder rerank on deduped top-20 -> MMR.
        # Standard two-stage retrieval deduplicates before reranking so near-identical
        # sliding-window chunks don't compete for cross-encoder attention. This may
        # improve Strategy 10 (currently 0.483 MRR vs BM25 0.573).
        if self.cross_encoder is not None and self.cross_encoder.model is not None:
            try:
                (
                    ce_reranked,
                    ce_remainder,
                    ce_scores_dict,
                ) = self.cross_encoder.rerank_pool_with_scores(
                    query,
                    rrf_wide,
                    self.corpus_texts,
                    pool_size=DEFAULT_CROSS_ENCODER_POOL_SIZE,
                )
                ce_deduped = deduplicate_results(
                    ce_reranked,
                    self.corpus_texts,
                    threshold=DEFAULT_DEDUP_THRESHOLD,
                    max_results=DEFAULT_DEDUP_MAX_RESULTS,
                )
                rankings["10. Cross-Encoder Re-rank"] = (
                    ce_deduped + ce_remainder,
                    ce_scores_dict,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranking failed (Strategy 10 skipped): %s", e)

        if self.st_model is not None and self.st_model.corpus_embeddings is not None:
            try:
                st_scores = self.st_model.score(query)
                st_rank = np.argsort(st_scores)[::-1].tolist()
                st_dict: Dict[int, float] = {i: float(st_scores[i]) for i in range(len(st_scores))}
                rankings["11. Sentence-Transformer (MiniLM)"] = (
                    st_rank,
                    st_dict,
                )
            except Exception as e:
                logger.warning("SentenceTransformer scoring failed (Strategy 11 skipped): %s", e)

        ad_scores = compute_adaptive_scores(query, b_scores, d_scores)
        ad_rank = np.argsort(ad_scores)[::-1].tolist()
        ad_dict: Dict[int, float] = {i: float(ad_scores[i]) for i in range(len(ad_scores))}
        rankings["12. Adaptive Hybrid"] = (ad_rank, ad_dict)

        if self.specter2 is not None and self.specter2.corpus_embeddings is not None:
            try:
                sp2_scores = self.specter2.score(query)
                sp2_rank = np.argsort(sp2_scores)[::-1].tolist()
                sp2_dict: Dict[int, float] = {
                    i: float(sp2_scores[i]) for i in range(len(sp2_scores))
                }
                rankings["13. SPECTER2 (Scientific Bi-Encoder)"] = (
                    sp2_rank,
                    sp2_dict,
                )
            except Exception as e:
                logger.warning("SPECTER2 scoring failed (Strategy 13 skipped): %s", e)

        # 7. Strategy 14: RRF + Graph + Dedup + MMR
        g_scores = self.graph_retriever.score(query)
        g_positive_indices = [idx for idx in np.argsort(g_scores)[::-1] if g_scores[idx] >= 0.2][:5]
        if g_positive_indices and np.max(g_scores) > 0:
            rrf_graph_wide, rrf_graph_scores = reciprocal_rank_fusion(
                b_rank,
                d_rank,
                k=DEFAULT_RRF_K,
                additional_rankings=[g_positive_indices],
                weights=[1.0, 1.0, 0.35],
            )
        else:
            rrf_graph_wide, rrf_graph_scores = rrf_wide, rrf_scores

        mmr_graph_pool = deduplicate_results(
            rrf_graph_wide,
            self.corpus_texts,
            threshold=DEFAULT_DEDUP_THRESHOLD,
            max_results=20,
        )
        rrf_graph_mmr = maximal_marginal_relevance(
            q_vec,
            mmr_graph_pool,
            self.tfidf.corpus_vectors,
            rrf_graph_scores,
            lambda_param=DEFAULT_MMR_LAMBDA,
            top_k=DEFAULT_MMR_TOP_K,
        )
        rankings["14. RRF + Graph + Dedup + MMR"] = (
            rrf_graph_mmr,
            rrf_graph_scores,
        )

        # Ablation cells (P0 - Factorial isolation of Graph, Dedup, MMR components)
        # Ablation 1: Graph only (no BM25/TF-IDF fusion, no post-processing)
        g_rank = [idx for idx in np.argsort(g_scores)[::-1] if g_scores[idx] >= 0.2][:50]
        g_dict: Dict[int, float] = {i: float(g_scores[i]) for i in range(len(g_scores))}
        rankings["Ablation: Graph only"] = (g_rank, g_dict)

        # Ablation 2: RRF + Graph (no dedup/MMR)
        if g_positive_indices and np.max(g_scores) > 0:
            rrf_graph_ablation, rrf_graph_ablation_scores = reciprocal_rank_fusion(
                b_rank,
                d_rank,
                k=DEFAULT_RRF_K,
                additional_rankings=[g_positive_indices],
                weights=[1.0, 1.0, 0.35],
            )
        else:
            rrf_graph_ablation, rrf_graph_ablation_scores = (
                rrf_wide,
                rrf_scores,
            )
        rankings["Ablation: RRF + Graph (no dedup/MMR)"] = (
            rrf_graph_ablation,
            rrf_graph_ablation_scores,
        )

        # Ablation 3: RRF + Graph + Dedup (no MMR)
        rrf_graph_dedup_ablation = deduplicate_results(
            rrf_graph_ablation,
            self.corpus_texts,
            threshold=DEFAULT_DEDUP_THRESHOLD,
            max_results=DEFAULT_DEDUP_MAX_RESULTS,
        )
        rankings["Ablation: RRF + Graph + Dedup (no MMR)"] = (
            rrf_graph_dedup_ablation,
            rrf_graph_ablation_scores,
        )

        # 8. Strategy 15: Qdrant Vector (ANN)
        if self.qdrant_retriever is not None:
            try:
                qd_scores = self.qdrant_retriever.score(query)
                qd_rank = np.argsort(qd_scores)[::-1].tolist()
                qd_dict: Dict[int, float] = {i: float(qd_scores[i]) for i in range(len(qd_scores))}
                rankings["15. Qdrant Vector (ANN)"] = (qd_rank, qd_dict)
            except Exception as e:
                logger.warning("Qdrant scoring failed (Strategy 15 skipped): %s", e)

        def _strategy_sort_key(name: str) -> int:
            try:
                return int(name.split(".")[0])
            except (ValueError, IndexError):
                return 999

        return dict(sorted(rankings.items(), key=lambda item: _strategy_sort_key(item[0])))

    def get_strategy_rankings(self, query: str) -> Dict[str, List[int]]:
        """
        Executes query retrieval and returns full ranked chunk indices for all active strategies.
        This provides single dispatch path parity for evaluation benchmarking.
        """
        rankings_with_scores = self.get_strategy_rankings_with_scores(query)
        return {name: ranked_indices for name, (ranked_indices, _) in rankings_with_scores.items()}

    def search(
        self, query: str, strategy: str = "rrf_dedup_mmr", top_k: int = 5
    ) -> List[SearchResult]:
        """
        Executes search using a specific strategy.

        Args:
            query: Search query string.
            strategy: Strategy key or full name.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects with genuine strategy scores.
        """
        canonical_strategy = STRATEGY_ALIASES.get(strategy.lower(), strategy)
        all_rankings_with_scores = self.get_strategy_rankings_with_scores(query)

        if canonical_strategy not in all_rankings_with_scores:
            matched = [k for k in all_rankings_with_scores if strategy.lower() in k.lower()]
            if matched:
                canonical_strategy = matched[0]
            else:
                available = list(all_rankings_with_scores.keys())
                raise KeyError(
                    f"Requested retrieval strategy '{strategy}' "
                    f"(canonical: '{canonical_strategy}') is not available or "
                    f"failed to index. Available strategies: {available}"
                )

        ranked_indices, score_dict = all_rankings_with_scores[canonical_strategy]
        ranked_indices = ranked_indices[:top_k]

        results: List[SearchResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = self.chunk_store.get_chunk(idx)
            if chunk is not None:
                score = float(score_dict.get(idx, 0.0))
                results.append(
                    SearchResult(
                        chunk_id=idx,
                        chunk=chunk,
                        score=score,
                        rank=rank,
                        strategy=canonical_strategy,
                    )
                )

        return results
