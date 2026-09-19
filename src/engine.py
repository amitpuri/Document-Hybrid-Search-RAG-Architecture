"""
Unified Hybrid Search & Generation Engine.

Binds Ingestion, Retrieval, and Generation pipelines into a
high-level Python API.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from src.common.types import GenerationResult, SearchResult
from src.config import CACHE_DIR, CORPUS_DIR, DEFAULT_STORAGE_BACKEND
from src.evaluation.harness import EvaluationHarness
from src.generation.pipeline import GenerationPipeline
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage import BaseChunkStore
from src.retrieval.pipeline import RetrievalPipeline


class HybridSearchEngine:
    """
    Enterprise-grade Hybrid Search & RAG Engine uniting:
    1. IngestionPipeline (PDF extraction, structured chunking, caching)
    2. RetrievalPipeline (14 hybrid sparse/dense/neural/fusion strategies)
    3. GenerationPipeline (Grounded context assembly and answer generation)
    """

    def __init__(
        self,
        chunk_store: BaseChunkStore,
        corpus_dir: Optional[Union[str, Path]] = CORPUS_DIR,
        cache_dir: Optional[Union[str, Path]] = CACHE_DIR,
        llm: Optional[str] = None,
        generator_config: Optional[dict] = None,
    ):
        self.chunk_store = chunk_store
        self.corpus_dir = corpus_dir
        self.cache_dir = cache_dir

        self.retrieval = RetrievalPipeline(
            chunk_store=chunk_store, corpus_dir=corpus_dir, cache_dir=cache_dir
        )

        # Build generation pipeline with optional provider configuration
        if generator_config:
            self.generation = GenerationPipeline(**generator_config)
        else:
            # Legacy behavior for backward compatibility
            self.generation = GenerationPipeline()

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Union[str, Path] = CORPUS_DIR,
        cache_dir: Union[str, Path] = CACHE_DIR,
        force_rebuild: bool = False,
        llm: Optional[str] = None,
        storage_backend: str = DEFAULT_STORAGE_BACKEND,
        generator_config: Optional[dict] = None,
    ) -> "HybridSearchEngine":
        """Factory method that runs ingestion and constructs the engine."""
        ingestion = IngestionPipeline(
            corpus_dir=corpus_dir,
            cache_dir=cache_dir,
            storage_backend=storage_backend,
        )
        chunk_store, _, _ = ingestion.run(force_rebuild=force_rebuild)
        engine = cls(
            chunk_store=chunk_store,
            corpus_dir=corpus_dir,
            cache_dir=cache_dir,
            llm=llm,
            generator_config=generator_config,
        )
        engine.retrieval.index()
        return engine

    def search(
        self, query: str, strategy: str = "rrf_dedup_mmr", top_k: int = 5
    ) -> List[SearchResult]:
        """Search the corpus using any of the 14 retrieval strategies."""
        return self.retrieval.search(query=query, strategy=strategy, top_k=top_k)

    def generate_answer(
        self, query: str, strategy: str = "rrf_graph_dedup_mmr", top_k: int = 3
    ) -> GenerationResult:
        """Retrieve context passages and generate grounded response."""
        results = self.search(query=query, strategy=strategy, top_k=top_k)

        # Collect query-relevant knowledge graph triplets
        graph_triplets = []
        if (
            hasattr(self.retrieval, "graph_retriever")
            and self.retrieval.graph_retriever is not None
        ):
            q_ents = self.retrieval.graph_retriever.identify_query_entities(query)
            for ent in q_ents[:3]:
                neighbors = self.retrieval.graph_retriever.graph_store.get_neighbors(ent, hops=1)
                for src, tgt, data in neighbors[:3]:
                    rel_type = data.get("relation_type", "RELATED_TO")
                    graph_triplets.append((src, rel_type, tgt))

        return self.generation.generate_from_results(
            query=query,
            results=results,
            strategy_used=strategy,
            graph_triplets=graph_triplets if graph_triplets else None,
        )

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Runs the 14-query x 14-strategy evaluation benchmark."""
        harness = EvaluationHarness(retrieval_pipeline=self.retrieval)
        return harness.run()
