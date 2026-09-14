"""
Document Hybrid Search & Generation Library.

Pipelines:
1. Ingestion Pipeline: src.ingestion
2. Retrieval Pipeline: src.retrieval
3. Generation Pipeline: src.generation
4. Evaluation Suite: src.evaluation
5. Unified Engine: src.engine.HybridSearchEngine
"""

from src.engine import HybridSearchEngine
from src.common.types import DocumentChunk, SearchResult, GenerationResult, MetricScores
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.pipeline import GenerationPipeline
from src.evaluation.harness import EvaluationHarness

__all__ = [
    "HybridSearchEngine",
    "DocumentChunk",
    "SearchResult",
    "GenerationResult",
    "MetricScores",
    "IngestionPipeline",
    "RetrievalPipeline",
    "GenerationPipeline",
    "EvaluationHarness",
]
