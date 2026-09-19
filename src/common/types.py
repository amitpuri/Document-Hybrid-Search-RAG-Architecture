"""
Common Data Contracts and Typed Models across Ingestion,
Retrieval, and Generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    """Represents a single text chunk with provenance metadata."""

    chunk_id: int
    doc_name: str
    page_num: int
    section: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_formatted_text(self) -> str:
        """Return the formatted chunk representation for retrieval."""
        return f"[{self.doc_name} | Page {self.page_num} | " f"§ {self.section}] {self.text}"


@dataclass
class SearchResult:
    """Represents a single retrieved result."""

    chunk_id: int
    chunk: DocumentChunk
    score: float
    rank: int
    strategy: str


@dataclass
class MetricScores:
    """Evaluation metrics for a single query."""

    mrr: float
    recall_1: float
    recall_3: float
    recall_5: float
    ndcg_5: float
    entity_coverage: float = 0.0
    relation_coverage: float = 0.0


@dataclass
class GenerationResult:
    """Represents the response from a RAG generation pipeline."""

    query: str
    answer: str
    citations: List[DocumentChunk]
    strategy_used: str
    # Per-citation confidence levels (parallel to citations list)
    # Will contain "high", "medium", "low" strings
    citation_confidence: Optional[List[str]] = None

    def __post_init__(self):
        """Ensure citation_confidence is initialized if None."""
        if self.citation_confidence is None:
            self.citation_confidence = []
