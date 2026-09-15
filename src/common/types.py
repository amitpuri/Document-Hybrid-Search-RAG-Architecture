"""
Common Data Contracts and Typed Models across Ingestion, Retrieval, and Generation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


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
        """Returns the formatted chunk representation used in retrieval and citations."""
        return f"[{self.doc_name} | Page {self.page_num} | § {self.section}] {self.text}"


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
