"""
Context Builder and Citation Formatter for RAG Generation.
Formats retrieved document chunks into clean, structured context blocks with provenance.
"""

from typing import List, Sequence, Optional, Tuple
from src.common.types import DocumentChunk, SearchResult


class ContextBuilder:
    """Builds formatted LLM context windows from retrieved chunks and graph elements."""

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def build_context(
        self,
        chunks_or_results: Sequence[DocumentChunk | SearchResult],
        graph_triplets: Optional[Sequence[Tuple[str, str, str] | str]] = None,
    ) -> str:
        """
        Formats chunks and knowledge graph triplets into structured context blocks
        with document, page, section, and knowledge-graph provenance.
        """
        context_blocks = []
        current_len = 0

        # 1. Knowledge Graph Context Block (if provided)
        if graph_triplets:
            kg_lines = ["[Knowledge Graph Relationships]"]
            for trip in graph_triplets:
                if isinstance(trip, (tuple, list)) and len(trip) >= 3:
                    kg_lines.append(f"- ({trip[0]}) --[{trip[1]}]--> ({trip[2]})")
                else:
                    kg_lines.append(f"- {str(trip)}")
            kg_block = "\n".join(kg_lines) + "\n"
            context_blocks.append(kg_block)
            current_len += len(kg_block)

        # 2. Document Chunk Passages
        chunks: List[DocumentChunk] = []
        for item in chunks_or_results:
            if isinstance(item, SearchResult):
                chunks.append(item.chunk)
            elif isinstance(item, DocumentChunk):
                chunks.append(item)

        for i, chunk in enumerate(chunks, start=1):
            header = f"[Source {i}: {chunk.doc_name} | Page {chunk.page_num} | § {chunk.section}]"
            block = f"{header}\n{chunk.text}\n"

            if current_len + len(block) > self.max_context_chars and len(context_blocks) > (1 if graph_triplets else 0):
                break

            context_blocks.append(block)
            current_len += len(block)

        return "\n".join(context_blocks)

