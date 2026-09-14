"""
Context Builder and Citation Formatter for RAG Generation.
Formats retrieved document chunks into clean, structured context blocks with provenance.
"""

from typing import List, Sequence
from src.common.types import DocumentChunk, SearchResult


class ContextBuilder:
    """Builds formatted LLM context windows from retrieved chunks."""

    def __init__(self, max_context_chars: int = 8000):
        self.max_context_chars = max_context_chars

    def build_context(self, chunks_or_results: Sequence[DocumentChunk | SearchResult]) -> str:
        """
        Formats chunks into a numbered context block with document, page, and section provenance.
        """
        chunks: List[DocumentChunk] = []
        for item in chunks_or_results:
            if isinstance(item, SearchResult):
                chunks.append(item.chunk)
            elif isinstance(item, DocumentChunk):
                chunks.append(item)

        context_blocks = []
        current_len = 0

        for i, chunk in enumerate(chunks, start=1):
            header = f"[Source {i}: {chunk.doc_name} | Page {chunk.page_num} | § {chunk.section}]"
            block = f"{header}\n{chunk.text}\n"

            if current_len + len(block) > self.max_context_chars and context_blocks:
                break

            context_blocks.append(block)
            current_len += len(block)

        return "\n".join(context_blocks)
