"""
Extensible Generator Implementation for Offline Grounded Synthesis and LLM Adapters.
Generates structured answers with exact source citations, and provides hook points
for external LLM APIs (OpenAI, Gemini, Anthropic, Ollama).
"""

from typing import List
from src.generation.base import BaseGenerator
from src.common.types import DocumentChunk, GenerationResult


class GroundedSynthesisGenerator(BaseGenerator):
    """
    Offline grounded answer synthesizer that summarizes relevant retrieved chunks
    with explicit section and page citations.
    Serves as an out-of-the-box generator and base adapter for cloud/local LLMs.
    """

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown"
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer="No relevant context passages were retrieved to answer the question.",
                citations=[],
                strategy_used=strategy_used
            )

        top_chunk = retrieved_chunks[0]
        # Synthesize key excerpt
        excerpt = top_chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 350:
            excerpt = excerpt[:347] + "..."

        citations_text = []
        for i, chunk in enumerate(retrieved_chunks[:3], start=1):
            citations_text.append(
                f"- [Source {i}] **{chunk.doc_name}** (Page {chunk.page_num}, § {chunk.section})"
            )

        answer = (
            f"Based on retrieved documentation using strategy '{strategy_used}':\n\n"
            f"**Key Finding**: \"{excerpt}\"\n\n"
            f"**Citations & Provenance**:\n" + "\n".join(citations_text)
        )

        return GenerationResult(
            query=query,
            answer=answer,
            citations=retrieved_chunks,
            strategy_used=strategy_used
        )
