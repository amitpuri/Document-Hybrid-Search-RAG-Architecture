"""
Offline Grounded Synthesis Generator (zero LLM calls).

GroundedSynthesisGenerator is a template-based extractive summarizer
that makes NO LLM API calls and requires NO API keys. It is the offline
default used automatically when no provider API key is configured. For
real LLM generation (Anthropic, OpenAI, Gemini), use the adapters in
src/generation/providers/ via src/generation/factory.py.
"""

from typing import List

from src.common.types import DocumentChunk, GenerationResult
from src.generation.base import BaseGenerator
from src.generation.confidence import compute_citation_confidences


class GroundedSynthesisGenerator(BaseGenerator):
    """
    Offline template-based answer synthesizer.

    Makes ZERO LLM API calls. Extracts and formats relevant passages
    from retrieved chunks with explicit section and page citations.
    Serves as the out-of-the-box offline default when no provider
    API key is configured.

    For real LLM generation, use the cloud provider adapters:
        - anthropic_generator.py (Anthropic / AWS Bedrock)
        - openai_generator.py   (OpenAI / Azure OpenAI)
        - gemini_generator.py   (Gemini / GCP Vertex AI)
    Auto-detected and instantiated by factory.py based on
    which API key environment variables are set.
    """

    def generate(
        self,
        query: str,
        formatted_context: str,
        retrieved_chunks: List[DocumentChunk],
        strategy_used: str = "unknown",
    ) -> GenerationResult:
        if not retrieved_chunks:
            return GenerationResult(
                query=query,
                answer=("No relevant context passages were retrieved " "to answer the question."),
                citations=[],
                strategy_used=strategy_used,
                citation_confidence=[],
            )

        top_chunk = retrieved_chunks[0]
        # Synthesize key excerpt
        excerpt = top_chunk.text.strip().replace("\n", " ")
        if len(excerpt) > 350:
            excerpt = excerpt[:347] + "..."

        citations_text = []
        for i, chunk in enumerate(retrieved_chunks[:3], start=1):
            citations_text.append(
                f"- [Source {i}] **{chunk.doc_name}** "
                f"(Page {chunk.page_num}, § {chunk.section})"
            )

        # Synthesize knowledge graph insights if present in context
        kg_section = ""
        if "[Knowledge Graph Relationships]" in formatted_context:
            lines = formatted_context.split("\n")
            kg_trips = [
                line.strip()
                for line in lines
                if line.strip().startswith("- (")
                or (line.strip().startswith("- ") and "-->" in line)
            ]
            if kg_trips:
                kg_section = "\n\n**Knowledge Graph Connections**:\n" + "\n".join(kg_trips[:4])

        answer = (
            f"Based on retrieved documentation using "
            f"strategy '{strategy_used}':\n\n"
            f'**Key Finding**: "{excerpt}"'
            f"{kg_section}\n\n"
            f"**Citations & Provenance**:\n" + "\n".join(citations_text)
        )

        # Calculate per-citation confidence (C3 - retrieval-native signal)
        # Since mock generator doesn't have access to actual fusion scores,
        # we simulate them based on rank position (higher rank = higher score)
        fusion_scores = {}
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            # Simulate fusion score: 1.0 for rank 1, decreasing by 0.1 per rank
            simulated_score = max(1.0 - (rank - 1) * 0.1, 0.0)
            fusion_scores[chunk.chunk_id] = simulated_score

        confidences = compute_citation_confidences(
            query=query,
            retrieved_chunks=retrieved_chunks,
            fusion_scores=fusion_scores,
            top_k=len(retrieved_chunks),
        )

        return GenerationResult(
            query=query,
            answer=answer,
            citations=retrieved_chunks,
            strategy_used=strategy_used,
            citation_confidence=[c.value for c in confidences],
        )
