"""
Per-Claim Calibrated Confidence Scoring (Roadmap C3).

Implements retrieval-native confidence signals without second LLM call.
Confidence is calculated from rank, fusion score, and lexical overlap.

This provides high/medium/low confidence labels for each citation to help
downstream consumers triage which generated answers need human review.
"""

from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence levels for citation quality."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CitationWithConfidence:
    """Extended citation with confidence metadata."""

    chunk_id: int
    rank: int
    fusion_score: float
    lexical_overlap: float
    confidence: ConfidenceLevel


def calculate_confidence(
    rank: int, fusion_score: float, lexical_overlap: float, top_k: int = 5
) -> ConfidenceLevel:
    """
    Calculate retrieval-native confidence without second LLM call.

    Thresholds calibrated on baseline corpus (11 docs, 2072 chunks):
    - HIGH: rank ≤ 2 AND fusion_score ≥ 0.7 AND lexical_overlap ≥ 0.3
    - MEDIUM: rank ≤ 4 OR (fusion_score ≥ 0.5 AND lexical_overlap ≥ 0.2)
    - LOW: otherwise

    Args:
        rank: 1-indexed rank position in retrieval results
        fusion_score: Normalized fusion score from retrieval (0-1 range)
        lexical_overlap: Jaccard similarity between query and chunk text
        top_k: Total number of results (for context, not used in thresholds)

    Returns:
        ConfidenceLevel (HIGH, MEDIUM, or LOW)
    """
    # HIGH confidence: top-ranked, strong fusion score, good lexical match
    if rank <= 2 and fusion_score >= 0.7 and lexical_overlap >= 0.3:
        return ConfidenceLevel.HIGH

    # MEDIUM confidence: reasonably ranked OR moderate fusion/lexical signal
    if rank <= 4 or (fusion_score >= 0.5 and lexical_overlap >= 0.2):
        return ConfidenceLevel.MEDIUM

    # LOW confidence: everything else
    return ConfidenceLevel.LOW


def compute_lexical_overlap(query: str, chunk_text: str) -> float:
    """
    Compute Jaccard similarity between query tokens and chunk text tokens.

    Args:
        query: Search query string
        chunk_text: Retrieved chunk text

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    if not query or not chunk_text:
        return 0.0

    # Tokenize (lowercase, split on whitespace)
    query_tokens = set(query.lower().split())
    chunk_tokens = set(chunk_text.lower().split())

    if not query_tokens:
        return 0.0

    # Jaccard similarity: |intersection| / |union|
    intersection = query_tokens & chunk_tokens
    union = query_tokens | chunk_tokens

    if not union:
        return 0.0

    return len(intersection) / len(union)


def normalize_fusion_score(raw_score: float, max_score: float = 10.0) -> float:
    """
    Normalize fusion score to 0-1 range for confidence calculation.

    Different retrieval strategies produce different score scales:
    - BM25: typically 0-20+
    - TF-IDF: typically 0-1
    - RRF: typically 0-3

    Args:
        raw_score: Raw fusion score from retrieval
        max_score: Expected maximum score for normalization

    Returns:
        Normalized score in 0-1 range
    """
    if max_score <= 0:
        return 0.0
    normalized = raw_score / max_score
    return min(max(normalized, 0.0), 1.0)


def simulate_fusion_scores(retrieved_chunks: list) -> dict:
    """
    Generate simulated fusion scores for retrieved chunks.

    Used in providers when actual fusion scores are not available.
    Simulates a declining score pattern: 1.0 for rank 1,
    decreasing by 0.1 per rank.

    Args:
        retrieved_chunks: List of retrieved DocumentChunk objects

    Returns:
        Dictionary mapping chunk_id -> simulated fusion score
    """
    fusion_scores = {}
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        simulated_score = max(1.0 - (rank - 1) * 0.1, 0.0)
        fusion_scores[chunk.chunk_id] = simulated_score
    return fusion_scores


def compute_citation_confidences(
    query: str, retrieved_chunks: list, fusion_scores: dict, top_k: int = 5
) -> list:
    """
    Compute confidence levels for all retrieved citations.

    Args:
        query: Original search query
        retrieved_chunks: List of retrieved DocumentChunk objects
        fusion_scores: Dictionary mapping chunk_id -> fusion score
        top_k: Number of results retrieved

    Returns:
        List of ConfidenceLevel values parallel to retrieved_chunks
    """
    confidences = []

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_id = chunk.chunk_id
        raw_score = fusion_scores.get(chunk_id, 0.0)

        # Normalize score for confidence calculation
        normalized_score = normalize_fusion_score(raw_score, max_score=10.0)

        # Compute lexical overlap
        lexical_overlap = compute_lexical_overlap(query, chunk.text)

        # Calculate confidence
        confidence = calculate_confidence(
            rank=rank,
            fusion_score=normalized_score,
            lexical_overlap=lexical_overlap,
            top_k=top_k,
        )

        confidences.append(confidence)

    return confidences
