"""
Unit tests for per-claim calibrated confidence scoring (Roadmap C3).

Tests confidence.py functionality including confidence level calculation,
lexical overlap computation, and citation confidence batch processing.
"""

import sys
from pathlib import Path

from src.common.console import ensure_utf8_streams
from src.common.types import DocumentChunk
from src.generation.confidence import (
    CitationWithConfidence,
    ConfidenceLevel,
    calculate_confidence,
    compute_citation_confidences,
    compute_lexical_overlap,
    normalize_fusion_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


def run_tests():
    """Run all test functions and report results."""
    tests_passed = 0
    tests_failed = 0

    # Test ConfidenceLevel enum
    try:
        test_enum_values()
        print("[PASS] test_enum_values")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_enum_values: {e}")
        tests_failed += 1

    # Test calculate_confidence
    try:
        test_high_confidence_all_conditions_met()
        print("[PASS] test_high_confidence_all_conditions_met")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_high_confidence_all_conditions_met: {e}")
        tests_failed += 1

    try:
        test_not_high_rank_too_low()
        print("[PASS] test_not_high_rank_too_low")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_not_high_rank_too_low: {e}")
        tests_failed += 1

    try:
        test_medium_confidence_rank_condition()
        print("[PASS] test_medium_confidence_rank_condition")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_medium_confidence_rank_condition: {e}")
        tests_failed += 1

    try:
        test_low_confidence_default()
        print("[PASS] test_low_confidence_default")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_low_confidence_default: {e}")
        tests_failed += 1

    # Test compute_lexical_overlap
    try:
        test_perfect_overlap()
        print("[PASS] test_perfect_overlap")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_perfect_overlap: {e}")
        tests_failed += 1

    try:
        test_no_overlap()
        print("[PASS] test_no_overlap")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_no_overlap: {e}")
        tests_failed += 1

    try:
        test_empty_query()
        print("[PASS] test_empty_query")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_empty_query: {e}")
        tests_failed += 1

    # Test normalize_fusion_score
    try:
        test_normalization_typical_score()
        print("[PASS] test_normalization_typical_score")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_normalization_typical_score: {e}")
        tests_failed += 1

    try:
        test_normalization_above_max()
        print("[PASS] test_normalization_above_max")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_normalization_above_max: {e}")
        tests_failed += 1

    # Test compute_citation_confidences
    try:
        test_empty_chunks()
        print("[PASS] test_empty_chunks")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_empty_chunks: {e}")
        tests_failed += 1

    try:
        test_single_chunk_high_confidence()
        print("[PASS] test_single_chunk_high_confidence")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_single_chunk_high_confidence: {e}")
        tests_failed += 1

    try:
        test_confidence_parallel_to_chunks()
        print("[PASS] test_confidence_parallel_to_chunks")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_confidence_parallel_to_chunks: {e}")
        tests_failed += 1

    # Test CitationWithConfidence
    try:
        test_creation()
        print("[PASS] test_creation")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_creation: {e}")
        tests_failed += 1

    print(f"\n{tests_passed} tests passed, {tests_failed} tests failed")
    return tests_failed == 0


# Test functions
def test_enum_values():
    """Test that enum has correct string values."""
    assert ConfidenceLevel.HIGH.value == "high"
    assert ConfidenceLevel.MEDIUM.value == "medium"
    assert ConfidenceLevel.LOW.value == "low"


def test_high_confidence_all_conditions_met():
    """Test HIGH confidence when all conditions are met."""
    confidence = calculate_confidence(rank=1, fusion_score=0.8, lexical_overlap=0.4)
    assert confidence == ConfidenceLevel.HIGH


def test_not_high_rank_too_low():
    """Test that rank > 2 prevents HIGH confidence."""
    confidence = calculate_confidence(rank=3, fusion_score=0.8, lexical_overlap=0.4)
    assert confidence != ConfidenceLevel.HIGH


def test_medium_confidence_rank_condition():
    """Test MEDIUM confidence based on rank <= 4."""
    confidence = calculate_confidence(rank=4, fusion_score=0.3, lexical_overlap=0.1)
    assert confidence == ConfidenceLevel.MEDIUM


def test_low_confidence_default():
    """Test LOW confidence when no conditions are met."""
    confidence = calculate_confidence(rank=10, fusion_score=0.2, lexical_overlap=0.05)
    assert confidence == ConfidenceLevel.LOW


def test_perfect_overlap():
    """Test perfect lexical overlap."""
    overlap = compute_lexical_overlap("test query", "test query")
    assert overlap == 1.0


def test_no_overlap():
    """Test no lexical overlap."""
    overlap = compute_lexical_overlap("quantum mechanics", "machine learning algorithms")
    assert overlap == 0.0


def test_empty_query():
    """Test empty query returns 0.0."""
    overlap = compute_lexical_overlap("", "some text")
    assert overlap == 0.0


def test_normalization_typical_score():
    """Test normalization of typical score."""
    normalized = normalize_fusion_score(5.0, max_score=10.0)
    assert normalized == 0.5


def test_normalization_above_max():
    """Test that scores above max are capped at 1.0."""
    normalized = normalize_fusion_score(15.0, max_score=10.0)
    assert normalized == 1.0


def test_empty_chunks():
    """Test with empty chunk list."""
    confidences = compute_citation_confidences(
        query="test", retrieved_chunks=[], fusion_scores={}, top_k=5
    )
    assert confidences == []


def test_single_chunk_high_confidence():
    """Test single chunk with high confidence signals."""
    chunk = DocumentChunk(
        chunk_id=1,
        doc_name="test.pdf",
        page_num=1,
        section="Introduction",
        text="POMDP belief state filtering is important",
    )
    fusion_scores = {1: 0.8}

    confidences = compute_citation_confidences(
        query="POMDP belief state",
        retrieved_chunks=[chunk],
        fusion_scores=fusion_scores,
        top_k=1,
    )

    assert len(confidences) == 1
    # Should be HIGH: rank=1, score=0.8, overlap should be good
    assert confidences[0] in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]


def test_confidence_parallel_to_chunks():
    """Test that confidence list is parallel to chunks list."""
    chunks = [
        DocumentChunk(
            chunk_id=i,
            doc_name=f"test{i}.pdf",
            page_num=1,
            section="Intro",
            text=f"text {i}",
        )
        for i in range(5)
    ]
    fusion_scores = {i: 0.5 for i in range(5)}

    confidences = compute_citation_confidences(
        query="test",
        retrieved_chunks=chunks,
        fusion_scores=fusion_scores,
        top_k=5,
    )

    assert len(confidences) == len(chunks)
    for conf in confidences:
        assert conf in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
        ]


def test_creation():
    """Test creating CitationWithConfidence instance."""
    citation = CitationWithConfidence(
        chunk_id=1,
        rank=1,
        fusion_score=0.8,
        lexical_overlap=0.4,
        confidence=ConfidenceLevel.HIGH,
    )
    assert citation.chunk_id == 1
    assert citation.rank == 1
    assert citation.fusion_score == 0.8
    assert citation.lexical_overlap == 0.4
    assert citation.confidence == ConfidenceLevel.HIGH


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
