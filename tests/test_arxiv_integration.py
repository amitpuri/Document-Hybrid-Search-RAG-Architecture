"""
Integration tests for arXiv ingestion pipeline (Phase 4).

Tests arxiv_fetcher.py functionality without requiring actual arXiv API calls
by using mock data or minimal real calls.
"""

import sys
from pathlib import Path

from src.common.console import ensure_utf8_streams
from src.ingestion.arxiv_fetcher import (
    ArxivFetchError,
    ArxivPaper,
    ArxivRateLimiter,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure UTF-8 output on Windows
ensure_utf8_streams()


def run_tests():
    """Run all test functions and report results."""
    tests_passed = 0
    tests_failed = 0

    # Test ArxivRateLimiter
    try:
        test_initialization()
        print("[PASS] test_initialization")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_initialization: {e}")
        tests_failed += 1

    try:
        test_wait_on_first_call()
        print("[PASS] test_wait_on_first_call")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_wait_on_first_call: {e}")
        tests_failed += 1

    try:
        test_record_success_resets_failures()
        print("[PASS] test_record_success_resets_failures")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_record_success_resets_failures: {e}")
        tests_failed += 1

    try:
        test_record_failure_increments_counter()
        print("[PASS] test_record_failure_increments_counter")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_record_failure_increments_counter: {e}")
        tests_failed += 1

    try:
        test_max_retries_exceeded()
        print("[PASS] test_max_retries_exceeded")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_max_retries_exceeded: {e}")
        tests_failed += 1

    try:
        test_exponential_backoff()
        print("[PASS] test_exponential_backoff")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_exponential_backoff: {e}")
        tests_failed += 1

    try:
        test_backoff_capped_at_max()
        print("[PASS] test_backoff_capped_at_max")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_backoff_capped_at_max: {e}")
        tests_failed += 1

    try:
        test_reset()
        print("[PASS] test_reset")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_reset: {e}")
        tests_failed += 1

    # Test ArxivPaper
    try:
        test_to_dict()
        print("[PASS] test_to_dict")
        tests_passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_to_dict: {e}")
        tests_failed += 1

    print(f"\n{tests_passed} tests passed, {tests_failed} tests failed")
    return tests_failed == 0


# Test functions
def test_initialization():
    """Test rate limiter initializes with correct defaults."""
    limiter = ArxivRateLimiter(min_delay=3.0, max_retries=5)
    assert limiter.min_delay == 3.0
    assert limiter.max_retries == 5
    assert limiter.consecutive_failures == 0
    assert limiter.last_request_time == 0.0


def test_wait_on_first_call():
    """Test that wait() sleeps on first call after initialization."""
    limiter = ArxivRateLimiter(min_delay=1.0)
    import time

    start = time.time()
    limiter.wait()
    elapsed = time.time() - start
    # First call should not sleep (no previous request)
    assert elapsed < 0.1


def test_record_success_resets_failures():
    """Test that recording success resets failure counter."""
    limiter = ArxivRateLimiter()
    limiter.consecutive_failures = 3
    limiter.record_success()
    assert limiter.consecutive_failures == 0


def test_record_failure_increments_counter():
    """Test that recording failure increments counter."""
    limiter = ArxivRateLimiter()
    limiter.record_failure()
    assert limiter.consecutive_failures == 1
    limiter.record_failure()
    assert limiter.consecutive_failures == 2


def test_max_retries_exceeded():
    """Test that exceeding max retries raises error."""
    limiter = ArxivRateLimiter(max_retries=2)
    limiter.consecutive_failures = 2
    try:
        limiter.record_failure()
        assert False, "Should have raised ArxivFetchError"
    except ArxivFetchError:
        pass  # Expected


def test_exponential_backoff():
    """Test that backoff increases exponentially."""
    limiter = ArxivRateLimiter(base_backoff=2.0, max_backoff=16.0)
    backoff1 = limiter.record_failure()
    backoff2 = limiter.record_failure()
    backoff3 = limiter.record_failure()
    assert backoff1 == 2.0
    assert backoff2 == 4.0
    assert backoff3 == 8.0


def test_backoff_capped_at_max():
    """Test that backoff is capped at max_backoff."""
    limiter = ArxivRateLimiter(base_backoff=2.0, max_backoff=8.0, max_retries=10)
    limiter.consecutive_failures = 5
    backoff = limiter.record_failure()
    assert backoff == 8.0


def test_reset():
    """Test that reset clears all state."""
    limiter = ArxivRateLimiter()
    limiter.consecutive_failures = 3
    limiter.last_request_time = 123.456
    limiter.reset()
    assert limiter.consecutive_failures == 0
    assert limiter.last_request_time == 0.0


def test_to_dict():
    """Test conversion to dictionary."""
    paper = ArxivPaper(
        arxiv_id="2301.07041",
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="Test abstract",
        published="2023-01-15",
        categories=["cs.AI", "cs.LG"],
        pdf_url="https://arxiv.org/pdf/2301.07041.pdf",
        primary_category="cs.AI",
    )
    result = paper.to_dict()
    assert result["arxiv_id"] == "2301.07041"
    assert result["title"] == "Test Paper"
    assert len(result["authors"]) == 2
    assert result["primary_category"] == "cs.AI"


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
