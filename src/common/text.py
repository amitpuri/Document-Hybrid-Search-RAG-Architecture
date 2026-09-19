"""
Text Processing and Tokenization Utilities.
"""

from typing import List, Set

STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "of",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "which",
    "can",
    "will",
    "would",
    "should",
    "could",
    "all",
    "any",
    "both",
    "each",
    "more",
}


def clean_text(text: str) -> str:
    """Replace non-alphanumeric characters with spaces and strip whitespace."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return " ".join(cleaned.split())


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Lowercases and cleans text into word tokens."""
    tokens = clean_text(text).split()
    if remove_stopwords:
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Computes Jaccard set similarity between two token sequences."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0
