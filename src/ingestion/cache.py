"""
Parameter- and State-Sensitive Disk Cache for Ingested Corpora and Embeddings.
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np
from src.common.types import DocumentChunk


def compute_cache_key(
    corpus_dir: str | Path,
    max_words: int = 120,
    overlap_sentences: int = 1,
    extra_tag: str = ""
) -> str:
    """
    Computes a cryptographic SHA-256 fingerprint over:
    1. Corpus directory absolute path
    2. Chunking parameters and extra identifier tag
    3. File names, modification times, and sizes of all PDF files
    """
    hasher = hashlib.sha256()
    abs_corpus = str(os.path.abspath(corpus_dir))
    hasher.update(abs_corpus.encode("utf-8"))
    hasher.update(f"max_words:{max_words},overlap_sents:{overlap_sentences},extra:{extra_tag}".encode("utf-8"))

    if os.path.exists(corpus_dir):
        pdf_files = sorted([f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf")])
        for f in pdf_files:
            full_path = os.path.join(corpus_dir, f)
            try:
                stat = os.stat(full_path)
                hasher.update(f"{f}:{stat.st_mtime}:{stat.st_size}".encode("utf-8"))
            except OSError:
                pass

    return hasher.hexdigest()[:16]


def load_cached_chunks(cache_path: str | Path) -> Optional[Dict[str, Any]]:
    """Loads cached chunk dictionary from pickle file if available."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_cached_chunks(cache_path: str | Path, data: Dict[str, Any]) -> None:
    """Saves chunk dictionary to pickle file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cached_embeddings(cache_path: str | Path) -> Optional[np.ndarray]:
    """Loads cached numpy embedding matrix if available."""
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            return None
    return None


def save_cached_embeddings(cache_path: str | Path, embeddings: np.ndarray) -> None:
    """Saves numpy embedding matrix."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
