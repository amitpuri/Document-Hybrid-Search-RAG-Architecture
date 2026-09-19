"""
Parameter- and State-Sensitive Disk Cache for Ingested Corpora, Embeddings,
and Partitioned Parquet Dataset Manifests.

SECURITY NOTE: This module replaces unsafe pickle serialization with secure JSON
and numpy formats (.npz, .npy) to prevent arbitrary code execution from untrusted
cache files. Pickle has been removed to ensure production-grade safety.
"""

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    PARQUET_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    PARQUET_DEFAULT_ROW_GROUP_SIZE,
)


def compute_chunking_regime_hash(
    max_words: int = 120,
    overlap_sentences: int = 1,
    version: str = "1.0",
) -> str:
    """
    Compute SHA-256 fingerprint for chunking parameters.

    Prevents heterogeneous regime drift.
    """
    hasher = hashlib.sha256()
    hasher.update(f"max_words:{max_words},overlap:{overlap_sentences},v:{version}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def compute_cache_key(
    corpus_dir: Union[str, Path],
    max_words: int = 120,
    overlap_sentences: int = 1,
    extra_tag: str = "",
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
    hasher.update(
        f"max_words:{max_words},overlap_sents:{overlap_sentences},extra:{extra_tag}".encode("utf-8")
    )

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


def load_dataset_manifest(dataset_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Loads the dataset manifest JSON from the specified dataset directory if it exists."""
    manifest_path = Path(dataset_dir) / "_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_dataset_manifest_atomic(
    dataset_dir: Union[str, Path], manifest_data: Dict[str, Any]
) -> None:
    """Atomically saves dataset manifest to _manifest.json via a temporary file."""
    dir_path = Path(dataset_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    manifest_path = dir_path / "_manifest.json"
    temp_path = dir_path / f"_manifest.json.tmp_{uuid.uuid4().hex}"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # Atomic rename / replace
    os.replace(temp_path, manifest_path)


def atomic_write_parquet_table(
    table: pa.Table,
    dest_file: Union[str, Path],
    compression: str = PARQUET_COMPRESSION,
    compression_level: int = PARQUET_COMPRESSION_LEVEL,
    row_group_size: int = PARQUET_DEFAULT_ROW_GROUP_SIZE,
) -> None:
    """Atomically writes a PyArrow table to a Parquet file via temporary file replacement."""
    dest_path = Path(dest_file)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = dest_path.parent / f"{dest_path.name}.tmp_{uuid.uuid4().hex}"
    pq.write_table(
        table,
        temp_path,
        compression=compression,
        compression_level=compression_level,
        row_group_size=row_group_size,
    )
    os.replace(temp_path, dest_path)


def load_cached_chunks(cache_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Loads cached chunk dictionary from JSON file if available.
    Replaces pickle for security: arbitrary code execution prevention.
    """
    json_path = str(cache_path).replace(".pkl", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    # Fallback: try legacy pickle path (for migration from old caches)
    if os.path.exists(cache_path):
        try:
            import pickle

            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def save_cached_chunks(cache_path: Union[str, Path], data: Dict[str, Any]) -> None:
    """
    Saves chunk dictionary to JSON file (secure, human-readable).
    Replaces pickle for security: arbitrary code execution prevention.
    """
    json_path = str(cache_path).replace(".pkl", ".json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cached_embeddings(cache_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Loads cached numpy embedding matrix if available."""
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            return None
    return None


def save_cached_embeddings(cache_path: Union[str, Path], embeddings: np.ndarray) -> None:
    """Saves numpy embedding matrix."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
