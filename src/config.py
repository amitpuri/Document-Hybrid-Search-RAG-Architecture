"""
Configuration and Defaults for Document Hybrid Search System.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
CACHE_DIR = BASE_DIR / ".cache"

# Automatically load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# Chunking Parameters
DEFAULT_MAX_WORDS = 120
DEFAULT_OVERLAP_SENTENCES = 1
MIN_CHUNK_WORDS = 25

# Retrieval Defaults
DEFAULT_RRF_K = 60
DEFAULT_DEDUP_THRESHOLD = 0.65
DEFAULT_DEDUP_MAX_RESULTS = 10
DEFAULT_MMR_LAMBDA = 0.7
DEFAULT_MMR_TOP_K = 5
DEFAULT_CROSS_ENCODER_POOL_SIZE = 50

# Neural Models
DEFAULT_BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SPECTER2_BASE_MODEL = "allenai/specter2_base"
DEFAULT_SPECTER2_ADAPTER = "allenai/specter2_proximity"

# PPMI Parameters
DEFAULT_PPMI_WINDOW_SIZE = 5
DEFAULT_PPMI_VOCAB_SIZE = 1500
DEFAULT_PPMI_MAX_CONTEXT = 50
