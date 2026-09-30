"""
Configuration and Defaults for Document Hybrid Search System.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE_DIR / "corpus"
CACHE_DIR = BASE_DIR / ".cache"
PARQUET_DATASET_DIR = CACHE_DIR / "chunks_dataset"

# Storage Backend Configuration
DEFAULT_STORAGE_BACKEND = "parquet"  # "parquet", "memory", or "qdrant"
PARQUET_COMPRESSION = "zstd"  # "zstd", "snappy", or "uncompressed"
PARQUET_COMPRESSION_LEVEL = 3  # Balanced compression ratio vs write throughput
PARQUET_DEFAULT_ROW_GROUP_SIZE = 64000  # Scalable row grouping (avoids tiny 1K blocks)
PARQUET_IN_MEMORY_THRESHOLD = (
    50000  # Automatic switch to mmap scanner when chunk count exceeds this
)

# Qdrant Vector Store Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "document_chunks")
QDRANT_VECTOR_SIZE = 384  # Default dimension for all-MiniLM-L6-v2
QDRANT_TIMEOUT = float(os.getenv("QDRANT_TIMEOUT", "10.0"))

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
#
# DOMAIN MISMATCH NOTE (P2):
# Both DEFAULT_BI_ENCODER_MODEL and DEFAULT_CROSS_ENCODER_MODEL are trained on
# MS MARCO — a web-search passage dataset. This causes significant domain mismatch
# on academic PDF prose and is the primary diagnosed cause of:
#   - Strategy 11 (MiniLM bi-encoder): 0.292 MRR vs BM25 0.573 MRR
#   - Strategy 10 (ms-marco cross-encoder): 0.483 MRR vs BM25 0.573 MRR
# Dense embeddings diffuse rare coined technical terms ("StarShell", "POMDP",
# "AgentRunner") across generic semantic neighborhoods, while BM25 retains exact
# lexical precision via IDF weighting.
#
# SPECTER2 (allenai/specter2_base) is purpose-trained on scientific citation graphs
# and uses ASYMMETRIC adapters: specter2_proximity for document passages and
# specter2_adhoc_query for search queries. Correct [CLS]-token pooling (index 0,
# NOT mean pooling) is required. Without the asymmetric adapters, SPECTER2 may
# score near zero on short keyword queries against long academic passages.
# See src/retrieval/retrievers/neural.py SPECTER2Retriever for implementation.
DEFAULT_BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_SPECTER2_BASE_MODEL = "allenai/specter2_base"
DEFAULT_SPECTER2_ADAPTER = "allenai/specter2_proximity"
DEFAULT_SPECTER2_QUERY_ADAPTER = "allenai/specter2_adhoc_query"

# PPMI Parameters
DEFAULT_PPMI_WINDOW_SIZE = 5
DEFAULT_PPMI_VOCAB_SIZE = 1500
DEFAULT_PPMI_MAX_CONTEXT = 50
