"""
Distributional Semantic PPMI Word Embeddings & Sparse Cosine Retriever.
Zero external dependency from-scratch implementation.
"""

import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from src.retrieval.retrievers.base import BaseRetriever
from src.common.text import STOPWORDS, clean_text
from src.ingestion.cache import compute_cache_key
from src.config import CACHE_DIR


def ppmi_tokenize(text: str) -> List[str]:
    """Tokenizes text for PPMI distributional embeddings."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
    return cleaned.split()


class PPMIEmbeddings:
    """
    Constructs term-term PPMI vectors from corpus co-occurrences.
    PPMI(w1, w2) = max(0, log2( P(w1, w2) / (P(w1) * P(w2)) ))
    """

    def __init__(
        self,
        tokenized_corpus: List[List[str]],
        window_size: int = 5,
        vocab_size: int = 1500,
        max_context_per_word: int = 50
    ):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.max_context_per_word = max_context_per_word

        # 1. Select top content vocabulary
        term_freqs = Counter()
        for doc in tokenized_corpus:
            content_tokens = [t for t in doc if t not in STOPWORDS and len(t) > 2]
            term_freqs.update(content_tokens)

        top_terms = [term for term, _ in term_freqs.most_common(vocab_size)]
        self.vocab: Set[str] = set(top_terms)

        # 2. Count co-occurrences within sliding window
        cooccur: Dict[str, Counter] = defaultdict(Counter)
        unigram_counts: Counter = Counter()
        total_cooccur = 0

        for doc in tokenized_corpus:
            filtered = [t for t in doc if t in self.vocab]
            n = len(filtered)
            for i, target in enumerate(filtered):
                unigram_counts[target] += 1
                start = max(0, i - window_size)
                end = min(n, i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        ctx = filtered[j]
                        cooccur[target][ctx] += 1
                        total_cooccur += 1

        self.total_cooccur = max(1, total_cooccur)
        self.unigram_counts = unigram_counts

        # 3. Build sparse PPMI vectors: keep top strongest associations
        self.word_vectors: Dict[str, Dict[str, float]] = {}
        for word, ctx_counts in cooccur.items():
            p_w = unigram_counts[word]
            if p_w == 0:
                continue

            word_ppmis = []
            for ctx, count in ctx_counts.items():
                p_c = unigram_counts[ctx]
                expected = (p_w * p_c * 2 * self.window_size) / self.total_cooccur
                if expected > 0:
                    pmi = math.log2((count * self.total_cooccur) / (p_w * p_c * 2 * self.window_size))
                    if pmi > 0:
                        word_ppmis.append((ctx, pmi))

            word_ppmis.sort(key=lambda x: x[1], reverse=True)
            self.word_vectors[word] = dict(word_ppmis[:self.max_context_per_word])

    def embed_text(self, tokens: List[str]) -> Tuple[Dict[str, float], float]:
        """Mean pools PPMI word vectors into a passage embedding vector."""
        combined = Counter()
        count = 0
        for token in tokens:
            if token in self.word_vectors:
                for ctx, weight in self.word_vectors[token].items():
                    combined[ctx] += weight
                count += 1

        if count == 0:
            return {}, 0.0

        avg_vec = {ctx: val / count for ctx, val in combined.items()}
        mag = math.sqrt(sum(v * v for v in avg_vec.values()))
        return avg_vec, mag


def sparse_cosine_similarity(
    vec_a: Dict[str, float],
    mag_a: float,
    vec_b: Dict[str, float],
    mag_b: float
) -> float:
    """Computes cosine similarity between two sparse dictionary vectors."""
    if mag_a == 0.0 or mag_b == 0.0 or not vec_a or not vec_b:
        return 0.0

    smaller, larger = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
    dot = sum(val * larger[k] for k, val in smaller.items() if k in larger)
    if dot <= 0:
        return 0.0

    return dot / (mag_a * mag_b)


class BM25FromScratch:
    """Pure-python standard library BM25 implementation."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_lengths = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, self.N)
        self.doc_freqs = [Counter(doc) for doc in corpus_tokens]
        self.idf = self._compute_idf()

    def _compute_idf(self) -> Dict[str, float]:
        df = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                df[term] += 1
        return {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        for i in range(self.N):
            doc_len = self.doc_lengths[i]
            freqs = self.doc_freqs[i]
            score = 0.0
            for term in query_tokens:
                if term in freqs:
                    f = freqs[term]
                    idf = self.idf.get(term, 0.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    score += idf * (f * (self.k1 + 1.0) / denom)
            scores[i] = score
        return scores


class PPMIRetriever(BaseRetriever):
    """Retriever utilizing PPMI distributional embeddings + BM25 scratch fusion."""

    def __init__(
        self,
        window_size: int = 5,
        vocab_size: int = 1500,
        max_context_per_word: int = 50,
        cache_dir: str | Path = CACHE_DIR,
    ):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.max_context_per_word = max_context_per_word
        self.cache_dir = Path(cache_dir)
        self.embedder: Optional[PPMIEmbeddings] = None
        self.bm25_scratch: Optional[BM25FromScratch] = None
        self.chunk_embeddings: List[Tuple[Dict[str, float], float]] = []
        self._corpus_size = 0

    def index(
        self,
        corpus_texts: List[str],
        corpus_dir: Optional[str | Path] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Indexes the corpus with PPMI embeddings and BM25, with disk caching."""
        # ── Try loading from cache ────────────────────────────────────────────
        cache_file: Optional[Path] = None
        if corpus_dir and not force_rebuild:
            cache_key = compute_cache_key(corpus_dir, extra_tag="ppmi")
            cache_file = self.cache_dir / f"ppmi_{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        state = pickle.load(f)
                    self.embedder = state["embedder"]
                    self.bm25_scratch = state["bm25_scratch"]
                    self.chunk_embeddings = state["chunk_embeddings"]
                    self._corpus_size = state["corpus_size"]
                    return
                except Exception:
                    pass  # cache corrupt, fall through to rebuild

        # ── Build from scratch ────────────────────────────────────────────────
        tokenized_corpus = [ppmi_tokenize(doc) for doc in corpus_texts]
        self.embedder = PPMIEmbeddings(
            tokenized_corpus,
            window_size=self.window_size,
            vocab_size=self.vocab_size,
            max_context_per_word=self.max_context_per_word
        )
        self.bm25_scratch = BM25FromScratch(tokenized_corpus)
        self.chunk_embeddings = [self.embedder.embed_text(toks) for toks in tokenized_corpus]
        self._corpus_size = len(corpus_texts)

        # ── Persist to cache ──────────────────────────────────────────────────
        if corpus_dir and cache_file is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump({
                        "embedder": self.embedder,
                        "bm25_scratch": self.bm25_scratch,
                        "chunk_embeddings": self.chunk_embeddings,
                        "corpus_size": self._corpus_size,
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass  # non-fatal: cache write failure

    def score(self, query: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("PPMIRetriever must be indexed before scoring.")
        q_toks = ppmi_tokenize(query)
        q_vec, q_mag = self.embedder.embed_text(q_toks)
        scores = [
            sparse_cosine_similarity(q_vec, q_mag, c_vec, c_mag)
            for c_vec, c_mag in self.chunk_embeddings
        ]
        return np.array(scores, dtype=np.float32)

    def retrieve_ppmi_bm25_rrf(self, query: str, k: int = 60) -> List[int]:
        """Runs Strategy 9: PPMI Semantic + BM25 RRF."""
        if self.embedder is None or self.bm25_scratch is None:
            raise RuntimeError("PPMIRetriever must be indexed before retrieving.")
        q_toks = ppmi_tokenize(query)
        q_vec, q_mag = self.embedder.embed_text(q_toks)
        ppmi_scores = [
            sparse_cosine_similarity(q_vec, q_mag, c_vec, c_mag)
            for c_vec, c_mag in self.chunk_embeddings
        ]
        bm25_scores = self.bm25_scratch.get_scores(q_toks)

        ppmi_ranking = sorted(range(self._corpus_size), key=lambda i: ppmi_scores[i], reverse=True)
        bm25_ranking = sorted(range(self._corpus_size), key=lambda i: bm25_scores[i], reverse=True)

        scores: Dict[int, float] = {}
        for rank, idx in enumerate(ppmi_ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        return sorted(scores, key=scores.get, reverse=True)
