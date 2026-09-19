"""
Distributional Semantic PPMI Word Embeddings & Sparse Cosine Retriever.
Zero external dependency from-scratch implementation.

SECURITY NOTE: This module uses .npz format (numpy compressed archive) instead of
pickle for caching to prevent arbitrary code execution from untrusted cache files.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.common.text import STOPWORDS
from src.config import CACHE_DIR
from src.ingestion.cache import compute_cache_key
from src.retrieval.retrievers.base import BaseRetriever


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
        max_context_per_word: int = 50,
    ):
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.max_context_per_word = max_context_per_word

        # 1. Select top content vocabulary
        term_freqs: Counter = Counter()
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
                    pmi = math.log2(
                        (count * self.total_cooccur) / (p_w * p_c * 2 * self.window_size)
                    )
                    if pmi > 0:
                        word_ppmis.append((ctx, pmi))

            word_ppmis.sort(key=lambda x: x[1], reverse=True)
            self.word_vectors[word] = dict(word_ppmis[: self.max_context_per_word])

    def embed_text(self, tokens: List[str]) -> Tuple[Dict[str, float], float]:
        """Mean pools PPMI word vectors into a passage embedding vector."""
        combined: Counter = Counter()
        count: float = 0.0
        for token in tokens:
            if token in self.word_vectors:
                for ctx, weight in self.word_vectors[token].items():
                    combined[ctx] += weight  # type: ignore[assignment]
                count += 1.0

        if count == 0:
            return {}, 0.0

        avg_vec = {ctx: val / count for ctx, val in combined.items()}
        mag = math.sqrt(sum(v * v for v in avg_vec.values()))
        return avg_vec, mag


def sparse_cosine_similarity(
    vec_a: Dict[str, float],
    mag_a: float,
    vec_b: Dict[str, float],
    mag_b: float,
) -> float:
    """Computes cosine similarity between two sparse dictionary vectors."""
    if mag_a == 0.0 or mag_b == 0.0 or not vec_a or not vec_b:
        return 0.0

    smaller, larger = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
    dot = sum(val * larger[k] for k, val in smaller.items() if k in larger)
    if dot <= 0:
        return 0.0

    return dot / (mag_a * mag_b)


def _serialize_ppmi_embeddings_to_json(embeddings: "PPMIEmbeddings") -> Dict:
    """Serializes PPMIEmbeddings to JSON-compatible dict."""
    return {
        "window_size": embeddings.window_size,
        "vocab_size": embeddings.vocab_size,
        "max_context_per_word": embeddings.max_context_per_word,
        "vocab": sorted(list(embeddings.vocab)),
        "total_cooccur": embeddings.total_cooccur,
        "unigram_counts": dict(embeddings.unigram_counts),
        "word_vectors": {word: dict(vec) for word, vec in embeddings.word_vectors.items()},
    }


def _deserialize_ppmi_embeddings_from_json(data: Dict) -> "PPMIEmbeddings":
    """Deserializes PPMIEmbeddings from JSON dict."""
    emb = PPMIEmbeddings.__new__(PPMIEmbeddings)
    emb.window_size = data["window_size"]
    emb.vocab_size = data["vocab_size"]
    emb.max_context_per_word = data["max_context_per_word"]
    emb.vocab = set(data["vocab"])
    emb.total_cooccur = data["total_cooccur"]
    emb.unigram_counts = Counter(data["unigram_counts"])
    emb.word_vectors = {word: dict(vec) for word, vec in data["word_vectors"].items()}
    return emb


def _serialize_bm25_from_scratch_to_json(bm25: "BM25FromScratch") -> Dict:
    """Serializes BM25FromScratch to JSON-compatible dict."""
    return {
        "k1": bm25.k1,
        "b": bm25.b,
        "N": bm25.N,
        "doc_lengths": bm25.doc_lengths,
        "avgdl": bm25.avgdl,
        "idf": dict(bm25.idf),
        "doc_freqs": [dict(counter) for counter in bm25.doc_freqs],
    }


def _deserialize_bm25_from_scratch_from_json(data: Dict) -> "BM25FromScratch":
    """Deserializes BM25FromScratch from JSON dict."""
    bm25 = BM25FromScratch.__new__(BM25FromScratch)
    bm25.k1 = data["k1"]
    bm25.b = data["b"]
    bm25.N = data["N"]
    bm25.doc_lengths = data["doc_lengths"]
    bm25.avgdl = data["avgdl"]
    bm25.idf = dict(data["idf"])
    bm25.doc_freqs = [Counter(freq_dict) for freq_dict in data["doc_freqs"]]
    bm25.corpus_tokens = []  # Not needed for scoring, can be empty
    return bm25


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
        df: Counter = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                df[term] += 1
        return {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
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
        cache_dir: Union[str, Path] = CACHE_DIR,
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
        corpus_dir: Optional[Union[str, Path]] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Index the corpus with PPMI embeddings and BM25.

        Uses disk caching (JSON + npz format, no pickle).
        """
        # ── Try loading from cache ────────────────────────────────────────────
        cache_base: Optional[Path] = None
        if corpus_dir and not force_rebuild:
            cache_key = compute_cache_key(corpus_dir, extra_tag="ppmi")
            cache_base = self.cache_dir / f"ppmi_{cache_key}"
            cache_json = Path(str(cache_base) + ".json")
            cache_npz = Path(str(cache_base) + ".npz")

            # Try loading from new JSON+NPZ format
            if cache_json.exists() and cache_npz.exists():
                try:
                    # Load embedder and BM25 metadata from JSON
                    with open(cache_json, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    self.embedder = _deserialize_ppmi_embeddings_from_json(meta["embedder"])
                    self.bm25_scratch = _deserialize_bm25_from_scratch_from_json(
                        meta["bm25_scratch"]
                    )
                    self._corpus_size = meta["corpus_size"]

                    # Load chunk embeddings from npz
                    with np.load(cache_npz, allow_pickle=False) as npz_data:
                        self.chunk_embeddings = []
                        for i in range(self._corpus_size):
                            mag = float(npz_data[f"mag_{i}"])
                            # Reconstruct sparse vector from saved keys/values
                            vec_dict = {}
                            if f"keys_{i}" in npz_data and f"vals_{i}" in npz_data:
                                keys = npz_data[f"keys_{i}"]
                                vals = npz_data[f"vals_{i}"]
                                vec_dict = {str(k): float(v) for k, v in zip(keys, vals)}
                            self.chunk_embeddings.append((vec_dict, mag))

                    return
                except Exception:
                    pass  # cache corrupt, fall through to rebuild

            # Legacy migration: try loading old pickle cache (for backwards compatibility)
            old_pkl_file = self.cache_dir / f"ppmi_{cache_key}.pkl"
            if old_pkl_file.exists():
                try:
                    import pickle as pkl

                    with open(old_pkl_file, "rb") as f:
                        state = pkl.load(f)
                    self.embedder = state["embedder"]
                    self.bm25_scratch = state["bm25_scratch"]
                    self.chunk_embeddings = state["chunk_embeddings"]
                    self._corpus_size = state["corpus_size"]
                    return
                except Exception:
                    pass  # migration failed, fall through to rebuild

        # ── Build from scratch ────────────────────────────────────────────────
        tokenized_corpus = [ppmi_tokenize(doc) for doc in corpus_texts]
        self.embedder = PPMIEmbeddings(
            tokenized_corpus,
            window_size=self.window_size,
            vocab_size=self.vocab_size,
            max_context_per_word=self.max_context_per_word,
        )
        self.bm25_scratch = BM25FromScratch(tokenized_corpus)
        self.chunk_embeddings = [self.embedder.embed_text(toks) for toks in tokenized_corpus]
        self._corpus_size = len(corpus_texts)

        # ── Persist to cache (JSON + NPZ format, secure, no pickle) ─────────────
        if corpus_dir and cache_base is not None:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # Save embedder and BM25 metadata as JSON
                meta = {
                    "embedder": _serialize_ppmi_embeddings_to_json(self.embedder),
                    "bm25_scratch": _serialize_bm25_from_scratch_to_json(self.bm25_scratch),
                    "corpus_size": self._corpus_size,
                }
                cache_json = Path(str(cache_base) + ".json")
                with open(cache_json, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # Save chunk embeddings as npz (sparse vectors)
                npz_dict = {}
                for i, (vec_dict, mag) in enumerate(self.chunk_embeddings):
                    npz_dict[f"mag_{i}"] = np.array(mag, dtype=np.float32)
                    if vec_dict:
                        keys = np.array(list(vec_dict.keys()), dtype=np.str_)
                        vals = np.array(list(vec_dict.values()), dtype=np.float32)
                        npz_dict[f"keys_{i}"] = keys
                        npz_dict[f"vals_{i}"] = vals

                cache_npz = Path(str(cache_base) + ".npz")
                np.savez_compressed(cache_npz, **npz_dict)

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

    def retrieve_ppmi_bm25_rrf_with_scores(
        self, query: str, k: int = 60
    ) -> Tuple[List[int], Dict[int, float]]:
        """Runs Strategy 9: PPMI Semantic + BM25 RRF and returns (ranked_indices, scores_dict)."""
        if self.embedder is None or self.bm25_scratch is None:
            raise RuntimeError("PPMIRetriever must be indexed before retrieving.")
        q_toks = ppmi_tokenize(query)
        q_vec, q_mag = self.embedder.embed_text(q_toks)
        ppmi_scores = [
            sparse_cosine_similarity(q_vec, q_mag, c_vec, c_mag)
            for c_vec, c_mag in self.chunk_embeddings
        ]
        bm25_scores = self.bm25_scratch.get_scores(q_toks)

        ppmi_ranking = sorted(
            range(self._corpus_size),
            key=lambda i: ppmi_scores[i],
            reverse=True,
        )
        bm25_ranking = sorted(
            range(self._corpus_size),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )

        scores: Dict[int, float] = {}
        for rank, idx in enumerate(ppmi_ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(bm25_ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        ranked = sorted(scores, key=lambda x: scores.get(x, 0.0), reverse=True)
        return ranked, scores

    def retrieve_ppmi_bm25_rrf(self, query: str, k: int = 60) -> List[int]:
        """Runs Strategy 9: PPMI Semantic + BM25 RRF."""
        ranked, _ = self.retrieve_ppmi_bm25_rrf_with_scores(query, k=k)
        return ranked
