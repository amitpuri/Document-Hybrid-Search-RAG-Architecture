"""
Knowledge Graph Retriever with Local and Global Search Modes.
Implements entity-neighborhood traversal (Local) and community-detection summarization (Global).
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Sequence, Any
import numpy as np

from src.retrieval.retrievers.base import BaseRetriever
from src.common.types import DocumentChunk
from src.ingestion.storage import BaseChunkStore
from src.ingestion.graph_extractor import (
    BaseGraphExtractor,
    HeuristicGraphExtractor,
    Entity,
    Relationship,
    normalize_entity_name,
)
from src.ingestion.graph_store import BaseGraphStore, NetworkXGraphStore
from src.config import CACHE_DIR


_COMMUNITY_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "have", "been",
    "what", "does", "how", "can", "are", "which", "into", "their", "when",
    "where", "model", "models", "paper", "study", "approach", "system",
    "using", "based", "such", "both", "each", "other", "some", "many",
}


class GraphRetriever(BaseRetriever):

    """
    Retriever leveraging a Knowledge Graph to score document chunks.
    Provides two query modes:
      1. Local: Traverses 1-2 hop neighborhoods around query entities.
      2. Global: Identifies high-modularity community clusters matching the query.
    """

    def __init__(
        self,
        graph_store: Optional[BaseGraphStore] = None,
        extractor: Optional[BaseGraphExtractor] = None,
        cache_dir: Optional[Path | str] = None,
        local_weight: float = 0.7,
        global_weight: float = 0.3,
    ):
        self.graph_store: BaseGraphStore = graph_store or NetworkXGraphStore()
        self.extractor: BaseGraphExtractor = extractor or HeuristicGraphExtractor()
        self.cache_dir: Path = Path(cache_dir) if cache_dir else Path(CACHE_DIR)
        self.local_weight = local_weight
        self.global_weight = global_weight

        self.corpus_size: int = 0
        self._entity_lookup: Dict[str, str] = {}  # lowercase_norm -> canonical_name
        self._indexed: bool = False

    def index(
        self,
        corpus_texts: List[str],
        chunk_store: Optional[BaseChunkStore] = None,
        corpus_dir: Optional[Path | str] = None,
        force_rebuild: bool = False,
    ) -> None:
        """
        Indexes the knowledge graph over the corpus.
        Extracts entities and relationships or loads them from disk cache.
        """
        self.corpus_size = len(corpus_texts)
        kg_cache_dir = self.cache_dir / "kg"
        kg_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = kg_cache_dir / "graph_store.json"

        # Check if pre-cached graph exists
        if not force_rebuild and cache_file.exists():
            try:
                self.graph_store.load(cache_file)
                self._build_lookup()
                self._indexed = True
                return
            except Exception:
                pass  # Rebuild on load failure

        # Gather chunks
        chunks: List[DocumentChunk] = []
        if chunk_store is not None:
            chunks = chunk_store.get_all_chunks()
        else:
            # Construct chunks from formatted texts
            for idx, text in enumerate(corpus_texts):
                chunks.append(DocumentChunk(
                    chunk_id=idx,
                    doc_name="Unknown",
                    page_num=1,
                    section="Overview",
                    text=text,
                ))

        # Extract entities & relationships
        entities, relationships = self.extractor.extract_from_chunks(chunks)

        for ent in entities:
            self.graph_store.add_entity(ent)

        for rel in relationships:
            self.graph_store.add_relationship(rel)

        # Pre-compute community detection
        _ = self.graph_store.get_communities()

        # Save to disk cache
        try:
            self.graph_store.save(cache_file)
        except Exception:
            pass

        self._build_lookup()
        self._indexed = True

    def _build_lookup(self) -> None:
        """Builds fast lowercased token and ngram lookup for entities."""
        self._entity_lookup.clear()
        for ent in self.graph_store.get_all_entities():
            norm = normalize_entity_name(ent.name).lower()
            if norm:
                self._entity_lookup[norm] = ent.name

    def identify_query_entities(self, query: str) -> List[str]:
        """Identifies entities mentioned in the query string."""
        q_lower = query.lower()
        matched: Set[str] = set()

        # 1. Exact or whole-word phrase matching against known entities
        for norm_name, canonical in self._entity_lookup.items():
            if norm_name in _COMMUNITY_STOPWORDS or len(norm_name) < 3:
                continue
            pattern = r"\b" + re.escape(norm_name) + r"\b"
            if re.search(pattern, q_lower):
                matched.add(canonical)

        # 2. Extract acronyms & capitalized tokens from query
        acronyms = re.findall(r"\b[A-Z]{2,10}\b", query)
        for acr in acronyms:
            norm_acr = acr.lower()
            if norm_acr in self._entity_lookup:
                matched.add(self._entity_lookup[norm_acr])

        # 3. Fuzzy sub-phrase matching for multi-word entities
        if not matched:
            q_words = set(re.findall(r"\b[a-z]{3,}\b", q_lower)) - _COMMUNITY_STOPWORDS
            for norm_name, canonical in self._entity_lookup.items():
                ent_words = set(re.findall(r"\b[a-z]{3,}\b", norm_name)) - _COMMUNITY_STOPWORDS
                if len(ent_words) >= 2 and ent_words.issubset(q_words):
                    matched.add(canonical)

        return sorted(list(matched))

    def local_search(self, query: str, hops: int = 2) -> np.ndarray:
        """
        Local Search Mode:
        Finds query entities, traverses 1-2 hops in the graph, and weights
        chunks linked to the traversed neighborhood.
        """
        scores = np.zeros(self.corpus_size, dtype=float)
        query_entities = self.identify_query_entities(query)

        if not query_entities:
            return scores

        visited_nodes: Set[str] = set()
        visited_edges: List[Tuple[str, str, Dict[str, Any]]] = []

        for q_ent in query_entities:
            visited_nodes.add(q_ent)
            ent_obj = self.graph_store.get_entity(q_ent)
            if ent_obj:
                # Direct entity match gets highest initial weight
                for cid in ent_obj.chunk_ids:
                    if 0 <= cid < self.corpus_size:
                        scores[cid] += 4.0

            # Traverse 1-2 hops
            neighbors = self.graph_store.get_neighbors(q_ent, hops=hops)
            visited_edges.extend(neighbors)

        for src, tgt, edge_data in visited_edges:
            w = float(edge_data.get("weight", 1.0))
            edge_chunks = edge_data.get("chunk_ids", set())
            for cid in edge_chunks:
                if 0 <= cid < self.corpus_size:
                    scores[cid] += 2.0 * min(w, 3.0)

            for node_name in (src, tgt):
                if node_name not in visited_nodes:
                    visited_nodes.add(node_name)
                    node_obj = self.graph_store.get_entity(node_name)
                    if node_obj:
                        for cid in node_obj.chunk_ids:
                            if 0 <= cid < self.corpus_size:
                                scores[cid] += 1.0

        max_s = np.max(scores)
        if max_s > 0:
            scores = scores / max_s

        return scores

    def global_search(self, query: str) -> np.ndarray:
        """
        Global Search Mode:
        Matches query against detected communities and scores chunks based on
        community membership and internal node centrality.
        """
        scores = np.zeros(self.corpus_size, dtype=float)
        communities = self.graph_store.get_communities()
        if not communities:
            return scores

        q_tokens = set(re.findall(r"\b[a-z]{3,}\b", query.lower())) - _COMMUNITY_STOPWORDS
        if not q_tokens:
            return scores

        # Score communities by token overlap
        scored_communities = []
        for comm_id, members in communities.items():
            comm_text = " ".join(members).lower()
            overlap_tokens = [t for t in q_tokens if t in comm_text]
            if len(overlap_tokens) >= 2 or (len(overlap_tokens) == 1 and any(len(t) >= 6 for t in overlap_tokens)):
                score_val = float(len(overlap_tokens)) / (len(members) ** 0.5 + 1e-6)
                scored_communities.append((score_val, comm_id))

        scored_communities.sort(key=lambda x: x[0], reverse=True)

        # Distribute scores only from top-2 distinct matching communities
        for c_score, comm_id in scored_communities[:2]:
            members = communities[comm_id]
            for m_name in members:
                ent = self.graph_store.get_entity(m_name)
                if ent:
                    for cid in ent.chunk_ids:
                        if 0 <= cid < self.corpus_size:
                            scores[cid] += c_score

        max_s = np.max(scores)
        if max_s > 0:
            scores = scores / max_s

        return scores

    def score(self, query: str) -> np.ndarray:
        """
        Blends Local and Global search scores.
        Returns a 1D numpy array of length corpus_size.
        """
        if not self._indexed:
            return np.zeros(self.corpus_size, dtype=float)

        local_scores = self.local_search(query, hops=2)
        has_local = np.max(local_scores) > 0

        if has_local:
            global_scores = self.global_search(query)
            return 0.85 * local_scores + 0.15 * global_scores
        else:
            return self.global_search(query)

