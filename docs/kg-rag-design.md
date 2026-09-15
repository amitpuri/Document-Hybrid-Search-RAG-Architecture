# Knowledge Graph RAG Layer Design (`docs/kg-rag-design.md`)

## 1. Overview & Architecture

This document describes the Knowledge Graph (KG) retrieval and generation extension for `Document-Hybrid-Search-RAG-Architecture`, inspired by the Microsoft GraphRAG and LightRAG paradigms in `awslabs/unified-kg-rag-on-aws`.

The implementation is engineered to run **completely locally** without proprietary cloud services (no Amazon Neptune, AWS Bedrock, or OpenSearch required). The graph storage and graph traversal are powered by **NetworkX** with structured JSON disk persistence and local vector/lexical indexes.

```
                    [ Ingestion Pipeline: PDFs -> DocumentChunks ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ Chunk Store ]                                 [ Graph Extraction Stage ]
   (Parquet / InMemory Store)                      (BaseGraphExtractor: Heuristic / LLM)
                  │                                               │
                  │                                      [ Entity Resolution ]
                  │                                  (Fuzzy matching: name + type)
                  │                                               │
                  │                                               ▼
                  │                                      [ BaseGraphStore ]
                  │                                   (NetworkXGraphStore + JSON)
                  │                                               │
                  ▼                                               ▼
   ┌─────────────────────────────── Retrieval Pipeline ───────────────────────────────┐
   │                                                                                  │
   │  BM25 (Sparse)       TF-IDF (Dense)      PPMI / Neural        GraphRetriever     │
   │  (Lexical tokens)    (Sublinear TF)      (Dense Embeds)     (Local & Global KG)  │
   │        │                    │                   │                     │          │
   │        └────────────┬───────┴───────────────────┴─────────────────────┘          │
   │                     ▼                                                            │
   │          Reciprocal Rank Fusion (RRF: BM25 + TF-IDF + Graph)                     │
   │                     │                                                            │
   │          Candidate Deduplication (Jaccard similarity threshold)                  │
   │                     │                                                            │
   │          Maximal Marginal Relevance (MMR diversity re-ranking)                   │
   │                     │                                                            │
   │                     ▼                                                            │
   │       Strategy 14: rrf_graph_dedup_mmr Ranked Chunks                             │
   └─────────────────────────────────────┬────────────────────────────────────────────┘
                                         │
                                         ▼
                             [ Generation Pipeline ]
                      (ContextBuilder with Graph Provenance)
                                         │
                                         ▼
                     [ Grounded Answer with Citations ]
```

---

## 2. Ingestion & Graph Extraction Stage

### 2.1 Execution Timing
Graph extraction runs **after chunking** and **alongside retrieval indexing**:
- Inputs: `Sequence[DocumentChunk]` from the chunk store.
- Outputs: Entities, relationships, and chunk-to-graph provenance indices stored in the `BaseGraphStore`.
- Deterministic Caching: Graph state is saved to `.cache/kg/graph_{corpus_hash}.json` keyed on the corpus directory state, chunking parameters, and extractor configuration.

### 2.2 Domain Entities & Relationships
Entities and relationships are represented as strongly-typed dataclasses:

```python
@dataclass
class Entity:
    name: str                       # e.g., "StarShell"
    entity_type: str                # e.g., "FRAMEWORK", "CONCEPT", "MODEL", "METRIC"
    description: str                # Short summary or definition
    chunk_ids: Set[int]             # Source chunks where entity is mentioned
    doc_names: Set[str]             # Source documents where entity appears
    metadata: Dict[str, Any]

@dataclass
class Relationship:
    source: str                     # Entity name
    target: str                     # Entity name
    relation_type: str              # e.g., "USES", "EXTENDS", "EVALUATES", "CONSTRAINS"
    description: str                # Synthesized context of relationship
    weight: float                   # Frequency / confidence score
    chunk_ids: Set[int]             # Provenance back-references
    metadata: Dict[str, Any]
```

### 2.3 Pluggable Graph Extractor Interface (`BaseGraphExtractor`)
Following the existing `BaseGenerator` pattern in `src/generation/base.py`:

```python
class BaseGraphExtractor(ABC):
    @abstractmethod
    def extract_from_chunks(
        self,
        chunks: Sequence[DocumentChunk]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extracts entities and relationships from a list of document chunks."""
        pass
```

Implementations:
1. **`HeuristicGraphExtractor`** (Default / Offline / Zero-Cost):
   - Extracts technical entities using capitalization, camelCase/snake_case terms, section titles, and scientific terminology patterns.
   - Detects relationships via sentence-level entity co-occurrence, syntactic dependency clues, and relational prepositional phrases.
   - Fast, reproducible, and runnable in offline CI/eval environments without paid API keys.
2. **`LLMGraphExtractor`** (Optional Production / High-Precision):
   - Uses prompt instructions with JSON schema outputs to extract nuanced entities and semantic predicates per chunk via `BaseGenerator` adapters (`OpenAI`, `Gemini`, `Anthropic`).

### 2.4 Entity Resolution & Deduplication
To resolve aliases across multiple papers (e.g., *"StarShell"* vs. *"StarShell Enterprise Agent"*, *"POMDP"* vs. *"Partially Observable Markov Decision Process"*):
1. **Case & Punctuation Normalization**: Canonical lowercase stripping of extraneous articles (*the*, *a*).
2. **Fuzzy & Levenshtein / Jaccard Matching**: Pairs of entities with same or compatible `entity_type` and high token-overlap/character similarity ($\ge 0.88$) are merged.
3. **Provenance Merging**: Merging unions their `chunk_ids`, `doc_names`, and consolidates descriptions.

---

## 3. Storage Abstraction (`BaseGraphStore` & `NetworkXGraphStore`)

Following `src/ingestion/storage.py`'s decoupled pattern:

```python
class BaseGraphStore(ABC):
    @abstractmethod
    def add_entity(self, entity: Entity) -> None: ...
    @abstractmethod
    def add_relationship(self, rel: Relationship) -> None: ...
    @abstractmethod
    def get_entity(self, name: str) -> Optional[Entity]: ...
    @abstractmethod
    def get_neighbors(self, name: str, hops: int = 1) -> List[Tuple[Entity, Relationship, Entity]]: ...
    @abstractmethod
    def get_communities(self) -> Dict[int, List[str]]: ...
    @abstractmethod
    def save(self, path: Path | str) -> None: ...
    @abstractmethod
    def load(self, path: Path | str) -> None: ...
```

### `NetworkXGraphStore`
- Internal representation: `networkx.MultiDiGraph` where nodes store `Entity` attributes and edges store `Relationship` attributes.
- Inverted index: Entity/Relationship $\to$ `chunk_ids` allows instant mapping from subgraphs back to document chunks.
- Persistence: JSON-serialized node-link format with atomic write.

---

## 4. Graph Retriever (`GraphRetriever`)

Inherits from `src/retrieval/retrievers/base.py:BaseRetriever`. Provides two query modes:

### 4.1 "Local" Search Mode (Entity Neighborhood Traversal)
Analogous to GraphRAG Local Search & LightRAG Local Retrieval:
1. **Query Entity Linking**: Matches keywords/entities from user query against graph nodes (exact match + BM25/fuzzy token match).
2. **Neighborhood Subgraph Traversal**: Explores 1- to 2-hop ego networks around matched entities.
3. **Chunk Scoring**: For every chunk tied to visited entities and edges, score is accumulated:
   $$\text{Score}_{\text{local}}(c) = \sum_{e \in \mathcal{E}_{\text{visited}} \cap c} \text{weight}(e) \cdot \text{relevance}(e, q) + \sum_{r \in \mathcal{R}_{\text{visited}} \cap c} \text{weight}(r)$$
4. Normalizes chunk scores into a probability/relative ranking.

### 4.2 "Global" Search Mode (Community Summarization)
Analogous to GraphRAG Global Search:
1. **Community Detection**: Computes non-overlapping communities using NetworkX's built-in Louvain algorithm (`nx.community.louvain_communities`).
2. **Community Topic Profiling**: Computes key terms and central bridge nodes (betweenness centrality) for each community.
3. **Query-to-Community Matching**: Scores communities against the query.
4. **Chunk Scoring**: Chunks belonging to relevant communities receive rank boosts based on internal node degree.

### 4.3 Combined Graph Scoring
`GraphRetriever.score(query)` returns a 1D `np.ndarray` of scores for all chunks in the corpus:
$$\text{Score}_{\text{graph}}(c) = 0.7 \cdot \text{Score}_{\text{local}}(c) + 0.3 \cdot \text{Score}_{\text{global}}(c)$$

---

## 5. Fusion Layer Integration: `rrf_graph_dedup_mmr`

In `src/retrieval/pipeline.py`:
1. `b_rank` (Pure BM25 rank)
2. `d_rank` (Pure TF-IDF rank)
3. `g_rank` (GraphRetriever rank)

We fuse them using 3-way Reciprocal Rank Fusion:
$$\text{RRF}(c) = \frac{1}{k + \text{rank}_b(c) + 1} + \frac{1}{k + \text{rank}_d(c) + 1} + \frac{1}{k + \text{rank}_g(c) + 1}$$

Where $k=60$. The fused candidates then pass through:
- **Deduplication**: Filters near-identical chunks ($\text{Jaccard} \ge 0.85$).
- **MMR (Maximal Marginal Relevance)**: Promotes diversity with $\lambda=0.7$.

New Strategy Alias:
`"rrf_graph_dedup_mmr"` $\to$ `"14. RRF + Graph + Dedup + MMR"`

---

## 6. Evaluation Metric: Deterministic Graph Coverage

In `src/evaluation/metrics.py`:
- Traditional IR metrics (`MRR`, `Recall@1,3,5`, `NDCG@5`) measure target chunk retrieval.
- **Entity/Relationship Coverage**: Given ground-truth entity keywords and relationship predicates for a query, checks whether the retrieved top-K context text covers the essential entities and relational verbs:
  $$\text{Coverage}_{\text{entity}} = \frac{|\{e \in \mathcal{E}_{\text{target}} : e \text{ in context}\}|}{|\mathcal{E}_{\text{target}}|}$$
  $$\text{Coverage}_{\text{rel}} = \frac{|\{r \in \mathcal{R}_{\text{target}} : r \text{ in context}\}|}{|\mathcal{R}_{\text{target}}|}$$
- Fully deterministic regex word-boundary matching (no LLM judge required).

---

## 7. Generation Layer Integration

In `src/generation/context.py` & `prompts.py`:
- `ContextBuilder` provides `build_graph_augmented_context(chunks, graph_triplets)`:
  ```
  [Knowledge Graph Context]
  - StarShell (FRAMEWORK) --[EXECUTES]--> Terminal Automation
  - POMDP (FORMALISM) --[UPDATES]--> Belief State Filtering

  [Source 1: 2604.00073v3.pdf | Page 5 | § 3.2 StarShell]
  ...
  ```
- `BaseGenerator` implementations (`OpenAI`, `Gemini`, `Anthropic`, and `GroundedSynthesisGenerator`) utilize the combined context to answer technical multi-hop questions with grounded citations.
