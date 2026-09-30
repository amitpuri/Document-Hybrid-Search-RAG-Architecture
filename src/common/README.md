# Common Domain Models & Utilities (`src/common`)

The `common` package defines the foundational domain contracts, data structures, and text-processing utilities shared across the **Ingestion**, **Retrieval**, and **Generation** pipelines.

---

## 📂 Module Breakdown

```
src/common/
├── __init__.py      # Public exports for common domain types and text utilities
├── types.py         # Strongly-typed domain dataclasses
└── text.py          # Tokenization, cleaning, and Jaccard similarity functions
```

---

## 🧱 Core Domain Models (`types.py`)

### 1. `DocumentChunk`
Represents a structured passage extracted from a PDF document, preserving fine-grained provenance metadata:
```python
@dataclass
class DocumentChunk:
    chunk_id: int          # Global 0-indexed position within the corpus
    doc_name: str          # PDF filename (e.g., '2605.23950v1.pdf')
    page_num: int          # 1-indexed page number in the source PDF
    section: str           # Active section heading (e.g., '§ 3 The Binding Constraint Thesis')
    text: str              # Pure chunk text content
    metadata: Dict[str, Any] = field(default_factory=dict)
```
- **`to_formatted_text() -> str`**: Formats the chunk into the canonical prefixed format:  
  `[doc_name | Page N | § Section] <text>`

### 2. `SearchResult`
Represents a scored and ranked result from a retrieval strategy:
```python
@dataclass
class SearchResult:
    chunk_id: int          # Index of the matched chunk
    chunk: DocumentChunk   # The underlying DocumentChunk record
    score: float           # Relative relevance or ranking score
    rank: int              # 1-indexed rank position in results
    strategy: str          # Name of the strategy used (e.g., '14. RRF + Graph + Dedup + MMR')
```

### 3. `MetricScores`
Container for standard information retrieval and structural coverage evaluation metrics:
```python
@dataclass
class MetricScores:
    mrr: float                   # Mean Reciprocal Rank (1 / rank_of_first_relevant_chunk)
    recall_1: float              # Binary recall at top 1
    recall_3: float              # Binary recall at top 3
    recall_5: float              # Binary recall at top 5
    ndcg_5: float                # Normalized Discounted Cumulative Gain at rank 5
    entity_coverage: float = 0.0 # Fraction of ground-truth query entities found in top-5 chunks
    relation_coverage: float = 0.0 # Fraction of target relational edges recovered in top-5 chunks
```

### 4. `GenerationResult`
Structured response from the RAG Generation pipeline:
```python
@dataclass
class GenerationResult:
    query: str                    # Original user query or question
    answer: str                   # Grounded answer synthesized by the generator
    citations: List[DocumentChunk]# Specific source chunks used for provenance
    strategy_used: str            # Retrieval strategy employed
```

---

## 🛠️ Text Utilities (`text.py`)

- **`clean_text(text: str) -> str`**: Normalizes whitespace and removes non-alphanumeric punctuation.
- **`tokenize(text: str, remove_stopwords: bool = False) -> List[str]`**: Produces lowercase word tokens with optional stopword filtering.
- **`jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float`**: Computes the set intersection over union:
  $$\mathrm{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  Used by the post-processing deduplication module to detect and discard near-duplicate chunks produced by sliding-window overlap.
- **`STOPWORDS`**: Curated set of high-frequency English functional words.

---

## 💡 Usage Example

```python
from src.common import DocumentChunk, SearchResult, MetricScores, tokenize, jaccard_similarity

# Create a chunk
chunk = DocumentChunk(
    chunk_id=1561,
    doc_name="2605.23950v1.pdf",
    page_num=4,
    section="§ 3 The Binding Constraint Thesis",
    text="For LLM agents operating on long-horizon tasks..."
)

print(chunk.to_formatted_text())
# -> "[2605.23950v1.pdf | Page 4 | § 3 The Binding Constraint Thesis] For LLM agents..."

# Check similarity between two candidate texts
tokens_1 = tokenize(chunk.text)
tokens_2 = tokenize("For LLM agents on long horizon tasks...")
sim = jaccard_similarity(tokens_1, tokens_2)
print(f"Jaccard Overlap: {sim:.2f}")

# Record IR and structural metrics
metrics = MetricScores(
    mrr=0.667,
    recall_1=0.571,
    recall_3=0.714,
    recall_5=0.857,
    ndcg_5=0.714,
    entity_coverage=0.786,
    relation_coverage=0.643
)
print(f"MRR: {metrics.mrr:.3f}, Entity Coverage: {metrics.entity_coverage:.1%}")
```
