# Generation Pipeline (`src/generation`)

The `generation` package implements **Pipeline 3** of the architecture. It connects retrieved document passages to language generation, establishing a production-ready Retrieval-Augmented Generation (RAG) framework with source attribution and grounded citation formatting.

---

## 📂 Module Breakdown

```
src/generation/
├── __init__.py      # Exports GenerationPipeline, ContextBuilder, prompt templates, generators
├── context.py       # ContextBuilder assembling formatted context with bracketed citations
├── prompts.py       # System prompt & instruction templates for grounded QA
├── base.py          # BaseGenerator abstract interface
├── mock.py          # GroundedSynthesisGenerator (local offline citation synthesizer)
└── pipeline.py      # GenerationPipeline orchestrator coordinating context and generation
```

---

## 🏗️ Architectural Components

### 1. Context Window Assembly (`context.py`)
Formats retrieved chunks into a clean, numbered context block with strict document provenance:
```
[Source 1: 2605.23950v1.pdf | Page 4 | § 3 The Binding Constraint Thesis]
For LLM agents operating on long-horizon tasks with comparable frontier models...

[Source 2: 2605.10223v1.pdf | Page 1 | § Abstract]
Beyond Autonomy: A Dynamic Tiered AgentRunner Framework...
```
Includes character budget management (`max_context_chars=8000`) to prevent context window overflow.

### 2. Prompt Engineering (`prompts.py`)
- **`SYSTEM_PROMPT`**: Enforces strict grounding guidelines:
  1. Compels the model to cite sources using `[Source N]` bracketed references.
  2. Forbids hallucination when information is absent.
  3. Demands attribution of technical definitions and theorems to specific papers and sections.
- **`format_qa_prompt(query, formatted_context)`**: Assembles context passages and the user's question into an instruction block.

### 3. Generator Abstractions & Adapters (`base.py`, `mock.py`)
- **`BaseGenerator`**: Abstract base class requiring `generate(query, formatted_context, retrieved_chunks, strategy_used) -> GenerationResult`.
- **`GroundedSynthesisGenerator`**: An out-of-the-box, zero-dependency offline generator. It extracts key findings, attributes statements to source documents/pages/sections, and outputs structured answers with citations without requiring cloud API keys.
- **Extensibility**: Clean plug-in interface ready to connect to external LLM providers (e.g. Google Gemini via `google-genai`, OpenAI via `openai`, Anthropic via `anthropic`, or local Ollama/vLLM endpoints).

---

## 💡 Usage Example

### Via CLI:
```bash
# Ask a question and generate a grounded answer with citations
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr --top-k 3
```

Output:
```
Based on retrieved documentation using strategy 'rrf_dedup_mmr':

**Key Finding**: "3 The Binding Constraint Thesis For LLM agents operating on long-horizon tasks with comparable frontier models, let B(M, H) denote the benchmark score of model M under harness H..."

**Citations & Provenance**:
- [Source 1] 2605.23950v1.pdf (Page 4, § 3 The Binding Constraint Thesis)
- [Source 2] 2605.23950v1.pdf (Page 4, § 3 The Binding Constraint Thesis)
- [Source 3] 2605.23950v1.pdf (Page 4, § 3 The Binding Constraint Thesis)
```

### Via Python API:
```python
from src.engine import HybridSearchEngine

engine = HybridSearchEngine.from_corpus("corpus/")

response = engine.generate_answer(
    query="What risk-tiering mechanisms does the AgentRunner framework apply?",
    strategy="rrf_dedup_mmr",
    top_k=3
)

print(response.answer)
for c in response.citations:
    print(f"Cited: {c.doc_name} (Page {c.page_num}, § {c.section})")
```
