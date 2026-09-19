# Generation Pipeline (`src/generation`)

The `generation` package implements **Pipeline 3** of the architecture. It connects retrieved document passages to language generation, establishing an enterprise-grade Retrieval-Augmented Generation (RAG) framework with source attribution, grounded citation formatting, client-side rate limiting, circuit breakers, and multi-provider deployment routing.

---

## 📂 Module Breakdown

```
src/generation/
├── __init__.py           # Exports GenerationPipeline, ContextBuilder, factory, router, providers
├── context.py            # ContextBuilder assembling formatted passages and graph relationships
├── prompts.py            # System prompt & instruction templates for grounded QA
├── confidence.py         # Per-claim calibrated confidence scoring (C3) - retrieval-native signals
├── base.py               # BaseGenerator abstract interface & GenerationResult contract
├── rate_limiter.py       # TokenBucket & ProviderRateLimiter (client-side RPM/TPM management)
├── router.py             # LLMRouter with fallback chain, error classification & circuit breaker
├── factory.py            # Auto-detection generator factory (env-var & provider routing)
├── mock.py               # GroundedSynthesisGenerator (local offline citation synthesizer)
├── providers/            # Multi-provider LLM adapters with deployment routes
│   ├── config.py         # Per-provider/route limit configuration (RPM/TPM from env)
│   ├── errors.py         # Normalized RateLimitError, CapacityError, AuthError
│   ├── anthropic_generator.py # Anthropic (direct + AWS Bedrock)
│   ├── openai_generator.py    # OpenAI (direct + Azure OpenAI)
│   └── gemini_generator.py    # Gemini (direct + GCP Vertex AI)
└── pipeline.py           # GenerationPipeline orchestrator coordinating context and generation
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
- **Graph-Enriched Context**: When using Knowledge Graph strategies (`rrf_graph_dedup_mmr`), `ContextBuilder` can attach 1-hop connected entity definitions and relations into the context header.
- **Budget Management**: Enforces `max_context_chars=8000` to prevent context window overflow while preserving complete citation boundaries.

### 2. Prompt Engineering (`prompts.py`)
- **`SYSTEM_PROMPT`**: Enforces strict grounding guidelines:
  1. Compels the model to cite sources using `[Source N]` bracketed references.
  2. Forbids hallucination when information is absent.
  3. Demands attribution of technical definitions, theorems, and mechanisms to specific papers and sections.
- **`format_qa_prompt(query, formatted_context)`**: Assembles context passages and the user's question into an instruction block.

### 3. Client-Side Rate Limiting (`rate_limiter.py`)
Protects cloud quotas and prevents HTTP 429 errors using dual-metered `TokenBucket` tracking:
- **Requests Per Minute (RPM)**
- **Tokens Per Minute (TPM)**: Tracks input tokens and estimated/reserved output tokens.
- **Quota Multipliers**: Supports provider-specific burn rates (e.g., AWS Bedrock 5× output token rate multiplier for Claude 3.7+).

### 4. Circuit Breakers & Fallback Chains (`router.py`)
- **Circuit Breaker**: Tracks consecutive failures per route. After $N$ consecutive failures (default: 3), the route trips into cooldown (default: 60s) and is bypassed.
- **Automated Fallback**: Automatically cascades across a configured list of providers and deployment routes (e.g., Anthropic Direct $\to$ AWS Bedrock $\to$ OpenAI Direct $\to$ Gemini Direct $\to$ Offline Mock).
- **Graceful Degradation**: If all cloud endpoints fail or no API keys are present, gracefully falls back to `GroundedSynthesisGenerator`.

### 5. Multi-Provider LLM Adapters (`providers/`)

| Provider | Direct Route | Enterprise / Cloud Route | Default Model | Fallback Model |
|---|---|---|---|---|
| **Anthropic** | `api.anthropic.com` | AWS Bedrock (`boto3`) | `claude-sonnet-5` | `claude-sonnet-4-6` |
| **OpenAI** | `api.openai.com` | Azure OpenAI | `gpt-5` | `gpt-4o-mini` |
| **Gemini** | `generativelanguage.googleapis.com` | GCP Vertex AI | `gemini-3.8-flash` | `gemini-2.5-flash` |
| **Mock** | *Offline Local* | *Offline Local* | `GroundedSynthesisGenerator` | *(no API key needed)* |
|
### 6. Per-Claim Calibrated Confidence (C3) - `confidence.py`
Retrieval-native confidence scoring without second LLM call:
- **Confidence Levels**: HIGH, MEDIUM, LOW based on retrieval signals
- **Signals Used**: Rank, fusion score, lexical overlap between query and chunk
- **Thresholds**:
  - HIGH: rank ≤ 2 AND fusion_score ≥ 0.7 AND lexical_overlap ≥ 0.3
  - MEDIUM: rank ≤ 4 OR (fusion_score ≥ 0.5 AND lexical_overlap ≥ 0.2)
  - LOW: otherwise
- **Display**: Confidence levels shown alongside citations as `[HIGH]/[MED]/[LOW]`
- **✅ COMPLETED 2026-09-18**

---

## 💡 Usage Example

### Via CLI:
```bash
# Default: auto-detects configured API key, falls back to offline mock
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_dedup_mmr

# Knowledge-Graph enriched generation (Strategy 14)
python -m src.cli ask "How does the Binding Constraint Thesis affect harness comparisons?" --strategy rrf_graph_dedup_mmr --graph-mode local

# Multi-provider route selection
python -m src.cli ask "..." --provider anthropic --route direct
python -m src.cli ask "..." --provider anthropic --route bedrock
python -m src.cli ask "..." --provider openai --route direct
python -m src.cli ask "..." --provider openai --route azure
python -m src.cli ask "..." --provider gemini --route direct
python -m src.cli ask "..." --provider gemini --route vertex

# Force offline mock mode (deterministic, zero external API keys required)
python -m src.cli ask "..." --provider mock
```

### Via Python API:
```python
from src.engine import HybridSearchEngine

engine = HybridSearchEngine.from_corpus("corpus/")

# Generate grounded answer with citations
response = engine.generate_answer(
    query="How does the Binding Constraint Thesis affect harness comparisons?",
    strategy="rrf_graph_dedup_mmr",
    top_k=3
)

print(response.answer)
print("\nStructured Citations:")
for c in response.citations:
    print(f"- Cited: {c.doc_name} | Page {c.page_num} | Section: {c.section}")
```
