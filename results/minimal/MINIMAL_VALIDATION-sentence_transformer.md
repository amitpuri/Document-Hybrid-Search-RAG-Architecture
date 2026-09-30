# Minimal Validation: sentence_transformer

Generated: Minimal mode test

## 📊 Empirical Benchmark Results

| # | Strategy Name | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
|---|---|---|---|---|---|---|---|
| 1 | sentence_transformer | 0.750 | 0.600 | 0.800 | 0.900 | 0.720 | Pure dense bi-encoder; diffuses rare coined terms |

> **Note:** This is a minimal validation with placeholder metrics.
> For full benchmark results, run the comprehensive validation suite.

## 🔍 Simple Retrieval Test (Minimal Mode)

### Query (a): "How does the Binding Constraint Thesis affect harness comparisons across models?"
*Ground-truth target: 2605.23950v1.pdf (Agent Harness Benchmarks & Binding Constraint Thesis)*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score | Duration |
|---|---|---|---|---|
| `sentence_transformer` | [2605.23950v1.pdf \| Page 1 \| § Introduction] | The Binding Constraint Thesis provides a theoretical framework for comparing agent harnesses across different models. This approach... | 0.950 | 125ms |

## 🤖 LLM Adapter Results

| Provider | Default Model | Route / Mode |
|---|---|---|
| **OpenAI** | `gpt-5.5` | Real API (Direct via `.env`) |
| **Anthropic** | `claude-sonnet-5` | Real API (Direct via `.env`) |
| **Gemini** | `gemini-3.8-flash` | Real API (Direct via `.env`) |

**Benchmark Query a:**
"How does the Binding Constraint Thesis affect harness comparisons across models?"
**Strategy:** sentence_transformer | **Corpus:** 1 PDFs, 50 chunks

> **Note:** LLM generation testing skipped in minimal mode.

## Validation Summary

**Total Retrieval Tests:** 1
**Successful Retrieval Tests:** 1
**Total Generation Tests:** 0
**Successful Generation Tests:** 0
