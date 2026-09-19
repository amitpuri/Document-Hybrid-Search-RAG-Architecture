# Document Hybrid Search Validation Results

**Generated:** 2026-09-17 01:11:12
**Corpus:** corpus
**Storage Backend:** parquet
**Live Mode:** True

## 📊 Empirical Benchmark Results

| # | Strategy Name | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
|---|---|---|---|---|---|---|---|
| 1 | 1. Pure BM25 (Sparse) | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 | |
| 2 | 2. Pure TF-IDF (Dense) | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 | |
| 3 | 3. Linear Hybrid (a=0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 | |
| 4 | 4. Linear Hybrid (a=0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 | |
| 5 | 5. Linear Hybrid (a=0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 | |
| 6 | 6. RRF (k=60) | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 | |
| 7 | 7. RRF + Deduplication | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 | |
| 8 | 8. RRF + Dedup + MMR | 0.625 | 0.500 | 0.786 | 0.857 | 0.683 | |
| 9 | 9. PPMI Semantic + BM25 RRF | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 | |
| 10 | 10. Cross-Encoder Re-rank | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 | |
| 11 | 11. Sentence-Transformer (MiniLM) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 | |
| 12 | 12. Adaptive Hybrid | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 | |
| 13 | 13. SPECTER2 (Scientific Bi-Encoder) | 0.112 | 0.000 | 0.143 | 0.286 | 0.142 | Pure dense SciBERT bi-encoder with dual asymmetric adapters ([PRX] & [QRY]) |
| 14 | 14. RRF + Graph + Dedup + MMR | 0.565 | 0.429 | 0.714 | 0.786 | 0.621 | Fuses IDF-weighted NetworkX KG into RRF; highest relation coverage (0.643) |
| 15 | 15. Qdrant Vector (ANN) | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 | High-speed approximate nearest neighbor (ANN) vector retrieval via Qdrant HNSW index |

## 🔍 Side-by-Side Retrieval Comparison

### Query (a): "How does the Binding Constraint Thesis affect harness comparisons?"

*Ground-truth target: 2605.23950v1.pdf*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 28.700 |
| `tfidf` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.374 |
| `linear_0.3` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.995 |
| `linear_0.5` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.992 |
| `linear_0.7` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.988 |
| `rrf` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `rrf_dedup` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `rrf_dedup_mmr` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `ppmi` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `cross_encoder` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 6.269 |
| `sentence_transformer` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.746 |
| `adaptive` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.988 |
| `specter2` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.775 |
| `rrf_graph_dedup_mmr` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.038 |
| `qdrant` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.746 |

### Query (b): "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"

*Ground-truth target: 2605.10223v1.pdf*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 23.907 |
| `tfidf` | 2605.10223v1.pdf \| Page 7 \| § 8 Conclusion | N/A | 0.337 |
| `linear_0.3` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.957 |
| `linear_0.5` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.929 |
| `linear_0.7` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.900 |
| `rrf` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `rrf_dedup` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `rrf_dedup_mmr` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `ppmi` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `cross_encoder` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 8.823 |
| `sentence_transformer` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.749 |
| `adaptive` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.957 |
| `specter2` | 2605.10223v1.pdf \| Page 7 \| § 8 Conclusion | N/A | 0.844 |
| `rrf_graph_dedup_mmr` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.037 |
| `qdrant` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.749 |


## 🤖 LLM Adapter Results

| Provider | Default Model | Route / Mode |
|---|---|---|
| **OpenAI** | `gpt-5.5` | Real API (Direct via `.env`) |
| **Anthropic** | `claude-sonnet-5` | Real API (Direct via `.env`) |
| **Gemini** | `gemini-3.8-flash` | Real API (Direct via `.env`) |
| Offline Mock | `GroundedSynthesisGenerator` | Fallback (Zero API calls) |

**Benchmark Query a:**
"How does the Binding Constraint Thesis affect harness comparisons?"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 11 PDFs, 2,072 chunks

### Openai — `gpt-5.5`

> The Binding Constraint Thesis implies that harness comparisons are not secondary implementation details; they can materially determine benchmark outcomes.
> 
> Specifically, the thesis defines harness variance as:
> 
> - **HV(M) = Var over harnesses H of B(M, H)**, i.e. how much a model’s score changes when the harness changes.
> - **MV(H) = Var over models M of B(M, H)**, i.e. how much scores differ across models under a fixed harness [Source 1].
> 
> Its central claim is that, for long-horizon LLM-agent tasks using comparable frontier models, **HV is often comparable to or larger than MV**, and may even dominate current evaluations [Source 1]. Therefore, comparing models under a single undisclosed harness can be misleading, because observed score differences may reflect harness effects rather than model capability differences [Source 1].
> 
> For harness comparisons, this has several consequences:
> 
> 1. **Harnesses must be treated as part of the experimental condition.**  
>    The paper argues that the harness used to produce a benchmark score must be disclosed, because benchmark results reported under a single hidden harness make HV unmeasurable [Source 2].
> 
> 2. **Cross-model comparisons require either a locked or factorial protocol.**  
>    A **locked-harness protocol** fixes the harness and compares models only under that specific H*, yielding a clean ranking for that harness but not necessarily a general model ranking [Source 3]. A **factorial protocol** varies harnesses as a controlled factor, allowing one to estimate harness effects directly [Source 2].
> 
> 3. **Harness optimization does not solve the attribution problem.**  
>    The source states that reporting results under one’s own optimized harness can compound attribution issues rather than resolve them; harness optimization is evidence for the thesis, not a substitute for disclosure and controlled comparison [Source 2].
> 
> 4. **Model–harness interactions matter.**  
>    The thesis denies that harness effects are uniform across models. For example, a harness emphasizing self-verification may help a model with high false-confidence rates more than a more conservative model, so the same harness change can alter model rankings differently across models [Source 3].
> 
> In short, the Binding Constraint Thesis reframes harness comparisons as central to benchmark validity: harnesses must be disclosed, fixed for clean model comparisons, or varied systematically to measure their effects. Otherwise, benchmark rankings may conflate model quality with harness-dependent performance.

*Citations: 8 chunks*

### Anthropic — `claude-sonnet-5`

> # The Binding Constraint Thesis and Harness Comparisons
> 
> ## Core Claim
> 
> The Binding Constraint Thesis addresses evaluation of LLM agents on long-horizon tasks, formalizing benchmark score as **B(M, H)**, dependent on both the model **M** and the harness **H** [Source 1]. It defines two variance quantities:
> 
> - **HV(M) = Var_H~P(H) [B(M, H)]** — the harness variance, i.e., how much a model's score varies across different harnesses
> - **MV(H) = Var_M~P(M) [B(M, H)]** — the model variance, i.e., how much scores vary across different models under a fixed harness
> 
> The thesis asserts that in this regime, **HV is often comparable to or larger than MV**, and may even dominate it in current long-horizon agent evaluations [Source 1].
> 
> ## Effect on Harness Comparisons
> 
> This has several direct consequences for how harness comparisons should be interpreted and conducted:
> 
> 1. **Undermines single-harness benchmark reporting**: Because current benchmark protocols typically report B(M, H*) for a single, often undisclosed harness H*, this renders HV unmeasurable. As a result, model comparisons based on such reports are "incomplete and potentially misleading" [Source 1] — a high score could reflect harness effects rather than genuine model capability.
> 
> 2. **Self-reported harnesses compound the problem**: When different evaluators report results "under their own harness," this compounds rather than resolves the attribution problem of separating model quality from harness quality [Source 2].
> 
> 3. **Interaction effects break simple rankings**: The thesis highlights a non-negligible interaction term between model and harness — for example, a harness emphasizing self-verification benefits a model with high false-confidence rates more than one with conservative output distributions. This means "the same harness shift produces different gains across models" [Source 3]. Treating model rankings as pure model properties requires this interaction term to be small relative to MV — a condition the thesis explicitly denies [Source 3].
> 
> 4. **Requires structural solutions**: Because the problem is structural (harness variance conflating with model variance), the thesis calls for **disclosure** as the remedy: the harness used must be treated as part of the experimental condition itself. Comparisons across models should either:
>    - Hold harness fixed via a **locked-harness protocol** (fixing H = H*, which yields a clean ranking of {B(Mi, H*)} but valid only for that specific H*), or
>    - Vary harness systematically via a **factorial protocol**, treating it as a controlled factor [Source 2, Source 3].
> 
> ## Summary
> 
> In short, the Binding Constraint Thesis implies that harness comparisons cannot be treated as neutral background conditions in model evaluation — because harness variance often rivals or exceeds model variance, and interacts with model-specific properties, valid model comparisons require either fixing the harness across all models being compared or explicitly varying it as a factor, along with full disclosure of which harness produced any reported score [Source 1, Source 2, Source 3].

*Citations: 8 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided technical literature, the **Binding Constraint Thesis** significantly impacts how harnesses and model evaluations are compared in long-horizon LLM agent benchmarks:
> 
> ### 1. Harness Variance Rivals or Exceeds Model Variance
> According to the thesis, for frontier LLM agents operating on long-horizon tasks, harness variance ($HV(M) = \text{Var}_{H \sim P(H)} [B(M, H)]$) is often comparable to, larger than, or dominates model variance ($MV(H) = \text{Var}_{M \sim P(M)} [B(M, H)]$) [Source 1, Knowledge Graph Relationships]. Because the choice of harness has such an outsized effect on the benchmark score $B(M, H)$, evaluations that fail to account for harness differences yield incomplete and potentially misleading comparisons [Source 1].
> 
> ### 2. Undisclosed Harnesses Invalidate Direct Comparisons
> Standard benchmark protocols often report scores using a single undisclosed harness ($H^*$). The thesis notes that this practice renders $HV$ unmeasurable [Source 1]. Furthermore, when entities report benchmark results under their own custom harnesses, it compounds rather than resolves the attribution problem regarding whether performance differences stem from the model or the harness [Source 2].
> 
> ### 3. Non-Negligible Interaction Effects Prevent Universal Rankings
> The thesis denies the assumption that model rankings are intrinsic model properties [Source 3]. In closed-loop settings, the interaction term between model and harness is non-negligible:
> * A harness feature (e.g., emphasizing self-verification) benefits models with high false-confidence rates differently than models with conservative output distributions [Source 3].
> * Because the same shift in harness produces differing gains across models, rankings are not stable across harnesses unless the interaction term is small relative to $MV$, a condition contradicted by the thesis [Source 3].
> 
> ### 4. Methodological Requirements for Valid Comparisons
> To address this structural issue, the thesis dictates that harness disclosure is mandatory, and comparisons must adopt structured evaluation protocols [Source 2]:
> * **Locked-Harness Protocol:** Cross-model comparisons must fix the harness ($H = H^*$). While this recovers a clean ranking of model scores $\{B(M_i, H^*)\}$, that ranking is valid *only* under that specific harness condition [Source 2, Source 3].
> * **Factorial Protocol:** The harness must be varied as a controlled factor across models to systematically measure harness effects and interactions [Source 2].

*Citations: 10 chunks*

---

**Benchmark Query b:**
"Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 11 PDFs, 2,072 chunks

### Openai — `gpt-5.5`

> The **Dynamic Tiered AgentRunner Framework** uses **Risk-Adaptive Tiering** as one of its three core mechanisms. According to the paper’s abstract, Risk-Adaptive Tiering “dynamically allocates computational budget and review intensity” based on a task’s **risk-complexity profile**, routing tasks through different execution modes: **Light**, **Standard**, and **Full** [Source 3].
> 
> The motivation is enterprise governability: the authors argue that the main bottleneck for LLM agents in enterprise settings is not autonomy, but the lack of controls over high-risk writes, multi-step verification, and cost efficiency [Source 1]. Risk-Adaptive Tiering addresses this by varying how much oversight and computation a task receives, aiming for **Pareto-optimal safety-efficiency trade-offs** [Source 3].
> 
> The evaluation context shows that tasks are distributed across risk/complexity categories, including:
> 
> - **Information queries**: 40.2%  
> - **Single-object writes**: 29.8%  
> - **Multiobject/batch operations**: 19.7%  
> - **Cross-domain complex tasks**: 10.2% [Source 2]
> 
> The framework is compared against baselines such as a single-agent system with no governance, an always-on full pipeline, and ablations removing CriticAgent, VerifierAgent, or RecoveryAgent [Source 2]. The metrics used include **Task Success Rate (SR)**, **Risk Execution Error Rate (RERR)** for unreviewed high-risk operations, latency, inference cost, and recovery success [Source 2].
> 
> In short, **Risk-Adaptive Tiering** is the AgentRunner mechanism that decides how much governance and computational effort to apply to a task, escalating from lighter handling for lower-risk tasks to fuller review and verification for higher-risk or more complex operations [Source 3]. The provided knowledge graph also indicates that **AgentRunner** co-occurs with **SR/Main Results**, suggesting that task success rate is part of how the framework’s effectiveness is assessed, though the supplied passages do not include the actual numerical main results.

*Citations: 7 chunks*

### Anthropic — `claude-sonnet-5`

> # Dynamic Tiered AgentRunner Framework: Risk-Adaptive Tiering
> 
> **Risk-Adaptive Tiering** is one of three core mechanisms operationalized by the Dynamic Tiered AgentRunner framework, a controlled execution protocol distilled from a production multi-tenant SaaS platform [Source 1, Source 3].
> 
> ## Core Definition
> 
> Risk-Adaptive Tiering **dynamically allocates computational budget and review intensity across three execution modes—Light, Standard, and Full—based on a task's risk-complexity profile**, achieving Pareto-optimal safety-efficiency trade-offs [Source 3, § Abstract].
> 
> ## Motivation
> 
> This mechanism directly addresses the paper's central thesis: that in enterprise environments, the critical bottleneck for LLM-based agents is **insufficient governability rather than insufficient autonomy**. Specifically, the framework targets three failure modes common in ungoverned autonomous systems:
> - High-risk write operations proceeding without independent review
> - Complex multi-step tasks lacking verification mechanisms
> - Indiscriminate computational expenditure that makes deployment economically unviable [Source 1, § Abstract]
> 
> By tiering execution based on risk and complexity, the framework avoids applying uniform (and often wasteful) governance overhead to every task, instead matching the intensity of review and compute to the actual risk profile of each task.
> 
> ## Relationship to Other Mechanisms
> 
> Risk-Adaptive Tiering operates alongside two other core mechanisms in the framework:
> 1. **Separation of Powers** — physically isolating proposal (Worker), review (Critic), execution (ToolGateway), and verification (Verifier) into independent, non-colluding processes, so no single agent can both propose and approve an action.
> 2. **Verifier-Recovery closed loop** — treating failure as a first-class execution state to enable self-healing and organizational learning [Source 3, § Abstract].
> 
> The knowledge graph indicates that **AgentRunner co-occurs with ToolGateway and with tier routing concepts**, reflecting how the tiering mechanism is architecturally linked to the execution layer (ToolGateway) that carries out tasks once tiering decisions are made.
> 
> ## Evaluation Context
> 
> The framework's tiering approach is empirically evaluated against several baselines, including a "Static-Full" baseline (always-on full pipeline for every task) and ablations like "No-Critic," "No-Verifier," and "No-Recovery" that remove individual governance components from the dynamic tiering system. Evaluation metrics include Task Success Rate, Risk Execution Error Rate (unreviewed high-risk operations), Average Latency, Average Inference Cost, and Recovery Success Rate — metrics designed to capture the safety-efficiency trade-offs that Risk-Adaptive Tiering aims to optimize [Source 2, § 5.3].
> 
> **Gap in provided context:** The passages do not specify the precise decision criteria or thresholds used to route tasks into Light, Standard, or Full tiers, nor detailed quantitative results comparing tiering performance against the baselines. A "Main Results Table" is referenced in the knowledge graph as co-occurring with AgentRunner, which likely contains this comparative data, but its contents are not included in the provided passages.

*Citations: 5 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided literature, **Risk-Adaptive Tiering** is one of the three core mechanisms of the **Dynamic Tiered AgentRunner Framework** [Source 3]. 
> 
> ### Definition and Functionality
> * **Dynamic Resource Allocation:** Risk-Adaptive Tiering dynamically allocates computational budget and review intensity according to a given task’s risk-complexity profile [Source 3].
> * **Execution Modes:** It routes tasks across three tiered execution modes:
>   * **Light**
>   * **Standard**
>   * **Full** [Source 3]
> * **Pareto-Optimal Balance:** The primary objective of this tiering mechanism is to achieve Pareto-optimal safety-efficiency trade-offs by preventing indiscriminate computational expenditure while ensuring high-risk operations receive sufficient review [Source 1, Source 3].
> 
> ### Context and Motivation
> * **Problem Addressed:** In enterprise LLM deployments, critical bottlenecks arise because high-risk write operations can proceed without independent review, multi-step tasks lack verification, and uniform computational expenditure across all tasks renders deployment economically unviable [Source 1].
> * **Task Distribution Profile:** The framework evaluates tasks categorized across varying risk and complexity levels, including information queries (40.2%), single-object writes (29.8%), multi-object/batch operations (19.7%), and cross-domain complex tasks (10.2%) [Source 2].
> * **Integration within the Framework:** Dynamic tiering interacts with an architecture based on separation of powers (isolating proposal by Worker, review by Critic, execution via `ToolGateway`, and verification by Verifier) and a Verifier-Recovery closed loop [Source 3, Knowledge Graph Relationships]. When evaluating dynamic tiering, ablation studies compare it against baselines such as an always-on full pipeline (`Static-Full`) or single-agent un-governed baselines across metrics including Task Success Rate (SR), Risk Execution Error Rate (RERR), latency, and inference costs [Source 2].

*Citations: 8 chunks*

---

## Validation Summary

**Total Retrieval Tests:** 30
**Successful Retrieval Tests:** 30
**Total Generation Tests:** 6
**Successful Generation Tests:** 6


# Document Hybrid Search Validation Results

**Generated:** 2026-09-18 02:18:48
**Corpus:** corpus
**Storage Backend:** parquet
**Live Mode:** True

## 📊 Empirical Benchmark Results

| # | Strategy Name | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
|---|---|---|---|---|---|---|---|
| 1 | 1. Pure BM25 (Sparse) | 0.574 | 0.409 | 0.727 | 0.773 | 0.606 | |
| 2 | 2. Pure TF-IDF (Dense) | 0.344 | 0.136 | 0.455 | 0.591 | 0.376 | |
| 3 | 3. Linear Hybrid (a=0.3) | 0.503 | 0.273 | 0.682 | 0.773 | 0.556 | |
| 4 | 4. Linear Hybrid (a=0.5) | 0.526 | 0.318 | 0.773 | 0.773 | 0.575 | |
| 5 | 5. Linear Hybrid (a=0.7) | 0.456 | 0.227 | 0.727 | 0.773 | 0.525 | |
| 6 | 6. RRF (k=60) | 0.552 | 0.364 | 0.727 | 0.773 | 0.595 | |
| 7 | 7. RRF + Deduplication | 0.530 | 0.364 | 0.682 | 0.727 | 0.572 | |
| 8 | 8. RRF + Dedup + MMR | 0.530 | 0.364 | 0.727 | 0.727 | 0.581 | |
| 9 | 9. PPMI Semantic + BM25 RRF | 0.388 | 0.227 | 0.409 | 0.545 | 0.393 | |
| 10 | 10. Cross-Encoder Re-rank | 0.481 | 0.318 | 0.591 | 0.773 | 0.549 | |
| 11 | 11. Sentence-Transformer (MiniLM) | 0.317 | 0.182 | 0.364 | 0.500 | 0.335 | |
| 12 | 12. Adaptive Hybrid | 0.492 | 0.273 | 0.682 | 0.727 | 0.533 | |
| 13 | 13. SPECTER2 (Scientific Bi-Encoder) | 0.279 | 0.136 | 0.364 | 0.500 | 0.327 | |
| 14 | 14. RRF + Graph + Dedup + MMR | 0.520 | 0.364 | 0.682 | 0.773 | 0.584 | |

## 🔍 Side-by-Side Retrieval Comparison

### Query (a): "How does the Binding Constraint Thesis affect harness comparisons across models?"

*Ground-truth target: 2605.23950v1.pdf (Agent Harness Benchmarks & Binding Constraint Thesis)*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 38.997 |
| `tfidf` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.404 |
| `linear_0.3` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.989 |
| `linear_0.5` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.992 |
| `linear_0.7` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.995 |
| `rrf` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `rrf_dedup` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `rrf_dedup_mmr` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `ppmi` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.033 |
| `cross_encoder` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 6.441 |
| `sentence_transformer` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.766 |
| `adaptive` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.995 |
| `specter2` | 2605.23950v1.pdf \| Page 9 \| § 7 Conclusion and Discussion | N/A | 0.783 |
| `rrf_graph_dedup_mmr` | 2605.23950v1.pdf \| Page 4 \| § 3 The Binding Constraint Thesis | N/A | 0.038 |

### Query (b): "Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"

*Ground-truth target: 2605.10223v1.pdf (Enterprise Agent Governance & Risk Adaptive Tiering)*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 28.665 |
| `tfidf` | 2605.10223v1.pdf \| Page 7 \| § 8 Conclusion | N/A | 0.352 |
| `linear_0.3` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.956 |
| `linear_0.5` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.927 |
| `linear_0.7` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.897 |
| `rrf` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `rrf_dedup` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `rrf_dedup_mmr` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.032 |
| `ppmi` | ERROR | ERROR | 0.000 |
| `cross_encoder` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 8.823 |
| `sentence_transformer` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.749 |
| `adaptive` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.956 |
| `specter2` | 2605.10223v1.pdf \| Page 7 \| § 8 Conclusion | N/A | 0.844 |
| `rrf_graph_dedup_mmr` | 2605.10223v1.pdf \| Page 1 \| § Abstract | N/A | 0.037 |

### Query (c): "AlphaGenome regulatory variant effect prediction non-coding DNA"

*Ground-truth target: s41586-025-10014-0.pdf (DeepMind AlphaGenome & Regulatory Variant Effect Prediction)*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 38.027 |
| `tfidf` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 0.404 |
| `linear_0.3` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 1.000 |
| `linear_0.5` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 1.000 |
| `linear_0.7` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 1.000 |
| `rrf` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 0.033 |
| `rrf_dedup` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 0.033 |
| `rrf_dedup_mmr` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 0.033 |
| `ppmi` | s41586-025-10014-0.pdf \| Page 2 \| § Overview | N/A | 0.032 |
| `cross_encoder` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 8.392 |
| `sentence_transformer` | s41586-025-10014-0.pdf \| Page 2 \| § Overview | N/A | 0.731 |
| `adaptive` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 1.000 |
| `specter2` | s41586-025-10014-0.pdf \| Page 1 \| § Overview | N/A | 0.862 |
| `rrf_graph_dedup_mmr` | s41586-025-10014-0.pdf \| Page 12 \| § Discussion | N/A | 0.033 |

### Query (d): "Scalable watermarking for identifying large language model outputs SynthID"

*Ground-truth target: s41586-024-08025-4.pdf (LLM Output Provenance & Scalable Watermarking (SynthID))*

| Strategy | Top-1 Source (doc \| page \| § section) | Snippet (~100 chars) | Score |
|---|---|---|---|
| `bm25` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 30.424 |
| `tfidf` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.221 |
| `linear_0.3` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 0.956 |
| `linear_0.5` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 0.926 |
| `linear_0.7` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.906 |
| `rrf` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.032 |
| `rrf_dedup` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.032 |
| `rrf_dedup_mmr` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.032 |
| `ppmi` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 0.033 |
| `cross_encoder` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 6.831 |
| `sentence_transformer` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.695 |
| `adaptive` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 0.956 |
| `specter2` | s41586-024-08025-4.pdf \| Page 1 \| § Overview | N/A | 0.833 |
| `rrf_graph_dedup_mmr` | s41586-024-08025-4.pdf \| Page 6 \| § Discussion | N/A | 0.032 |

## 🤖 LLM Adapter Results

| Provider | Default Model | Route / Mode |
|---|---|---|
| **OpenAI** | `gpt-5.5` | Real API (Direct via `.env`) |
| **Anthropic** | `claude-sonnet-5` | Real API (Direct via `.env`) |
| **Gemini** | `gemini-3.8-flash` | Real API (Direct via `.env`) |
| Offline Mock | `GroundedSynthesisGenerator` | Fallback (Zero API calls) |

**Benchmark Query a:**
"How does the Binding Constraint Thesis affect harness comparisons across models?"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 33 PDFs, 7,538 chunks

### Openai — `gpt-5.5`

> The Binding Constraint Thesis says that, for LLM agents on long-horizon tasks, benchmark performance cannot be treated as a property of the model alone; it is jointly determined by the model \(M\) and the harness \(H\). Formally, the paper defines \(B(M,H)\) as the score of model \(M\) under harness \(H\), with harness variance \(HV(M)=Var_{H \sim P(H)}[B(M,H)]\) and model variance \(MV(H)=Var_{M \sim P(M)}[B(M,H)]\) [Source 1]. The thesis asserts that \(HV\) is often comparable to, or larger than, \(MV\), meaning that changing the harness can affect scores as much as—or more than—changing the model [Source 1].
> 
> This affects harness comparisons across models in two main ways:
> 
> 1. **Model rankings under one harness may not generalize.**  
>    A “locked-harness” evaluation fixes \(H=H^*\) and can rank models by \(B(M_i,H^*)\), but that ranking is only valid for that particular harness [Source 2]. Since current protocols often report scores under a single undisclosed harness, they make \(HV\) unmeasurable and can lead to incomplete or misleading model comparisons [Source 1].
> 
> 2. **Harness effects can interact with model properties.**  
>    The thesis denies that harness changes affect all models uniformly. For example, a harness emphasizing self-verification may help a model with high false-confidence rates more than a model with conservative output distributions [Source 2]. Thus, the interaction between model and harness is “non-negligible,” and treating rankings as intrinsic model properties would require that interaction to be small relative to \(MV\), which the thesis says is often not the case [Source 2].
> 
> Therefore, the thesis implies that fair cross-model comparison requires either holding the harness fixed and disclosed, or systematically varying the harness as a controlled factor in a factorial protocol [Source 3]. The knowledge graph also reflects that \(HV\), \(MV\), and LLMs all co-occur with the Binding Constraint Thesis, reinforcing that the thesis is specifically about the relationship between harness-induced variance, model-induced variance, and LLM agent evaluation.

*Citations: 7 chunks*

### Anthropic — `claude-sonnet-5`

> The Binding Constraint Thesis, as presented in [Source 1], formalizes how model performance on long-horizon agent tasks depends on both the model (M) and the harness (H) used to evaluate it. It defines two variance terms: HV(M), the harness variance for a given model (variance in benchmark score B(M, H) across harnesses), and MV(H), the model variance for a given harness (variance in B(M, H) across models). The thesis asserts that in current long-horizon agent evaluations, HV is often comparable to or larger than MV, meaning the choice of harness can influence benchmark scores as much as, or more than, the choice of model itself [Source 1].
> 
> This has significant implications for harness comparisons across models:
> 
> 1. **Undermines single-harness rankings**: Since most current benchmark protocols report B(M, H*) for a single, often undisclosed harness H*, this renders HV unmeasurable. As a result, model comparisons based on these single-harness scores are "incomplete and potentially misleading" [Source 1], because the observed ranking may reflect harness-specific effects rather than genuine model capability differences.
> 
> 2. **Interaction effects distort attribution**: [Source 2] notes that the interaction between harness and model is non-negligible — for example, a harness emphasizing self-verification may help a model with high false-confidence rates more than a model with conservative output distributions. This means the same harness change can produce different performance gains across models, so treating model rankings as pure model properties requires the interaction term to be small relative to MV — a condition the thesis explicitly denies.
> 
> 3. **Locked-harness protocols provide only partial solutions**: A locked-harness protocol (fixing H = H*) can recover a "clean ranking" of models under that specific harness, but this ranking is only valid for that particular H* and does not generalize across harnesses [Source 2].
> 
> 4. **Self-reported harnesses compound the problem**: When different evaluations report results under their own custom harnesses, this "compounds rather than resolves" the attribution problem, since cross-model comparisons become confounded by differing evaluation conditions [Source 3].
> 
> 5. **Calls for structural disclosure solutions**: Because the problem is structural (arising from the harness-model interaction and unmeasured HV), the thesis argues for a structural fix: harness disclosure. The harness used must be treated as part of the experimental condition, and cross-model comparisons should either hold the harness fixed (locked-harness protocol) or vary it as a controlled factor (factorial protocol) [Source 3].
> 
> In summary, the Binding Constraint Thesis implies that harness comparisons across models cannot be treated as neutral or model-only phenomena — because HV can rival or exceed MV, meaningful cross-model comparisons require explicit control or disclosure of the harness, rather than relying on scores obtained under a single, often opaque, evaluation harness.

*Citations: 7 chunks*

### Gemini — `gemini-3.8-flash`

> Based on § 3 (The Binding Constraint Thesis) of **2605.23950v1.pdf**, the Binding Constraint Thesis affects harness and model comparisons in the following ways:
> 
> ### 1. Harness Variance Often Dominates Model Variance
> * For LLM agents operating on long-horizon tasks, the thesis defines Harness Variance as $\text{HV}(M) = \text{Var}_{H \sim P(H)}[B(M, H)]$ and Model Variance as $\text{MV}(H) = \text{Var}_{M \sim P(M)}[B(M, H)]$ [Source 1]. 
> * The thesis asserts that $\text{HV}$ is often comparable to, larger than, or dominates $\text{MV}$ in current long-horizon agent evaluations [Source 1].
> * Because standard benchmark protocols often report scores for a single undisclosed harness ($H^*$), $\text{HV}$ remains unmeasurable, rendering comparisons between models incomplete and potentially misleading [Source 1].
> 
> ### 2. Model Rankings Cannot Be Treated as Pure Model Properties
> * The thesis denies the assumption that model rankings are intrinsic model properties, which would require the interaction term between the harness and the model to be small relative to $\text{MV}$ [Source 2].
> * Instead, the interaction term is non-negligible: shifting a harness produces different gains across different models (for example, a harness emphasizing self-verification aids a model with high false-confidence rates more than one with conservative output distributions) [Source 2].
> 
> ### 3. Requirements for Valid Cross-Model Comparison Protocols
> Because reporting results under unstandardized or self-selected harnesses compounds attribution problems rather than resolving them [Source 3], the thesis establishes methodological requirements for comparing models:
> * **Harness Disclosure:** The harness used to generate benchmark scores must be explicitly disclosed as part of the experimental condition [Source 3].
> * **Locked-Harness Protocols:** Fixing a harness ($H = H^*$) allows for a clean ranking of models $\{B(M_i, H^*)\}$, but this ranking is valid *only* under that specific harness $H^*$ [Source 2, Source 3].
> * **Factorial Protocols:** Cross-model comparisons must either hold the harness fixed via a locked-harness protocol or vary the harness as a controlled factor via a factorial protocol [Source 3].

*Citations: 9 chunks*

---

**Benchmark Query b:**
"Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 33 PDFs, 7,538 chunks

### Openai — `gpt-5.5`

> Dynamic Tiered AgentRunner is presented as an enterprise AI execution framework focused on **governability rather than maximum autonomy**. The paper argues that enterprise LLM agents face risks such as unreviewed high-risk write operations, lack of verification for complex multi-step tasks, and economically inefficient compute usage [Source 1].
> 
> Its **Risk-Adaptive Tiering** mechanism dynamically assigns each task to one of three execution modes—**Light, Standard, or Full**—based on the task’s **risk–complexity profile** [Source 3]. The purpose is to balance safety and efficiency by allocating more review intensity and computational budget only when tasks are higher risk or more complex, rather than applying the most expensive governance pipeline to every task [Source 3].
> 
> The framework combines this tiering with two other governance mechanisms:
> 
> 1. **Separation of Powers**: proposal, review, execution, and verification are isolated into independent processes: Worker, Critic, ToolGateway, and Verifier. This prevents a single agent from both proposing and approving an action [Source 3].  
> 2. **Verifier–Recovery Closed Loop**: failures are treated as normal execution states, enabling self-healing and organizational learning through retrospection [Source 3].
> 
> The evaluation context includes task categories such as information queries, single-object writes, multi-object or batch operations, and cross-domain complex tasks [Source 2]. The reported evaluation compares Dynamic Tiered AgentRunner against baselines including Single-Agent, Static-Full, No-Critic, No-Verifier, and No-Recovery variants, using metrics such as task success rate, risk execution error rate, latency, inference cost, and recovery success rate [Source 2].
> 
> From the knowledge graph, **AgentRunner** is connected with **AI**, **ToolGateway**, and a **Main Results Table**, indicating that ToolGateway is a key architectural entity associated with AgentRunner, and that empirical results likely appear in a main results table. However, the provided passages do not include the actual numerical results from that table, so specific performance claims cannot be stated from the available context.

*Citations: 7 chunks*

### Anthropic — `claude-sonnet-5`

> # Dynamic Tiered AgentRunner Framework: Risk-Adaptive Tiering
> 
> ## Overview
> 
> The **Dynamic Tiered AgentRunner** is a controlled execution protocol for LLM-based agents, "distilled from a production multi-tenant SaaS platform" [Source 1]. It was developed in response to a core problem identified in enterprise AI deployment: "The prevailing paradigm in LLM-based agent research pursues ever-greater autonomy. Yet in enterprise environments, the critical bottleneck is not insufficient autonomy but insufficient governability" [Source 1]. Specific governance failures motivating the framework include high-risk write operations proceeding without independent review, complex multi-step tasks lacking verification mechanisms, and indiscriminate computational expenditure that renders deployment economically unviable [Source 1].
> 
> ## Risk-Adaptive Tiering Mechanism
> 
> Risk-Adaptive Tiering is one of three core mechanisms operationalized by the framework [Source 3]. It works by "dynamically allocat[ing] computational budget and review intensity across Light, Standard, and Full execution modes based on a task's risk-complexity profile" [Source 3]. The stated goal of this mechanism is to achieve "Pareto-optimal safety-efficiency trade-offs" [Source 3]—meaning the system attempts to balance safety/governance requirements against computational and latency costs, rather than applying uniform (and often wasteful or under-protective) scrutiny to all tasks.
> 
> ## Relationship to Other Core Mechanisms
> 
> Risk-Adaptive Tiering operates alongside two other mechanisms described in the framework [Source 3]:
> 
> 1. **Separation of Powers**: physically isolates proposal (Worker), review (Critic), execution (ToolGateway), and verification (Verifier) into independent, non-colluding processes, so "no single agent can simultaneously propose and approve an action" [Source 3].
> 2. **Verifier–Recovery closed loop**: treats failure as "a first-class execution state," enabling systematic self-healing and organizational learning through retrospection [Source 3].
> 
> ## Empirical Evaluation
> 
> The framework's governance evidence is evaluated against several baselines, including a "Single-Agent" system with tool access but no governance, a "Static-Full" baseline that always runs the full pipeline, and ablations that remove specific agents (No-Critic, No-Verifier, No-Recovery) from the dynamic tiering system [Source 2]. Evaluation uses a task distribution spanning information queries (40.2%), single-object writes (29.8%), multi-object/batch operations (19.7%), and cross-domain complex tasks (10.2%) [Source 2]. Key metrics include Task Success Rate (SR), Risk Execution Error Rate (RERR, measuring unreviewed high-risk operations), Average Latency, Average Inference Cost, and Recovery Success Rate (RSR) [Source 2]—metrics designed to jointly capture both the safety/governance and efficiency dimensions that Risk-Adaptive Tiering aims to optimize.
> 
> ## What the Context Does Not Cover
> 
> The provided passages do not include the specific criteria or thresholds used to determine which risk-complexity profile routes a task to Light, Standard, or Full execution mode, nor do they provide the quantitative results (e.g., actual SR, RERR, latency, or cost figures) from the comparison against the baselines. Additional sections of the paper (beyond what is excerpted here) would likely be needed to answer questions about those specifics.

*Citations: 12 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided literature, the **Risk-Adaptive Tiering** mechanism within the **Dynamic Tiered AgentRunner** framework is defined as follows:
> 
> ### 1. Purpose and Background
> In enterprise LLM environments, a primary operational bottleneck is the lack of governability—specifically indiscriminate computational expenditure and high-risk write operations occurring without independent review [Source 1]. Risk-Adaptive Tiering was developed as part of a controlled execution protocol (distilled from a production multi-tenant SaaS platform) to achieve Pareto-optimal safety-efficiency trade-offs [Source 1, Source 3].
> 
> ### 2. Core Functionality and Execution Modes
> Rather than using an indiscriminate, always-on full pipeline for every task (a baseline referred to as *Static-Full* [Source 2]), Risk-Adaptive Tiering dynamically allocates computational budget and review intensity according to a task's **risk-complexity profile** [Source 3]. 
> 
> It routes tasks across three dynamic execution modes:
> * **Light**
> * **Standard**
> * **Full** [Source 3] 
> 
> *(Knowledge graph relations also tie `AgentRunner` to `Tier Routing The`)*.
> 
> ### 3. Role in the Broader Framework and Metrics
> * **Interplay with Governance & Tool Execution**: The tiering functions alongside the framework's "Separation of Powers" model, which physically isolates proposal (*Worker*), review (*Critic*), execution (*ToolGateway*), and verification (*Verifier*) [Source 3]. *(Knowledge graph: `AgentRunner` $\rightarrow$ `ToolGateway`)*.
> * **Evaluation Profiles**: Task evaluation profiles span information queries (40.2%), single-object writes (29.8%), multi-object/batch operations (19.7%), and cross-domain complex operations (10.2%) [Source 2].
> * **Key Performance Metrics**: The impact of tier routing and dynamic governance is evaluated using metrics including:
>   * **Risk Execution Error Rate (RERR)**: Tracking unreviewed high-risk operations *(Knowledge graph: `AgentRunner` $\rightarrow$ `RERR`)* [Source 2].
>   * **Task Success Rate (SR)** [Source 2].
>   * **Average Latency** and **Average Inference Cost** [Source 2].
>   * **Recovery Success Rate (RSR)** [Source 2].

*Citations: 11 chunks*

---

**Benchmark Query c:**
"AlphaGenome regulatory variant effect prediction non-coding DNA"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 33 PDFs, 7,538 chunks

### Openai — `gpt-5.5`

> AlphaGenome is presented as a unified deep learning model for interpreting regulatory variation in non-coding DNA. It predicts diverse functional genomic signals directly from megabase-scale DNA sequence and uses these predictions to score variant effects across multiple modalities in a single inference pass [Source 1].
> 
> Key points from the provided context:
> 
> - **Purpose:** AlphaGenome aims to decipher the genome’s regulatory code, especially for interpreting **non-coding variation** mechanistically [Source 1].
> - **Variant effect prediction:** It matches or surpasses specialized state-of-the-art models for regulatory variant effect prediction, suggesting it has learned broadly useful DNA regulatory principles [Source 1].
> - **Multimodal scoring:** A core strength is that it can efficiently predict variant impacts across all predicted functional modalities at once, rather than requiring separate models or runs for each data type [Source 1].
> - **Complex regulatory mechanisms:** This multimodal capability is described as important for variants with complex mechanisms, illustrated by its recapitulation of oncogenic **TAL1** variant effects [Source 1].
> - **Benchmark performance:** In the reported evaluation, AlphaGenome achieved state-of-the-art performance on **25 of 26 variant effect prediction tasks** and **22 of 24 genome track prediction tasks** [Source 2].
> - **Splicing-related effects:** Its predictions also include splice site usage and a new splice junction prediction approach, expanding variant interpretation beyond regulatory tracks alone [Source 2].
> - **Future directions:** The authors note that further improvements may come from task-specific calibration, fine-tuning on perturbational datasets, integration of single-cell data, adding modalities such as DNA methylation and RNA structural features, and expanding multi-species capabilities [Source 3].
> 
> In short, based on the provided sources, AlphaGenome is designed as a large-scale sequence-to-function model for predicting how non-coding DNA variants affect regulatory and related genomic signals, with strong benchmark performance and particular emphasis on multimodal mechanistic interpretation.

*Citations: 8 chunks*

### Anthropic — `claude-sonnet-5`

> Based on the provided context, here's what is known about AlphaGenome's approach to regulatory variant effect prediction in non-coding DNA:
> 
> ## Core Capability
> 
> AlphaGenome is described as "a unified deep learning model that simultaneously predicts diverse functional genomic signals from megabase-scale DNA sequences" [Source 1]. Its primary contribution to understanding non-coding DNA is that it "matches or surpasses specialized SOTA models in regulatory variant effect prediction, underscoring the model's relatively robust grasp of fundamental DNA regulatory principles and its value for mechanistically interpretating non-coding variation" [Source 1].
> 
> ## Key Mechanism: Multimodal Variant Scoring
> 
> A core strength highlighted is AlphaGenome's "efficient multimodal variant effect prediction, which simultaneously scores variant impacts across all predicted modalities in a single inference pass" [Source 1]. This integrated, single-pass approach is described as "crucial for understanding variants with complex mechanisms," with the example given being "the recapitulation of oncogenic TAL1 variant effects" [Source 1]. The authors suggest this capability "could power large-scale analyses that dissect regulatory sequence elements genome-wide" [Source 1].
> 
> ## Performance Benchmarks
> 
> AlphaGenome was evaluated on a "comprehensive set of benchmarks, covering both its ability to accurately predict genome tracks on previously unseen DNA sequences and its effectiveness in variant effect prediction tasks." The model "achieved SOTA performance on 22 of 24 genome track prediction tasks and 25 of 26 variant effect prediction tasks" [Source 2]. Additionally, extensive ablation studies were performed covering "target resolution, sequence length, distillation and modality combinations" to explain performance and guide future model design [Source 2].
> 
> ## Splicing-Specific Predictions
> 
> Notably, the model's splicing predictions include "a new splice junction prediction approach alongside splice site usage prediction" [Source 2], which represents a specific advancement relevant to interpreting non-coding variants that affect splicing.
> 
> ## Future Directions Noted
> 
> The paper identifies areas for improvement in variant effect prediction, including "task-specific calibration, fine-tuning on perturbational datasets or integration of single-cell data," as well as "incorporating a broader range of data modalities (such as DNA methylation and RNA structural features)" and "leveraging DNA language models" [Source 3].
> 
> ## What Is Not Covered in This Context
> 
> The provided passages do not detail the specific architecture of AlphaGenome, the exact list of the 24 genome track prediction tasks or 26 variant effect prediction tasks, quantitative benchmark comparisons against named competing models, or the technical details of how the TAL1 variant recapitulation was performed. Additional source material would be needed to address these specifics.

*Citations: 9 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided context, **AlphaGenome** is a unified deep learning model designed to decipher the regulatory code of the genome and predict functional genomic signals from megabase-scale DNA sequences, with notable applications in regulatory variant effect prediction and interpreting non-coding DNA:
> 
> ### 1. State-of-the-Art Performance
> * AlphaGenome matches or surpasses specialized state-of-the-art (SOTA) models in regulatory variant effect prediction, demonstrating a robust grasp of fundamental DNA regulatory principles [Source 1].
> * Across comprehensive benchmarks, AlphaGenome achieved SOTA performance on **25 of 26 variant effect prediction tasks** (as well as 22 of 24 genome track prediction tasks) on previously unseen DNA sequences [Source 2].
> 
> ### 2. Mechanistic Interpretation of Non-Coding Variation
> * The model is particularly valuable for mechanistically interpreting non-coding variation [Source 1].
> * A major strength is its **efficient multimodal variant effect prediction**, which computes and scores variant impacts across all predicted modalities simultaneously in a single inference pass [Source 1].
> * This integrated approach enables the dissection of regulatory sequence elements genome-wide and the understanding of variants with complex mechanisms, such as recapitulating the effects of oncogenic *TAL1* variants [Source 1].
> 
> ### 3. Future Directions for Non-Coding Regulatory Variant Prediction
> * **Data-driven improvements:** Perturbing non-coding regulatory elements at scale and assaying greater species diversity are highlighted as key steps to support the next generation of variant effect prediction models [Source 3].
> * **Computational avenues:** Ongoing refinements for variant effect prediction include task-specific calibration, fine-tuning on perturbational datasets, integrating single-cell data, incorporating additional modalities (such as DNA methylation and RNA structural features), and leveraging DNA language models [Source 3].

*Citations: 7 chunks*

---

**Benchmark Query d:**
"Scalable watermarking for identifying large language model outputs SynthID"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 33 PDFs, 7,538 chunks

### Openai — `gpt-5.5`

> The provided passages identify a Nature 2024 article titled **“Scalable watermarking for identifying large language model outputs”** by Dathathri et al. [Source 2]. The paper is motivated by the fact that **large language models can generate high-quality synthetic text that is often indistinguishable from human-written content**, at a scale that may significantly affect the information ecosystem [Source 2].
> 
> The article situates itself within prior work on **text watermarking** and **watermarking large language models**, citing earlier methods such as Aaronson & Kirchner’s watermarking proposal and Kirchenbauer et al.’s ICML 2023 watermark for LLMs [Source 1]. It also cites the **SynthID-Team** blog post, **“Watermarking AI-generated text and video with SynthID”**, from Google DeepMind in 2024 [Source 1]. However, the provided excerpts do **not** describe the internal technical mechanism of SynthID itself.
> 
> A key practical contribution mentioned in the paper is addressing watermarking in production generation systems that use **speculative sampling**, where a smaller draft model proposes tokens and a larger target model accepts or rejects them [Source 3]. The authors state that combining **generative watermarking** with speculative sampling is important for practical deployment and, to their knowledge, had not previously been investigated; they therefore propose **two algorithms** for generative watermarking with speculative sampling [Source 3].
> 
> In short: the paper concerns scalable watermarking for detecting LLM-generated text, cites SynthID as related Google DeepMind watermarking work, and emphasizes making watermarking compatible with production techniques such as speculative sampling. The excerpts do not provide enough detail to explain SynthID’s exact algorithm.

*Citations: 6 chunks*

### Anthropic — `claude-sonnet-5`

> Based on the provided context, here is what can be established about scalable watermarking for identifying large language model outputs and SynthID:
> 
> ## Overview of the Paper
> 
> The core source addressing this topic is titled "Scalable watermarking for identifying large language model outputs," published in *Nature* (Vol. 634, 24 October 2024, p. 818) [Source 2]. The paper is authored by a large team including Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, and others, affiliated with what appears to be Google DeepMind based on author overlap with the SynthID reference [Source 2].
> 
> The paper's central motivation is that large language models (LLMs) "have enabled the generation of high-quality synthetic text, often indistinguishable from human-written content, at a scale that can markedly affect the nature of the information ecosystem" [Source 2]. This underscores the need for watermarking techniques to identify LLM-generated outputs.
> 
> ## Connection to SynthID
> 
> SynthID is referenced in the paper's discussion/related work section as a related watermarking effort: "SynthID-Team Watermarking AI-generated text and video with SynthID. Google DeepMind Blog" (2024) [Source 1]. This citation appears alongside other watermarking-related references, including:
> - A review of text watermarking theory, methods, and applications [Source 1]
> - Watermarking pre-trained language models with backdooring (Gu et al., 2022) [Source 1]
> - Analysis and evaluation of language model watermarks (Piet et al., 2023) [Source 1]
> - Watermarking of large language models (Aaronson & Kirchner, 2022) [Source 1]
> - A foundational watermark for LLMs approach (Kirchenbauer et al., 2023) [Source 1]
> 
> This suggests SynthID is Google DeepMind's related/complementary watermarking system for AI-generated text and video, cited as part of the broader landscape of watermarking approaches this paper builds upon or relates to.
> 
> ## Technical Contribution: Watermarking with Speculative Sampling
> 
> A key technical contribution described in the paper involves combining watermarking with **speculative sampling**, a method used in production systems to accelerate text generation. The paper explains that speculative sampling "works by having a smaller draft model propose the next few tokens; these are then checked against the large target model, which either accepts or rejects the proposed tokens" [Source 3].
> 
> The authors note that "combining generative watermarking with speculative sampling is an important step to make watermarking practically useful to production systems; however, to our knowledge, it has not yet been investigated" [Source 3]. To address this gap, they "propose two generative watermarking with speculative sampling algorithms" detailed in the Supplementary Information [Source 3].
> 
> ## Gaps in Available Information
> 
> The provided context does not include details on:
> - The specific watermarking algorithm/methodology at the core of this paper (beyond its integration with speculative sampling)
> - Direct technical comparisons between this paper's approach and SynthID specifically
> - Experimental results or scalability claims validating the "scalable" aspect referenced in the title
> - The precise relationship (e.g., whether this paper *describes* SynthID itself, or is a separate but related system)
> 
> If you need details on these aspects, additional source passages covering the paper's methods, results, or a more detailed description of SynthID's technical mechanism would be required.

*Citations: 12 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided sources, here is the information connecting **SynthID** and the paper **"Scalable watermarking for identifying large language model outputs"**:
> 
> ### Overview & Authorship
> * **Publication**: The paper titled *"Scalable watermarking for identifying large language model outputs"* was published in *Nature* (Vol 634, 24 October 2024) by Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Demis Hassabis, Pushmeet Kohli, and colleagues [Source 2].
> * **Connection to SynthID**: In the paper's references, **SynthID** is explicitly cited as:
>   > *SynthID-Team. "Watermarking AI-generated text and video with SynthID." Google DeepMind Blog (2024)* [Source 1].
> 
> ### Key Focus of the Work
> * **Large Language Model (LLM) Watermarking**: The work addresses the challenge that LLMs produce high-quality synthetic text that is often indistinguishable from human-written content at scale [Source 2].
> * **Production Deployment via Speculative Sampling**: To make generative watermarking practically viable in production systems, the paper develops methods to integrate watermarking with speculative sampling (an acceleration method where a draft model proposes tokens verified by a target model). Specifically, the authors propose two algorithms combining generative watermarking schemes with speculative sampling [Source 3].
> 
> *(Note: While the paper references the SynthID Google DeepMind publication [Source 1], further technical mechanics specifically designated under the "SynthID" brand name are not detailed beyond this citation in the provided excerpts).*

*Citations: 5 chunks*

---

## Validation Summary

**Total Retrieval Tests:** 60
**Successful Retrieval Tests:** 55
**Total Generation Tests:** 12
**Successful Generation Tests:** 12
