# Document Hybrid Search Validation Results

**Generated:** 2026-09-17 02:23:27
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
| 14 | 14. RRF + Graph + Dedup + MMR | 0.565 | 0.429 | 0.714 | 0.786 | 0.621 | |
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

> The Binding Constraint Thesis implies that harness comparisons are not a secondary implementation detail; they can be central to interpreting benchmark results for LLM agents.
> 
> Specifically, it defines benchmark performance as \(B(M,H)\), where both the model \(M\) and the harness \(H\) affect the score. It then defines harness variance as
> 
> \[
> HV(M)=Var_{H\sim P(H)}[B(M,H)]
> \]
> 
> and model variance as
> 
> \[
> MV(H)=Var_{M\sim P(M)}[B(M,H)].
> \]
> 
> The thesis claims that, for long-horizon LLM-agent tasks with comparable frontier models, harness variance is often comparable to or larger than model variance [Source 1]. In other words, changing the harness may affect scores as much as—or more than—changing the model.
> 
> This affects harness comparisons in three main ways:
> 
> 1. **Single-harness benchmark results are incomplete.**  
>    If a benchmark reports only \(B(M,H^*)\) for one undisclosed harness \(H^*\), then \(HV\) cannot be measured, making model comparisons potentially misleading [Source 1].
> 
> 2. **Harness comparisons must be treated as controlled experimental conditions.**  
>    The paper argues that the harness used to produce a score must be disclosed and included as part of the experimental condition. Cross-model comparisons should either hold the harness fixed in a “locked-harness protocol” or vary it deliberately in a “factorial protocol” [Source 2].
> 
> 3. **Harness effects may interact with model properties.**  
>    The thesis denies that harness shifts affect all models uniformly. For example, a harness emphasizing self-verification may help a model with high false-confidence rates more than a more conservative model. Thus, the model–harness interaction term is non-negligible, so comparing harnesses independently of models can be misleading [Source 3].
> 
> So, under the Binding Constraint Thesis, harness comparisons are valid only when the harness is disclosed and either fixed or systematically varied. The thesis also warns that a “better” harness for one model may not be equally better for another, because harness effects can interact strongly with model behavior [Source 3].

*Citations: 5 chunks*

### Anthropic — `claude-sonnet-5`

> # The Binding Constraint Thesis and Harness Comparisons
> 
> ## Core Claim
> 
> The Binding Constraint Thesis addresses evaluation of LLM agents on long-horizon tasks, formalizing the benchmark score B(M, H) as a function of both model M and harness H [Source 1]. It defines two variance terms:
> 
> - **HV(M)** = harness variance — the variance in a model's score across different harnesses: HV(M) = Var_{H~P(H)}[B(M, H)]
> - **MV(H)** = model variance — the variance across different models under a fixed harness: MV(H) = Var_{M~P(M)}[B(M, H)]
> 
> The thesis asserts that in the regime of long-horizon agent evaluation with comparable frontier models, **HV is often comparable to or larger than MV**, and may even dominate it [Source 1]. This means that differences in the harness (the scaffolding, tooling, and prompting infrastructure used to run an agent) can produce score variation as large as, or larger than, differences between the models themselves.
> 
> ## Consequences for Harness Comparisons
> 
> 1. **Undermines single-harness benchmark reporting**: Because current benchmark protocols typically report B(M, H*) for a single, often undisclosed harness H*, HV becomes unmeasurable. This renders model comparisons "incomplete and potentially misleading" [Source 1], since one cannot tell whether an observed ranking reflects genuine model capability differences or artifacts of the specific harness used.
> 
> 2. **Self-reported harnesses compound the problem**: When different evaluators report results "under their own harness," this compounds rather than resolves the attribution problem, since scores become tied to non-standardized, hidden experimental conditions [Source 2].
> 
> 3. **Non-negligible interaction effects**: The thesis emphasizes that harness and model effects are not additive or separable — there is a meaningful interaction term. For example, a harness emphasizing self-verification may help a model with high false-confidence rates more than a model with conservative output distributions, so "the same harness shift produces different gains across models" [Source 3]. This means model rankings are not stable properties of the models alone; they depend on the harness in a way that can even reorder rankings across different harnesses.
> 
> 4. **Rankings are only valid under fixed conditions**: Treating model rankings as intrinsic model properties would require the interaction term to be small relative to MV — a condition the thesis explicitly denies [Source 3]. As a result, a ranking derived from a "locked-harness protocol" (fixing H = H*) only yields a clean, valid comparison for that specific H*, not a generalizable statement about relative model quality [Source 3].
> 
> ## Proposed Structural Solution
> 
> Because this is framed as a structural methodological problem, the paper argues it requires a structural solution: **disclosure** [Source 2]. Specifically:
> 
> - The harness used to produce any benchmark score must be treated as part of the experimental condition (not left implicit or hidden).
> - Cross-model comparisons should either **hold the harness fixed** across all models being compared (the "locked-harness protocol") or **treat harness as a controlled experimental factor** and vary it systematically (the "factorial protocol") [Source 2].
> 
> This ensures that harness effects are made visible and accounted for, rather than confounded with model capability, thereby making cross-model comparisons more reliable and interpretable.

*Citations: 9 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided literature, the **Binding Constraint Thesis** significantly impacts how benchmark and harness comparisons are conducted and interpreted for LLM agents on long-horizon tasks [Source 1, Source 2, Source 3]. 
> 
> The thesis, which co-occurs centrally with harness variance ($HV$), model variance ($MV$), and LLMs [Knowledge Graph Relationships], asserts that harness variance $HV(M) = \text{Var}_{H \sim P(H)} [B(M, H)]$ is often comparable to, larger than, or dominates model variance $MV(H) = \text{Var}_{M \sim P(M)} [B(M, H)]$ [Source 1: § 3].
> 
> Its primary effects on comparisons include:
> 
> 1. **Renders Single/Undisclosed Harness Comparisons Incomplete and Misleading:**
>    * Standard benchmark protocols that report scores $B(M, H^*)$ using a single undisclosed harness $H^*$ leave $HV$ unmeasurable. This makes model comparisons incomplete and potentially misleading [Source 1: § 3].
>    * Reporting results under distinct, self-selected harnesses compounds attribution problems rather than resolving them [Source 2: § 3].
> 
> 2. **Invalidates Treating Rankings as Purely Model Properties:**
>    * The interaction term between model and harness is non-negligible [Source 3: § 3]. For example, a harness emphasizing self-verification aids a model with high false-confidence rates more than one with conservative output distributions, meaning harness shifts affect models differently [Source 3: § 3].
>    * Treating model rankings as intrinsic model properties requires this interaction term to be small relative to $MV$—an assumption that the Binding Constraint Thesis explicitly denies [Source 3: § 3].
> 
> 3. **Necessitates Structural Protocol Changes (Disclosure, Locked-Harness, and Factorial Protocols):**
>    * The harness must be formally disclosed as part of the experimental condition [Source 2: § 3].
>    * Valid cross-model comparisons require either:
>      * **A Locked-Harness Protocol:** Holding the harness fixed ($H = H^*$). While this recovers a clean ranking of $\{B(M_i, H^*)\}$ across models, that ranking is strictly valid *only* under that specific $H^*$ [Source 2: § 3, Source 3: § 3].
>      * **A Factorial Protocol:** Varying the harness as a controlled experimental factor across evaluations [Source 2: § 3].

*Citations: 10 chunks*

---

**Benchmark Query b:**
"Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 11 PDFs, 2,072 chunks

### Openai — `gpt-5.5`

> Dynamic Tiered AgentRunner’s **Risk-Adaptive Tiering** is a governance mechanism for enterprise AI execution that dynamically adjusts both **computational budget** and **review intensity** according to a task’s **risk-complexity profile** [Source 3].
> 
> According to the paper’s abstract, the framework uses three execution modes:
> 
> - **Light**
> - **Standard**
> - **Full**
> 
> Tasks are routed among these tiers based on how risky or complex they are, with the goal of achieving a **Pareto-optimal safety-efficiency trade-off**—that is, improving safety without applying expensive full governance to every task [Source 3].
> 
> The motivation is that in enterprise settings, the main bottleneck is not lack of agent autonomy, but lack of **governability**: high-risk write operations may proceed without independent review, multi-step tasks may lack verification, and indiscriminate computation can make deployment economically impractical [Source 1].
> 
> In the evaluation setup, tasks were distributed across several categories of increasing operational risk and complexity:
> 
> - Information queries: **40.2%**
> - Single-object writes: **29.8%**
> - Multi-object or batch operations: **19.7%**
> - Cross-domain complex tasks: **10.2%** [Source 2]
> 
> The framework was compared against baselines such as a single unguided agent, an always-on full pipeline, and ablations removing the CriticAgent, VerifierAgent, or RecoveryAgent [Source 2]. Metrics included **Task Success Rate (SR)**, **Risk Execution Error Rate (RERR)**, latency, inference cost, and Recovery Success Rate (RSR) [Source 2].
> 
> In short, **Risk-Adaptive Tiering** in Dynamic Tiered AgentRunner is the mechanism that routes tasks to different governance levels so that simple, low-risk tasks avoid unnecessary overhead, while higher-risk or more complex operations receive stronger review and verification [Source 3].

*Citations: 7 chunks*

### Anthropic — `claude-sonnet-5`

> # Dynamic Tiered AgentRunner Framework: Risk-Adaptive Tiering
> 
> ## Overview
> 
> The **Dynamic Tiered AgentRunner** is a controlled execution protocol for enterprise AI systems, distilled from a production multi-tenant SaaS platform [Source 1]. It was developed to address a critical gap in the prevailing LLM-based agent research paradigm, which the authors argue over-emphasizes autonomy while neglecting **governability**. As stated in the paper: "the critical bottleneck is not insufficient autonomy but insufficient governability: high-risk write operations proceed without independent review, complex multi-step tasks lack verification mechanisms, and indiscriminate computational expenditure renders deployment economically unviable" [Source 1].
> 
> ## Risk-Adaptive Tiering Mechanism
> 
> Risk-Adaptive Tiering is one of **three core mechanisms** operationalized by the framework [Source 3]. Specifically, it:
> 
> - **Dynamically allocates computational budget and review intensity** across three distinct execution modes: **Light**, **Standard**, and **Full** [Source 3].
> - Bases this allocation on a **task's risk-complexity profile**, meaning that the level of scrutiny and resources applied scales with how risky and/or complex a given task is [Source 3].
> - Is designed to achieve **Pareto-optimal safety-efficiency trade-offs**, balancing the need for oversight (safety) against computational cost and latency (efficiency) [Source 3].
> 
> ## Relationship to Other Framework Components
> 
> Risk-Adaptive Tiering operates alongside two other core mechanisms described in the framework:
> 
> 1. **Separation of Powers** — physically isolating proposal (Worker), review (Critic), execution (ToolGateway), and verification (Verifier) roles into independent, non-colluding processes, ensuring no single agent can both propose and approve an action [Source 3].
> 2. **Verifier-Recovery closed loop** — treating failure as a first-class execution state, enabling self-healing and organizational learning [Source 3].
> 
> ## Evaluation Evidence
> 
> The paper's empirical evaluation (§5.3) tests the tiering approach against several baselines, including a **Static-Full** configuration (always-on full pipeline for every task) and ablations such as **No-Critic**, **No-Verifier**, and **No-Recovery** (dynamic tiering with the respective agent removed) [Source 2]. These are evaluated across a workload distributed among information queries (40.2%), single-object writes (29.8%), multi-object/batch operations (19.7%), and cross-domain complex tasks (10.2%) [Source 2]. Metrics used include Task Success Rate (SR), Risk Execution Error Rate (RERR, i.e., unreviewed high-risk operations), Average Latency, Average Inference Cost, and Recovery Success Rate (RSR) [Source 2].
> 
> ## Gaps in Available Information
> 
> The provided context does not include the specific quantitative results comparing Risk-Adaptive Tiering against the baselines (e.g., exact SR, RERR, latency, or cost figures), nor does it detail the precise criteria distinguishing when a task is routed to Light, Standard, or Full execution mode. The knowledge graph indicates that "AgentRunner" co-occurs with "Main Results," "SR," and "Tier Routing," suggesting these details exist elsewhere in the source document, but they are not present in the passages provided here.

*Citations: 11 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided scientific literature, **Risk-Adaptive Tiering** is one of the three core mechanisms introduced in the **Dynamic Tiered AgentRunner** framework ([Source 1], [Source 3]).
> 
> ### Overview and Purpose
> In enterprise environments, unrestricted LLM autonomy presents critical governability bottlenecks—such as unreviewed high-risk write operations, absent verification, and indiscriminate computational expenditure ([Source 1]). To address these issues, the Dynamic Tiered AgentRunner framework was designed as a controlled execution protocol distilled from a production multi-tenant SaaS platform ([Source 1], [Source 3]).
> 
> ### Mechanism of Risk-Adaptive Tiering
> Risk-Adaptive Tiering operates through the following design principles:
> 
> * **Dynamic Allocation of Resources:** It dynamically allocates both computational budget and review intensity based on an incoming task's specific **risk-complexity profile** ([Source 3]).
> * **Execution Modes (Tiers):** The framework structures execution across three distinct modes:
>   1. **Light**
>   2. **Standard**
>   3. **Full** ([Source 3]).
> * **Pareto-Optimal Trade-offs:** By adjusting the review depth and compute budget according to task risk (e.g., differentiating between simple information queries, single-object writes, batch operations, or cross-domain complex tasks [Source 2]), it achieves Pareto-optimal safety-efficiency trade-offs without incurring unnecessary costs on low-risk tasks ([Source 1], [Source 3]).
> 
> ### System Context and Governance Integration
> Risk-Adaptive Tiering functions alongside other modular agents and components in the execution framework:
> * It selectively engages specialized, non-colluding roles—such as the **Worker** (proposal), **Critic** / `CriticAgent` (review), **ToolGateway** (execution), and **Verifier** / `VerifierAgent` (verification)—depending on the active execution tier ([Source 2], [Source 3], [Knowledge Graph Relationships]). 
> * Evaluating the tiered architecture against static baselines (such as an always-on `Static-Full` pipeline or single-agent baselines) demonstrates its ability to balance Task Success Rate (SR), Risk Execution Error Rate (RERR), Average Latency, and Average Inference Cost ([Source 2]).

*Citations: 13 chunks*

---

## Validation Summary

**Total Retrieval Tests:** 30
**Successful Retrieval Tests:** 30
**Total Generation Tests:** 6
**Successful Generation Tests:** 6
