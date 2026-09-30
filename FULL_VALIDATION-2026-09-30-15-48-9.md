# Document Hybrid Search Validation Results

**Generated:** 2026-09-30 15:48:19
**Corpus:** corpus
**Storage Backend:** parquet
**Live Mode:** True

## 📊 Empirical Benchmark Results

| # | Strategy Name | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 | Key Characteristic |
|---|---|---|---|---|---|---|---|
| 1 | 1. Pure BM25 (Sparse) | 0.421 | 0.263 | 0.526 | 0.632 | 0.458 | |
| 2 | 2. Pure TF-IDF (Sparse Vector Space) | 0.199 | 0.053 | 0.263 | 0.368 | 0.208 | |
| 3 | 3. Linear Hybrid (a=0.3) | 0.378 | 0.211 | 0.474 | 0.579 | 0.408 | |
| 4 | 4. Linear Hybrid (a=0.5) | 0.376 | 0.211 | 0.474 | 0.474 | 0.370 | |
| 5 | 5. Linear Hybrid (a=0.7) | 0.271 | 0.105 | 0.368 | 0.474 | 0.294 | |
| 6 | 6. RRF (k=60) | 0.393 | 0.263 | 0.474 | 0.474 | 0.382 | |
| 7 | 7. RRF + Deduplication | 0.349 | 0.263 | 0.421 | 0.421 | 0.349 | |
| 8 | 8. RRF + Dedup + MMR | 0.354 | 0.263 | 0.421 | 0.526 | 0.397 | |
| 9 | 9. PPMI Semantic + BM25 RRF | 0.257 | 0.105 | 0.316 | 0.368 | 0.247 | |
| 10 | 10. Cross-Encoder Re-rank | 0.441 | 0.316 | 0.474 | 0.684 | 0.490 | |
| 11 | 11. Sentence-Transformer (MiniLM) | 0.326 | 0.211 | 0.368 | 0.368 | 0.303 | |
| 12 | 12. Adaptive Hybrid | 0.366 | 0.211 | 0.474 | 0.579 | 0.401 | |
| 13 | 14. RRF + Graph + Dedup + MMR | 0.407 | 0.368 | 0.421 | 0.526 | 0.435 | |
| 14 | Ablation: Graph only | 0.245 | 0.158 | 0.211 | 0.316 | 0.232 | |
| 15 | Ablation: RRF + Graph (no dedup/MMR) | 0.457 | 0.368 | 0.474 | 0.474 | 0.428 | |
| 16 | Ablation: RRF + Graph + Dedup (no MMR) | 0.412 | 0.368 | 0.421 | 0.421 | 0.395 | |
## 🤖 LLM Adapter Results

| Provider | Default Model | Route / Mode |
|---|---|---|
| **OpenAI** | `gpt-5.5` | Real API (Direct via `.env`) |
| **Anthropic** | `claude-sonnet-5` | Real API (Direct via `.env`) |
| **Gemini** | `gemini-3.8-flash` | Real API (Direct via `.env`) |

**Benchmark Query a:**
"How does the Binding Constraint Thesis affect harness comparisons across models?"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 37 PDFs, 8,637 chunks

### Openai — `gpt-5.5`

> The Binding Constraint Thesis says that in long-horizon LLM agent evaluations, benchmark outcomes depend not only on the model but also strongly on the **harness** used to run the model. Formally, it defines benchmark score as \(B(M,H)\), with harness variance \(HV(M)=Var_H[B(M,H)]\) and model variance \(MV(H)=Var_M[B(M,H)]\). The thesis claims that, for comparable frontier models, harness variance is often comparable to or larger than model variance, so differences in scores may reflect harness effects as much as—or more than—model capability differences [Source 1].
> 
> This affects harness comparisons across models in three main ways:
> 
> 1. **Model rankings are not purely model properties.**  
>    A model’s benchmark score under one harness \(H^*\) only gives a ranking for that specific harness. A “locked-harness” comparison can cleanly rank models as \(\{B(M_i,H^*)\}\), but the result should not be generalized to other harnesses unless harness effects are known to be small [Source 2].
> 
> 2. **Harness–model interactions can change relative performance.**  
>    The thesis emphasizes that the interaction term is non-negligible: for example, a harness that emphasizes self-verification may help a model with high false-confidence rates more than it helps a more conservative model. Thus, the same harness modification can produce different gains across different models [Source 2].
> 
> 3. **Comparisons using different undisclosed or self-optimized harnesses are confounded.**  
>    If different groups report results using their own harnesses, attribution becomes unclear: a higher score may come from the model, the harness, or their interaction. The thesis therefore argues that the harness must be disclosed and treated as part of the experimental condition [Source 3].
> 
> So, under the Binding Constraint Thesis, valid cross-model comparison requires either a **locked-harness protocol**, where all models are evaluated under the same harness, or a **factorial protocol**, where harnesses are varied as a controlled experimental factor [Source 3]. Without this, harness comparisons across models are incomplete and potentially misleading.

*Citations: 5 chunks*

### Anthropic — `claude-sonnet-5`

> Based on the provided sources, the Binding Constraint Thesis has significant implications for how harness comparisons across models should be understood and conducted.
> 
> ## Core Claim of the Thesis
> 
> The Binding Constraint Thesis asserts that for LLM agents operating on long-horizon tasks with comparable frontier models, **harness variance (HV) is often comparable to or larger than model variance (MV)**, and may even dominate it in many current long-horizon agent evaluations [Source 1]. Formally, given B(M, H) as the benchmark score of model M under harness H:
> - HV(M) = Var over harnesses (H ~ P(H)) of B(M, H)
> - MV(H) = Var over models (M ~ P(M)) of B(M, H)
> 
> The thesis holds that HV is not negligible relative to MV in this regime [Source 1].
> 
> ## Effect on Harness Comparisons Across Models
> 
> **1. Model-harness interactions are non-trivial and asymmetric.** The thesis emphasizes that the interaction term between harness and model is non-negligible: a harness change does not affect all models equally. For example, a harness that emphasizes self-verification helps a model with high false-confidence rates more than a model with a conservative output distribution—meaning "the same harness shift produces different gains across models" [Source 2]. This directly undermines any comparison across models under a single harness, since the ranking may reflect harness-specific interactions rather than intrinsic model quality.
> 
> **2. Treating model rankings as stable model properties requires an assumption the thesis rejects.** According to Source 2, interpreting cross-model rankings as reflecting true model capability requires the interaction term to be small relative to MV—a condition the Binding Constraint Thesis explicitly denies [Source 2].
> 
> **3. Current benchmarking practice obscures this problem.** Because current protocols report B(M, H*) for a single, often undisclosed harness H*, harness variance (HV) becomes unmeasurable, rendering "model comparisons incomplete and potentially misleading" [Source 1]. Some evaluations even use different harnesses per model or the model developer's own harness, which "compound[s] rather than resolve[s] attribution" rather than fixing the problem [Source 3].
> 
> **4. A "locked-harness protocol" produces valid but narrow comparisons.** Fixing H = H* allows a clean ranking of {B(Mi, H*)} across models—but this ranking is only valid under that specific harness and cannot be generalized as a statement about intrinsic model superiority [Source 2].
> 
> ## Implications: The Need for Structural Solutions
> 
> Because the problem is structural—arising from the confounding of harness and model effects—the thesis argues that the solution must also be structural: **disclosure of the harness as part of the experimental condition**, combined with either:
> - A **locked-harness protocol**, holding H fixed across all models being compared, or
> - A **factorial protocol**, treating harness as a controlled experimental factor varied systematically across models [Source 3].
> 
> In summary, the Binding Constraint Thesis implies that harness comparisons across models cannot be taken at face value under current single-harness benchmarking practices, because harness-driven variance and harness-model interactions can be as large as or larger than genuine model differences. Valid cross-model comparison requires explicit harness disclosure and either locking or systematically varying the harness as an experimental factor [Source 1], [Source 2], [Source 3].

*Citations: 11 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the technical literature in § 3 of **2605.23950v1.pdf**, the **Binding Constraint Thesis** fundamentally challenges how harness and model comparisons are interpreted in long-horizon LLM agent evaluations:
> 
> ### 1. Harness Variance Dominates or Rivals Model Variance
> For LLM agents operating on long-horizon tasks with comparable frontier models, the thesis defines:
> * Harness variance: $HV(M) = \text{Var}_{H \sim P(H)} [B(M, H)]$
> * Model variance: $MV(H) = \text{Var}_{M \sim P(M)} [B(M, H)]$
> 
> The thesis asserts that $HV$ is often comparable to, larger than, or dominates $MV$ [Source 1]. Because typical benchmark protocols report scores $B(M, H^*)$ for a single undisclosed harness $H^*$, $HV$ remains unmeasurable, rendering model comparisons incomplete and potentially misleading [Source 1].
> 
> ### 2. Model Rankings Cannot Be Treated as Inherent Model Properties
> Treating model rankings as intrinsic properties of models requires the model-harness interaction term to be small relative to $MV$—a condition that the Binding Constraint Thesis explicitly denies [Source 2]:
> * The interaction term between models and harnesses is non-negligible [Source 2].
> * A shift in harness produces differing gains across different models (for instance, a harness prioritizing self-verification provides greater gains to a model with high false-confidence rates than to a model with conservative output distributions) [Source 2].
> 
> ### 3. Invalidity of Independent Harness Evaluations
> When models are evaluated and reported under their own separate harnesses, attribution issues are compounded rather than resolved [Source 3].
> 
> ### 4. Required Evaluation Methodologies
> Because of the thesis's implications, valid cross-model comparisons require structural solutions and disclosure [Source 3]:
> * **Locked-Harness Protocol:** Fixes the harness ($H = H^*$) across all models. While this recovers a clean ranking $\{B(M_i, H^*)\}$ across models, that ranking is valid *only* under that specific harness $H^*$ [Source 2, Source 3].
> * **Factorial Protocol:** Varies the harness systematically as a controlled factor across models [Source 3].
> * **Mandatory Disclosure:** The harness must be explicitly disclosed and treated as a formal part of the experimental condition [Source 3].

*Citations: 10 chunks*

---

**Benchmark Query b:**
"Dynamic Tiered AgentRunner Framework Risk Adaptive Tiering"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 37 PDFs, 8,637 chunks

### Openai — `gpt-5.5`

> The **Dynamic Tiered AgentRunner Framework** is presented as a governable enterprise AI execution protocol designed to address risks in LLM-based agent systems, especially where autonomy alone is insufficient for safe deployment [Source 1].
> 
> Its **Risk-Adaptive Tiering** mechanism dynamically assigns each task to one of multiple execution modes—**Light, Standard, or Full**—based on the task’s **risk-complexity profile** [Source 3]. The purpose is to balance:
> 
> - **Safety / governance intensity**, especially for high-risk operations  
> - **Computational cost**, avoiding unnecessary full-pipeline execution for low-risk tasks  
> - **Latency and efficiency**, improving deployability in enterprise SaaS settings [Source 1], [Source 3]
> 
> According to the abstract, this tiering is intended to achieve **Pareto-optimal safety–efficiency trade-offs**, meaning the framework aims to improve governance and safety without indiscriminately increasing cost or latency for every task [Source 3].
> 
> The framework is evaluated against several baselines, including:
> 
> - **Single-Agent**: an LLM agent with tool access but no governance  
> - **Static-Full**: always uses the full pipeline for every task  
> - **No-Critic**, **No-Verifier**, and **No-Recovery** ablations [Source 2]
> 
> The evaluation uses metrics such as:
> 
> - **Task Success Rate (SR)**  
> - **Risk Execution Error Rate (RERR)**, defined as unreviewed high-risk operations  
> - **Average Latency**  
> - **Average Inference Cost**  
> - **Recovery Success Rate (RSR)** [Source 2]
> 
> The knowledge graph also links **AgentRunner** with **SR**, **ToolGateway**, and **LLM**, which aligns with the paper’s framing: AgentRunner is an LLM-based execution framework whose governed execution includes a ToolGateway and is evaluated using success-rate metrics [Source 2], [Source 3].
> 
> A key point is that Risk-Adaptive Tiering is not described as a standalone classifier in the provided context; rather, it is one component of a broader controlled execution protocol that also includes **Separation of Powers** and a **Verifier–Recovery closed loop** [Source 3].

*Citations: 10 chunks*

### Anthropic — `claude-sonnet-5`

> # Dynamic Tiered AgentRunner Framework: Risk-Adaptive Tiering
> 
> The **Dynamic Tiered AgentRunner** is a controlled execution protocol for enterprise AI systems, distilled from a production multi-tenant SaaS platform [Source 1]. It was developed in response to a specific problem: in enterprise environments, the critical bottleneck for LLM-based agents is not insufficient autonomy but **insufficient governability**—manifesting as unreviewed high-risk write operations, a lack of verification mechanisms for complex multi-step tasks, and economically unsustainable indiscriminate computational spending [Source 1].
> 
> ## Risk-Adaptive Tiering Mechanism
> 
> Risk-Adaptive Tiering is one of three core mechanisms operationalized by the framework [Source 3]. Specifically, it:
> 
> - **Dynamically allocates computational budget and review intensity** across three distinct execution modes: **Light, Standard, and Full** [Source 3].
> - Bases this allocation on a task's **risk-complexity profile**, meaning the system adapts its scrutiny level according to how risky and/or complex a given task is [Source 3].
> - Is designed to achieve **Pareto-optimal safety-efficiency trade-offs**, meaning it aims to maximize safety and efficiency simultaneously without unnecessary compromises on either dimension [Source 3].
> 
> ## Relationship to the Broader Framework
> 
> Risk-Adaptive Tiering does not operate in isolation. It works alongside two other core mechanisms described in the framework:
> 
> 1. **Separation of Powers** — physically isolating proposal (Worker), review (Critic), execution (ToolGateway), and verification (Verifier) roles into independent, non-colluding processes, ensuring no single agent can both propose and approve an action [Source 3].
> 2. **Verifier-Recovery closed loop** — treating failure as a first-class execution state, enabling systematic self-healing and organizational learning through retrospection [Source 3].
> 
> ## Evaluation Evidence
> 
> The framework's governance capabilities, including the tiering mechanism, are evaluated using specific metrics such as **Task Success Rate (SR)** and **Risk Execution Error Rate (RERR)**—the latter specifically measuring unreviewed high-risk operations [Source 2]. According to the knowledge graph relationships, AgentRunner co-occurs with both SR and RERR as evaluation metrics, indicating these are central to assessing how well the risk-tiering approach balances safety and task performance.
> 
> The evaluation setup compares the full Dynamic Tiered AgentRunner against several baselines, including a **Single-Agent** baseline (no governance), a **Static-Full** baseline (always-on full pipeline regardless of risk level), and ablations that remove specific components (**No-Critic**, **No-Verifier**, **No-Recovery**) [Source 2]. This suggests the risk-adaptive tiering is empirically contrasted against a non-adaptive "always full pipeline" approach to demonstrate its efficiency benefits.
> 
> ## What the Context Does Not Specify
> 
> The provided passages do not detail the exact criteria or thresholds used to classify a task's risk-complexity profile into Light, Standard, or Full tiers, nor do they provide the specific quantitative results (e.g., SR, RERR, latency, or cost figures) comparing tiered execution against the baselines. Additional source material would be needed to address those specifics.

*Citations: 10 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided context, the **Risk-Adaptive Tiering** mechanism in the **Dynamic Tiered AgentRunner Framework** is defined as follows:
> 
> ### Overview and Purpose
> The **Dynamic Tiered AgentRunner** framework addresses enterprise limitations in standard LLM agent research—where unconstrained autonomy leads to unreviewed high-risk write operations, lack of verification, and excessive computational expense [Source 1, § Abstract; KG]. 
> 
> ### Risk-Adaptive Tiering Mechanism
> **Risk-Adaptive Tiering** is one of the three core operational mechanisms distilled from a production multi-tenant SaaS platform [Source 1, Source 3, § Abstract]. Its primary functions include:
> * **Dynamic Resource Allocation:** It dynamically allocates both computational budget and review intensity according to a task's specific risk-complexity profile [Source 3, § Abstract].
> * **Execution Modes:** It routes tasks across three distinct execution tiers [Source 3, § Abstract; KG]:
>   1. **Light**
>   2. **Standard**
>   3. **Full**
> * **Safety-Efficiency Optimization:** By matching execution intensity to task risk, it is designed to achieve Pareto-optimal trade-offs between safety and computational efficiency [Source 3, § Abstract].
> 
> ### System Context and Evaluation
> * **Complementary Governance:** Risk-Adaptive Tiering operates alongside two other framework mechanisms: a physical **Separation of Powers** (isolating `Worker`, `Critic`, `ToolGateway`, and `Verifier` into independent processes) and a **VerifierRecovery closed loop** for handling failure and self-healing [Source 3, § Abstract].
> * **Evaluation Metrics:** In enterprise evaluation contexts, dynamic tiering configurations are assessed against baselines (such as *Static-Full* pipelines or ablations without critics, verifiers, or recovery agents) using metrics including Task Success Rate (SR), Risk Execution Error Rate (RERR), latency, inference cost, and Recovery Success Rate (RSR) [Source 2, § 5.3; KG].

*Citations: 7 chunks*

---

**Benchmark Query c:**
"Procedural Graphs Self-Evolving Execution Structures for LLM Agents"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 37 PDFs, 8,637 chunks

### Openai — `gpt-5.5`

> “**Procedural Graphs: Self-Evolving Execution Structures for LLM Agents**” introduces the **Procedural Graph (PG)** as an explicit, editable representation of procedural knowledge for LLM agents [Source 2].
> 
> According to the paper’s conclusion, a Procedural Graph gives an agent a queryable way to decide **“what to do next”** by connecting the agent’s current progress with relevant transitions and execution advice, while still preserving the model’s reasoning flexibility [Source 2]. The method is positioned as an alternative to memory-based approaches: across tasks and model families, PG reportedly delivers consistent gains over memory-based baselines [Source 2]. The paper also compares memory mechanisms in its main experiments, as indicated by Table 6 [Source 1].
> 
> A key contribution is **self-evolution**: Procedural Graphs can be built from minimal initializations and can repair expert-provided priors when those priors initially hurt performance [Source 2]. This supports the broader claim that agents can learn and revise procedural knowledge from execution feedback **without updating model weights** [Source 2].
> 
> The provided knowledge graph also connects **“Agents Knowledge Graph Procedural”** with **“Evolving Execution Structures,”** and further links **“Evolving Execution Structures”** to task-like structures such as **“Extract Calculate Check Answer”** and **“Graph Mona Lisa Da.”** This suggests the paper’s theme is about representing agent execution procedures as graph-like structures that evolve over time.
> 
> One reported empirical result is that, for **Claude Sonnet 4.6**, PG achieves the highest survival rate of **58.0%**, a **14.0-point gain over the baseline** [Source 3].
> 
> A noted tradeoff is that procedural guidance can increase token use even when it reduces the number of solver steps; the authors suggest future work could reuse guidance across steps or generate it selectively [Source 2].

*Citations: 8 chunks*

### Anthropic — `claude-sonnet-5`

> # Procedural Graphs: Self-Evolving Execution Structures for LLM Agents
> 
> Based on the provided context, here is what can be established about this work:
> 
> ## Core Concept
> 
> The paper introduces the **Procedural Graph (PG)**, described as "an explicit and editable representation of procedural knowledge that gives LLM agents a queryable answer to what to do next" [Source 2]. The key design goal is that PG "connects the agent's current progress with relevant transitions and execution advice while preserving reasoning flexibility" [Source 2].
> 
> ## Self-Evolution Mechanism
> 
> A central feature highlighted in the conclusion is that the system is **self-evolving**: "Self-evolution builds effective graphs from minimal initializations and repairs expert priors that initially hinder performance" [Source 2]. This suggests the graph structure is not fixed upfront but develops iteratively based on the agent's execution experience, with the framework supporting "learning and revising procedural knowledge from execution feedback without updating model weights" [Source 2] — meaning the approach improves agent performance without requiring retraining or fine-tuning of the underlying LLM.
> 
> ## Empirical Results
> 
> The paper reports performance comparisons against memory-based baselines:
> - PG "delivers consistent gains over memory-based baselines" across different tasks and model families [Source 2].
> - In experiments involving **Claude Sonnet 4.6**, PG "achieves the highest survival rate of 58.0% (a 14.0-point gain over the baseline)" [Source 3], indicating a substantial improvement in agent robustness/success on the evaluated task.
> - The paper also includes a comparison table of memory mechanisms used across the main experiments (Table 6), suggesting PG was evaluated against multiple alternative memory approaches [Source 1].
> 
> ## Trade-offs and Future Work
> 
> The authors note an important limitation: **guidance increases token use even when it reduces solver steps** [Source 2]. This implies a cost/efficiency trade-off between the token overhead of providing procedural guidance versus the benefit of shortening the number of steps needed to solve a task. The authors suggest future work could address this by finding ways to "reuse guidance across steps or generate it selectively" [Source 2].
> 
> ## Gaps in Available Context
> 
> The provided passages do not include details on:
> - The precise architecture or formal definition of how the Procedural Graph is structured (nodes, edges, transitions).
> - The specific self-evolution algorithm/update rule used to build and repair the graph.
> - Full experimental setup, baseline methods, or datasets/tasks beyond the single Claude Sonnet 4.6 result cited.
> - What "MCP" and "DGM" (noted as co-occurring with "Agents" in the knowledge graph) specifically refer to in relation to this paper — the context does not clarify these acronyms or their connection to Procedural Graphs.
> 
> If you need these additional details, further excerpts from the paper's methodology and experimental sections would be required.

*Citations: 9 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided literature, **Procedural Graphs (PG)** represent self-evolving execution structures designed for Large Language Model (LLM) agents [Source 2, Knowledge Graph Relationships]. 
> 
> Key aspects of this framework include:
> 
> ### 1. Definition and Core Functionality
> * **Explicit & Editable Representation:** A Procedural Graph provides an explicit, editable representation of procedural knowledge that offers LLM agents a queryable answer to "what to do next" [Source 2].
> * **Execution Guidance:** It links an agent’s current progress to relevant transitions and execution advice, maintaining reasoning flexibility during task performance [Source 2]. 
> * **No Weight Updates:** The framework enables learning, refining, and revising procedural knowledge entirely from execution feedback without requiring updates to the underlying model weights [Source 2].
> 
> ### 2. Self-Evolution Mechanism
> * **Dynamic Refinement:** Through self-evolution, the system can construct effective graphs starting from minimal initializations [Source 2].
> * **Repairing Priors:** It can automatically diagnose and repair expert priors that might otherwise hinder agent performance [Source 2].
> 
> ### 3. Empirical Performance
> * **Comparison to Baselines:** Across various model families and tasks, PG delivers consistent gains over traditional memory-based baselines [Source 1, Source 2].
> * **Task Success:** For example, when evaluated with Claude Sonnet 4.6, PG achieved a 58.0% survival rate, marking a 14.0-point improvement over the baseline [Source 3].
> 
> ### 4. Trade-offs and Future Directions
> * **Token Efficiency:** While guidance from the Procedural Graph reduces overall solver steps, it increases token usage [Source 2].
> * **Future Work:** Suggested improvements include generating guidance selectively or reusing guidance across multiple steps to optimize token consumption [Source 2].

*Citations: 10 chunks*

---

**Benchmark Query d:**
"Analyzing and Predicting Token Consumption in Agentic Coding Tasks"
**Strategy:** RRF + Dedup + MMR | **Corpus:** 37 PDFs, 8,637 chunks

### Openai — `gpt-5.5`

> The paper **“Analyzing and Predicting Token Consumption in Agentic Coding Tasks”** presents what it describes as the **first systematic study of token consumption patterns in agentic coding tasks** [Source 1]. It is motivated by the rapid growth of LLM token usage as AI agents are adopted in complex workflows [Source 1].
> 
> The study focuses on three main questions:
> 
> 1. **Where do AI agents spend tokens?**  
> 2. **Which models are more token-efficient?**  
> 3. **Can agents predict their token usage before task execution?** [Source 1]
> 
> According to the paper’s methodology and findings, the authors compare token usage across three coding-related settings: **code reasoning**, **coding chat**, and **agentic coding** [Source 2]. They report that **agentic tasks are uniquely expensive**, and that **input tokens drive the cost of AI agents** [Source 2]. The paper also examines variation in token usage across tasks and runs, whether higher token costs correlate with task completion, and whether expert judgments of task difficulty align with actual agent token costs [Source 2].
> 
> In its conclusion, the paper states that **agentic tasks produce complex token-usage dynamics** and that **predicting token consumption before execution remains fundamentally challenging for frontier models** [Source 3]. The authors suggest that these findings may support future work on more controllable and transparent pricing schemes for AI agents [Source 3].
> 
> From the provided knowledge graph, the topic is also connected to broader entities such as **AGENTIC**, **CODING**, **AGENTS**, and **AI**, reinforcing that the work sits at the intersection of agentic AI systems and coding workflows.

*Citations: 8 chunks*

### Anthropic — `claude-sonnet-5`

> # Analyzing and Predicting Token Consumption in Agentic Coding Tasks
> 
> Based on the provided sources, this refers to a research paper by Longju Bai, Zhemin Huang, Xingyao Wang, Jiao Sun, Rada Mihalcea, Erik Brynjolfsson, Alex Pentland, and Jiaxin Pei, affiliated with University of Michigan, Stanford University, All Hands AI, Google Deepmind, Microsoft AI, and Massachusetts Institute of Technology [Source 1].
> 
> ## Core Research Questions
> 
> The paper is motivated by the rapid growth in LLM token consumption driven by the wide adoption of AI agents in complex human workflows. The authors identify three central questions that arise when agents are deployed on token-intensive tasks [Source 1]:
> 
> 1. **Where do AI agents spend the tokens?**
> 2. **Which models are more token-efficient?**
> 3. **Can agents predict their token usage before task execution?**
> 
> The paper explicitly presents itself as "the first systematic study of token consumption patterns in agentic coding tasks" [Source 1].
> 
> ## Methodology and Key Findings
> 
> The study conducts a systematic comparison across three coding-related task types [Source 2]:
> - **Code reasoning** (referencing Gu et al., 2024)
> - **Coding chat** (referencing Crystalcare AI, 2023)
> - **Agentic coding** (referencing Jimenez et al., 2024; Chowdhury et al., 2024)
> 
> A key finding highlighted is that **agentic tasks are uniquely expensive, and input tokens drive the cost of AI Agents** [Source 2]. The analysis in this section (§2/§3 of the paper) also examines:
> - Variance in token usage across tasks and runs
> - Whether higher token costs correlate with greater task completion success
> - Whether expert perceptions of task difficulty align with actual agent token costs [Source 2]
> 
> ## Conclusions
> 
> The paper's conclusion (§8) summarizes that this is the first systematic study empirically evaluating whether models can predict their own token usage before execution. The authors find that **agentic tasks lead to complex token usage dynamics**, and that **predicting potential token consumption before task execution remains a fundamentally challenging problem for frontier models** [Source 3].
> 
> The authors suggest their findings offer new insights into agent behavior that could inspire future work on building **more controllable and transparent agent pricing schemes** [Source 3].
> 
> ## Note on Available Information
> 
> The provided sources do not include the detailed experimental setup (e.g., specific datasets, models tested, or benchmark names used for token prediction), nor the specific quantitative results comparing model token-efficiency. The knowledge graph relationships provided (e.g., linking "AGENTIC" with "RAPTOR," "RAG," "RETRIEVAL," or "CODING" with "DARWIN," "GPT-6") do not appear to be substantively connected to the content of this specific paper based on the source passages given, and thus cannot be used to further ground claims about this work's methodology or findings.

*Citations: 8 chunks*

### Gemini — `gemini-3.8-flash`

> Based on the provided literature, **"Analyzing and Predicting Token Consumption in Agentic Coding Tasks"** is the first systematic study investigating LLM token consumption patterns and token predictability in complex agent workflows [Source 1, Source 3].
> 
> ---
> 
> ### 1. Research Motivation and Core Questions
> The increasing integration of AI agents into complex human workflows has led to rapid growth in token consumption [Source 1]. This raises three fundamental questions regarding deployment efficiency and cost [Source 1]:
> 1. **Where do AI agents spend the tokens?**
> 2. **Which models are more token-efficient?**
> 3. **Can agents predict their token usage before task execution?**
> 
> ---
> 
> ### 2. Token Consumption Patterns in Agentic Coding
> Key findings on consumption dynamics include:
> * **Unique Expense and Cost Drivers:** Agentic coding tasks are uniquely expensive compared to other coding tasks, with **input tokens driving the primary cost** of AI agents [Source 2].
> * **Task Comparison:** The study establishes a systematic comparison of token usage across three distinct coding-related paradigms [Source 2]:
>   * **Agentic coding** (e.g., Jimenez et al., 2024; Chowdhury et al., 2024)
>   * **Code reasoning** (e.g., Gu et al., 2024)
>   * **Coding chat** (e.g., Crystalcare AI, 2023)
> * **Variances and Performance:** The study evaluates the variance in token consumption across different runs and tasks, investigating whether higher token expenditures correlate with higher task completion rates, as well as whether human expert perceptions of task difficulty align with actual agent token costs [Source 2]. In connection with agentic workflows, coding tasks frequently relate to long-horizon execution [Knowledge Graph Relationships].
> 
> ---
> 
> ### 3. Predictability of Token Usage
> * **Predictive Capability:** Empirically evaluating whether models can forecast their own token consumption before task execution shows that **predicting token usage prior to execution remains a fundamentally challenging problem for frontier models** due to complex token dynamics [Source 3].
> * **Implications:** Insights from this work aim to inform the development of more controllable, predictable, and transparent pricing schemes for AI agents [Source 3].

*Citations: 8 chunks*

---

## Validation Summary

**Total Retrieval Tests:** 72
**Successful Retrieval Tests:** 64
**Total Generation Tests:** 12
**Successful Generation Tests:** 12
