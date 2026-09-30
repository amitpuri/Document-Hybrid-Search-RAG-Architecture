# Retrieval Strategies: From Hybrid Text Search to Graph-Enhanced Multimodal Retrieval

> **Where this file lives:** `docs/retrieval-strategies.md`. Links are relative to that location (`../src/...`).
>
> **What it covers:**
> - **Part I** explains the 12 retrieval strategies benchmarked in this repository.
> - **Part II** is a deep dive into one recent paper that extends retrieval with a knowledge graph and multiple modalities: *Liao et al., "A study on the GraphRAG semantic retrieval algorithm for multimodal data," Discover Artificial Intelligence, 2026.*
> - **Part III** shows how the paper's ideas could be added to this codebase.
>
> **Convention:** anything described as *implemented* comes from the project [README](../README.md). Anything under "proposed" is a suggestion, not existing code.

---

## Contents

1. [Why retrieval strategy matters](#1-why-retrieval-strategy-matters)
2. [Part I: Strategies in this repo](#part-i-strategies-in-this-repo)
3. [Part II: The paper in depth](#part-ii-the-paper-in-depth)
4. [Part III: Bringing the paper's ideas into this repo](#part-iii-bringing-the-papers-ideas-into-this-repo)
5. [Decision guide](#decision-guide)
6. [Glossary](#glossary)
7. [Sources and licensing note](#sources-and-licensing-note)

---

## 1. Why retrieval strategy matters

A RAG system retrieves evidence, then generates an answer from it. The generator can only work with what retrieval hands it, so retrieval sets the ceiling on answer quality. A retrieval strategy is a decision about **how** candidates are found, combined, cleaned, and ordered.

Four properties are worth measuring:

| Property | Question it answers | Typical metric |
|---|---|---|
| Recall | Did the right evidence appear at all? | Recall@k |
| Ranking | Was it near the top? | MRR, NDCG |
| Diversity | Is the list free of repeats? | Qualitative, MMR |
| Provenance | Can each fact be traced to a source? | Citation accuracy |

---

# Part I: Strategies in this repo

## 2. Architecture recap

The platform is split into three decoupled pipelines (see the [README](../README.md)):

| Pipeline | Location | Role |
|---|---|---|
| Ingestion | [`src/ingestion/`](../src/ingestion) | PDF extraction, section-aware chunking, cached storage |
| Retrieval | [`src/retrieval/`](../src/retrieval) | 12 benchmarked strategies: retrievers, fusion, post-processing |
| Generation | [`src/generation/`](../src/generation) | Context building with source headers, grounded prompts |

Evaluation lives in [`src/evaluation/`](../src/evaluation) (MRR, Recall@1/3/5, NDCG@5 on 14 curated queries). The unified entry points are [`src/engine.py`](../src/engine.py) and [`src/cli.py`](../src/cli.py).

## 3. The strategy catalogue

| # | Strategy | CLI flag | Family |
|---|---|---|---|
| 1 | BM25 | `bm25` | Sparse |
| 2 | TF-IDF | `tfidf` | Sparse-style |
| 3 to 5 | Linear hybrid (three weightings) | `linear_0.3`, `linear_0.5`, `linear_0.7` | Score fusion |
| 6 | Reciprocal Rank Fusion | `rrf` | Rank fusion |
| 7 | RRF + Jaccard deduplication | `rrf_dedup` | Fusion + cleanup |
| 8 | RRF + dedup + MMR | `rrf_dedup_mmr` | Fusion + cleanup + diversity |
| 9 | PPMI semantics + BM25, fused by RRF | `ppmi` | Distributional hybrid |
| 10 | Cross-encoder re-rank over a wide pool | `cross_encoder` | Two-stage |
| 11 | MiniLM sentence-transformer | `sentence_transformer` | Dense |
| 12 | Adaptive hybrid heuristic | `adaptive` | Query-aware |

Try one:

```bash
python -m src.cli search "POMDP belief state filtering" --strategy rrf_dedup_mmr --top-k 5
```

## 4. The concepts behind each family

### 4.1 Sparse retrieval: BM25 and TF-IDF

Sparse methods match the **words themselves**. TF-IDF rewards terms that are frequent in a chunk but rare in the corpus. BM25 refines this with term-frequency saturation (the tenth repeat counts for less than the second) and length normalization.

- **Good at:** exact terms such as acronyms, model names, identifiers, and error codes.
- **Weak at:** synonyms and paraphrase ("car" versus "automobile").

### 4.2 Dense retrieval: bi-encoders

A bi-encoder (here, MiniLM) embeds the query and each chunk into vectors independently. Retrieval is nearest-neighbor search.

- **Good at:** meaning, paraphrase, conceptual questions.
- **Weak at:** rare exact terms, and it depends heavily on how well the embedding model fits the domain.

### 4.3 Distributional semantics: PPMI

Positive Pointwise Mutual Information builds word-association signals from co-occurrence counts in the corpus itself. No neural network is needed. It sits between keyword matching and learned embeddings, and here it is fused with BM25.

### 4.4 Cross-encoder re-ranking

A cross-encoder reads the query and a candidate **together** and outputs one relevance score. It is more accurate than a bi-encoder but runs once per candidate, so it is used on a shortlist drawn from a wide first-stage pool.

### 4.5 Fusion

Sparse and dense retrievers fail in different ways, so combining them helps. Two approaches are benchmarked:

**Linear combination** blends the two score types with a weight. It needs score normalization, because BM25 scores and vector similarities live on different scales, and the best weight varies by dataset.

**Reciprocal Rank Fusion (RRF)** uses only rank positions:

```
RRF(doc) = sum over ranked lists of 1 / (k + rank(doc))      with k = 60
```

Documents ranked well in several lists rise. There is almost nothing to tune, which makes RRF a robust default.

**Adaptive hybrid** uses heuristics about the query to decide how to weight or route signals.

### 4.6 Post-processing

- **Jaccard deduplication** drops results whose word sets overlap too much with an already-kept result. This matters because overlapping chunks waste context-window space.
- **Maximal Marginal Relevance (MMR)** greedily picks the next result that is both relevant to the query and *different* from results already chosen, trading a little relevance for broader coverage.

## 5. Benchmark results

Measured by the repo on **14 ground-truth queries over 11 research PDFs (354 pages, 2,072 chunks)**. Reproduce with `python run_eval.py`.

| Strategy | MRR | Recall@1 | Recall@3 | Recall@5 | NDCG@5 |
|---|---|---|---|---|---|
| BM25 | 0.573 | 0.429 | 0.714 | 0.786 | 0.612 |
| TF-IDF | 0.392 | 0.143 | 0.500 | 0.714 | 0.448 |
| Linear (0.3) | 0.554 | 0.357 | 0.714 | 0.786 | 0.595 |
| Linear (0.5) | 0.524 | 0.286 | 0.714 | 0.857 | 0.599 |
| Linear (0.7) | 0.488 | 0.214 | 0.714 | 0.857 | 0.573 |
| RRF | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 |
| RRF + dedup | 0.629 | 0.500 | 0.714 | 0.857 | 0.678 |
| **RRF + dedup + MMR** | 0.625 | 0.500 | **0.786** | 0.857 | **0.683** |
| PPMI + BM25 | 0.402 | 0.214 | 0.500 | 0.643 | 0.437 |
| Adaptive | 0.494 | 0.286 | 0.571 | 0.786 | 0.549 |
| Cross-encoder | 0.483 | 0.286 | 0.571 | 0.857 | 0.567 |
| MiniLM dense | 0.292 | 0.143 | 0.357 | 0.571 | 0.339 |

**How to read it**

1. **Fusion wins.** The RRF family leads on MRR, Recall@1, and NDCG@5.
2. **BM25 is a hard baseline to beat** on technical text full of exact terms, and it clearly beat the MiniLM dense retriever here.
3. **Rank fusion beat score blending** in every setting tested.
4. **More machinery is not automatically better.** The cross-encoder and adaptive strategies did not beat plain RRF.
5. **Small sample warning.** With 14 queries, one query changes Recall@1 by about 7 points. Treat gaps between close strategies (for example RRF versus RRF + dedup + MMR) as inconclusive.

---

# Part II: The paper in depth

**Citation:** Liao, C., Ye, P., Huang, A., Huang, S., Yuan, X. *A study on the GraphRAG semantic retrieval algorithm for multimodal data.* Discover Artificial Intelligence 6:1060 (2026). DOI: [10.1007/s44163-026-02023-3](https://doi.org/10.1007/s44163-026-02023-3)

## 6. The problem the paper targets

The strategies in Part I all treat evidence as **independent text chunks**. The paper argues that this breaks down when:

- similar-looking text is retrieved but the **chain of evidence is broken**, meaning linked facts are not connected;
- **image content** and **structured fields** (tables) carry facts that text lacks;
- the generator has no principled way to judge **how credible** the retrieved evidence is.

The domain is electrical equipment (transformers, switchgear, cables, and so on), where a question like "what caused this alarm?" needs a maintenance report, an infrared photo, and an alarm-log row together.

## 7. Architecture and the paper's parameter choices

Five stages: data ingestion, semantic encoding, graph construction, retrieval enhancement, controlled generation.

| Stage | What happens | Key parameters |
|---|---|---|
| Ingestion | Parse text, detect images, clean table fields in parallel | 256-token text segments with 64 overlap; images resized to 224x224; table fields with over 8% missing values excluded from the graph |
| Encoding | Specialist encoders per modality, then projection to a shared space | MiniLM (384-D), YOLOv8m + CLIP ViT-B/32 for image regions, FT-Transformer (320-D); shared space is 640-D |
| Graph construction | Extract entities, merge aliases, link relations | 6 node types, 9 relation types, alias-merge threshold 0.81, entity-link threshold 0.74 |
| Retrieval enhancement | Vector recall, graph expansion, diffusion, aggregation, re-ranking | Top-120 recall, 3 hops (4 only in a sensitivity test), max 300 nodes, paths of 2 to 5 edges, edges under 0.58 dropped |
| Controlled generation | Cited, evidence-bounded answering | Qwen2.5-7B-Instruct, up to 12 evidence units, credibility threshold 0.69 |

Infrastructure: FAISS-GPU (IVFPQ) for vectors and Neo4j 5.12 for the graph.

**A note on the image encoder dimension.** The text says CLIP produces a 512-D vector, but the paper's encoding table lists the image branch as 640-D. The extra dimensions are most plausibly the class confidence, bounding-box coordinates, and texture descriptors the paper says are appended. The concatenation arithmetic supports this: 384 + 640 + 320 = 1344, the size of the paper's fused tri-modal vector.

## 8. The equations, explained in plain words

The paper defines 14 equations. Grouped by purpose:

### 8.1 Preprocessing quality (Eq. 1)

A per-modality data quality score: reward **completeness** and **retention after deduplication** and **structural consistency**, subtract the **anomaly ratio**. It acts as a gate so that poor input does not pollute the graph.

### 8.2 Alignment (Eqs. 3 to 5)

- **Cross-modal attention (Eq. 3).** For a text unit, compute a softmax over candidate image regions of the scaled dot product between projected text and image vectors, **plus a positional correction** built from entity co-occurrence, page proximity, and heading hierarchy. In effect: a region on the same page under the same heading as the text is more likely to be its match.
- **Fusion vector (Eq. 4).** A weighted sum of the three projected modality vectors plus a bias, passed through layer normalization. (The paper's notation reuses one projection symbol for all three modalities, which appears to be a typesetting slip, since the text describes a separate structured-field matrix.)
- **Alignment loss (Eq. 5).** Contrastive loss, plus a term matching the mean of text and image features, plus a term matching the covariance of text and structured-field features, plus a **graph adjacency term** that pulls together vectors of nodes connected in the graph. The three extra weights are 0.31, 0.27, and 0.42.

Training setup: batch size 96, a negative-sample queue of 4,096, temperature 0.073. Positives are units from the same gold evidence chain. Hard negatives are same-entity-type items with a different identity.

### 8.3 Graph expansion (Eqs. 6 and 7)

- **Entity expansion probability (Eq. 6).** A sigmoid over: vector similarity to the query, the summed prior weight of the relation types involved, co-occurrence count discounted by time difference between sources, minus a penalty for very generic, high-frequency entities.
- **Path confidence (Eq. 7).** The product of edge weight times relation reliability along the path, multiplied by an exponential length penalty. Paths need a confidence above 0.46 to survive.

### 8.4 Candidate scoring and diffusion (Eqs. 8 to 10)

- **Initial document score (Eq. 8).** Combines vector cosine, the fraction of query entities covered by the document, and the summed reliability of graph paths connecting the query to the document (damped by a log of the path count).
- **Diffusion (Eq. 9).** Iteratively mixes each node's score with its neighbors' scores through a degree-normalized adjacency matrix, keeping a share of the original score each round. This is the same family of computation as personalized PageRank. Settings: coefficient 0.57, 5 rounds.
- **Final recall score (Eq. 10).** Initial score plus the diffused score, minus a duplication penalty, plus a reward for covering text, image, and structured evidence.

### 8.5 Aggregation (Eqs. 11 to 13)

- **Path strength (Eq. 11).** Average of relation-type weight times edge confidence, scaled by how many text, image, and table hits support the path, and discounted for wide source spans.
- **Retention utility (Eq. 12).** Path strength **minus a penalty for similarity to paths already selected**, **plus a reward for modal complementarity**. This is structurally the same idea as MMR in this repo, with a graph-aware relevance term and a cross-modality bonus in place of the plain novelty term.
- **Selection (Eq. 13).** Choose the set of paths maximizing total utility subject to a token budget and a cap on item count.

### 8.6 Final re-ranking (Eq. 14)

A sigmoid over: semantic match, source-time consistency, graph path strength, log of the cross-verification count, minus cross-modal conflict penalty, minus duplicate penalty. A six-layer cross-encoder supplies the semantic component. Thresholds: semantic consistency 0.69, source reliability 0.73, conflict penalty 0.25.

### 8.7 Funnel summary

```
Top-120 vector recall
   -> 3-hop graph expansion (<= 300 nodes, paths of 2-5 edges, edges >= 0.58)
   -> diffusion + path scoring, cache <= 260 paths
   -> aggregation to <= 28 fragments within the token budget
   -> cross-encoder re-ranks top 72, keeps 18
   -> compression to <= 12 evidence units
   -> cited, evidence-bounded generation
```

## 9. How the paper evaluates

- **Corpus:** 18,640 Chinese and English text units, 7,820 equipment images, 42,300 table records, 96,500 normalized entities, 128,700 relation triples.
- **Queries:** 3,600 hand-built, split 70/10/20 into 2,520 training, 360 validation, and 720 test queries.
- **Test mix:** 210 text fact lookups, 170 image-text matching, 160 structured-field lookups, 180 multi-hop relation questions.
- **Annotation:** three engineers plus a senior adjudicator. Agreement (Cohen's kappa) of 0.91 for entity identity, 0.86 for relation labels, 0.88 for relevance labels.
- **Leakage control:** splitting is done at the equipment or source-document-group level, and near-duplicate texts are grouped before splitting. This is a good practice that many retrieval papers skip.
- **Generation judging:** three blinded domain experts decomposed answers into atomic claims and checked entailment, citation location, entity and temporal consistency, and whether rejection was required (kappa 0.84). An LLM judge was used only as a secondary check.
- **Metrics:** Precision@5, Recall@10, MRR, NDCG@10, evidence hit rate, cross-modal consistency, citation accuracy, answer fidelity, answer support rate, hallucination rate, no-answer accuracy, plus latency and graph-size statistics.

## 10. Results in depth

### 10.1 Overall and by task (720 test queries)

| Task | Queries | P@5 | R@10 | MRR | NDCG@10 | Evidence hit | Latency |
|---|---|---|---|---|---|---|---|
| Text fact lookup | 210 | 0.892 | 0.918 | 0.874 | 0.901 | 91.6% | 184 ms |
| Image-text matching | 170 | 0.861 | 0.887 | 0.846 | 0.872 | 88.9% | 216 ms |
| Table field lookup | 160 | 0.879 | 0.904 | 0.858 | 0.886 | 90.3% | 198 ms |
| Multi-hop relations | 180 | 0.834 | 0.862 | 0.811 | 0.847 | 86.7% | 243 ms |
| **Combined** | 720 | 0.867 | 0.893 | 0.848 | 0.876 | 89.4% | 211 ms |

Multi-hop questions are the hardest and slowest, which is where graph expansion should matter most.

### 10.2 Baseline ladder (P@5 / R@10, full test set)

| Method | P@5 / R@10 |
|---|---|
| BM25 | 0.721 / 0.754 |
| Dense retrieval | 0.812 / 0.835 |
| BM25 + dense hybrid | 0.824 / 0.847 |
| Multimodal vector retrieval | 0.838 / 0.861 |
| Multimodal RAG | 0.846 / 0.868 |
| KG-RAG | 0.842 / 0.864 |
| Standard GraphRAG | 0.855 / 0.878 |
| **Proposed** | **0.867 / 0.893** |

The proposed system beats standard GraphRAG by about 1.2 points in Precision@5 and 1.5 in Recall@10. Most of the climb from dense retrieval to standard GraphRAG comes from adding graph structure at all.

**Contrast with this repo.** In the paper's corpus, BM25 is the *weakest* baseline. In this repo's corpus, BM25 is the *strongest single retriever* and MiniLM the weakest. The metrics and corpora differ, so the numbers cannot be compared directly, but the opposite orderings illustrate a real lesson: **which retriever is best is corpus-dependent, so always benchmark on your own data.**

### 10.3 Does alignment actually work?

Direct alignment evaluation on image-text and field-text queries:

| Direction | Recall@1 / 5 / 10 before | After |
|---|---|---|
| Text to image | 0.461 / 0.684 / 0.773 | 0.632 / 0.824 / 0.891 |
| Image to text | 0.438 / 0.661 / 0.752 | 0.608 / 0.807 / 0.876 |
| Field to text | 0.512 / 0.731 / 0.819 | 0.689 / 0.861 / 0.918 |

Mean cosine similarity of true cross-modal pairs rose from 0.54 to 0.76, while hard negatives dropped from 0.31 to 0.24. The margin between them widened from 0.23 to 0.52, which is the clearest single piece of evidence that the shared space is doing its job.

### 10.4 Adding modalities

| Input | Queries | P@5 | R@10 | Latency |
|---|---|---|---|---|
| Text | 180 | 0.824 | 0.851 | 172 ms |
| Image | 120 | 0.791 | 0.826 | 205 ms |
| Structured | 110 | 0.808 | 0.839 | 188 ms |
| Text + image | 105 | 0.858 | 0.881 | 231 ms |
| Text + structured | 95 | 0.872 | 0.897 | 219 ms |
| Image + structured | 70 | 0.846 | 0.873 | 244 ms |
| All three | 40 | 0.904 | 0.928 | 268 ms |

Accuracy climbs with each added modality, and so does latency. Table fields help text more than images do because they supply attributes the prose omits. **Caveat:** the headline 0.904 figure rests on only 40 queries.

### 10.5 Ablations: which parts earn their keep

| Variant | P@5 / R@10 | Other effect |
|---|---|---|
| Full system | 0.867 / 0.893 | Evidence hit 89.4%, answer fidelity 93.2% |
| No cross-modal alignment | 0.834 / 0.858 | Evidence hit 84.7%, fidelity 87.6% |
| No graph expansion | 0.842 / 0.861 | Evidence hit 84.9% |
| No alias merging | 0.846 / 0.871 | Evidence hit 86.2% |
| No path-credibility constraints | 0.851 / 0.876 | Evidence hit 86.8% |
| Conflict penalty set to zero | 0.853 / 0.879 | Fidelity falls to 89.6% |
| No redundancy suppression | 0.858 / 0.883 | Fidelity falls to 90.5% |
| Cross-encoder replaced by graph score | 0.849 / 0.874 | Latency drops to 224 ms |
| No no-hit rejection | | Hallucination rises from 6.8% to 12.9% |
| No controlled-generation constraints | | Fidelity 84.3%, hallucination 15.7% |
| No evidence identifiers | | Citation accuracy falls from 94.1% to 82.6% |

Reading across: retrieval-side components (alignment, graph expansion, alias merging) move accuracy by 2 to 3 points each, while generation-side controls move **faithfulness** most.

**Hop depth:**

| Hops | Recall@10 | Latency |
|---|---|---|
| 2 | 0.879 | 241 ms |
| 3 | 0.893 | 283 ms |
| 4 | 0.887 | 342 ms |

More hops past three add noise and cost.

### 10.6 Interpretability of path evidence

On 360 complex queries, adding knowledge-graph path reasoning changed:

| Measure | Without | With |
|---|---|---|
| Evidence-chain completeness | 78.3% | 91.5% |
| Entity coverage | 81.6% | 93.2% |
| Path readability | 0.74 | 0.88 |
| Conflict suppression | 69.4% | 86.7% |

Paths of three hops showed the highest credibility (0.92 for entity relations), supporting the case for "moderate" chain lengths.

### 10.7 Cost and scaling

Per-stage latency (average): encoding 38 ms, vector recall 47 ms, graph expansion 82 ms, re-ranking 64 ms, aggregation 52 ms. These sum to the reported 283 ms. Graph expansion is the largest single stage.

| Entities | Avg latency | P95 | Peak VRAM | Throughput | Recall@10 |
|---|---|---|---|---|---|
| 10K | 146 ms | 228 ms | 6.8 GB | 6.8 QPS | 0.902 |
| 50K | 201 ms | 312 ms | 8.4 GB | 5.0 QPS | 0.899 |
| 100K | 283 ms | 421 ms | 11.6 GB | 3.5 QPS | 0.893 |
| 1M | 612 ms | 948 ms | 22.9 GB | 1.6 QPS | 0.879 |

Beyond about 100K entities, Neo4j traversal becomes the dominant cost. The cross-encoder adds a nearly fixed 64 to 79 ms and hurts throughput under concurrency. Cutting its budget from 72 to 48 candidates at 1M entities raised throughput from 1.6 to 2.1 QPS at a cost of 0.8 points of Recall@10.

## 11. Case studies from the paper

**Success trace.** The question asked what caused a high-oil-temperature alarm on a specific transformer and which maintenance action the evidence supports. Query entities (device, event, timestamp, target action) were linked into the graph with similarities of 0.94 and 0.91. The Top-120 stage returned 46 text units, 31 image regions, 28 table records, and 15 graph-associated units. After three-hop expansion and filtering, four pieces of evidence remained: a maintenance-report passage about an oxidized contactor terminal preventing a cooling fan from starting, an infrared image region showing a roughly 86 °C hotspot at that terminal, an alarm-table row, and a graph path linking device, fan, contactor, and image. Aggregation removed two duplicate maintenance descriptions and one image taken a week earlier. The cross-encoder scored the four survivors between 0.89 and 0.94, and the generated answer cited all four.

**Failure trace.** For a question about pump cavitation, an early configuration confused the pump with a similarly named pressure-sensor alias and treated a partly occluded image region as cavitation evidence. The vibration record contained no cavitation alarm. The initial evidence score of 0.71 produced a wrong positive answer. After entity-type compatibility, timestamp, and conflict penalties were applied, the score fell to 0.54, below the 0.58 threshold, and the system correctly declined. The authors identify **alias collisions, stale evidence, and weak image regions** as the main remaining failure sources.

## 12. Critical reading

**Strengths**

- Leakage-aware partitioning and expert, blinded claim-level evaluation of generation.
- Ablations that isolate individual components, including generation-side controls.
- Honest reporting of cost, including scaling to 1M entities and a failure case.
- A concrete conflict-resolution rule (keep both values, resolve only on a margin of 0.18 or more).

**Weaknesses and inconsistencies to be aware of**

| Issue | Detail |
|---|---|
| Figure versus text | Figure 1 shows a 512-D space, Top-80 recall, and two-hop expansion. The text specifies 640-D, Top-120, and three hops. Trust the text. |
| Context budget | 4,096 tokens in most places, 6,144 in the optimization constraint (Eq. 13). |
| Item caps | 12 evidence units, 18 after re-ranking, 28 fragments, and 6 per group appear at different stages; the funnel is coherent but poorly signposted. |
| Initial retrieval index | The text describes FAISS IVFPQ with Top-120 recall in one place and an HNSW index with 180 candidates in another. The pseudocode uses Top-120. |
| Latency | 211 ms in the task table versus 283 ms in the abstract and ablations. |
| Cross-references | Several point to the wrong figure or table (for example a results discussion citing Figure 8 where Figure 9 is meant). |
| Sample sizes | The tri-modal headline result uses 40 queries. |
| Data access | The corpus is proprietary and single-domain (electrical equipment); data is available only on request. |
| Novelty over standard GraphRAG | The measured margin is about 1 to 1.5 points. |
| No lexical baseline inside the pipeline | The paper's first stage is vector recall only. BM25 appears solely as a weak baseline. |

---

# Part III: Bringing the paper's ideas into this repo

## 13. Side-by-side comparison

| Dimension | This repo | The paper |
|---|---|---|
| Data | Text chunks from PDFs | Text, image regions, table records |
| Retrieved unit | Chunk | Document-entity-relation-path unit |
| First-stage recall | BM25 and/or dense, fused with RRF | Dense vector recall only |
| Combining signals | RRF, linear blend, adaptive heuristic | Learned weighted scoring plus graph diffusion |
| Structure | None (flat chunk list) | Knowledge graph in Neo4j |
| Redundancy control | Jaccard dedup, MMR | Redundancy penalty plus modal-complementarity reward |
| Re-ranking | Cross-encoder over a wide pool | Six-layer cross-encoder over 72 candidates |
| Generation control | Grounded prompts, bracketed source headers | Mandatory citations, thresholded rejection, conflict flags |
| Evaluation | 14 queries, MRR / Recall@1-3-5 / NDCG@5 | 720 test queries, 11+ metrics incl. hallucination and no-answer accuracy |
| Cost | Low, runs locally | Graph database, several encoders, 283 ms typical |

## 14. Where the repo already aligns with the paper

| Paper idea | Repo counterpart (implemented) |
|---|---|
| Wide first-stage pool then re-rank | Cross-encoder over a wide candidate pool |
| Redundancy suppression | Jaccard deduplication |
| Relevance-plus-novelty selection (Eq. 12) | MMR |
| Provenance for every fact | Context builder with bracketed source headers |
| Grounded, citation-oriented generation | Grounded instruction prompt templates and a citation-producing local generator |

## 15. Proposed extensions (not implemented)

These are suggestions for adopting the paper's ideas incrementally, ordered from cheapest to most involved. They fit the repo's decoupled structure.

**1. Broaden evaluation first (cheap, high value)**
Add metrics the paper uses that the harness lacks: evidence hit rate (any gold chunk in the top 10), Precision@k, and, once generation is evaluated, citation accuracy and hallucination rate. Add a small set of **unanswerable queries** to measure no-answer accuracy. Extend the 14-query set, since small samples limit every conclusion in Section 5.
*Where:* [`src/evaluation/`](../src/evaluation) (`metrics.py`, `dataset.py`).

**2. Add rejection to generation (cheap)**
Introduce a minimum evidence-score threshold below which the generator answers "insufficient evidence," and validate that every claim carries a source tag. The paper's ablations show this is the single most effective hallucination control.
*Where:* [`src/generation/`](../src/generation) (`context.py`, `prompts.py`).

**3. Make MMR complementarity-aware (cheap)**
Mirror Eq. 12 by adding a bonus when a candidate comes from a different section or document than those already selected. That is the text-only analogue of the paper's modal-complementarity reward.
*Where:* [`src/retrieval/postprocessing/`](../src/retrieval/postprocessing).

**4. A lightweight chunk graph as a 13th strategy (moderate)**
Build an in-memory graph (for example with `networkx`) where nodes are chunks and entities (method names, acronyms, section headings) and edges are containment, co-occurrence, and adjacency. Expand from the top RRF hits by one to three hops, then run diffusion. Compare against `rrf_dedup_mmr` on the harness. This tests the paper's central claim on a real corpus without a graph database.
*Where:* a new `src/retrieval/graph/` module, registered in the retrieval pipeline dispatch.

**5. Cross-modal ingestion (larger)**
Extend ingestion to extract figures and tables from the PDFs, encode them, and link them to the surrounding text by page proximity and heading, which is the same positional signal as the paper's correction term in Eq. 3.
*Where:* [`src/ingestion/`](../src/ingestion).

**6. Ablation harness (moderate)**
Add flags to switch off individual components (dedup, MMR, graph expansion, rejection) and report the deltas, following the paper's ablation design.

**Suggested improvement to the paper's design:** its first stage is dense-only. Given this repo's finding that BM25 plus fusion is hard to beat on term-heavy technical text, starting graph expansion from **RRF-fused sparse and dense hits** is a plausible improvement. This is an inference from the two sources, not something either one tests.

## 16. Sketch: graph expansion with diffusion

Illustrative pseudocode for proposal 4. It is not part of the repo.

```python
import numpy as np

def graph_expand(seed_scores, adj, hops=3, alpha=0.57, rounds=5):
    """
    seed_scores: dict node -> initial score (e.g. from rrf_dedup_mmr hits)
    adj:         scipy/numpy adjacency matrix over chunk+entity nodes
    """
    nodes = list(seed_scores)
    s0 = np.zeros(adj.shape[0]); 
    for n, v in seed_scores.items():
        s0[n] = v

    # degree-normalise: A* = D^-1/2 A D^-1/2
    d = np.asarray(adj.sum(axis=1)).ravel()
    d_inv_sqrt = np.where(d > 0, d ** -0.5, 0.0)
    A = adj.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :])

    s = s0.copy()
    for _ in range(rounds):
        s = alpha * (A @ s) + (1 - alpha) * s0   # personalised-PageRank style diffusion

    return s   # blend with the original score, then dedup / MMR as before
```

Restrict the graph to `hops` neighbors of the seeds, prune weak edges, and cap the node count as the paper does (about 300 nodes, edges below a weight threshold), so the added latency stays bounded.

---

## Decision guide

1. **Start with BM25 plus dense retrieval fused by RRF.** Cheap, robust, and strong on this repo's benchmark.
2. **Add dedup and MMR** when results are repetitive or the context window is tight.
3. **Add a cross-encoder** only after measuring. It did not help on this repo's corpus and costs 64 to 79 ms in the paper's setup.
4. **Add a graph** when questions chain facts, entities appear under many aliases, or you need explainable evidence paths.
5. **Go multimodal** only when answers genuinely live in images or tables, and budget for the extra latency.
6. **Constrain generation in every case:** require citations and allow "insufficient evidence."
7. **Benchmark on your own data.** The same baselines rank in opposite orders across the two corpora discussed here.

---

## Glossary

| Term | Meaning |
|---|---|
| Sparse retrieval | Word-matching methods such as BM25 and TF-IDF |
| Dense retrieval | Embedding-based nearest-neighbor search |
| Bi-encoder | Encodes query and document separately; fast |
| Cross-encoder | Reads query and document together; accurate but slower |
| RRF | Rank fusion using positions only |
| MMR | Re-ranking that balances relevance and novelty |
| Knowledge graph | Entities (nodes) connected by typed relations (edges) |
| Hop | One step along a graph edge |
| Graph diffusion | Spreading relevance scores across connected nodes |
| Alias merging | Recognizing that different mentions are one entity |
| Hard negative | A tricky wrong example, similar to the right one, used in training |
| Broken evidence chain | Needed evidence exists but retrieval fails to connect it |
| No-hit rejection | Declining to answer when evidence is too weak |
| MRR | Mean of 1/rank of the first correct result |
| Recall@k / Precision@k | Share of relevant items found in the top k / share of the top k that is relevant |
| NDCG@k | Ranking quality that rewards relevant items placed higher |
| Evidence hit rate | Share of queries with at least one gold evidence unit in the top 10 |
| Answer fidelity | Share of answer claims supported by retrieved evidence |

---

## Sources and licensing note

- Liao, C., Ye, P., Huang, A., Huang, S., Yuan, X. *A study on the GraphRAG semantic retrieval algorithm for multimodal data.* Discover Artificial Intelligence 6:1060 (2026). https://doi.org/10.1007/s44163-026-02023-3
- This repository's [README](../README.md) for the strategy list, architecture, and benchmark table.

The paper is published under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 license. This document is an independent explanation and analysis that paraphrases the paper's ideas and reports its numbers with attribution. It does not reproduce the paper's text or figures. Anyone extending it should keep to that approach and check the license terms for their use.
