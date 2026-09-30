"""
Evaluation Harness for Benchmarking All 15 Retrieval Strategies.
"""

from typing import Dict, List, Optional

import numpy as np

from src.common.types import MetricScores
from src.evaluation.dataset import EVAL_DATASET, validate_ground_truth
from src.evaluation.metrics import evaluate_ranking
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.pipeline import RetrievalPipeline


class EvaluationHarness:
    """Orchestrates quantitative evaluation over the benchmark dataset."""

    def __init__(
        self,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
        dataset: Optional[List[Dict]] = None,
        storage_backend: str = "parquet",
    ):
        self.pipeline = retrieval_pipeline
        self.dataset = dataset or EVAL_DATASET
        self.storage_backend = storage_backend

    def run(self) -> Dict[str, Dict[str, float]]:
        """
        Runs evaluation across all dataset queries and active strategies.
        Returns a dictionary of aggregated metrics per strategy.
        """
        if self.pipeline is None:
            print("Running ingestion pipeline...")
            ingestion = IngestionPipeline(storage_backend=self.storage_backend)
            chunk_store, total_pages, pdf_paths = ingestion.run()
            print(
                f"Loaded {len(pdf_paths)} PDFs | Total Pages: {total_pages} | "
                f"Chunks: {len(chunk_store)}\n"
            )

            self.pipeline = RetrievalPipeline(chunk_store)

        corpus_texts = self.pipeline.corpus_texts

        # TODO (P3 — Evaluation Rigor): With n=14 queries, a single flipped
        # query shifts MRR by 1/14 ≈ 0.071 — larger than several reported
        # gaps between strategies. Should be treated as directional, not
        # statistically robust. Future improvements:
        #   1. Expand src/evaluation/dataset.py to 50+ queries for
        #      tighter confidence.
        #   2. Add bootstrap confidence intervals to the output table
        #      so README numbers report uncertainty.
        #   3. Audit whether DEFAULT_RRF_K=60,
        #      DEFAULT_DEDUP_THRESHOLD=0.65, and DEFAULT_MMR_LAMBDA=0.7
        #      in config.py were tuned against these same 14 queries.
        #      If so, split into a dev set (for tuning) and a held-out
        #      test set (for reported numbers) to avoid overfitting.
        # 1. Sanity check ground-truth
        validate_ground_truth(self.dataset, corpus_texts)
        print("Ground-truth sanity check passed.\n")

        # 2. Index pipeline models
        print("Indexing retrieval models (BM25, TF-IDF, Graph, PPMI, Neural)...")
        self.pipeline.index()

        # 3. Determine active strategy names (sorted by pipeline)
        sample_query = self.dataset[0]["query"]
        sample_rankings = self.pipeline.get_strategy_rankings(sample_query)
        strategy_names = list(sample_rankings.keys())

        results: Dict[str, List[MetricScores]] = {name: [] for name in strategy_names}

        multihop_indices = [i for i, item in enumerate(self.dataset) if item.get("is_multihop")]
        print(
            f"Evaluating {len(self.dataset)} queries "
            f"({len(multihop_indices)} multi-hop) x {len(strategy_names)} "
            f"strategies...\n"
        )

        for item in self.dataset:
            q_text = item["query"]
            target_doc = item["target_doc"]
            target_chunk_idx = item.get("target_chunk_idx")
            target_entities = item.get("target_entities")
            target_relations = item.get("target_relations")

            try:
                rankings = self.pipeline.get_strategy_rankings(q_text)
            except Exception as exc:
                print(
                    f"[WARNING] Failed to retrieve rankings for query "
                    f"{q_text!r}: {exc}. Skipping query."
                )
                continue

            for name in strategy_names:
                ranked = rankings.get(name, [])
                try:
                    metrics = evaluate_ranking(
                        ranked,
                        corpus_texts,
                        target_doc,
                        target_chunk_idx=target_chunk_idx,
                        target_entities=target_entities,
                        target_relations=target_relations,
                    )
                    results[name].append(metrics)
                except Exception as exc:
                    print(
                        f"[WARNING] Evaluation failed for query {q_text!r} "
                        f"on strategy {name!r}: {exc}"
                    )

        # 4. Aggregate & Print Summary Table
        print(
            f"\n{'Retrieval Strategy':<38}"
            f"{'MRR':<8}{'R@1':<8}{'R@3':<8}{'R@5':<8}{'NDCG@5':<9}"
            f"{'EntCov':<9}{'RelCov':<8}"
        )
        print("=" * 96)

        summary_metrics: Dict[str, Dict[str, float]] = {}

        for name in strategy_names:
            ml = results[name]
            if not ml:
                continue
            avg_mrr = float(np.mean([m.mrr for m in ml]))
            avg_r1 = float(np.mean([m.recall_1 for m in ml]))
            avg_r3 = float(np.mean([m.recall_3 for m in ml]))
            avg_r5 = float(np.mean([m.recall_5 for m in ml]))
            avg_ndcg = float(np.mean([m.ndcg_5 for m in ml]))
            avg_ent = float(np.mean([m.entity_coverage for m in ml]))
            avg_rel = float(np.mean([m.relation_coverage for m in ml]))

            summary_metrics[name] = {
                "mrr": avg_mrr,
                "recall_1": avg_r1,
                "recall_3": avg_r3,
                "recall_5": avg_r5,
                "ndcg_5": avg_ndcg,
                "entity_coverage": avg_ent,
                "relation_coverage": avg_rel,
            }

            print(
                f"{name:<38}"
                f"{avg_mrr:<8.3f}"
                f"{avg_r1:<8.3f}"
                f"{avg_r3:<8.3f}"
                f"{avg_r5:<8.3f}"
                f"{avg_ndcg:<9.3f}"
                f"{avg_ent:<9.3f}"
                f"{avg_rel:<8.3f}"
            )

        # 5. Multi-Hop Query Breakdown for Key Strategies
        if multihop_indices:
            print("\n" + "-" * 96)
            print(
                f"Multi-Hop Benchmark Breakdown "
                f"({len(multihop_indices)} complex reasoning queries):"
            )
            print(
                f"{'Retrieval Strategy':<38}{'MRR':<8}{'R@1':<8}{'R@3':<8}"
                f"{'R@5':<8}{'NDCG@5':<9}{'EntCov':<9}{'RelCov':<8}"
            )
            print("-" * 96)
            for name in strategy_names:
                ml = results[name]
                if not ml:
                    continue
                mh_ml = [ml[i] for i in multihop_indices if i < len(ml)]
                if not mh_ml:
                    continue
                mh_mrr = float(np.mean([m.mrr for m in mh_ml]))
                mh_r1 = float(np.mean([m.recall_1 for m in mh_ml]))
                mh_r3 = float(np.mean([m.recall_3 for m in mh_ml]))
                mh_r5 = float(np.mean([m.recall_5 for m in mh_ml]))
                mh_ndcg = float(np.mean([m.ndcg_5 for m in mh_ml]))
                mh_ent = float(np.mean([m.entity_coverage for m in mh_ml]))
                mh_rel = float(np.mean([m.relation_coverage for m in mh_ml]))
                print(
                    f"{name:<38}"
                    f"{mh_mrr:<8.3f}"
                    f"{mh_r1:<8.3f}"
                    f"{mh_r3:<8.3f}"
                    f"{mh_r5:<8.3f}"
                    f"{mh_ndcg:<9.3f}"
                    f"{mh_ent:<9.3f}"
                    f"{mh_rel:<8.3f}"
                )

        print()
        return summary_metrics
