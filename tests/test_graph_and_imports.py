"""
Unit and smoke tests for the Knowledge Graph retrieval extension,
graph storage roundtrip, and package-wide import sanity.
"""

import importlib
from pathlib import Path
import pkgutil
import pytest
import numpy as np

from src.common.types import DocumentChunk
from src.ingestion.graph_extractor import HeuristicGraphExtractor, Entity, Relationship
from src.ingestion.graph_store import NetworkXGraphStore
from src.ingestion.storage import InMemoryChunkStore
from src.retrieval.retrievers.graph import GraphRetriever


def test_import_all_src_modules():
    """
    Smoke test: walks and imports every module under the `src` package.
    Catches any NameError, ImportError, or missing typing annotations at collection time.
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    assert src_dir.exists(), f"Source directory {src_dir} not found"

    imported_count = 0
    failures = []

    for module_info in pkgutil.walk_packages([str(src_dir)], prefix="src."):
        mod_name = module_info.name
        try:
            mod = importlib.import_module(mod_name)
            assert mod is not None
            imported_count += 1
        except Exception as exc:
            failures.append((mod_name, str(exc)))

    assert not failures, f"Failed to import {len(failures)} module(s): {failures}"
    assert imported_count >= 20, f"Expected at least 20 modules to be imported, found {imported_count}"


@pytest.fixture
def sample_technical_chunks():
    """A minimal synthetic fixture of technical chunks with known concepts and relationships."""
    texts = [
        "Partially Observed Markov Decision Processes (POMDP) model sequential decision problems under uncertainty.",
        "A POMDP maintains a belief state filtering probability distribution over latent world states.",
        "StarShell provides a terminal automation agent interface for enterprise workflows and secure CLI execution.",
        "The Binding Constraint Thesis states that execution harness bottlenecks dominate LLM reasoning benchmarks.",
        "In enterprise settings, AgentRunner applies Risk Adaptive Tiering to govern automated actions.",
        "Dynamic Tiered AgentRunner Framework coordinates multi-agent subtasks with policy enforcement.",
    ]
    return [
        DocumentChunk(
            chunk_id=i,
            doc_name=f"paper_{i // 2}.pdf",
            page_num=1 + (i % 3),
            section=f"Section {i + 1} Architecture",
            text=text,
        )
        for i, text in enumerate(texts)
    ]


def test_heuristic_extractor_and_store_roundtrip(sample_technical_chunks, tmp_path):
    """
    Tests that HeuristicGraphExtractor extracts non-zero entities and relationships,
    and that NetworkXGraphStore can add, atomically save to JSON, and reload them losslessly.
    """
    extractor = HeuristicGraphExtractor()
    entities, relationships = extractor.extract_from_chunks(sample_technical_chunks)

    # Assert non-zero and bounded entities & relationships
    assert len(entities) > 0, "Expected non-zero extracted entities"
    assert len(entities) < 100, f"Expected bounded entity count, got {len(entities)}"
    assert len(relationships) > 0, "Expected non-zero extracted relationships"
    assert len(relationships) < 200, f"Expected bounded relationship count, got {len(relationships)}"

    # Check that known technical entities appear
    entity_names = {e.name for e in entities}
    assert any("POMDP" in name for name in entity_names), "POMDP should be extracted"
    assert any("StarShell" in name for name in entity_names), "StarShell should be extracted"

    # Populate graph store
    store = NetworkXGraphStore()
    for ent in entities:
        store.add_entity(ent)
    for rel in relationships:
        store.add_relationship(rel)

    assert len(store) == len(entities)
    assert store.number_of_edges == len(relationships)

    # Save to disk and reload
    save_file = tmp_path / "graph_roundtrip.json"
    store.save(save_file)
    assert save_file.exists()

    loaded_store = NetworkXGraphStore()
    loaded_store.load(save_file)

    assert len(loaded_store) == len(store)
    assert loaded_store.number_of_edges == store.number_of_edges

    # Verify attributes and provenance are preserved
    sample_ent = store.get_all_entities()[0]
    loaded_ent = loaded_store.get_entity(sample_ent.name)
    assert loaded_ent is not None
    assert loaded_ent.entity_type == sample_ent.entity_type
    assert loaded_ent.chunk_ids == sample_ent.chunk_ids


def test_graph_retriever_index_and_score(sample_technical_chunks, tmp_path):
    """
    Tests that GraphRetriever indexes a chunk store and returns a same-length numpy array
    of finite float scores within [0.0, 1.0].
    """
    corpus_texts = [c.text for c in sample_technical_chunks]
    chunk_store = InMemoryChunkStore(sample_technical_chunks)

    retriever = GraphRetriever(cache_dir=tmp_path)
    retriever.index(corpus_texts=corpus_texts, chunk_store=chunk_store, force_rebuild=True)

    query = "POMDP belief state filtering"
    scores = retriever.score(query)

    assert isinstance(scores, np.ndarray), f"Expected np.ndarray, got {type(scores)}"
    assert len(scores) == len(sample_technical_chunks), "Scores array length must match corpus size"
    assert np.all(np.isfinite(scores)), "All scores must be finite"
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0), "Scores must be normalized in [0, 1]"

    # Chunks 0 and 1 mention POMDP and belief state filtering, so at least one should score > 0
    assert scores[0] > 0.0 or scores[1] > 0.0, "Relevant chunks should receive positive graph scores"
