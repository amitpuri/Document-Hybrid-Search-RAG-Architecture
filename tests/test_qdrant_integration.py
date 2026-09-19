"""
Unit and integration tests for Qdrant storage and vector retrieval.
Tests QdrantChunkStore, QdrantRetriever, filtering, and BaseChunkStore.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
from src.common.console import ensure_utf8_streams
from src.common.types import DocumentChunk
from src.ingestion.storage import _QDRANT_AVAILABLE, QdrantChunkStore
from src.retrieval.retrievers.qdrant import QdrantRetriever

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure UTF-8 output on Windows
ensure_utf8_streams()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except ImportError:
    QdrantClient = None
    qmodels = None


class TestQdrantIntegration(unittest.TestCase):
    """Test suite for Qdrant chunk store and retriever."""

    def setUp(self):
        if not _QDRANT_AVAILABLE:
            self.skipTest("qdrant-client not installed")

        # Use an isolated in-memory client for fast, deterministic testing
        self.client = QdrantClient(":memory:")
        self.collection_name = "test_chunks"
        self.vector_size = 4

        self.sample_chunks = [
            DocumentChunk(
                chunk_id=0,
                doc_name="paper_a.pdf",
                page_num=1,
                section="Introduction",
                text=("Deep learning architectures for retrieval " "augmented generation."),
                metadata={"category": "ai"},
            ),
            DocumentChunk(
                chunk_id=1,
                doc_name="paper_a.pdf",
                page_num=2,
                section="Methods",
                text=("Reciprocal rank fusion combines " "sparse and dense rankings."),
                metadata={"category": "ir"},
            ),
            DocumentChunk(
                chunk_id=2,
                doc_name="paper_b.pdf",
                page_num=1,
                section="Overview",
                text=("Knowledge graph traversal enhances " "POMDP belief state filtering."),
                metadata={"category": "robotics"},
            ),
        ]

        self.sample_vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]

    def test_chunk_store_lifecycle(self):
        """
        Test creating QdrantChunkStore, adding chunks, and
        retrieving them.
        """
        store = QdrantChunkStore(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            client=self.client,
            chunks=self.sample_chunks,
        )

        self.assertEqual(len(store), 3)

        # Single get
        c0 = store.get_chunk(0)
        self.assertIsNotNone(c0)
        self.assertEqual(c0.chunk_id, 0)
        self.assertEqual(c0.doc_name, "paper_a.pdf")
        self.assertEqual(c0.section, "Introduction")

        # Non-existent get
        self.assertIsNone(store.get_chunk(999))

        # Batch lookup
        batch = store.get_chunks([2, 0])
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].chunk_id, 2)
        self.assertEqual(batch[1].chunk_id, 0)

        # get_all_chunks maintains chunk_id ascending order
        all_chunks = store.get_all_chunks()
        self.assertEqual(len(all_chunks), 3)
        self.assertEqual([c.chunk_id for c in all_chunks], [0, 1, 2])

        # get_texts format
        texts = store.get_texts()
        self.assertEqual(len(texts), 3)
        self.assertTrue(texts[0].startswith("[paper_a.pdf | Page 1 | § Introduction]"))

    def test_server_side_filtering(self):
        """Test metadata filtering via Qdrant FieldConditions."""
        store = QdrantChunkStore(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            client=self.client,
            chunks=self.sample_chunks,
        )

        # Filter by doc_name
        results_a = store.filter(doc_name="paper_a.pdf")
        self.assertEqual(len(results_a), 2)
        self.assertTrue(all(c.doc_name == "paper_a.pdf" for c in results_a))

        # Filter by page_num
        results_p2 = store.filter(page_num=2)
        self.assertEqual(len(results_p2), 1)
        self.assertEqual(results_p2[0].chunk_id, 1)

        # Filter by section
        results_sec = store.filter(section="Methods")
        self.assertEqual(len(results_sec), 1)
        self.assertEqual(results_sec[0].section, "Methods")

    def test_vector_updates_and_retrieval(self):
        """Test updating vectors and querying through QdrantRetriever."""
        store = QdrantChunkStore(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            client=self.client,
            chunks=self.sample_chunks,
        )

        # Update vectors
        store.update_vectors([0, 1, 2], self.sample_vectors)

        # Initialize retriever with existing client
        retriever = QdrantRetriever(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            client=self.client,
        )
        retriever._corpus_size = len(self.sample_chunks)

        # Mock model.encode to return sample vector
        class MockModel:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0, 0.0, 0.0]])

        retriever.model = MockModel()

        scores = retriever.score("dummy query")
        self.assertEqual(len(scores), 3)
        # Point 0 has exact match [1, 0, 0, 0] -> score should be highest
        self.assertEqual(np.argmax(scores), 0)
        self.assertAlmostEqual(scores[0], 1.0, places=3)

        # Top-k ANN search
        top_results = retriever.search_top_k("dummy query", top_k=2)
        self.assertEqual(len(top_results), 2)
        self.assertEqual(top_results[0].chunk.chunk_id, 0)
        self.assertEqual(top_results[0].rank, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
